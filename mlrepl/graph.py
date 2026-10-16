"""Typed DAG representation of a procedural shape program.

Parameters are stored *unconstrained* (raw) so optimizers can act on them directly;
JSON stores human-readable constrained values. Nodes may be shared (DAG, not tree).
"""
from __future__ import annotations

import copy
import itertools
import json
from typing import Iterable, Optional

import torch

from .ops import OPS, EvalCtx

# Description-length weights (bits-ish). Used as the structural prior in search.
NODE_COST = 4.0
PARAM_COST = 0.5
INSTANCE_PARAM_COST = 0.5  # same as any param; instancing saves by sharing the prototype + node overhead


class Node:
    __slots__ = ("id", "op", "inputs", "raw", "attrs")

    def __init__(self, id: str, op: str, inputs: Iterable[str] = (), params: Optional[dict] = None,
                 attrs: Optional[dict] = None, k: Optional[int] = None):
        spec = OPS[op]
        self.id, self.op, self.inputs = id, op, list(inputs)
        self.attrs = dict(spec.attrs, **(attrs or {}))
        params = dict(params or {})
        if spec.category == "inst" and k is None:
            k = len(params["t"]) if "t" in params else 1
        self.raw = {}
        for name, ps in spec.params.items():
            if name in params:
                val = torch.as_tensor(params[name], dtype=torch.float32)
                if ps.per_instance:
                    val = val.reshape((-1,) + tuple(ps.shape))
                else:
                    val = val.reshape(ps.shape)
            elif ps.optional:
                continue
            else:
                val = ps.default_value(k or 1)
            self.raw[name] = ps.to_raw(val)

    # -- values -------------------------------------------------------------
    @property
    def spec(self):
        return OPS[self.op]

    def values(self) -> dict:
        return {n: self.spec.params[n].from_raw(r) for n, r in self.raw.items()}

    @property
    def device(self) -> torch.device:
        return next(iter(self.raw.values())).device if self.raw else torch.device("cpu")

    def set_value(self, name: str, value):
        ps = self.spec.params[name]
        dev = self.device
        self.raw[name] = ps.to_raw(torch.as_tensor(value, dtype=torch.float32).to(dev))

    def to(self, device) -> "Node":
        self.raw = {k: v.detach().to(device) for k, v in self.raw.items()}
        return self

    def value(self, name: str) -> torch.Tensor:
        return self.spec.params[name].from_raw(self.raw[name])

    @property
    def k(self) -> int:  # instance count
        return int(self.raw["t"].shape[0]) if self.spec.category == "inst" else 1

    def nparams(self) -> int:
        return int(sum(r.numel() for r in self.raw.values()))

    def clone(self) -> "Node":
        n = Node.__new__(Node)
        n.id, n.op, n.inputs = self.id, self.op, list(self.inputs)
        n.raw = {k: v.detach().clone() for k, v in self.raw.items()}
        n.attrs = copy.deepcopy(self.attrs)
        return n

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": list(self.inputs),
            "params": {k: v.detach().cpu().tolist() for k, v in self.values().items()},
            "attrs": self.attrs,
        }


class Graph:
    def __init__(self, nodes: Iterable[Node] = (), output: Optional[str] = None, device=None):
        self.nodes: dict[str, Node] = {n.id: n for n in nodes}
        self.output = output
        self._ids = itertools.count()
        self.device = torch.device(device) if device is not None else (
            next(iter(self.nodes.values())).device if self.nodes else torch.device("cpu"))

    def to(self, device) -> "Graph":
        """Move all parameters to `device` in place (returns self)."""
        self.device = torch.device(device)
        for n in self.nodes.values():
            n.to(self.device)
        return self

    # -- construction helpers ----------------------------------------------
    def new_id(self, prefix: str) -> str:
        while True:
            nid = f"{prefix}{next(self._ids)}"
            if nid not in self.nodes:
                return nid

    def add(self, op: str, inputs: Iterable[str] = (), params=None, attrs=None, id: Optional[str] = None, k=None) -> str:
        nid = id or self.new_id(op[:3])
        self.nodes[nid] = Node(nid, op, inputs, params, attrs, k=k).to(self.device)
        return nid

    def placed(self, prim: str, pos=(0, 0, 0), rot=(0, 0, 0), scale=1.0, attrs=None, **prim_params) -> str:
        """prim -> transform, the most common motif. Returns the transform id."""
        p = self.add(prim, params=prim_params, attrs=attrs)
        return self.add("transform", [p], params={"t": list(pos), "r": list(rot), "s": [scale]})

    # -- structure ------------------------------------------------------------
    def topo(self, root: Optional[str] = None) -> list[str]:
        root = root or self.output
        order, seen = [], set()

        def visit(n):
            if n in seen:
                return
            seen.add(n)
            for c in self.nodes[n].inputs:
                visit(c)
            order.append(n)

        visit(root)
        return order

    def parents(self) -> dict[str, list[str]]:
        par = {n: [] for n in self.nodes}
        for n in self.reachable():
            for c in self.nodes[n].inputs:
                par[c].append(n)
        return par

    def reachable(self) -> list[str]:
        return self.topo()

    def prune_unreachable(self) -> "Graph":
        keep = set(self.reachable())
        self.nodes = {k: v for k, v in self.nodes.items() if k in keep}
        return self

    def replace_refs(self, old: str, new: str):
        for n in self.nodes.values():
            n.inputs = [new if i == old else i for i in n.inputs]
        if self.output == old:
            self.output = new

    def clone(self) -> "Graph":
        g = Graph([n.clone() for n in self.nodes.values()], self.output, device=self.device)
        return g

    # -- parameters -------------------------------------------------------------
    def raw_params(self) -> list[torch.Tensor]:
        return [r for nid in self.reachable() for r in self.nodes[nid].raw.values()]

    def requires_grad_(self, flag: bool = True) -> "Graph":
        for n in self.nodes.values():
            for k, r in n.raw.items():
                n.raw[k] = r.detach().requires_grad_(flag)
        return self

    # -- prior -----------------------------------------------------------------
    def complexity(self) -> float:
        c = 0.0
        for nid in self.reachable():
            n = self.nodes[nid]
            c += NODE_COST
            if n.spec.category == "inst":
                c += INSTANCE_PARAM_COST * sum(r.numel() for k, r in n.raw.items() if k != "logit")
            else:  # appearance params are not structure
                c += PARAM_COST * sum(r.numel() for k, r in n.raw.items() if k != "color")
        return c

    def counts(self) -> dict:
        out = {}
        for nid in self.reachable():
            op = self.nodes[nid].op
            out[op] = out.get(op, 0) + 1
        return out

    # -- evaluation ----------------------------------------------------------------
    def sdf(self, p: torch.Tensor, ctx: Optional[EvalCtx] = None, root: Optional[str] = None) -> torch.Tensor:
        ctx = ctx or EvalCtx()
        vals_cache: dict[str, dict] = {}

        def fn(nid):
            node = self.nodes[nid]
            if nid not in vals_cache:
                vals_cache[nid] = node.values()
            children = [fn(c) for c in node.inputs]
            spec = node.spec
            if ctx.color and spec.category == "prim":
                v = vals_cache[nid]
                col = v["color"] if "color" in v else torch.tensor(
                    node.attrs.get("material", {}).get("base_color", (0.7, 0.7, 0.7)), dtype=torch.float32,
                    device=self.device)

                def f(q):
                    d = spec.sdf(v, children, q, ctx)
                    return torch.cat([d[..., None], col.expand(*d.shape, 3)], -1)

                return f

            def f(q):
                return spec.sdf(vals_cache[nid], children, q, ctx)

            return f

        return fn(root or self.output)(p)

    __call__ = sdf

    # -- io ------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"output": self.output, "nodes": [self.nodes[n].to_dict() for n in self.topo()]}

    def to_json(self, path=None, indent=1) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        g = cls(output=d["output"])
        for nd in d["nodes"]:
            g.nodes[nd["id"]] = Node(nd["id"], nd["op"], nd.get("inputs", []), nd.get("params"), nd.get("attrs"))
        return g

    @classmethod
    def from_json(cls, s_or_path: str) -> "Graph":
        if s_or_path.strip().startswith("{"):
            return cls.from_dict(json.loads(s_or_path))
        with open(s_or_path) as f:
            return cls.from_dict(json.load(f))

    def pretty(self) -> str:
        lines = []

        def fmt_vals(n):
            if n.spec.category == "inst":
                alive = int((n.value("logit") >= 0).sum())
                return f"K={n.k} alive={alive}" + (" +deform" if "dk" in n.raw else "")
            parts = []
            for k, v in n.values().items():
                v = v.detach().flatten().tolist()
                parts.append(f"{k}=" + ",".join(f"{x:.3g}" for x in v))
            return " ".join(parts)

        def rec(nid, depth, seen):
            n = self.nodes[nid]
            shared = " (shared)" if nid in seen else ""
            lines.append("  " * depth + f"{n.op}[{nid}] {fmt_vals(n)}{shared}")
            if nid in seen:
                return
            seen.add(nid)
            for c in n.inputs:
                rec(c, depth + 1, seen)

        rec(self.output, 0, set())
        return "\n".join(lines)
