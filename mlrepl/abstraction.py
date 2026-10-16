"""Library learning: mine reusable sub-programs ("macros") from solved graphs, and use them as
structured proposals during search.

v1 (this file): exact-skeleton subtree mining with compression gain, DreamCoder-flavoured but
without anti-unification over op choices. A macro is a complete parameterized subtree; using it
inlines a fresh copy (so the program stays a plain executable graph) under a transform fitted to
the residual region. Parameter values from the mined occurrences are kept as the init prior.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch

from .graph import NODE_COST, Graph
from .ops import EvalCtx


@dataclass
class Macro:
    name: str
    signature: str
    template: Graph  # template.output is the macro root
    count: int
    programs: int  # number of distinct programs it occurs in
    gain: float
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    extent: float = 1.0

    def to_dict(self):
        return {"name": self.name, "signature": self.signature, "count": self.count, "programs": self.programs,
                "gain": self.gain, "center": self.center.tolist(), "extent": self.extent,
                "template": self.template.to_dict()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["signature"], Graph.from_dict(d["template"]), d["count"], d["programs"], d["gain"],
                   np.asarray(d["center"]), d["extent"])


def _subtree(g: Graph, root: str) -> Graph:
    keep = g.topo(root)
    sub = Graph([g.nodes[n].clone() for n in keep], root)
    return sub


def _skeleton(g: Graph, nid: str) -> str:
    n = g.nodes[nid]
    tag = n.op + ("+dk" if "dk" in n.raw else "")
    if not n.inputs:
        return tag
    kids = [_skeleton(g, c) for c in n.inputs]
    if n.op in ("union", "intersect"):
        kids.sort()
    return f"{tag}({','.join(kids)})"


def _extent(g: Graph, n=4096):
    P = torch.rand(n, 3) * 4 - 2
    with torch.no_grad():
        inside = P[g.sdf(P, EvalCtx(hard_gates=True)) < 0].numpy()
    if len(inside) < 8:
        return np.zeros(3), 1.0
    return inside.mean(0), float(np.linalg.norm(inside.max(0) - inside.min(0)))


def mine(programs: list[Graph], min_nodes: int = 3, max_nodes: int = 10, min_count: int = 2,
         top_k: int = 8) -> list[Macro]:
    """Frequent complete subtrees across a corpus, ranked by compression gain
    gain = (count - 1) * NODE_COST * nodes  (bits saved if the subtree became one library call)."""
    occ: dict[str, list] = {}
    for pi, g in enumerate(programs):
        for nid in g.reachable():
            size = len(g.topo(nid))
            if size < min_nodes or size > max_nodes:
                continue
            occ.setdefault(_skeleton(g, nid), []).append((pi, g, nid, size))
    macros = []
    for sig, lst in occ.items():
        if len(lst) < min_count:
            continue
        size = lst[0][3]
        progs = len({pi for pi, *_ in lst})
        gain = (len(lst) - 1) * NODE_COST * size
        # template = the occurrence with median parameter norm (a typical instance)
        subs = [_subtree(g, nid) for _, g, nid, _ in lst]
        tmpl = subs[len(subs) // 2]
        c, e = _extent(tmpl)
        macros.append(Macro(f"m{len(macros)}", sig, tmpl, len(lst), progs, gain, c, e))
    # drop macros strictly contained in a better macro with the same occurrence count
    macros.sort(key=lambda m: (-m.gain, -m.programs))
    out = []
    for m in macros:
        if any(m.signature in o.signature and m.count <= o.count for o in out):
            continue
        out.append(m)
        if len(out) == top_k:
            break
    for i, m in enumerate(out):
        m.name = f"m{i}"
    return out


def save_library(macros: list[Macro], path: str):
    with open(path, "w") as f:
        json.dump([m.to_dict() for m in macros], f, indent=1)


def load_library(path: str) -> list[Macro]:
    with open(path) as f:
        return [Macro.from_dict(d) for d in json.load(f)]


def instantiate(g: Graph, macro: Macro, part: np.ndarray) -> str:
    """Inline a copy of the macro into g, placed/scaled onto the point cluster. Returns the new root id."""
    idmap = {}
    for nid in macro.template.topo():
        n = macro.template.nodes[nid].clone()
        new = g.new_id("mac")
        idmap[nid] = new
        n.id = new
        n.inputs = [idmap[i] for i in n.inputs]
        g.nodes[new] = n
    root = idmap[macro.template.output]
    c = part.mean(0)
    ext = float(np.linalg.norm(part.max(0) - part.min(0))) if len(part) > 1 else macro.extent
    s = max(ext / max(macro.extent, 1e-6), 1e-3)
    t = c - s * macro.center
    return g.add("transform", [root], params={"t": t.tolist(), "r": [0, 0, 0], "s": [s]})
