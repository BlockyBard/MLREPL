"""Structure search over graphs: data-guided mutations + inner continuous fit + MDL-style score.

score(G) = (1-w)(1 - IoU_hard(G)) + w (1 - surface_recall(G)) + beta * complexity(G)

IoU is measured with exact CSG semantics (tau = 0, hard gates), so every score is the score
of a realizable mesh. Mutations are proposed from residuals (missing volume -> add, spurious
volume -> subtract) and from structural rewrites (instancify, deform, swap, remove).
Per-mutation acceptance statistics are recorded so mutation strategies can be compared.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.transform import Rotation

from .fit import FitConfig, FitResult, fit, score
from .graph import Graph
from .mathutil import DEFORM_KINDS
from .ops import PRIMITIVES, EvalCtx
from .targets import Target

# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------


@dataclass
class Residual:
    fn_parts: list  # list of (N,3) arrays: connected regions of missing volume
    fp_parts: list  # spurious volume
    fn_frac: float
    fp_frac: float


def components(pts: np.ndarray, lo, hi, res=24, min_pts=8) -> list:
    if len(pts) < min_pts:
        return []
    idx = np.clip(((pts - lo) / (hi - lo) * res).astype(int), 0, res - 1)
    grid = np.zeros((res,) * 3, bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    pl = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
    parts = [pts[pl == i] for i in range(1, n + 1)]
    parts = [p for p in parts if len(p) >= min_pts]
    return sorted(parts, key=len, reverse=True)


def residual(g: Optional[Graph], target: Target) -> Residual:
    P = target.eval_points
    gt = (target.eval_occ > 0.5).detach().cpu().numpy()
    if g is None:
        pred = np.zeros_like(gt)
    else:
        with torch.no_grad():
            pred = (g.sdf(P, EvalCtx(tau=0.0, hard_gates=True)) < 0).detach().cpu().numpy()
    Pn = P.detach().cpu().numpy()
    lo, hi = target.lo.detach().cpu().numpy(), target.hi.detach().cpu().numpy()
    n_in = max(gt.sum(), 1)
    fn, fp = gt & ~pred, ~gt & pred
    return Residual(components(Pn[fn], lo, hi), components(Pn[fp], lo, hi), fn.sum() / n_in, fp.sum() / n_in)


# ---------------------------------------------------------------------------
# Primitive initialization from a point cluster (moment matching)
# ---------------------------------------------------------------------------


def _frame(xa, ya, za):
    M = np.stack([xa, ya, za], 1)
    if np.linalg.det(M) < 0:
        M[:, 2] *= -1
    return Rotation.from_matrix(M).as_rotvec()


def prim_from_points(kind: str, pts: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return (prim params, translation, axis-angle) whose solid has the cluster's 2nd moments."""
    c = pts.mean(0)
    C = np.cov((pts - c).T) + 1e-6 * np.eye(3)
    w, V = np.linalg.eigh(C)  # ascending
    v3, v2, v1 = np.maximum(w, 1e-6)
    e3, e2, e1 = V[:, 0], V[:, 1], V[:, 2]
    if kind == "sphere":
        return {"r": [float(np.sqrt(5 * w.mean()))]}, c, np.zeros(3)
    if kind == "box":
        return {"b": np.sqrt(3 * np.array([v2, v1, v3])).tolist(), "round": [1e-3]}, c, _frame(e2, e1, e3)
    if kind in ("cylinder", "capsule"):
        r = float(2 * np.sqrt(0.5 * (v2 + v3)))
        h = float(np.sqrt(3 * v1))
        if kind == "capsule":
            h = max(h - 0.5 * r, 0.01)
        return {"r": [r], "h": [h]}, c, _frame(e2, e1, e3)
    if kind == "torus":
        R = float(np.sqrt(v1 + v2))
        r = float(max(2 * np.sqrt(v3), 0.15 * R))
        return {"R": [R], "r": [min(r, 0.6 * R)]}, c, _frame(e1, e3, e2)
    raise ValueError(kind)


def _prim_extents(kind, v):
    if kind == "sphere":
        return np.full(3, v["r"][0])
    if kind == "box":
        return np.asarray(v["b"])
    if kind == "cylinder":
        return np.array([v["r"][0], v["h"][0], v["r"][0]])
    if kind == "capsule":
        return np.array([v["r"][0], v["h"][0] + v["r"][0], v["r"][0]])
    if kind == "torus":
        e = v["R"][0] + v["r"][0]
        return np.array([e, v["r"][0], e])
    raise ValueError(kind)


def _prim_from_extents(kind, e):
    rxz = float(0.5 * (e[0] + e[2]))
    if kind == "sphere":
        return {"r": [float(np.mean(e))]}
    if kind == "box":
        return {"b": [float(x) for x in e], "round": [1e-3]}
    if kind == "cylinder":
        return {"r": [rxz], "h": [float(e[1])]}
    if kind == "capsule":
        return {"r": [rxz], "h": [max(float(e[1]) - rxz, 0.01)]}
    if kind == "torus":
        return {"R": [0.7 * rxz], "r": [max(0.3 * rxz, 0.01)]}
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Graph surgery helpers
# ---------------------------------------------------------------------------


def _placed(g: Graph, kind: str, pts: np.ndarray, attrs=None) -> str:
    prm, t, r = prim_from_points(kind, pts)
    return g.placed(kind, pos=t.tolist(), rot=r.tolist(), attrs=attrs, **prm)


def union_with(g: Graph, nid: str):
    root = g.nodes[g.output]
    if root.op == "union":
        root.inputs.append(nid)
    else:
        g.output = g.add("union", [g.output, nid])


def _parts(g: Graph) -> list:
    root = g.nodes[g.output]
    return list(root.inputs) if root.op == "union" else [g.output]


def _placed_prims(g: Graph):
    """(transform_id, prim_id) pairs where transform's only input is a primitive."""
    out = []
    for nid in g.reachable():
        n = g.nodes[nid]
        if n.op == "transform":
            c = g.nodes[n.inputs[0]]
            if c.spec.category == "prim":
                out.append((nid, c.id))
    return out


def signature(g: Graph) -> str:
    def rec(nid):
        n = g.nodes[nid]
        tag = n.op + ("+dk" if "dk" in n.raw else "")
        if n.spec.category == "pattern":
            tag += f"[{n.attrs.get('axis', n.attrs.get('n'))}]"
        if not n.inputs:
            return tag
        kids = [rec(c) for c in n.inputs]
        if n.op in ("union", "intersect"):
            kids.sort()
        return f"{tag}({','.join(kids)})"

    return rec(g.output)


# ---------------------------------------------------------------------------
# Mutations: (parent graph, residual, rng) -> child graph or None
# ---------------------------------------------------------------------------


def local_crop(part: np.ndarray, rng, min_pts=12) -> np.ndarray:
    """RANSAC-style locality: a ball around a random seed point of the region, so a primitive can
    lock onto one sub-part (a tabletop) instead of moment-matching the whole region (table+legs)."""
    if len(part) < 2 * min_pts:
        return part
    ext = np.linalg.norm(part.max(0) - part.min(0))
    rad = ext * rng.uniform(0.15, 0.6)
    seed = part[rng.integers(len(part))]
    crop = part[np.linalg.norm(part - seed, axis=1) < rad]
    return crop if len(crop) >= min_pts else part


def _pick_part(parts, rng):
    part = parts[0] if rng.random() < 0.6 else parts[rng.integers(len(parts))]
    return local_crop(part, rng) if rng.random() < 0.5 else part


def best_kind(part: np.ndarray, target: Target, rng, g: Optional[Graph] = None, kinds=None, steps: int = 40) -> str:
    """Pick the primitive kind that best explains a residual region in isolation. The local problem
    uses the target's (surface-concentrated) volume samples inside the region's box, minus the points
    the current graph already covers (union makes those don't-care). Each kind is moment-matched,
    quick-fitted, and compared by local IoU. Cheap: a few hundred points, 40 steps."""
    from .fit import FitConfig, fit as _fit

    kinds = kinds or PRIMITIVES
    lo, hi = part.min(0), part.max(0)
    pad = 0.25 * (hi - lo) + 0.03
    dev = target.device
    lo_t = torch.tensor(lo - pad, dtype=torch.float32, device=dev)
    hi_t = torch.tensor(hi + pad, dtype=torch.float32, device=dev)
    P = target.points
    m = ((P >= lo_t) & (P <= hi_t)).all(1)
    if g is not None and m.any():
        with torch.no_grad():
            m[m.clone()] &= g.sdf(P[m], EvalCtx(hard_gates=True)) >= 0
    if m.sum() < 30 or target.occ[m].sum() < 10:
        return str(rng.choice(kinds))
    pts, occ = P[m], target.occ[m]
    S = target.surface
    ms = ((S >= lo_t) & (S <= hi_t)).all(1)
    surf = S[ms] if ms.sum() >= 8 else pts[occ > 0.5][:64]
    loc = Target(pts, occ, surf, lo_t, hi_t, pts, occ)
    best = (-1.0, kinds[0])
    for kind in kinds:
        h = Graph()
        h.output = _placed(h, kind, part)
        r = _fit(h, loc, FitConfig(steps=steps, nv=1024, ns=256, densify_every=10 ** 9))
        if r.iou > best[0]:
            best = (r.iou, kind)
    return best[1]


def m_add(g, res, rng, target=None):
    if not res.fn_parts:
        return None
    part = _pick_part(res.fn_parts, rng)
    kind = best_kind(part, target, rng, g) if (target is not None and rng.random() < 0.6) else str(rng.choice(PRIMITIVES))
    g = g.clone()
    union_with(g, _placed(g, kind, part))
    return g


def m_subtract(g, res, rng):
    if not res.fp_parts:
        return None
    part = _pick_part(res.fp_parts, rng)
    g = g.clone()
    cutter = _placed(g, str(rng.choice(["box", "cylinder", "sphere", "capsule"])), part)
    # subtract from the top-level part that currently covers most of the spurious region
    pts = torch.tensor(part, dtype=torch.float32, device=g.device)
    parts = _parts(g)
    with torch.no_grad():
        cover = [(g.sdf(pts, EvalCtx(hard_gates=True), root=p) < 0).float().mean().item() for p in parts]
    host = parts[int(np.argmax(cover))]
    sub = g.add("subtract", [host, cutter])
    if g.output == host:
        g.output = sub
    else:
        root = g.nodes[g.output]
        root.inputs = [sub if i == host else i for i in root.inputs]
    return g


def m_remove(g, res, rng):
    g = g.clone()
    cands = []
    for nid in g.reachable():
        n = g.nodes[nid]
        if n.op in ("union", "intersect") and len(n.inputs) >= 2:
            cands += [("drop", nid, i) for i in range(len(n.inputs))]
        if n.op == "subtract" or n.spec.category == "deform":
            cands.append(("bypass", nid, 0))
    if not cands:
        return None
    kind, nid, i = cands[rng.integers(len(cands))]
    n = g.nodes[nid]
    if kind == "drop":
        n.inputs.pop(i)
        if len(n.inputs) == 1:
            g.replace_refs(nid, n.inputs[0])
    else:
        g.replace_refs(nid, n.inputs[0])
    return g.prune_unreachable()


def m_swap(g, res, rng):
    pp = _placed_prims(g)
    if not pp:
        return None
    g = g.clone()
    _, pid = pp[rng.integers(len(pp))]
    old = g.nodes[pid]
    kind = str(rng.choice([k for k in PRIMITIVES if k != old.op]))
    ext = _prim_extents(old.op, {k: v.detach().cpu().numpy() for k, v in old.values().items()})
    from .graph import Node

    g.nodes[pid] = Node(pid, kind, [], _prim_from_extents(kind, ext), old.attrs)
    return g


DEFORM_SCAN = (-1.6, -0.9, -0.4, 0.4, 0.9, 1.6)


def m_deform(g, res, rng, target=None):
    """Wrap a placed primitive in a deformer. The strength is initialized by a cheap scan (no fitting):
    gradients from k ~ 0 rarely reach a large twist/bend, since the loss is flat-ish near zero."""
    pp = _placed_prims(g)
    if not pp:
        return None
    g = g.clone()
    tid, pid = pp[rng.integers(len(pp))]
    kind = str(rng.choice(DEFORM_KINDS))
    d = g.add(kind, [pid], params={"k": [float(rng.normal() * 0.2)]})
    g.nodes[tid].inputs[0] = d
    if target is not None:
        P = target.eval_points[:15000]
        gt = target.eval_occ[:15000] > 0.5
        best = (-1.0, None)
        for k in (float(g.nodes[d].value("k")),) + DEFORM_SCAN:
            g.nodes[d].set_value("k", torch.tensor([k]))
            with torch.no_grad():
                pr = g.sdf(P, EvalCtx(hard_gates=True)) < 0
            v = float((pr & gt).sum() / max(int((pr | gt).sum()), 1))
            if v > best[0]:
                best = (v, k)
        g.nodes[d].set_value("k", torch.tensor([best[1]]))
    return g


def m_to_instance(g, res, rng):
    """Turn a placed primitive into a 1-instance node so the fitter's densify step can grow it."""
    pp = [x for x in _placed_prims(g)]
    if not pp:
        return None
    g = g.clone()
    tid, pid = pp[rng.integers(len(pp))]
    t = g.nodes[tid].values()
    inst = g.add("instance", [pid], params={"t": [t["t"].tolist()], "r": [t["r"].tolist()], "s": [t["s"].tolist()],
                                            "logit": [3.0]})
    g.replace_refs(tid, inst)
    return g.prune_unreachable()


def m_instancify(g, res, rng):
    """Merge >=2 same-kind placed primitives under one union into a single instance node (MDL compression)."""
    g = g.clone()
    root = g.nodes[g.output]
    if root.op != "union":
        return None
    groups = {}
    for c in root.inputs:
        n = g.nodes[c]
        if n.op == "transform" and g.nodes[n.inputs[0]].spec.category == "prim":
            groups.setdefault(g.nodes[n.inputs[0]].op, []).append(c)
        if n.spec.category == "inst" and g.nodes[n.inputs[0]].spec.category == "prim":
            groups.setdefault(g.nodes[n.inputs[0]].op, []).append(c)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return None
    kind = str(rng.choice(sorted(groups)))
    members = groups[kind]
    poses, exts = [], []
    for c in members:
        n = g.nodes[c]
        pv = {k: v.detach().cpu().numpy() for k, v in g.nodes[n.inputs[0]].values().items()}
        e = _prim_extents(kind, pv)
        v = {k: x.detach().cpu().numpy() for k, x in n.values().items()}
        if n.op == "transform":
            poses.append((v["t"], v["r"], float(v["s"][0]), e))
        else:
            for i in range(n.k):
                poses.append((v["t"][i], v["r"][i], float(v["s"][i][0]), e))
        exts.append(e)
    proto_ext = np.exp(np.mean([np.log(p[3] * p[2]) for p in poses], 0))
    proto = g.add(kind, params=_prim_from_extents(kind, proto_ext))
    scales = [float(np.exp(np.mean(np.log(p[3] * p[2] / proto_ext)))) for p in poses]
    inst = g.add("instance", [proto], params={"t": [p[0].tolist() for p in poses], "r": [p[1].tolist() for p in poses],
                                              "s": [[s] for s in scales], "logit": [3.0] * len(poses)})
    root.inputs = [c for c in root.inputs if c not in members] + [inst]
    if len(root.inputs) == 1:
        g.replace_refs(root.id, inst)
    return g.prune_unreachable()


def m_instance_deform(g, res, rng):
    insts = [n for n in (g.nodes[i] for i in g.reachable()) if n.spec.category == "inst" and "dk" not in n.raw]
    if not insts:
        return None
    g = g.clone()
    n = g.nodes[insts[rng.integers(len(insts))].id]
    n.set_value("dk", torch.randn(n.k, 3, generator=torch.Generator().manual_seed(int(rng.integers(1 << 31)))) * 0.05)
    return g


def m_macro(g, res, rng, library):
    """Library proposal: inline a mined macro onto a missing region (or as the whole shape)."""
    from .abstraction import instantiate

    if not library or not res.fn_parts:
        return None
    gains = np.array([m.gain * m.programs for m in library], float)
    mac = library[rng.choice(len(library), p=gains / gains.sum())]
    part = _pick_part(res.fn_parts, rng)
    g = g.clone()
    union_with(g, instantiate(g, mac, part))
    return g


MUTATIONS: dict[str, Callable] = {
    "add": m_add,
    "subtract": m_subtract,
    "remove": m_remove,
    "swap": m_swap,
    "deform": m_deform,
    "to_instance": m_to_instance,
    "instancify": m_instancify,
    "instance_deform": m_instance_deform,
}

DEFAULT_WEIGHTS = {"add": 3.0, "subtract": 1.5, "remove": 1.0, "swap": 1.0, "deform": 1.0, "to_instance": 0.7,
                   "instancify": 1.0, "instance_deform": 0.5, "macro": 2.0,
                   # regularity detectors (regularity.py)
                   "pattern": 2.0, "mirror": 1.0, "symmetrize": 0.7, "plane": 1.5, "count": 0.5}


# ---------------------------------------------------------------------------
# Search driver
# ---------------------------------------------------------------------------


@dataclass
class SearchConfig:
    beam: int = 3
    proposals_per_parent: int = 6
    generations: int = 10
    patience: int = 3
    beta: float = 1e-3
    fit: FitConfig = field(default_factory=lambda: FitConfig(steps=150, nv=2048, ns=512))
    final_fit: FitConfig = field(default_factory=lambda: FitConfig(steps=500))
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    seed: int = 0
    time_budget: float = 600.0  # seconds (wall clock; depends on machine load)
    max_evals: int = 0  # if > 0, budget in candidate fits instead (reproducible across machines/load)
    adaptive: bool = True  # bandit re-weighting of mutations by observed improvement rate
    simplify: bool = True  # final deterministic simplification sweep
    regularity: bool = True  # detector moves: pattern (array/grid/radial), mirror, symmetrize, plane, count
    w_surf_score: float = 0.5  # surface-recall weight in the score (0 = pure volume IoU, the v0 behaviour)
    simplify_fit: FitConfig = field(default_factory=lambda: FitConfig(steps=120, nv=2048, ns=512))
    library: list = field(default_factory=list)  # mined Macros (abstraction.mine)
    macro_init: bool = True  # also seed the search with each macro fitted to the whole target


@dataclass
class Candidate:
    graph: Graph
    score: float
    iou: float
    origin: str
    gen: int


@dataclass
class SearchResult:
    best: Candidate
    history: list  # per generation: dict(best score, iou, complexity, signature)
    mutation_stats: dict  # name -> [proposed, fit, improved_parent, entered_beam]
    evaluated: int
    seconds: float


SIZE_PARAMS = {"sphere": ["r"], "box": ["b", "round"], "cylinder": ["r", "h"], "capsule": ["r", "h"],
               "torus": ["R", "r"]}


def canonicalize(g: Graph) -> Graph:
    """Semantics-preserving cleanup so equivalent programs get equal signatures:
    empty instances dropped, K=1 instances -> transforms, single-child unions collapsed,
    nested unions flattened, instance scale gauge moved into primitive prototypes."""
    g = g.clone()
    changed = True
    while changed:
        changed = False
        for nid in list(g.reachable()):
            n = g.nodes.get(nid)
            if n is None:
                continue
            if n.spec.category == "inst" and n.k == 0:
                par = [p for p in g.nodes.values() if nid in p.inputs and p.op == "union"]
                if par:
                    for p in par:
                        p.inputs = [i for i in p.inputs if i != nid]
                    changed = True
            elif n.spec.category == "inst" and n.k == 1 and "dk" not in n.raw:
                v = n.values()
                tid = g.add("transform", [n.inputs[0]], params={"t": v["t"][0].tolist(), "r": v["r"][0].tolist(),
                                                               "s": v["s"][0].tolist()})
                g.replace_refs(nid, tid)
                changed = True
            elif n.op == "array" and int(n.attrs.get("n", 2)) == 1:
                g.replace_refs(nid, n.inputs[0])
                changed = True
            elif n.op == "union":
                flat = []
                for c in n.inputs:
                    cn = g.nodes[c]
                    flat += cn.inputs if cn.op == "union" else [c]
                if flat != n.inputs:
                    n.inputs, changed = flat, True
                if len(n.inputs) == 1:
                    g.replace_refs(nid, n.inputs[0])
                    changed = True
            elif n.spec.category == "inst" and g.nodes[n.inputs[0]].op in SIZE_PARAMS:
                s = n.value("s").detach()
                m = float(torch.exp(torch.log(s).mean()))
                if abs(m - 1) > 1e-3:
                    proto = g.nodes[n.inputs[0]]
                    for name in SIZE_PARAMS[proto.op]:
                        proto.set_value(name, proto.value(name).detach() * m)
                    n.set_value("s", s / m)
            g.prune_unreachable()
    # transform(prim) gauge: fold transform scale into the primitive
    for tid, pid in _placed_prims(g):
        t, p = g.nodes[tid], g.nodes[pid]
        s = float(t.value("s"))
        if abs(s - 1) > 1e-3 and p.op in SIZE_PARAMS and len(g.parents()[pid]) == 1:
            for name in SIZE_PARAMS[p.op]:
                p.set_value(name, p.value(name).detach() * s)
            t.set_value("s", torch.ones(1))
    return g


W_SURF = 0.5  # module default; SearchConfig.w_surf_score overrides per search


def _evaluate(g: Graph, target: Target, fcfg: FitConfig, beta: float, origin: str, gen: int,
              w_surf: Optional[float] = None) -> Candidate:
    w = W_SURF if w_surf is None else w_surf
    fcfg = FitConfig(**{**fcfg.__dict__, "beta": beta, "w_surf_score": w})
    r = fit(g, target, fcfg)
    cg = canonicalize(r.graph)
    return Candidate(cg, score(cg, target, beta, w_surf=w), r.iou, origin, gen)


def initial_candidates(target: Target, n_local: int = 2, seed: int = 0) -> list[Graph]:
    """One global moment-matched primitive per kind, plus `n_local` locally-cropped ones per kind."""
    rng = np.random.default_rng(seed)
    inside = target.eval_points[target.eval_occ > 0.5].detach().cpu().numpy()
    out = []
    for kind in PRIMITIVES:
        for j in range(1 + n_local):
            g = Graph()
            g.output = _placed(g, kind, inside if j == 0 else local_crop(inside, rng))
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _cand_to_dict(c: Candidate) -> dict:
    return {"graph": c.graph.to_dict(), "score": c.score, "iou": c.iou, "origin": c.origin, "gen": c.gen}


def _cand_from_dict(d: dict, device) -> Candidate:
    return Candidate(Graph.from_dict(d["graph"]).to(device), d["score"], d["iou"], d["origin"], d["gen"])


def target_fingerprint(target: Target) -> str:
    import hashlib

    h = hashlib.sha1()
    for t in (target.points, target.occ, target.surface):
        h.update(t.detach().cpu().numpy().astype("float32").tobytes()[:1 << 20])
        h.update(str(tuple(t.shape)).encode())
    return h.hexdigest()[:16]


def _atomic_json(path: str, obj):
    import json
    import os

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _jsonl(path: str, obj):
    import json

    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

Evaluator = Callable[[list, Target, "SearchConfig", int], list]
"""(list of (graph, origin)), target, cfg, gen -> list of Candidate (same order). Lets callers swap in
parallel/batched evaluation (see parallel.py) without touching the search logic."""


def serial_evaluator(jobs: list, target: Target, cfg: "SearchConfig", gen: int) -> list:
    return [_evaluate(g, target, cfg.fit, cfg.beta, origin, gen) for g, origin in jobs]


def search(target: Target, cfg: Optional[SearchConfig] = None, init: Optional[list[Graph]] = None,
           log: Callable = print, checkpoint_dir: Optional[str] = None, resume: bool = False,
           evaluator: Optional[Evaluator] = None) -> SearchResult:
    """Beam search over graph structures.

    checkpoint_dir: if set, the full search state is written there after every generation
    (state.json, atomic) and one line per generation is appended to log.jsonl.
    resume: continue from checkpoint_dir/state.json (the target must be the same; checked by fingerprint).
            Pass cfg to change the budget (generations / max_evals) of the resumed run; omit it to reuse
            the checkpointed config.
    evaluator: how a generation's candidates get fitted (default: one after another).
    """
    import os

    from .config import from_dict as cfg_from_dict, to_dict as cfg_to_dict

    cfg_given = cfg is not None
    cfg = cfg or SearchConfig()
    evaluator = evaluator or serial_evaluator
    global W_SURF
    W_SURF = cfg.w_surf_score
    dev = target.device
    fp = target_fingerprint(target)
    state_path = os.path.join(checkpoint_dir, "state.json") if checkpoint_dir else None
    log_path = os.path.join(checkpoint_dir, "log.jsonl") if checkpoint_dir else None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    muts = dict(MUTATIONS)
    muts["add"] = lambda g, res, rng: m_add(g, res, rng, target)
    muts["deform"] = lambda g, res, rng: m_deform(g, res, rng, target)
    if cfg.regularity:
        from . import regularity as REG

        muts["pattern"] = REG.m_pattern
        muts["mirror"] = REG.m_mirror
        muts["symmetrize"] = REG.make_m_symmetrize(target)
        muts["plane"] = REG.make_m_plane(target)
        muts["count"] = REG.m_count
    if cfg.library:
        muts["macro"] = lambda g, res, rng: m_macro(g, res, rng, cfg.library)
    names = list(muts)
    prior = np.array([cfg.weights.get(k, 0.0) for k in names])

    if resume and state_path and os.path.exists(state_path):
        import json

        with open(state_path) as f:
            st = json.load(f)
        if st["target_fingerprint"] != fp:
            raise ValueError("checkpoint was made for a different target")
        if not cfg_given and st.get("config"):  # an explicit cfg wins, e.g. to extend the budget on resume
            cfg = cfg_from_dict(SearchConfig, st["config"])
        rng = np.random.default_rng()
        rng.bit_generator.state = st["rng"]
        torch.manual_seed(cfg.seed + st["gen"])
        beam = [_cand_from_dict(c, dev) for c in st["beam"]]
        best = _cand_from_dict(st["best"], dev)
        stats = {k: list(v) for k, v in st["stats"].items()}
        for k in names:
            stats.setdefault(k, [0, 0, 0, 0])
        history, evaluated, stall, start_gen = st["history"], st["evaluated"], st["stall"], st["gen"] + 1
        t0 = time.time() - st["elapsed"]
        done_searching = st.get("phase") != "search"
        log(f"resumed at gen {st['gen']} (evaluated {evaluated}, best {best.score:.4f}, phase {st.get('phase')})")
        if st.get("phase") == "done":
            return SearchResult(best, history, stats, evaluated, st["elapsed"])
    else:
        rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        t0 = time.time()
        stats = {k: [0, 0, 0, 0] for k in names}
        stats["init"] = [0, 0, 0, 0]
        seeds = list(init) if init else initial_candidates(target)
        if cfg.library and cfg.macro_init:
            from .abstraction import instantiate

            inside = target.eval_points[target.eval_occ > 0.5].detach().cpu().numpy()
            for mac in cfg.library:
                g0 = Graph()
                g0.output = instantiate(g0, mac, inside)
                seeds.append(g0)
        pool = evaluator([(g, "init") for g in seeds], target, cfg, 0)
        evaluated = len(pool)
        beam = _select(pool, cfg.beam)
        best = beam[0]
        history = [_hist(best, 0)]
        stall, start_gen, done_searching = 0, 1, False
        log(f"gen 0  best score {best.score:.4f} iou {best.iou:.3f}  {signature(best.graph)}")

    def checkpoint(gen: int, phase: str):
        if not state_path:
            return
        _atomic_json(state_path, {
            "version": 1, "gen": gen, "phase": phase, "evaluated": evaluated, "stall": stall,
            "elapsed": time.time() - t0, "rng": rng.bit_generator.state, "target_fingerprint": fp,
            "beam": [_cand_to_dict(c) for c in beam], "best": _cand_to_dict(best), "stats": stats,
            "history": history, "config": cfg_to_dict(cfg)})

    def weights():
        if not cfg.adaptive:
            return prior / prior.sum()
        # posterior-mean improvement rate (Beta(1,3) prior), floor keeps every mutation explorable
        rate = np.array([(stats[k][2] + 1) / (stats[k][1] + 4) for k in names])
        w = prior * np.maximum(rate, 0.05)
        return w / w.sum()

    if not resume or start_gen == 1:
        checkpoint(0, "search")
    gen = start_gen - 1
    for gen in range(start_gen, cfg.generations + 1):
        if done_searching or stall >= cfg.patience or _out_of_budget(cfg, t0, evaluated):
            break
        # 1) propose (sequential, rng-driven: keeps runs reproducible regardless of evaluator)
        jobs, parents = [], []
        for parent in beam:
            res = residual(parent.graph, target)
            for _ in range(cfg.proposals_per_parent):
                for _attempt in range(4):
                    name = names[rng.choice(len(names), p=weights())]
                    stats[name][0] += 1
                    child = muts[name](parent.graph, res, rng)
                    if child is not None:
                        jobs.append((child, name))
                        parents.append(parent)
                        stats[name][1] += 1
                        break
        if cfg.max_evals > 0:  # never overshoot an evaluation budget
            keep = max(cfg.max_evals - evaluated, 0)
            jobs, parents = jobs[:keep], parents[:keep]
        # 2) evaluate (serial, parallel or batched)
        children = evaluator(jobs, target, cfg, gen)
        evaluated += len(children)
        for c, parent in zip(children, parents):
            if c.score < parent.score:
                stats[c.origin][2] += 1
        # 3) select
        beam = _select(beam + children, cfg.beam)
        for c in beam:
            if c.gen == gen:
                stats[c.origin][3] += 1
        if beam[0].score < best.score - 1e-4:
            best, stall = beam[0], 0
        else:
            stall += 1
        history.append(_hist(best, gen))
        elapsed = time.time() - t0
        log(f"gen {gen}  best score {best.score:.4f} iou {best.iou:.3f} via {best.origin}  {signature(best.graph)}"
            f"  [{elapsed:.0f}s]")
        if log_path:
            _jsonl(log_path, {"gen": gen, "evaluated": evaluated, "seconds": round(elapsed, 2),
                              "best_score": best.score, "best_iou": best.iou, "complexity": best.graph.complexity(),
                              "signature": signature(best.graph),
                              "children": [[c.origin, round(c.score, 5)] for c in children]})
        checkpoint(gen, "search")
    checkpoint(gen, "simplify")
    if cfg.simplify:
        best, n_eval = simplify(best, target, cfg, log)
        evaluated += n_eval
        history.append(_hist(best, len(history)))
    final = _evaluate(best.graph, target, cfg.final_fit, cfg.beta, best.origin, best.gen)
    if final.score > best.score:
        final = best
    best = final
    checkpoint(gen, "done")
    if log_path:
        _jsonl(log_path, {"final": True, "evaluated": evaluated, "seconds": round(time.time() - t0, 2),
                          "best_score": final.score, "best_iou": final.iou, "signature": signature(final.graph)})
    return SearchResult(final, history, stats, evaluated, time.time() - t0)


def simplification_moves(g: Graph, regularity: bool = True) -> list[tuple[str, Graph]]:
    """Every single-step structural reduction of g: drop a union/intersect child, bypass a subtract or
    deformer, instancify each same-kind group, swap each primitive kind."""
    out = []
    for nid in g.reachable():
        n = g.nodes[nid]
        if n.op in ("union", "intersect") and len(n.inputs) >= 2:
            for i in range(len(n.inputs)):
                h = g.clone()
                hn = h.nodes[nid]
                hn.inputs.pop(i)
                if len(hn.inputs) == 1:
                    h.replace_refs(nid, hn.inputs[0])
                out.append(("drop", h.prune_unreachable()))
        if n.op == "subtract" or n.spec.category == "deform":
            h = g.clone()
            h.replace_refs(nid, n.inputs[0])
            out.append(("bypass", h.prune_unreachable()))
    rng = np.random.default_rng(0)
    root = g.nodes[g.output]
    if root.op == "union":
        kinds = set()
        for c in root.inputs:
            n = g.nodes[c]
            if n.op in ("transform", "instance") and g.nodes[n.inputs[0]].spec.category == "prim":
                kinds.add(g.nodes[n.inputs[0]].op)
        for kind in sorted(kinds):
            for _ in range(3):  # instancify picks a group at random; a few draws cover the groups
                h = m_instancify(g, None, rng)
                if h is not None:
                    out.append(("instancify", h))
                    break
    if regularity:
        from .regularity import deterministic_rewrites

        out += deterministic_rewrites(g)
    return out


def simplify(best: Candidate, target: Target, cfg: SearchConfig, log: Callable, max_rounds: int = 4):
    """Greedy MDL descent: apply the best-scoring reduction until none improves the score."""
    n_eval = 0
    for _ in range(max_rounds):
        moves = simplification_moves(best.graph, cfg.regularity)
        seen, cands = {signature(best.graph)}, []
        for name, h in moves:
            sig = signature(h)
            if sig in seen:
                continue
            seen.add(sig)
            cands.append(_evaluate(h, target, cfg.simplify_fit, cfg.beta, f"simplify:{name}", best.gen))
            n_eval += 1
        if not cands:
            break
        top = min(cands, key=lambda c: c.score)
        if top.score >= best.score - 1e-4:
            break
        log(f"simplify: {top.origin} score {best.score:.4f} -> {top.score:.4f}  {signature(top.graph)}")
        best = top
    return best, n_eval


def _out_of_budget(cfg: SearchConfig, t0: float, evaluated: int) -> bool:
    if cfg.max_evals > 0:
        return evaluated >= cfg.max_evals
    return time.time() - t0 > cfg.time_budget


def _select(cands: list[Candidate], k: int) -> list[Candidate]:
    """Top-k by score, at most one per structural signature (keeps the beam diverse)."""
    seen, out = set(), []
    for c in sorted(cands, key=lambda c: c.score):
        s = signature(c.graph)
        if s in seen:
            continue
        seen.add(s)
        out.append(c)
        if len(out) == k:
            break
    return out


def _hist(c: Candidate, gen: int) -> dict:
    return {"gen": gen, "score": c.score, "iou": c.iou, "complexity": c.graph.complexity(),
            "signature": signature(c.graph)}
