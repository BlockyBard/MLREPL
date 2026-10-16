"""Structure search over graphs: data-guided mutations + inner continuous fit + MDL-style score.

score(G) = (1 - IoU_hard(G)) + beta * complexity(G)

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
    gt = (target.eval_occ > 0.5).numpy()
    if g is None:
        pred = np.zeros_like(gt)
    else:
        with torch.no_grad():
            pred = (g.sdf(P, EvalCtx(tau=0.0, hard_gates=True)) < 0).numpy()
    Pn = P.numpy()
    lo, hi = target.lo.numpy(), target.hi.numpy()
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


def m_add(g, res, rng):
    if not res.fn_parts:
        return None
    part = _pick_part(res.fn_parts, rng)
    g = g.clone()
    union_with(g, _placed(g, str(rng.choice(PRIMITIVES)), part))
    return g


def m_subtract(g, res, rng):
    if not res.fp_parts:
        return None
    part = _pick_part(res.fp_parts, rng)
    g = g.clone()
    cutter = _placed(g, str(rng.choice(["box", "cylinder", "sphere", "capsule"])), part)
    # subtract from the top-level part that currently covers most of the spurious region
    pts = torch.tensor(part, dtype=torch.float32)
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
    ext = _prim_extents(old.op, {k: v.detach().numpy() for k, v in old.values().items()})
    from .graph import Node

    g.nodes[pid] = Node(pid, kind, [], _prim_from_extents(kind, ext), old.attrs)
    return g


def m_deform(g, res, rng):
    pp = _placed_prims(g)
    if not pp:
        return None
    g = g.clone()
    tid, pid = pp[rng.integers(len(pp))]
    kind = str(rng.choice(DEFORM_KINDS))
    d = g.add(kind, [pid], params={"k": [float(rng.normal() * 0.2)]})
    g.nodes[tid].inputs[0] = d
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
        pv = {k: v.detach().numpy() for k, v in g.nodes[n.inputs[0]].values().items()}
        e = _prim_extents(kind, pv)
        v = {k: x.detach().numpy() for k, x in n.values().items()}
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
    n.set_value("dk", torch.randn(n.k, 3) * 0.05)
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
                   "instancify": 1.0, "instance_deform": 0.5, "macro": 2.0}


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
    time_budget: float = 600.0  # seconds
    adaptive: bool = True  # bandit re-weighting of mutations by observed improvement rate
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


def _evaluate(g: Graph, target: Target, fcfg: FitConfig, beta: float, origin: str, gen: int) -> Candidate:
    fcfg = FitConfig(**{**fcfg.__dict__, "beta": beta})
    r = fit(g, target, fcfg)
    cg = canonicalize(r.graph)
    return Candidate(cg, score(cg, target, beta), r.iou, origin, gen)


def initial_candidates(target: Target, n_local: int = 2, seed: int = 0) -> list[Graph]:
    """One global moment-matched primitive per kind, plus `n_local` locally-cropped ones per kind."""
    rng = np.random.default_rng(seed)
    inside = target.eval_points[target.eval_occ > 0.5].numpy()
    out = []
    for kind in PRIMITIVES:
        for j in range(1 + n_local):
            g = Graph()
            g.output = _placed(g, kind, inside if j == 0 else local_crop(inside, rng))
            out.append(g)
    return out


def search(target: Target, cfg: Optional[SearchConfig] = None, init: Optional[list[Graph]] = None,
           log: Callable = print) -> SearchResult:
    cfg = cfg or SearchConfig()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    muts = dict(MUTATIONS)
    if cfg.library:
        muts["macro"] = lambda g, res, rng: m_macro(g, res, rng, cfg.library)
    stats = {k: [0, 0, 0, 0] for k in muts}
    stats["init"] = [0, 0, 0, 0]
    evaluated = 0
    seeds = list(init) if init else initial_candidates(target)
    if cfg.library and cfg.macro_init:
        from .abstraction import instantiate

        inside = target.eval_points[target.eval_occ > 0.5].numpy()
        for mac in cfg.library:
            g0 = Graph()
            g0.output = instantiate(g0, mac, inside)
            seeds.append(g0)
    pool = []
    for g in seeds:
        pool.append(_evaluate(g, target, cfg.fit, cfg.beta, "init", 0))
        evaluated += 1
    beam = _select(pool, cfg.beam)
    best = beam[0]
    history = [_hist(best, 0)]
    log(f"gen 0  best score {best.score:.4f} iou {best.iou:.3f}  {signature(best.graph)}")
    stall = 0
    names = list(muts)
    prior = np.array([cfg.weights.get(k, 0.0) for k in names])

    def weights():
        if not cfg.adaptive:
            return prior / prior.sum()
        # posterior-mean improvement rate (Beta(1,3) prior), floor keeps every mutation explorable
        rate = np.array([(stats[k][2] + 1) / (stats[k][1] + 4) for k in names])
        w = prior * np.maximum(rate, 0.05)
        return w / w.sum()
    for gen in range(1, cfg.generations + 1):
        children = []
        for parent in beam:
            res = residual(parent.graph, target)
            for _ in range(cfg.proposals_per_parent):
                for _attempt in range(4):
                    name = names[rng.choice(len(names), p=weights())]
                    stats[name][0] += 1
                    child = muts[name](parent.graph, res, rng)
                    if child is not None:
                        break
                else:
                    continue
                stats[name][1] += 1
                c = _evaluate(child, target, cfg.fit, cfg.beta, name, gen)
                evaluated += 1
                if c.score < parent.score:
                    stats[name][2] += 1
                children.append(c)
            if time.time() - t0 > cfg.time_budget:
                break
        beam = _select(beam + children, cfg.beam)
        for c in beam:
            if c.gen == gen:
                stats[c.origin][3] += 1
        if beam[0].score < best.score - 1e-4:
            best, stall = beam[0], 0
        else:
            stall += 1
        history.append(_hist(best, gen))
        log(f"gen {gen}  best score {best.score:.4f} iou {best.iou:.3f} via {best.origin}  {signature(best.graph)}"
            f"  [{time.time() - t0:.0f}s]")
        if stall >= cfg.patience or time.time() - t0 > cfg.time_budget:
            break
    final = _evaluate(best.graph, target, cfg.final_fit, cfg.beta, best.origin, best.gen)
    if final.score > best.score:
        final = best
    return SearchResult(final, history, stats, evaluated, time.time() - t0)


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
