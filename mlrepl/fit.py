"""Continuous fitting of a fixed-structure graph to a target, with the soft->hard annealing that
keeps the optimum realizable as exact mesh CSG.

Schedule (fractions of `steps`):
  [0, anneal_end)   boolean temperature tau: tau0 -> 0, occupancy sharpness eps: eps0 -> eps1
  [anneal_end, 1]   tau == 0 (exact CSG semantics), gates pushed to binary
Instance nodes are densified from false-negative clusters and pruned by ablation importance
(like splat densify/prune). A differentiable presence gate (soft erosion by -log sigmoid(logit))
exists in ops.instance but is OFF by default: it is degenerate with primitive size (the optimizer
shrinks presence and grows radii to compensate), so discrete leave-one-out pruning is used instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from .graph import Graph
from .ops import EvalCtx
from .targets import Target


@dataclass
class FitConfig:
    steps: int = 300
    lr: float = 0.03
    lr_final: float = 0.05  # cosine decay to lr * lr_final
    lr_logit: float = 0.08
    tau0: float = 0.03
    eps0: float = 0.03
    eps1: float = 0.006
    anneal_end: float = 0.7
    occ_loss: str = "robust"  # robust: bounded L2 on soft occupancy (+ BCE warmup) | bce
    w_bce_warm: float = 0.3  # BCE weight at start of robust schedule, decays to 0 by anneal_end
    w_surf: float = 3.0
    surf_delta0: float = 0.1  # Geman-McClure scale for the surface term (annealed to surf_delta1)
    surf_delta1: float = 0.02
    soft_gates: bool = False  # experimental differentiable presence (see module doc)
    w_count: float = 2e-3
    w_binarize: float = 0.02
    gate_scale: float = 0.05
    nv: int = 4096
    ns: int = 1024
    densify_every: int = 40
    densify_until: float = 0.6
    densify_max_new: int = 4
    densify_min_frac: float = 0.01
    prune_gain: float = 0.004  # remove an instance if it explains less than this fraction of the inside volume
    prune_after: float = 0.15
    freeze_gauge: bool = True
    seed: int = 0
    beta: float = 1e-3  # complexity weight in score


@dataclass
class FitResult:
    graph: Graph
    iou: float
    loss: float
    score: float
    history: list = field(default_factory=list)


def iou(g: Graph, target: Target, ctx: Optional[EvalCtx] = None) -> float:
    ctx = ctx or EvalCtx(tau=0.0, hard_gates=True)
    with torch.no_grad():
        pred = g.sdf(target.eval_points, ctx) < 0
    gt = target.eval_occ > 0.5
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return inter / max(union, 1)


def score(g: Graph, target: Target, beta: float, iou_val: Optional[float] = None) -> float:
    iou_val = iou(g, target) if iou_val is None else iou_val
    return (1.0 - iou_val) + beta * g.complexity()


def _instance_nodes(g):
    return [g.nodes[n] for n in g.reachable() if g.nodes[n].spec.category == "inst"]


def _frozen(g: Graph) -> set:
    """Gauge fixing: a transform's scale is redundant with its primitive's size params."""
    fr = set()
    for nid in g.reachable():
        n = g.nodes[nid]
        if n.op == "transform" and g.nodes[n.inputs[0]].spec.category == "prim":
            fr.add((nid, "s"))
    return fr


def _make_opt(g: Graph, cfg: FitConfig):
    frozen = _frozen(g) if cfg.freeze_gauge else set()
    g.requires_grad_(True)
    main, logits = [], []
    for nid in g.reachable():
        n = g.nodes[nid]
        for name, r in n.raw.items():
            if (nid, name) in frozen:
                r.requires_grad_(False)
                continue
            if name == "logit" and not cfg.soft_gates:
                r.requires_grad_(False)
                continue
            (logits if name == "logit" else main).append(r)
    groups = [{"params": main, "lr": cfg.lr}]
    if logits:
        groups.append({"params": logits, "lr": cfg.lr_logit})
    return torch.optim.Adam(groups) if main or logits else None


def loss_terms(g: Graph, batch, ctx: EvalCtx, eps: float, cfg: FitConfig, late: bool, a: float = 1.0):
    """a in [0,1]: annealing progress. The robust losses are bounded per point, so volume a
    candidate cannot explain (e.g. legs when fitting only a tabletop) stops pulling on it;
    residual-driven mutations are responsible for explaining that volume instead."""
    pts, occ, surf, w = batch
    d = g.sdf(pts, ctx)
    if cfg.occ_loss == "bce":
        l_occ = F.binary_cross_entropy_with_logits(-d / eps, occ, weight=w)
        l_surf = g.sdf(surf, ctx).abs().mean()
    else:
        p = torch.sigmoid(-d / eps)
        sq = (p - occ) ** 2
        l_occ = (sq * w).mean() if w is not None else sq.mean()
        wb = cfg.w_bce_warm * (1 - a)
        if wb > 0:
            l_occ = l_occ + wb * F.binary_cross_entropy_with_logits(-d / eps, occ, weight=w)
        delta = cfg.surf_delta0 * (1 - a) + cfg.surf_delta1 * a
        ds = g.sdf(surf, ctx)
        l_surf = (ds * ds / (ds * ds + delta * delta)).mean() * delta
    loss = l_occ + cfg.w_surf * l_surf
    terms = {"occ": l_occ.item(), "surf": l_surf.item()}
    for n in (_instance_nodes(g) if cfg.soft_gates else []):
        pr = torch.sigmoid(n.raw["logit"])
        loss = loss + cfg.w_count * pr.sum()
        if late:
            loss = loss + cfg.w_binarize * (pr * (1 - pr)).sum()
    return loss, terms


# ---------------------------------------------------------------------------
# Splat-style instance densification / pruning
# ---------------------------------------------------------------------------


def _kmeans(x: torch.Tensor, k: int, iters: int = 10, gen=None):
    c = x[torch.randperm(len(x), generator=gen)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(len(c)):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    a = torch.cdist(x, c).argmin(1)
    return c, torch.bincount(a, minlength=len(c))


def _mismatch(g: Graph, target: Target) -> float:
    with torch.no_grad():
        pred = g.sdf(target.points, EvalCtx(tau=0.0, hard_gates=True)) < 0
    gt = target.occ > 0.5
    return float((pred != gt).sum()) / max(int(gt.sum()), 1)


def _remove_rows(n, keep: torch.Tensor):
    for k in list(n.raw):
        n.raw[k] = n.raw[k].detach()[keep].clone()


def prune_dead(g: Graph) -> int:
    removed = 0
    for n in _instance_nodes(g):
        keep = n.raw["logit"].detach() >= 0
        if not keep.all():
            removed += int((~keep).sum())
            _remove_rows(n, keep)
    return removed


def prune_instances(g: Graph, target: Target, gain: float, max_iter: int = 16) -> int:
    """Greedy leave-one-out: repeatedly drop the least useful live instance while its removal costs
    less than `gain` (mismatch, as a fraction of the target's inside volume)."""
    removed = prune_dead(g)
    for _ in range(max_iter):
        base = _mismatch(g, target)
        best = None
        for n in _instance_nodes(g):
            lg = n.raw["logit"]
            for i in range(n.k):
                old = lg[i].item()
                lg.data[i] = -10.0
                delta = _mismatch(g, target) - base
                lg.data[i] = old
                if best is None or delta < best[0]:
                    best = (delta, n, i)
        if best is None or best[0] >= gain:
            break
        _, n, i = best
        keep = torch.ones(n.k, dtype=torch.bool)
        keep[i] = False
        _remove_rows(n, keep)
        removed += 1
    return removed


def densify_instances(g: Graph, target: Target, cfg: FitConfig, gen=None) -> int:
    insts = [n for n in _instance_nodes(g) if n.k > 0]
    if not insts:
        return 0
    with torch.no_grad():
        pred_in = g.sdf(target.points, EvalCtx(tau=0.0, hard_gates=True)) < 0
    fn = (target.occ > 0.5) & ~pred_in
    n_in = int((target.occ > 0.5).sum())
    if fn.sum() < max(20, cfg.densify_min_frac * n_in):
        return 0
    pts = target.points[fn]
    k = min(cfg.densify_max_new, max(1, len(pts) // 50))
    cents, sizes = _kmeans(pts, k, gen=gen)
    added = 0
    for c, sz in zip(cents, sizes):
        if sz < 20:
            continue
        # attach to the instance node whose nearest live instance is closest; copy its pose/shape data
        best = None
        for n in insts:
            t = n.value("t").detach()
            j = int((t - c).norm(dim=1).argmin())
            dist = float((t[j] - c).norm())
            if best is None or dist < best[0]:
                best = (dist, n, j)
        _, n, j = best
        for name in list(n.raw):
            r = n.raw[name].detach()
            new = r[j:j + 1].clone()
            if name == "t":
                new = n.spec.params["t"].to_raw(c[None])
            if name == "logit":
                new = torch.full_like(new, 3.0)
            n.raw[name] = torch.cat([r, new], 0)
        added += 1
    return added


# ---------------------------------------------------------------------------


def fit(g: Graph, target: Target, cfg: Optional[FitConfig] = None, verbose: bool = False) -> FitResult:
    cfg = cfg or FitConfig()
    g = g.clone()
    gen = torch.Generator().manual_seed(cfg.seed)
    opt = _make_opt(g, cfg)
    hist = []
    if opt is not None:
        for step in range(cfg.steps):
            f = step / max(cfg.steps - 1, 1)
            a = min(f / cfg.anneal_end, 1.0)
            tau = cfg.tau0 * (1 - a)
            eps = math.exp((1 - a) * math.log(cfg.eps0) + a * math.log(cfg.eps1))
            ctx = EvalCtx(tau=tau, gate_scale=cfg.gate_scale, hard_gates=not cfg.soft_gates)
            late = f >= cfg.anneal_end
            batch = target.batch(cfg.nv, cfg.ns, gen)
            lr_mult = cfg.lr_final + (1 - cfg.lr_final) * 0.5 * (1 + math.cos(math.pi * f))
            for grp, base in zip(opt.param_groups, (cfg.lr, cfg.lr_logit)):
                grp["lr"] = base * lr_mult
            opt.zero_grad()
            loss, terms = loss_terms(g, batch, ctx, eps, cfg, late, a)
            if not loss.requires_grad:  # e.g. every instance pruned away: nothing left to optimize
                break
            loss.backward()
            opt.step()
            hist.append(loss.item())
            if _instance_nodes(g) and step > 0 and step % cfg.densify_every == 0:
                changed = 0
                if f > cfg.prune_after:
                    changed += prune_instances(g, target, cfg.prune_gain)
                if f < cfg.densify_until:
                    changed += densify_instances(g, target, cfg, gen)
                if changed:
                    opt = _make_opt(g, cfg)
            if verbose and step % 50 == 0:
                print(f"  step {step:4d} loss {loss.item():.4f} tau {tau:.3f} eps {eps:.3f} {terms}")
        # realizable endpoint: drop dead / useless instances
        prune_instances(g, target, cfg.prune_gain)
    g.requires_grad_(False)
    v = iou(g, target)
    return FitResult(g, v, hist[-1] if hist else float("nan"), score(g, target, cfg.beta, v), hist)
