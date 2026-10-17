"""Faster candidate evaluation.

Two independent mechanisms:

* `ProcessEvaluator` — fits a generation's candidates in parallel worker processes (different
  structures can't share kernels, so on CPU the win comes from cores). Each worker receives the
  target once (pool initializer) and then only small pickled graphs. Works with `spawn` (Windows).

* `fit_multistart` — fits B jittered copies of ONE structure in a single batched pass with
  `torch.func.vmap`: the B parameter sets are stacked along a leading dim, the graph's SDF is
  evaluated through vmap, and one Adam step updates all copies. On a GPU this costs about the same
  as one fit; it trades the local-minimum lottery of a single start for the best of B.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn.functional as F

from .fit import FitConfig, FitResult, clamp_params, iou, score
from .graph import Graph
from .ops import EvalCtx
from .targets import Target

# ---------------------------------------------------------------------------
# Parallel evaluation of different structures
# ---------------------------------------------------------------------------

_W_TARGET: Optional[Target] = None


def _init_worker(target: Target, threads: int):
    global _W_TARGET
    torch.set_num_threads(threads)
    _W_TARGET = target


def _work(args):
    from . import search as S

    g, origin, gen, fit_cfg, beta, w_surf, warm = args
    c = S._evaluate(g, _W_TARGET, fit_cfg, beta, origin, gen, w_surf=w_surf, warm_ids=warm)
    c.graph.to("cpu")
    return c


class ProcessEvaluator:
    """Drop-in `evaluator=` for search(). Use on CPU; on a single GPU prefer the serial evaluator plus
    `fit_multistart` (several processes contending for one GPU rarely helps)."""

    def __init__(self, workers: int = 0, threads_per_worker: int = 1, start_method: Optional[str] = None):
        self.workers = workers or max(1, (os.cpu_count() or 2))
        self.threads = threads_per_worker
        self.ctx = mp.get_context(start_method or ("fork" if os.name == "posix" else "spawn"))
        self.pool = None
        self._fp = None

    def _ensure(self, target: Target):
        from .search import target_fingerprint

        fp = target_fingerprint(target)
        if self.pool is None or fp != self._fp:
            self.close()
            cpu_target = Target(**{k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                                   for k, v in target.__dict__.items()})
            self.pool = ProcessPoolExecutor(self.workers, mp_context=self.ctx, initializer=_init_worker,
                                            initargs=(cpu_target, self.threads))
            self._fp = fp

    def __call__(self, jobs, target: Target, cfg, gen: int):
        from . import search as S

        if not jobs:
            return []
        self._ensure(target)
        args = [(j[0].clone().to("cpu"), j[1], gen, cfg.fit, cfg.beta, S.W_SURF, j[2] if len(j) > 2 else ())
                for j in jobs]
        out = list(self.pool.map(_work, args))
        for c in out:
            c.graph.to(target.device)
        return out

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)
            self.pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Batched multi-start of one structure (torch.func.vmap)
# ---------------------------------------------------------------------------


def _jitter(g: Graph, b: int, scale: float, gen: torch.Generator) -> dict:
    """Raw-parameter dict for start b (b = 0 is the unperturbed graph)."""
    out = {}
    for nid, n in g.nodes.items():
        for name, r in n.raw.items():
            r0 = r.detach()
            if b > 0 and name in ("t", "r", "k", "dk"):
                amp = scale * (0.5 if name == "t" else 1.0)
                r0 = r0 + amp * torch.randn(r0.shape, generator=gen).to(r0.device)
            elif b > 0 and n.spec.params[name].kind == "pos":
                r0 = r0 + 0.5 * scale * torch.randn(r0.shape, generator=gen).to(r0.device)
            out[(nid, name)] = r0
    return out


def fit_multistart(g: Graph, target: Target, starts: int = 4, cfg: Optional[FitConfig] = None,
                   jitter: float = 0.15) -> tuple[FitResult, list]:
    """Fit `starts` jittered copies of g at once; return (best FitResult, all scores).
    Instance densify/prune are structure changes and are not batched: run a normal `fit` afterwards
    if the graph has instances whose count should adapt."""
    cfg = cfg or FitConfig()
    g = g.clone().to(target.device)
    gen = torch.Generator().manual_seed(cfg.seed)
    keys = [(nid, name) for nid in g.reachable() for name in g.nodes[nid].raw]
    starts_raw = [_jitter(g, b, jitter, gen) for b in range(starts)]
    stacked = {k: torch.stack([s[k] for s in starts_raw]).requires_grad_(True) for k in keys}
    frozen = {k for k in keys if k[1] == "logit" or
              (g.nodes[k[0]].op == "transform" and g.nodes[g.nodes[k[0]].inputs[0]].spec.category == "prim"
               and k[1] == "s")}
    for k in frozen:
        stacked[k].requires_grad_(False)
    from .fit import param_group

    groups = {}
    for k in keys:
        if k in frozen:
            continue
        groups.setdefault(param_group(g.nodes[k[0]], k[1]), []).append(stacked[k])
    opt = torch.optim.Adam([{"params": p, "lr": cfg.lr * cfg.lr_mult.get(n, 1.0), "base": cfg.lr * cfg.lr_mult.get(n, 1.0)}
                            for n, p in groups.items()])

    def sdf_with(raws: dict, pts: torch.Tensor, ctx: EvalCtx) -> torch.Tensor:
        saved = {k: g.nodes[k[0]].raw[k[1]] for k in keys}
        try:
            for k in keys:
                g.nodes[k[0]].raw[k[1]] = raws[k]
            return g.sdf(pts, ctx)
        finally:
            for k, v in saved.items():
                g.nodes[k[0]].raw[k[1]] = v

    def loss_one(raws, pts, occ, surf, eps, delta, wb, ctx):
        d = sdf_with(raws, pts, ctx)
        p = torch.sigmoid(-d / eps)
        l_occ = ((p - occ) ** 2).mean() + wb * F.binary_cross_entropy_with_logits(-d / eps, occ)
        ds = sdf_with(raws, surf, ctx)
        l_surf = (ds * ds / (ds * ds + delta * delta)).mean() * delta
        return l_occ + cfg.w_surf * l_surf

    vloss = torch.func.vmap(loss_one, in_dims=(0, None, None, None, None, None, None, None))
    pos_bound = cfg.pos_bound * 0.5 * target.diag
    for step in range(cfg.steps):
        f = step / max(cfg.steps - 1, 1)
        a = min(f / cfg.anneal_end, 1.0)
        tau = cfg.tau0 * (1 - a)
        eps = math.exp((1 - a) * math.log(cfg.eps0) + a * math.log(cfg.eps1))
        delta = cfg.surf_delta0 * (1 - a) + cfg.surf_delta1 * a
        wb = cfg.w_bce_warm * (1 - a)
        ctx = EvalCtx(tau=tau, hard_gates=True)
        pts, occ, surf, _ = target.batch(cfg.nv, cfg.ns, gen)
        lr_mult = cfg.lr_final + (1 - cfg.lr_final) * 0.5 * (1 + math.cos(math.pi * f))
        for grp in opt.param_groups:
            grp["lr"] = grp["base"] * lr_mult
        opt.zero_grad()
        losses = vloss(stacked, pts, occ, surf, eps, delta, wb, ctx)
        losses.sum().backward()  # copies are independent: each gets exactly its own gradient
        for grp in opt.param_groups:
            for p in grp["params"]:
                if p.grad is not None:
                    p.grad.nan_to_num_(0.0)
        opt.step()
        with torch.no_grad():  # clamp each copy like the single fit does
            for k in keys:
                g.nodes[k[0]].raw[k[1]] = stacked[k]
            clamp_params(g, pos_bound)
    # pick the best start by the (exact-CSG) score
    results = []
    for b in range(starts):
        h = g.clone()
        for k in keys:
            h.nodes[k[0]].raw[k[1]] = stacked[k][b].detach().clone()
        h.requires_grad_(False)
        v = iou(h, target)
        results.append((score(h, target, cfg.beta, v, cfg.w_surf_score), v, h))
    results.sort(key=lambda x: x[0])
    sc, v, best = results[0]
    return FitResult(best, v, float("nan"), sc, [], cfg.steps, 0), [r[0] for r in results]
