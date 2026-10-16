"""Target observations. Every input modality is reduced to the same supervision:
volume samples with inside/outside labels + surface samples (+ optional SDF values).

Implemented here: SDF functions / graphs, dense meshes, depth maps (back-projected points
+ free-space carving along rays), and Gaussian splat centers. Multi-view RGB is handled by
the renderer loss (render.py), not by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from skimage import measure

from .ops import EvalCtx


@dataclass
class Target:
    points: torch.Tensor  # (N,3) volume samples
    occ: torch.Tensor  # (N,) float 0/1 (or soft confidence)
    surface: torch.Tensor  # (M,3) surface samples
    lo: torch.Tensor
    hi: torch.Tensor
    eval_points: Optional[torch.Tensor] = None  # held-out uniform samples for IoU
    eval_occ: Optional[torch.Tensor] = None
    weight: Optional[torch.Tensor] = None  # per-volume-sample confidence (depth carving)

    @property
    def diag(self) -> float:
        return float((self.hi - self.lo).norm())

    def batch(self, nv: int, ns: int, gen: torch.Generator = None):
        iv = torch.randint(len(self.points), (min(nv, len(self.points)),), generator=gen)
        is_ = torch.randint(len(self.surface), (min(ns, len(self.surface)),), generator=gen)
        w = None if self.weight is None else self.weight[iv]
        return self.points[iv], self.occ[iv], self.surface[is_], w


def _grid(lo, hi, res):
    axes = [np.linspace(lo[i], hi[i], res) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1)


def sample_triangles(V: np.ndarray, F: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
    idx = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def _assemble(occ_fn, surface: np.ndarray, lo, hi, n_vol, n_eval, near_sigma, rng) -> Target:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n_near = n_vol // 2
    near = surface[rng.integers(len(surface), size=n_near)] + rng.normal(size=(n_near, 3)) * near_sigma
    uni = lo + rng.random((n_vol - n_near, 3)) * (hi - lo)
    pts = np.concatenate([near, uni])
    ev = lo + rng.random((n_eval, 3)) * (hi - lo)
    f = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return Target(f(pts), f(occ_fn(pts)), f(surface), f(lo), f(hi), f(ev), f(occ_fn(ev)))


def from_sdf(fn: Callable[[np.ndarray], np.ndarray], lo=(-1, -1, -1), hi=(1, 1, 1), res=96, n_vol=20000,
             n_surf=6000, n_eval=20000, seed=0) -> Target:
    """fn maps (N,3) float array -> (N,) signed distance (negative inside)."""
    rng = np.random.default_rng(seed)
    grid = _grid(lo, hi, res)
    vol = fn(grid.reshape(-1, 3)).reshape(grid.shape[:3])
    if vol.min() >= 0 or vol.max() <= 0:
        raise ValueError("target has no surface inside the bounds")
    step = (np.asarray(hi) - np.asarray(lo)) / (res - 1)
    V, F, _, _ = measure.marching_cubes(vol, 0.0, spacing=tuple(step))
    V = V + np.asarray(lo)
    surf = sample_triangles(V, F, n_surf, rng)
    sigma = 0.02 * float(np.linalg.norm(np.asarray(hi) - np.asarray(lo)))
    return _assemble(lambda p: (fn(p) < 0).astype(np.float32), surf, lo, hi, n_vol, n_eval, sigma, rng)


def from_graph(g, lo=(-1, -1, -1), hi=(1, 1, 1), **kw) -> Target:
    ctx = EvalCtx(tau=0.0, hard_gates=True)

    def fn(p):
        out = []
        with torch.no_grad():
            for s in range(0, len(p), 65536):
                out.append(g.sdf(torch.tensor(p[s:s + 65536], dtype=torch.float32), ctx).numpy())
        return np.concatenate(out)

    return from_sdf(fn, lo, hi, **kw)


def from_mesh(V: np.ndarray, F: np.ndarray, normalize=True, n_vol=20000, n_surf=6000, n_eval=20000, seed=0,
              pad=0.1) -> Target:
    """Dense (watertight-ish) mesh target; inside test via generalized winding number."""
    from .realize import winding_number

    rng = np.random.default_rng(seed)
    V = np.asarray(V, float)
    if normalize:
        c = 0.5 * (V.min(0) + V.max(0))
        V = (V - c) / (0.5 * (V.max(0) - V.min(0)).max()) * (1 - pad)
    lo, hi = np.full(3, -1.0), np.full(3, 1.0)
    surf = sample_triangles(V, F, n_surf, rng)
    occ_fn = lambda p: (winding_number(V, F, p) > 0.5).astype(np.float32)  # noqa: E731
    return _assemble(occ_fn, surf, lo, hi, n_vol, n_eval, 0.04, rng)


def from_depth(depth: np.ndarray, K: np.ndarray, cam_to_world: np.ndarray, n_free_per_ray=4, max_points=20000,
               seed=0) -> Target:
    """Single (or call per-view and merge) depth map. Surface = back-projected pixels; free space =
    points sampled before the hit along each ray (labelled outside); a thin band behind the hit is
    labelled inside with reduced confidence. Pixels with depth <= 0 are background (ray all free)."""
    rng = np.random.default_rng(seed)
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    dirs = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1)
    R, t = cam_to_world[:3, :3], cam_to_world[:3, 3]
    d = depth.reshape(-1)
    dirs = dirs.reshape(-1, 3) @ R.T
    hit = d > 0
    surf = t + dirs[hit] * d[hit, None]
    far = np.where(hit, d, d[hit].max() * 1.5 if hit.any() else 3.0)
    sel = rng.choice(len(d), min(len(d), max_points // (n_free_per_ray + 1)), replace=False)
    fr = rng.random((len(sel), n_free_per_ray)) * 0.97
    free = t + dirs[sel, None, :] * (fr * far[sel, None])[..., None]
    behind_sel = sel[hit[sel]]
    behind = t + dirs[behind_sel] * (d[behind_sel] + rng.random(len(behind_sel)) * 0.03)[:, None]
    pts = np.concatenate([free.reshape(-1, 3), behind])
    occ = np.concatenate([np.zeros(free.shape[0] * free.shape[1]), np.ones(len(behind))])
    w = np.concatenate([np.ones(free.shape[0] * free.shape[1]), np.full(len(behind), 0.5)])
    lo, hi = surf.min(0) - 0.1, surf.max(0) + 0.1
    f = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return Target(f(pts), f(occ), f(surf), f(lo), f(hi), weight=f(w))


def gaussian_normals(rotations: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Unoriented splat normals: the axis of the smallest scale. rotations are (N,4) wxyz quaternions."""
    w, x, y, z = (rotations / np.linalg.norm(rotations, axis=1, keepdims=True)).T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    j = np.argmin(scales, axis=1)
    return R[np.arange(len(R)), :, j]


def from_gaussians(means: np.ndarray, opacities: np.ndarray, scales: np.ndarray, normals: np.ndarray = None,
                   n_vol=20000, n_eval=20000, seed=0, min_opacity=0.5, k=8, res=96) -> Target:
    """Gaussian splats -> solid target via a point-set-surface signed distance: for a query point, the
    Gaussian-weighted average of n_i . (p - mu_i) over the k nearest confident splats.

    `normals` must be oriented (outward); with 3DGS, take the smallest-scale axis (`gaussian_normals`)
    and orient it toward the camera that sees the splat. Splat centers lie on the surface, so
    unoriented density thresholding only gives a leaky shell; that is why orientation is required."""
    from scipy.spatial import cKDTree

    if normals is None:
        raise ValueError("from_gaussians needs oriented normals (see gaussian_normals + camera orientation)")
    keep = opacities > min_opacity
    mu, nrm = means[keep], normals[keep]
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    sc = np.maximum(np.asarray(scales)[keep].reshape(len(mu), -1).max(-1), 1e-3)
    tree = cKDTree(mu)
    h = 2.0 * float(np.median(sc))

    def fn(p):
        dist, idx = tree.query(p, k=k)
        w = np.exp(-0.5 * (dist / h) ** 2) + 1e-12
        sd = ((p[:, None, :] - mu[idx]) * nrm[idx]).sum(-1)
        return (w * sd).sum(1) / w.sum(1)

    lo, hi = mu.min(0) - 0.1, mu.max(0) + 0.1
    return from_sdf(fn, lo, hi, res=res, n_vol=n_vol, n_surf=min(len(mu), 6000), n_eval=n_eval, seed=seed)
