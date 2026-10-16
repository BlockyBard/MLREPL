"""Sphere-traced previews of graphs (no GPU needed). Also returns depth, so the same tracer can
produce depth-map targets for testing the depth pipeline."""
from __future__ import annotations

import math

import numpy as np
import torch

from .ops import EvalCtx


def look_at(eye, center=(0, 0, 0), up=(0, 1, 0)) -> np.ndarray:
    eye, center, up = (np.asarray(x, float) for x in (eye, center, up))
    f = center - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = r, -u, f, eye  # camera +z looks forward, +y down (image rows)
    return M


def intrinsics(res: int, fov_deg: float = 40.0) -> np.ndarray:
    f = 0.5 * res / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1.0]])


@torch.no_grad()
def trace(g, cam_to_world: np.ndarray, K: np.ndarray, res: int = 160, steps: int = 96, far: float = 8.0,
          step_scale: float = 0.6):
    ys, xs = np.mgrid[0:res, 0:res]
    d = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1)
    d = d.reshape(-1, 3) @ cam_to_world[:3, :3].T
    zscale = 1.0 / np.linalg.norm(d, axis=1)  # depth along camera z = t * zscale... (d has z=1 in cam frame)
    d = d * zscale[:, None]
    o = torch.tensor(np.broadcast_to(cam_to_world[:3, 3], d.shape).copy(), dtype=torch.float32)
    d = torch.tensor(d, dtype=torch.float32)
    t = torch.zeros(len(d))
    ctx = EvalCtx(tau=0.0, hard_gates=True)
    hit = torch.zeros(len(d), dtype=torch.bool)
    for _ in range(steps):
        p = o + d * t[:, None]
        s = g.sdf(p, ctx)
        hit |= s.abs() < 1e-3
        t = torch.where(hit, t, t + step_scale * s.clamp(min=-0.05))
        if bool(((t > far) | hit).all()):
            break
    p = o + d * t[:, None]
    e = 1e-3
    n = torch.stack([g.sdf(p + torch.tensor(v) * e, ctx) - g.sdf(p - torch.tensor(v) * e, ctx)
                     for v in np.eye(3, dtype=np.float32)], -1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-9)
    hit = hit & (t < far)
    depth = (t * torch.tensor(zscale, dtype=torch.float32)).where(hit, torch.zeros_like(t))
    return hit.reshape(res, res).numpy(), n.reshape(res, res, 3).numpy(), depth.reshape(res, res).numpy()


def shade(hit, n, cam_to_world, color=(0.85, 0.6, 0.4)) -> np.ndarray:
    L = np.array([0.4, 0.8, 0.45])
    L /= np.linalg.norm(L)
    V = -cam_to_world[:3, 2]
    diff = np.clip((n * L).sum(-1), 0, 1)
    h = L + V
    h /= np.linalg.norm(h)
    spec = np.clip((n * h).sum(-1), 0, 1) ** 32
    rgb = np.asarray(color) * (0.25 + 0.75 * diff[..., None]) + 0.25 * spec[..., None]
    bg = np.array([0.95, 0.95, 0.96])
    return np.where(hit[..., None], np.clip(rgb, 0, 1), bg)


def preview(g, res=160, eye=(2.6, 1.8, 3.2), color=(0.85, 0.6, 0.4)) -> np.ndarray:
    M = look_at(eye)
    hit, n, _ = trace(g, M, intrinsics(res), res)
    return shade(hit, n, M, color)


def save_grid(rows, path, titles=None, row_labels=None):
    """rows: list of lists of HxWx3 images."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R, C = len(rows), len(rows[0])
    fig, axes = plt.subplots(R, C, figsize=(2.4 * C, 2.4 * R), squeeze=False)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            ax = axes[i][j]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            if titles and i == 0:
                ax.set_title(titles[j], fontsize=9)
            if row_labels and j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
