"""Numerically safe differentiable helpers shared by SDF ops (torch) and mesh realization (numpy)."""
from __future__ import annotations

import math

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Rotations (axis-angle, the parameterization stored in graphs)
# ---------------------------------------------------------------------------


def axis_angle_to_matrix(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula, stable (and with finite gradients) at w == 0. (...,3) -> (...,3,3)."""
    th2 = (w * w).sum(-1, keepdim=True)
    th = torch.sqrt(th2 + 1e-12)
    a = torch.sin(th) / th
    half = 0.5 * th
    b = 0.5 * (torch.sin(half) / half) ** 2  # == (1 - cos th) / th^2 without cancellation
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    k = torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*w.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(k)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def axis_angle_to_matrix_np(w) -> np.ndarray:
    return axis_angle_to_matrix(torch.as_tensor(np.asarray(w, dtype=np.float64))).numpy()


# ---------------------------------------------------------------------------
# Safe "c" functions: f(u)/u style quotients that are smooth through u == 0
# ---------------------------------------------------------------------------


def atanc(u: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """atan(u)/u with a series branch; the untaken branch never sees u == 0, so grads stay finite."""
    small = u.abs() < eps
    u_safe = torch.where(small, torch.ones_like(u), u)
    return torch.where(small, 1.0 - u * u / 3.0, torch.atan(u_safe) / u_safe)


# ---------------------------------------------------------------------------
# Domain deformations ("vertex shader" deformers), all along the local +Y axis.
#
# Forward maps move mesh vertices (used at realization/export time, i.e. what a
# WPO vertex shader would do). Inverse maps move SDF query points back into the
# undeformed frame, so  inside(D(shape)) at q  <=>  inside(shape) at D^-1(q).
# Composite order (forward): taper -> twist -> bend.
# ---------------------------------------------------------------------------

DEFORM_KINDS = ("taper", "twist", "bend")


def _taper_factor(y, k, lib):
    f = 1.0 + k * y
    return lib.clip(f, 0.05, None) if lib is np else torch.clamp(f, min=0.05)


def deform_inverse(kind: str, q: torch.Tensor, k: torch.Tensor):
    """Return (q_undeformed, distance_scale). k broadcasts against q[..., 0]."""
    x, y, z = q.unbind(-1)
    if kind == "twist":  # rotate xz by -k*y
        a = -k * y
        c, s = torch.cos(a), torch.sin(a)
        return torch.stack([c * x - s * z, y, s * x + c * z], -1), torch.ones_like(x)
    if kind == "taper":  # forward scales xz by f(y) = 1 + k*y
        f = _taper_factor(y, k, torch)
        return torch.stack([x / f, y, z / f], -1), f
    if kind == "bend":
        # Forward: y-axis bent along a circle of curvature k towards +x (see realize.deform_forward).
        a = 1.0 - k * x
        b = k * y
        rho = torch.sqrt(a * a + b * b + 1e-12)
        x0 = (2.0 * x - k * (x * x + y * y)) / (1.0 + rho)
        a_c = torch.clamp(a, min=1e-3)
        y0 = (y / a_c) * atanc(b / a_c)
        return torch.stack([x0, y0, z], -1), torch.ones_like(x)
    raise ValueError(kind)


def deform_forward_np(kind: str, v: np.ndarray, k) -> np.ndarray:
    """Forward vertex deformation (numpy, float64). v: (N,3); k: scalar."""
    k = float(k)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    if kind == "twist":
        a = k * y
        c, s = np.cos(a), np.sin(a)
        return np.stack([c * x - s * z, y, s * x + c * z], -1)
    if kind == "taper":
        f = _taper_factor(y, k, np)
        return np.stack([x * f, y, z * f], -1)
    if kind == "bend":
        th = k * y
        # sin(th)/k and (1-cos th)/k written as y*sinc forms: stable at k -> 0
        sin_over_k = y * np.sinc(th / math.pi)
        half = 0.5 * th
        one_minus_cos_over_k = y * np.sin(half) * np.sinc(half / math.pi)
        X = one_minus_cos_over_k + x * np.cos(th)
        Y = sin_over_k - x * np.sin(th)
        return np.stack([X, Y, z], -1)
    raise ValueError(kind)


def apply_deform_chain_inverse(q: torch.Tensor, ks: dict):
    """Inverse of the forward chain taper->twist->bend. ks maps kind -> k tensor. Returns (q0, dist_scale)."""
    scale = torch.ones_like(q[..., 0])
    for kind in reversed(DEFORM_KINDS):
        if kind in ks:
            q, sc = deform_inverse(kind, q, ks[kind])
            scale = scale * sc
    return q, scale


def apply_deform_chain_forward_np(v: np.ndarray, ks: dict) -> np.ndarray:
    for kind in DEFORM_KINDS:
        if kind in ks and abs(float(ks[kind])) > 1e-9:
            v = deform_forward_np(kind, v, ks[kind])
    return v
