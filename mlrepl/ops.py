"""Op registry: every op has a differentiable SDF semantics (here) and an exact mesh semantics (realize.py).

Value type flowing along graph edges is a *shape* (a signed distance field in the node's frame).
Evaluating a node means: given query points p (..., 3), return distances (...).
Children are passed as callables so frame-changing ops (transform, deform, instance) can query
their children at transformed points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .mathutil import DEFORM_KINDS, apply_deform_chain_inverse, axis_angle_to_matrix

BIG = 1e3


@dataclass
class EvalCtx:
    """Global evaluation settings, annealed by the fitter.

    tau:        boolean softness (log-sum-exp temperature). 0 => exact CSG (realizable).
    gate_scale: how far (in scene units per nat) a low-presence instance is pushed outward.
    hard_gates: if True, instances with logit < 0 are removed exactly (realization semantics).
    """

    tau: float = 0.0
    gate_scale: float = 0.05
    hard_gates: bool = False


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class ParamSpec:
    shape: tuple
    kind: str = "real"  # real | pos
    default: object = 0.0
    per_instance: bool = False  # leading K dimension (instance ops)
    optional: bool = False

    def to_raw(self, v: torch.Tensor) -> torch.Tensor:
        v = torch.as_tensor(v, dtype=torch.float32)
        if self.kind == "pos":
            v = torch.clamp(v, min=1e-6)
            return torch.where(v > 20, v, torch.log(torch.expm1(v)))
        return v.clone()

    def from_raw(self, r: torch.Tensor) -> torch.Tensor:
        if self.kind == "pos":
            return F.softplus(r)
        return r

    def default_value(self, k: int = 1) -> torch.Tensor:
        shape = ((k,) if self.per_instance else ()) + tuple(self.shape)
        return torch.full(shape, float(self.default)) if not isinstance(self.default, (list, tuple)) else torch.tensor(
            self.default, dtype=torch.float32
        ).expand(shape).clone()


@dataclass
class OpSpec:
    name: str
    category: str  # prim | xform | deform | bool | inst
    arity: Optional[int]  # None => n-ary (>= 1)
    params: dict
    sdf: Callable
    doc: str = ""
    attrs: dict = field(default_factory=dict)  # attr defaults


OPS: dict[str, OpSpec] = {}


def register(spec: OpSpec):
    OPS[spec.name] = spec
    return spec


# ---------------------------------------------------------------------------
# Primitives (canonical frame, centered at origin, long axis = +Y)
# ---------------------------------------------------------------------------


def _sphere(v, ch, p, ctx):
    return p.norm(dim=-1) - v["r"][..., 0]


def _box(v, ch, p, ctx):
    rnd = v["round"][..., 0]
    b = torch.clamp(v["b"] - rnd[..., None], min=1e-4)
    q = p.abs() - b
    outside = torch.clamp(q, min=0).norm(dim=-1)
    inside = torch.clamp(q.max(-1).values, max=0)
    return outside + inside - rnd


def _cylinder(v, ch, p, ctx):
    r, h = v["r"][..., 0], v["h"][..., 0]
    dx = torch.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2 + 1e-12) - r
    dy = p[..., 1].abs() - h
    d = torch.stack([dx, dy], -1)
    return torch.clamp(d, min=0).norm(dim=-1) + torch.clamp(d.max(-1).values, max=0)


def _capsule(v, ch, p, ctx):
    r, h = v["r"][..., 0], v["h"][..., 0]
    y = torch.clamp(p[..., 1], -h, h)
    q = torch.stack([p[..., 0], p[..., 1] - y, p[..., 2]], -1)
    return q.norm(dim=-1) - r


def _torus(v, ch, p, ctx):
    R, r = v["R"][..., 0], v["r"][..., 0]
    qx = torch.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2 + 1e-12) - R
    return torch.sqrt(qx * qx + p[..., 1] ** 2 + 1e-12) - r


P1 = lambda d: ParamSpec((1,), "pos", d)  # noqa: E731

register(OpSpec("sphere", "prim", 0, {"r": P1(0.5)}, _sphere, "sphere of radius r"))
register(
    OpSpec(
        "box",
        "prim",
        0,
        {"b": ParamSpec((3,), "pos", 0.5), "round": ParamSpec((1,), "pos", 1e-4)},
        _box,
        "box with half-extents b and edge rounding",
    )
)
register(OpSpec("cylinder", "prim", 0, {"r": P1(0.3), "h": P1(0.5)}, _cylinder, "Y-axis cylinder, half-height h"))
register(OpSpec("capsule", "prim", 0, {"r": P1(0.2), "h": P1(0.4)}, _capsule, "Y-axis capsule, segment half-length h"))
register(OpSpec("torus", "prim", 0, {"R": P1(0.5), "r": P1(0.15)}, _torus, "torus in XZ plane"))

PRIMITIVES = [n for n, s in OPS.items() if s.category == "prim"]

# ---------------------------------------------------------------------------
# Transform: p_local = R^T (p - t) / s ;  d = s * child(p_local)
# ---------------------------------------------------------------------------


def _to_local(p, t, r, s):
    """p (*B,3); t,r (K,3); s (K,1) -> (K,*B,3). Works with K==1 for plain transforms."""
    K = t.shape[0]
    nb = p.dim() - 1
    view = (K,) + (1,) * nb
    R = axis_angle_to_matrix(r)  # K,3,3
    d = p.unsqueeze(0) - t.view(*view, 3)
    q = torch.einsum("kij,k...i->k...j", R, d)
    return q / s.view(*view, 1)


def _transform(v, ch, p, ctx):
    t, r, s = v["t"][None], v["r"][None], v["s"][None]
    q = _to_local(p, t, r, s)[0]
    return ch[0](q) * s[0, 0]


register(
    OpSpec(
        "transform",
        "xform",
        1,
        {"t": ParamSpec((3,), "real", 0.0), "r": ParamSpec((3,), "real", 0.0), "s": P1(1.0)},
        _transform,
        "rigid transform + uniform scale (axis-angle rotation)",
    )
)

# ---------------------------------------------------------------------------
# Deformers (domain warps realizable as vertex displacement)
# ---------------------------------------------------------------------------


def _make_deform(kind):
    def fn(v, ch, p, ctx):
        q, sc = apply_deform_chain_inverse(p, {kind: v["k"][..., 0]})
        return ch[0](q) * sc

    return fn


for _k in DEFORM_KINDS:
    register(OpSpec(_k, "deform", 1, {"k": ParamSpec((1,), "real", 0.0)}, _make_deform(_k), f"{_k} along local Y"))

# ---------------------------------------------------------------------------
# Booleans. tau > 0 => log-sum-exp relaxation (gradients reach every child);
# tau == 0 => exact min/max, i.e. exactly the mesh CSG that realize.py produces.
# ---------------------------------------------------------------------------


def softmin(ds: torch.Tensor, tau: float, dim: int = 0) -> torch.Tensor:
    if tau <= 0:
        return ds.min(dim).values
    return -tau * torch.logsumexp(-ds / tau, dim)


def softmax(ds: torch.Tensor, tau: float, dim: int = 0) -> torch.Tensor:
    return -softmin(-ds, tau, dim)


def _union(v, ch, p, ctx):
    return softmin(torch.stack([c(p) for c in ch], 0), ctx.tau)


def _intersect(v, ch, p, ctx):
    return softmax(torch.stack([c(p) for c in ch], 0), ctx.tau)


def _subtract(v, ch, p, ctx):
    return softmax(torch.stack([ch[0](p), -ch[1](p)], 0), ctx.tau)


register(OpSpec("union", "bool", None, {}, _union, "n-ary union"))
register(OpSpec("intersect", "bool", None, {}, _intersect, "n-ary intersection"))
register(OpSpec("subtract", "bool", 2, {}, _subtract, "a minus b"))

# ---------------------------------------------------------------------------
# Instancing: one prototype child, K poses, per-instance deformation (vertex-shader
# custom data) and per-instance presence logits (additive/removal like splats).
# ---------------------------------------------------------------------------


def _instance(v, ch, p, ctx):
    t = v["t"]
    K = t.shape[0]
    if K == 0:
        return torch.full(p.shape[:-1], BIG, dtype=p.dtype)
    s = v["s"]
    q = _to_local(p, t, v["r"], s)  # K,*B,3
    nb = p.dim() - 1
    view = (K,) + (1,) * nb
    scale = torch.ones_like(q[..., 0])
    if "dk" in v:  # (K,3): taper, twist, bend
        ks = {kind: v["dk"][:, i].view(view) for i, kind in enumerate(DEFORM_KINDS)}
        q, scale = apply_deform_chain_inverse(q, ks)
    d = ch[0](q) * scale * s.view(view)
    logit = v["logit"].view(view)
    if ctx.hard_gates:
        d = torch.where(logit >= 0, d, torch.full_like(d, BIG))
    else:
        d = d + ctx.gate_scale * F.softplus(-logit)  # = -gate_scale * log(sigmoid(logit))
    return softmin(d, ctx.tau, 0)


register(
    OpSpec(
        "instance",
        "inst",
        1,
        {
            "t": ParamSpec((3,), "real", 0.0, per_instance=True),
            "r": ParamSpec((3,), "real", 0.0, per_instance=True),
            "s": ParamSpec((1,), "pos", 1.0, per_instance=True),
            "logit": ParamSpec((), "real", 3.0, per_instance=True),
            "dk": ParamSpec((3,), "real", 0.0, per_instance=True, optional=True),
        },
        _instance,
        "K posed copies of the child; per-instance deform (taper,twist,bend) and presence",
    )
)


def param_count(op: str, values: dict) -> int:
    return int(sum(v.numel() for v in values.values()))
