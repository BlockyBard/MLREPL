"""Exact mesh semantics for graphs: hard CSG via manifold3d, deformers as vertex warps.

This is the 'realizability' half of the contract: for tau == 0 and hard gates, the SDF
evaluated by Graph.sdf and the mesh produced here describe the same solid (up to
tessellation of curved primitives).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from manifold3d import CrossSection, Manifold, Mesh, OpType

from .graph import Graph
from .mathutil import DEFORM_KINDS, apply_deform_chain_forward_np, axis_angle_to_matrix_np

DEFAULT_PALETTE = [
    (0.80, 0.42, 0.25), (0.30, 0.55, 0.80), (0.45, 0.70, 0.35), (0.85, 0.75, 0.30),
    (0.60, 0.40, 0.70), (0.35, 0.70, 0.70), (0.75, 0.35, 0.45), (0.55, 0.55, 0.55),
]


@dataclass
class RealizeOpts:
    segments: int = 48
    refine_len: float = 0.03  # edge length before vertex deformation
    deform_eps: float = 1e-4  # |k| below this => no warp


@dataclass
class MaterialTable:
    """Maps manifold original ids (one per realized primitive) back to source node + PBR material."""

    by_oid: dict = field(default_factory=dict)

    def material_for(self, node_id, attrs) -> dict:
        mat = dict(attrs.get("material", {}))
        if "base_color" not in mat:
            mat["base_color"] = DEFAULT_PALETTE[hash(node_id) % len(DEFAULT_PALETTE)]
        mat.setdefault("roughness", 0.6)
        mat.setdefault("metallic", 0.0)
        return mat


def _np(t):
    return t.detach().cpu().double().numpy()


def _affine(t, r, s):
    R = axis_angle_to_matrix_np(r) * float(s)
    return np.concatenate([R, np.asarray(t, dtype=np.float64).reshape(3, 1)], 1)


def _primitive(op, v, opts: RealizeOpts) -> Manifold:
    seg = opts.segments
    if op == "sphere":
        return Manifold.sphere(float(v["r"][0]), seg)
    if op == "box":
        b, rnd = v["b"], float(v["round"][0])
        rnd = min(rnd, float(b.min()) * 0.999)
        if rnd < 1e-3:
            return Manifold.cube(tuple(2 * b), True)
        inner = b - rnd
        pts = [Manifold.sphere(rnd, max(8, seg // 2)).translate(tuple(inner * np.array(sg)))
               for sg in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)]
        return Manifold.batch_hull(pts)
    if op == "cylinder":
        r, h = float(v["r"][0]), float(v["h"][0])
        return Manifold.cylinder(2 * h, r, -1.0, seg, True).rotate((-90.0, 0.0, 0.0))
    if op == "capsule":
        r, h = float(v["r"][0]), float(v["h"][0])
        sph = Manifold.sphere(r, seg)
        return Manifold.batch_hull([sph.translate((0, h, 0)), sph.translate((0, -h, 0))])
    if op == "torus":
        R, r = float(v["R"][0]), float(v["r"][0])
        return CrossSection.circle(r, max(12, seg // 2)).translate((R, 0)).revolve(seg).rotate((90.0, 0.0, 0.0))
    raise ValueError(op)


def warp(m: Manifold, ks: dict, opts: RealizeOpts) -> Manifold:
    ks = {k: float(v) for k, v in ks.items() if abs(float(v)) > opts.deform_eps}
    if not ks:
        return m
    m = m.refine_to_length(opts.refine_len)
    return m.warp_batch(lambda v: apply_deform_chain_forward_np(np.asarray(v), ks))


def realize(g: Graph, root=None, opts: RealizeOpts = None, mats: MaterialTable = None) -> Manifold:
    """Bake the (sub)graph into a single exact manifold."""
    opts = opts or RealizeOpts()
    mats = mats if mats is not None else MaterialTable()
    memo = {}

    def rec(nid) -> Manifold:
        if nid in memo:
            return memo[nid]
        n = g.nodes[nid]
        v = {k: _np(x) for k, x in n.values().items()}
        cat = n.spec.category
        if cat == "prim":
            m = _primitive(n.op, v, opts).as_original()
            mats.by_oid[m.original_id()] = (nid, mats.material_for(nid, n.attrs))
        elif cat == "xform":
            m = rec(n.inputs[0]).transform(_affine(v["t"], v["r"], v["s"][0]))
        elif cat == "deform":
            m = warp(rec(n.inputs[0]), {n.op: v["k"][0]}, opts)
        elif n.op == "union":
            m = Manifold.batch_boolean([rec(c) for c in n.inputs], OpType.Add)
        elif n.op == "intersect":
            m = Manifold.batch_boolean([rec(c) for c in n.inputs], OpType.Intersect)
        elif n.op == "subtract":
            m = rec(n.inputs[0]) - rec(n.inputs[1])
        elif cat == "inst":
            proto = rec(n.inputs[0])
            parts = []
            for i in range(n.k):
                if v["logit"][i] < 0:
                    continue
                mi = proto
                if "dk" in v:
                    mi = warp(mi, dict(zip(DEFORM_KINDS, v["dk"][i])), opts)
                parts.append(mi.transform(_affine(v["t"][i], v["r"][i], v["s"][i][0])))
            m = Manifold.batch_boolean(parts, OpType.Add) if parts else Manifold()
        else:
            raise ValueError(n.op)
        memo[nid] = m
        return m

    return rec(root or g.output)


def manifold_to_arrays(m: Manifold):
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3].astype(np.float64), np.asarray(mesh.tri_verts).astype(np.int64)


def manifold_material_runs(m: Manifold, mats: MaterialTable):
    """Split a baked manifold into (verts, faces, material, source_node) per original primitive."""
    mesh = m.to_mesh()
    V = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    F = np.asarray(mesh.tri_verts).astype(np.int64)
    run_idx = np.asarray(mesh.run_index).astype(np.int64)
    if run_idx.size and run_idx[-1] == 3 * len(F):
        run_idx = run_idx // 3
    out = {}
    for j, oid in enumerate(np.asarray(mesh.run_original_id)):
        a, b = run_idx[j], run_idx[j + 1]
        nid, mat = mats.by_oid.get(int(oid), (f"oid{oid}", {"base_color": (0.7, 0.7, 0.7), "roughness": 0.6, "metallic": 0.0}))
        out.setdefault(nid, (mat, []))[1].append(F[a:b])
    return V, [(nid, mat, np.concatenate(fs)) for nid, (mat, fs) in out.items()]


def mesh_volume(m: Manifold) -> float:
    return float(m.volume())


# ---------------------------------------------------------------------------
# Containment for verification / mesh targets: generalized winding number
# ---------------------------------------------------------------------------


def winding_number(V: np.ndarray, F: np.ndarray, P: np.ndarray, budget: float = 4e6) -> np.ndarray:
    """Jacobson et al. generalized winding number. ~1 inside, ~0 outside. O(N*T); fine for tests."""
    tri = V[F]  # T,3,3
    chunk = max(1, int(budget // max(len(F), 1)))
    out = np.empty(len(P))
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, None, :]  # n,1,1,3
        d = tri[None] - p  # n,T,3,3
        a, b, c = d[..., 0, :], d[..., 1, :], d[..., 2, :]
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = np.einsum("ntk,ntk->nt", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("ntk,ntk->nt", a, b) * lc + np.einsum("ntk,ntk->nt", b, c) * la \
            + np.einsum("ntk,ntk->nt", c, a) * lb
        out[s:s + chunk] = np.arctan2(det, den).sum(-1) / (2 * np.pi)
    return out


def mesh_contains(m_or_VF, P: np.ndarray) -> np.ndarray:
    if isinstance(m_or_VF, Manifold):
        V, F = manifold_to_arrays(m_or_VF)
    else:
        V, F = m_or_VF
    return winding_number(V, F, P) > 0.5
