"""Game-engine export: structured glTF scene with PBR materials and real instancing.

Structure kept in the export:
  * top-level union children become separate scene nodes (not one welded blob)
  * instance nodes become ONE prototype mesh referenced by K scene nodes (GPU-instancable)
  * per-instance deformation coefficients are written to node extras (taper, twist, bend) so an
    engine can reproduce them with a WPO vertex shader + per-instance custom data; optionally the
    deformed copies are also baked
  * boolean subtrees are baked to exact CSG meshes, split by source primitive for materials
"""
from __future__ import annotations

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

from .graph import Graph
from .mathutil import DEFORM_KINDS
from .realize import MaterialTable, RealizeOpts, _affine, _np, manifold_material_runs, realize, warp


def _tm(V, F, mat) -> trimesh.Trimesh:
    m = trimesh.Trimesh(V, F, process=False)
    c = list(mat["base_color"]) + [1.0]
    m.visual = trimesh.visual.TextureVisuals(
        material=PBRMaterial(baseColorFactor=c[:4], roughnessFactor=mat["roughness"], metallicFactor=mat["metallic"])
    )
    return m


def _add_baked(scene, g, nid, name, opts, transform=None):
    mats = MaterialTable()
    m = realize(g, nid, opts, mats)
    if m.is_empty():
        return
    V, runs = manifold_material_runs(m, mats)
    for src, mat, F in runs:
        used = np.unique(F)
        remap = -np.ones(len(V), dtype=np.int64)
        remap[used] = np.arange(len(used))
        scene.add_geometry(_tm(V[used], remap[F], mat), node_name=f"{name}/{src}", geom_name=f"{name}/{src}",
                           transform=transform)


def expand_instances(g: Graph, nid: str):
    """Flatten chains of transform / pattern (mirror, array, radial) / plain instance nodes into
    [(proto_id, 4x4 world-from-proto)], or None if the subtree is not pure instancing
    (booleans, deformers, per-instance deformation stop the expansion)."""
    import torch

    from .ops import pattern_offsets

    n = g.nodes[nid]
    cat = n.spec.category
    if cat not in ("xform", "pattern", "inst"):
        return [(nid, np.eye(4))]
    if cat == "inst" and "dk" in n.raw:
        return None
    sub = expand_instances(g, n.inputs[0])
    if sub is None:
        return None
    v = {k: _np(x) for k, x in n.values().items()}
    if cat == "xform":
        A = np.eye(4)
        A[:3] = _affine(v["t"], v["r"], v["s"][0])
        mats = [A]
    elif cat == "pattern":
        R, t = pattern_offsets(n.op, n.attrs, {k: torch.as_tensor(x) for k, x in v.items()}, dtype=torch.float64)
        mats = []
        for i in range(R.shape[0]):
            A = np.eye(4)
            A[:3, :3], A[:3, 3] = R[i].numpy(), t[i].numpy()
            mats.append(A)
    else:
        mats = []
        for i in range(n.k):
            if v["logit"][i] >= 0:
                A = np.eye(4)
                A[:3] = _affine(v["t"][i], v["r"][i], v["s"][i][0])
                mats.append(A)
    return [(p, A @ B) for A in mats for p, B in sub]


def _add_instanced(scene, g, pid, expansion, opts):
    """One shared mesh per prototype, referenced by one scene node per copy."""
    for proto_id in dict.fromkeys(p for p, _ in expansion):
        mats = MaterialTable()
        V, runs = manifold_material_runs(realize(g, proto_id, opts, mats), mats)
        geoms = []
        for src, mat, F in runs:
            used = np.unique(F)
            remap = -np.ones(len(V), dtype=np.int64)
            remap[used] = np.arange(len(used))
            geoms.append((f"{pid}/proto/{proto_id}/{src}", _tm(V[used], remap[F], mat)))
        for i, (p, A) in enumerate(x for x in expansion if x[0] == proto_id):
            for gname, geom in geoms:
                node_name = f"{pid}/{proto_id}/i{i}/{gname.rsplit('/', 1)[-1]}"
                if gname not in scene.geometry:
                    scene.add_geometry(geom, geom_name=gname, node_name=node_name, transform=A)
                else:
                    scene.graph.update(frame_to=node_name, frame_from=scene.graph.base_frame, matrix=A, geometry=gname)


def to_scene(g: Graph, opts: RealizeOpts = None, bake_deformed_instances: bool = True) -> trimesh.Scene:
    opts = opts or RealizeOpts()
    scene = trimesh.Scene()
    root = g.nodes[g.output]
    parts = root.inputs if root.op == "union" else [g.output]
    for pid in parts:
        n = g.nodes[pid]
        exp = expand_instances(g, pid)
        if exp is not None and len(exp) > 1:  # pure instancing (incl. mirror/array/radial): share meshes
            _add_instanced(scene, g, pid, exp, opts)
            continue
        if n.spec.category != "inst":
            _add_baked(scene, g, pid, pid, opts)
            continue
        v = {k: _np(x) for k, x in n.values().items()}
        proto_id = n.inputs[0]
        mats = MaterialTable()
        proto = realize(g, proto_id, opts, mats)
        V, runs = manifold_material_runs(proto, mats)
        proto_geoms = []
        for src, mat, F in runs:
            gname = f"{pid}/proto/{src}"
            used = np.unique(F)
            remap = -np.ones(len(V), dtype=np.int64)
            remap[used] = np.arange(len(used))
            proto_geoms.append((gname, _tm(V[used], remap[F], mat)))
        for i in range(n.k):
            if v["logit"][i] < 0:
                continue
            A = np.eye(4)
            A[:3] = _affine(v["t"][i], v["r"][i], v["s"][i][0])
            dk = dict(zip(DEFORM_KINDS, v["dk"][i].tolist())) if "dk" in v else {}
            deformed = any(abs(x) > opts.deform_eps for x in dk.values())
            node_name = f"{pid}/i{i}"
            if deformed and bake_deformed_instances:
                mi = warp(proto, dk, opts)
                Vi, runs_i = manifold_material_runs(mi, mats)
                for src, mat, F in runs_i:
                    scene.add_geometry(_tm(Vi, F, mat), node_name=f"{node_name}/{src}", geom_name=f"{node_name}/{src}",
                                       transform=A)
            else:
                for gname, geom in proto_geoms:
                    if gname not in scene.geometry:
                        scene.add_geometry(geom, geom_name=gname, node_name=f"{node_name}/{gname}", transform=A)
                    else:
                        scene.graph.update(frame_to=f"{node_name}/{gname}", frame_from=scene.graph.base_frame,
                                           matrix=A, geometry=gname)
            scene.graph.update(frame_to=node_name, frame_from=scene.graph.base_frame,
                               extras={"source": pid, "instance": i, "deform": dk})
    return scene


def export_glb(g: Graph, path: str, **kw) -> trimesh.Scene:
    scene = to_scene(g, **kw)
    scene.export(path)
    return scene
