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


def to_scene(g: Graph, opts: RealizeOpts = None, bake_deformed_instances: bool = True) -> trimesh.Scene:
    opts = opts or RealizeOpts()
    scene = trimesh.Scene()
    root = g.nodes[g.output]
    parts = root.inputs if root.op == "union" else [g.output]
    for pid in parts:
        n = g.nodes[pid]
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
