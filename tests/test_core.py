import math

import numpy as np
import pytest
import torch

from mlrepl import library
from mlrepl.abstraction import instantiate, mine
from mlrepl.fit import FitConfig, fit, iou
from mlrepl.graph import Graph
from mlrepl.mathutil import axis_angle_to_matrix, deform_forward_np, deform_inverse
from mlrepl.ops import EvalCtx
from mlrepl.realize import mesh_contains, realize
from mlrepl.search import canonicalize, residual, signature
from mlrepl.targets import from_depth, from_graph

torch.manual_seed(0)


def test_rotation_orthonormal_and_finite_grad_at_zero():
    w = torch.zeros(3, requires_grad=True)
    R = axis_angle_to_matrix(w)
    R.sum().backward()
    assert torch.isfinite(w.grad).all()
    w2 = torch.randn(16, 3)
    R2 = axis_angle_to_matrix(w2)
    assert torch.allclose(R2 @ R2.transpose(-1, -2), torch.eye(3).expand(16, 3, 3), atol=1e-5)
    assert torch.allclose(torch.det(R2), torch.ones(16), atol=1e-5)


@pytest.mark.parametrize("kind,k", [("twist", 1.7), ("taper", -0.6), ("bend", 1.2), ("bend", 1e-7), ("twist", 0.0)])
def test_deform_inverse_matches_forward(kind, k):
    v = np.random.default_rng(0).uniform(-0.6, 0.6, (500, 3))
    fwd = deform_forward_np(kind, v, k)
    back, _ = deform_inverse(kind, torch.tensor(fwd, dtype=torch.float64), torch.tensor(k, dtype=torch.float64))
    assert np.abs(back.numpy() - v).max() < 1e-6


def test_deform_grads_finite_at_zero():
    for kind in ("twist", "taper", "bend"):
        k = torch.zeros((), requires_grad=True)
        q, _ = deform_inverse(kind, torch.randn(64, 3), k)
        q.sum().backward()
        assert torch.isfinite(k.grad), kind


@pytest.mark.parametrize("name", list(library.ALL))
def test_sdf_and_mesh_realization_agree(name):
    g = library.ALL[name]()
    P = torch.rand(1500, 3) * 2.2 - 1.1
    occ_sdf = (g.sdf(P, EvalCtx(tau=0, hard_gates=True)) < 0).numpy()
    occ_mesh = mesh_contains(realize(g), P.numpy().astype(np.float64))
    assert (occ_sdf == occ_mesh).mean() > 0.995


def test_soft_booleans_converge_to_hard():
    g = library.mug()
    P = torch.rand(2000, 3) * 2 - 1
    hard = g.sdf(P, EvalCtx(tau=0))
    errs = [(g.sdf(P, EvalCtx(tau=t)) - hard).abs().max().item() for t in (0.05, 0.01, 0.001)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 5e-3


def test_json_roundtrip_and_canonicalize_preserve_sdf():
    P = torch.rand(2000, 3) * 2 - 1
    for f in library.ALL.values():
        g = f()
        g2 = Graph.from_json(g.to_json())
        c = canonicalize(g)
        assert torch.allclose(g.sdf(P), g2.sdf(P), atol=1e-5)
        assert torch.allclose(g.sdf(P), c.sdf(P), atol=1e-4)


def _table(legs, top_t=(0, 0.5, 0), legr=0.06):
    g = Graph()
    top = g.placed("box", pos=top_t, b=[0.8, 0.06, 0.5])
    leg = g.add("cylinder", params={"r": [legr], "h": [0.45]})
    inst = g.add("instance", [leg], params={"t": legs, "r": [[0, 0, 0]] * len(legs), "s": [[1]] * len(legs),
                                            "logit": [3] * len(legs)})
    g.output = g.add("union", [top, inst])
    return g


@pytest.fixture(scope="module")
def table_target():
    return from_graph(library.table())


def test_fit_recovers_perturbed_parameters(table_target):
    init = _table([[x * 0.7, 0.1, z * 0.6] for x in (-0.7, 0.7) for z in (-0.4, 0.4)], top_t=(0.1, 0.35, 0), legr=0.1)
    assert iou(init, table_target) < 0.2
    r = fit(init, table_target, FitConfig(steps=300))
    assert r.iou > 0.95


def test_instance_count_is_discovered(table_target):
    r = fit(_table([[0, 0, 0]], top_t=(0, 0.45, 0)), table_target, FitConfig(steps=400))
    inst = [n for n in r.graph.nodes.values() if n.op == "instance"][0]
    assert inst.k == 4 and r.iou > 0.95


def test_residual_finds_missing_part(table_target):
    g = Graph()
    g.output = g.placed("box", pos=(0, 0.5, 0), b=[0.8, 0.06, 0.5])
    res = residual(g, table_target)
    assert len(res.fn_parts) >= 4  # four legs
    assert res.fp_frac < 0.05


def test_export_shares_instance_meshes(tmp_path):
    from mlrepl.export import export_glb

    sc = export_glb(library.fence(), str(tmp_path / "f.glb"))
    assert len(sc.geometry) == 2  # one post mesh + one rail mesh
    assert len(sc.graph.nodes_geometry) == 8  # 6 posts + 2 rails


def test_depth_target_labels_are_consistent():
    from mlrepl.render import intrinsics, look_at, trace

    g = library.mug()
    M = look_at((2.5, 1.5, 2.5))
    K = intrinsics(64)
    hit, _, depth = trace(g, M, K, 64)
    T = from_depth(depth, K, M)
    with torch.no_grad():
        d = g.sdf(T.points)
    free = T.occ < 0.5
    assert (d[free] > -1e-2).float().mean() > 0.98  # carved free space really is outside
    assert g.sdf(T.surface).abs().mean() < 0.01


def test_mining_finds_shared_motif():
    progs = [library.table(), library.stool(), canonicalize(library.table())]
    lib = mine(progs, min_nodes=3)
    assert lib, "expected at least one macro"
    assert any("instance" in m.signature for m in lib)
    g = Graph()
    g.output = instantiate(g, lib[0], np.random.default_rng(0).uniform(-0.5, 0.5, (200, 3)))
    assert torch.isfinite(g.sdf(torch.rand(10, 3))).all()


def test_box_rounding_larger_than_extent_stays_realizable():
    # regression: found by the benchmark (bent pipe), SDF allowed round > min(b), mesh clamped it
    g = Graph()
    b = g.add("box", params={"b": [0.13, 0.77, 0.13], "round": [0.2]})
    g.output = g.add("bend", [b], params={"k": [1.4]})
    P = torch.rand(3000, 3) * 2.2 - 1.1
    occ_sdf = (g.sdf(P) < 0).numpy()
    occ_mesh = mesh_contains(realize(g), P.numpy().astype(np.float64))
    assert (occ_sdf == occ_mesh).mean() > 0.995
