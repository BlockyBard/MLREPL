import math

import numpy as np
import pytest
import torch

from mlrepl import library
from mlrepl import regularity as R
from mlrepl.export import to_scene
from mlrepl.graph import Graph, GraphError
from mlrepl.ops import EvalCtx
from mlrepl.realize import mesh_contains, realize
from mlrepl.search import canonicalize, signature
from mlrepl.targets import from_graph


def _agree(g, n=2500):
    P = torch.rand(n, 3) * 2.4 - 1.2
    a = (g.sdf(P, EvalCtx(hard_gates=True)) < 0).numpy()
    b = mesh_contains(realize(g), P.numpy().astype(np.float64))
    return (a == b).mean()


def _gear_radial():
    g = Graph()
    tooth = g.placed("box", pos=(0.6, 0, 0), rot=(0, 0.2, 0), b=[0.09, 0.12, 0.08])
    g.output = g.add("union", [g.add("radial", [tooth], attrs={"n": 12}),
                               g.add("cylinder", params={"r": [0.55], "h": [0.1]})])
    return g


def test_pattern_ops_sdf_matches_mesh():
    g = Graph()
    post = g.add("box", params={"b": [0.05, 0.5, 0.05]})
    g.output = g.add("array", [post], params={"d": [0.3, 0, 0]}, attrs={"n": 6})
    assert _agree(g) > 0.999
    assert _agree(_gear_radial()) > 0.999
    h = Graph()
    ch = h.placed("capsule", pos=(0.3, 0.1, 0.45), rot=(0.3, 0.5, 0.2), r=[0.1], h=[0.3])
    h.output = h.add("mirror", [ch], params={"c": [0.1]}, attrs={"axis": 2})
    assert _agree(h) > 0.999


def test_pattern_params_get_gradients_and_validate():
    g = Graph()
    post = g.add("box", params={"b": [0.05, 0.5, 0.05]})
    g.output = g.add("array", [post], params={"d": [0.3, 0, 0]}, attrs={"n": 4})
    g.requires_grad_(True)
    g.sdf(torch.rand(500, 3) * 2 - 1).sum().backward()
    assert g.nodes[g.output].raw["d"].grad.abs().sum() > 0
    g.nodes[g.output].attrs["n"] = 0
    with pytest.raises(GraphError, match="integer n"):
        g.validate()


def test_patterns_export_as_shared_instances():
    sc = to_scene(_gear_radial())
    assert len(sc.geometry) == 2 and len(sc.graph.nodes_geometry) == 13  # 12 teeth share one mesh + disc


@pytest.mark.parametrize("name,kind", [("fence", "array"), ("gear", "radial"), ("table", "grid"),
                                       ("bracket", "array"), ("stool", "radial")])
def test_detectors_rewrite_library_patterns_exactly(name, kind):
    g = library.ALL[name]()
    rw = dict(R.deterministic_rewrites(g))
    assert kind in rw
    h = rw[kind]
    h.validate()
    P = torch.rand(4000, 3) * 2 - 1
    mismatch = ((g.sdf(P) < 0) != (h.sdf(P) < 0)).float().mean()
    assert mismatch < 0.005  # the rewrite reproduces the shape before any fitting


def test_patterns_compress_large_repeats():
    for name in ("gear", "fence", "colonnade"):
        g = library.ALL[name]()
        best = min(h.complexity() for _, h in R.deterministic_rewrites(g))
        assert best < g.complexity() - 5


def test_no_false_regularity_on_irregular_controls():
    for name in ("rubble", "cairn"):
        g = library.ALL[name]()
        assert R.deterministic_rewrites(g) == []
        assert R.symmetry_planes(from_graph(g, n_eval=20000)) == []


def test_symmetry_planes_found_on_symmetric_targets():
    planes = R.symmetry_planes(from_graph(library.table(), n_eval=20000))
    assert {ax for ax, _ in planes} >= {0, 2}
    assert all(abs(c) < 0.03 for _, c in planes)


def test_plane_slab_proposal_is_a_thin_box():
    gt = library.bookshelf()
    T = from_graph(gt, n_eval=20000)
    g = Graph()
    g.output = g.placed("box", pos=(0, 0, 0), b=[0.6, 0.92, 0.26])
    from mlrepl.search import residual

    res = residual(g, T)
    part = res.fp_parts[0] if res.fp_parts else T.eval_points[T.eval_occ > 0.5].numpy()
    h = R.plane_slab(None, part, T, np.random.default_rng(0))
    assert h is not None
    box = next(n for n in h.nodes.values() if n.op == "box")
    b = box.value("b").detach().numpy()
    assert b.min() < 0.5 * np.sort(b)[1]  # one dimension much thinner than the others


def test_canonicalize_drops_unit_array():
    g = Graph()
    s = g.add("sphere", params={"r": [0.3]})
    g.output = g.add("array", [s], params={"d": [0.3, 0, 0]}, attrs={"n": 1})
    assert signature(canonicalize(g)) == "sphere"
