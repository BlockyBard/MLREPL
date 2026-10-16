import numpy as np
import torch

from mlrepl import library
from mlrepl.diffrender import (Camera, ImageFitConfig, RenderOpts, carve, fit_images, make_views, orbit_cameras,
                               render)
from mlrepl.fit import iou
from mlrepl.graph import Graph
from mlrepl.ops import EvalCtx
from mlrepl.render import intrinsics, look_at
from mlrepl.targets import from_gaussians, from_graph
from mlrepl.realize import manifold_to_arrays, realize


def _central_rays(n=24):
    cam = Camera(look_at((0, 0, 3.0)), intrinsics(n, 20.0), n, n)
    return cam.rays()


def test_implicit_depth_gradient_matches_finite_difference():
    g = Graph()
    g.output = g.placed("sphere", pos=(0.05, 0.02, 0.0), r=[0.6])
    o, d, z = _central_rays()
    ctx, opts = EvalCtx(hard_gates=True), RenderOpts()
    g.requires_grad_(True)
    r = render(g, o, d, z, ctx, opts, want_color=False)
    hit = r["hit"].clone()
    assert hit.float().mean() > 0.3
    r["depth"][hit].sum().backward()
    ana = torch.cat([g.nodes["tra1"].raw["t"].grad, g.nodes["sph0"].raw["r"].grad])

    def val(dt=torch.zeros(3), dr=0.0):
        h = g.clone()
        with torch.no_grad():
            h.nodes["tra1"].raw["t"] += dt
            h.nodes["sph0"].raw["r"] += dr
        rr = render(h, o, d, z, ctx, opts, want_color=False)
        return float(rr["depth"][hit].sum())

    e = 1e-3
    fd = [(val(dt=torch.eye(3)[i] * e) - val(dt=-torch.eye(3)[i] * e)) / (2 * e) for i in range(3)]
    fd.append((val(dr=e) - val(dr=-e)) / (2 * e))
    fd = torch.tensor(fd)
    assert torch.allclose(ana, fd, rtol=0.1, atol=0.5 + 0.02 * fd.abs().max())


def test_render_colors_follow_csg():
    g = library.mug()
    cam = orbit_cameras(1, res=32)[0]
    o, d, z = cam.rays()
    r = render(g, o, d, z, EvalCtx(hard_gates=True))
    assert r["hit"].any()
    alb = r["albedo"][r["hit"]]
    # the mug body is authored ceramic, its interior (cut by an un-materialed cutter) is default gray
    assert ((alb - torch.tensor([0.9, 0.88, 0.82])).abs().sum(-1) < 1e-4).any()


def test_image_fit_is_stationary_at_ground_truth():
    gt = library.table()
    views = make_views(gt, orbit_cameras(6, res=40), with_depth=True)
    g, _ = fit_images(gt, views, ImageFitConfig(steps=40, rays=768, fit_color=False))
    assert iou(g, from_graph(gt, n_eval=20000)) > 0.98


def test_image_fit_recovers_colors():
    gt = library.table()
    views = make_views(gt, orbit_cameras(6, res=40), with_depth=False)
    g, _ = fit_images(gt, views, ImageFitConfig(steps=80, rays=768, fit_geometry=False))
    cols = [g.nodes[n].value("color") for n in g.nodes if "color" in g.nodes[n].raw]
    assert cols and all((c - torch.tensor([0.55, 0.36, 0.2])).abs().max() < 0.05 for c in cols)


def test_carving_with_depth_beats_visual_hull():
    gt = library.table()
    res = 48
    ax = np.linspace(-1.1, 1.1, res)
    P = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    with torch.no_grad():
        occ_gt = (gt.sdf(torch.tensor(P, dtype=torch.float32)) < 0).numpy().reshape(res, res, res)
    ious = {}
    for depth in (False, True):
        oc = carve(make_views(gt, orbit_cameras(9, res=64), with_depth=depth), res=res)
        ious[depth] = (oc & occ_gt).sum() / (oc | occ_gt).sum()
        assert (occ_gt & ~oc).sum() / occ_gt.sum() < 0.15  # carving keeps most true volume (thin parts lose some at low res)
    assert ious[True] > ious[False] and ious[True] > 0.75


def test_gaussian_splat_target_is_solid():
    gt = library.mug()
    V, F = manifold_to_arrays(realize(gt))
    rng = np.random.default_rng(0)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(fn, axis=1)
    idx = rng.choice(len(F), 8000, p=area / area.sum())
    u, v = rng.random((2, 8000))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    mu = tri[idx, 0] + u[:, None] * (tri[idx, 1] - tri[idx, 0]) + v[:, None] * (tri[idx, 2] - tri[idx, 0])
    normals = fn[idx] / area[idx, None]  # manifold3d meshes are outward-wound
    T = from_gaussians(mu, np.ones(len(mu)), np.full((len(mu), 3), 0.015), normals)
    with torch.no_grad():
        gt_occ = gt.sdf(T.eval_points) < 0
    pred = T.eval_occ > 0.5
    assert float((pred & gt_occ).sum() / (pred | gt_occ).sum()) > 0.85
