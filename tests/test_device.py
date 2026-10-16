"""Device-agnostic smoke tests. Always run on CPU; also run on CUDA when available (e.g. your GPU box),
which is where device mix-ups would actually surface."""
import pytest
import torch

from mlrepl import library
from mlrepl.diffrender import ImageFitConfig, fit_images, make_views, orbit_cameras
from mlrepl.fit import FitConfig, fit, iou
from mlrepl.search import SearchConfig, search
from mlrepl.targets import from_graph

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_graph_and_target_move_together(device):
    g = library.table().to(device)
    assert all(r.device.type == device for n in g.nodes.values() for r in n.raw.values())
    T = from_graph(library.table(), n_eval=5000).to(device)
    assert T.points.device.type == device
    assert iou(g, T) > 0.95
    c = g.clone()
    assert c.device.type == device and next(iter(c.nodes.values())).raw  # clone keeps device
    nid = c.add("sphere", params={"r": [0.1]})
    assert c.nodes[nid].raw["r"].device.type == device  # new nodes land on the graph's device


@pytest.mark.parametrize("device", DEVICES)
def test_fit_and_short_search_on_device(device):
    T = from_graph(library.bracket(), n_eval=8000).to(device)
    r = fit(library.bracket(), T, FitConfig(steps=20, nv=512, ns=128))
    assert r.graph.device.type == device and r.iou > 0.9
    s = search(T, SearchConfig(max_evals=4, beam=1, proposals_per_parent=2, simplify=False,
                               fit=FitConfig(steps=10, nv=256, ns=64), final_fit=FitConfig(steps=10, nv=256, ns=64)),
               log=lambda *_: None)
    assert s.best.graph.device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_render_fit_on_device(device):
    gt = library.table().to(device)
    views = make_views(gt, orbit_cameras(4, res=24), with_depth=True)
    assert views.device.type == device
    g, hist = fit_images(gt, views, ImageFitConfig(steps=5, rays=256))
    assert g.device.type == device and all(h == h for h in hist)
