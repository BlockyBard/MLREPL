"""End-to-end solvers from raw observations.

images (RGB + mask [+ depth]) ──carve──► volumetric proxy ──search──► graph ──render-fit──► graph + colors
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .diffrender import ImageFitConfig, ImageTarget, fit_images, target_from_views, view_metrics
from .graph import Graph
from .search import SearchConfig, SearchResult, search


@dataclass
class ImageSolve:
    graph: Graph
    proxy_graph: Graph
    search: SearchResult
    metrics_proxy: dict
    metrics_final: dict
    fit_history: list = field(default_factory=list)


def solve_from_images(views: ImageTarget, search_cfg: Optional[SearchConfig] = None,
                      fit_cfg: Optional[ImageFitConfig] = None, held_out: Optional[ImageTarget] = None,
                      carve_res: int = 72, log: Callable = print) -> ImageSolve:
    proxy = target_from_views(views, res=carve_res, n_eval=40000)
    r = search(proxy, search_cfg or SearchConfig(), log=log)
    g0 = r.best.graph
    g, hist = fit_images(g0, views, fit_cfg or ImageFitConfig())
    ev = held_out or views
    return ImageSolve(g, g0, r, view_metrics(g0, ev), view_metrics(g, ev), hist)
