"""Recover graphs from rendered multi-view images (RGB + mask, optionally depth).

    python examples/image_benchmark.py --out results/images --only mug,table --views 8 --res 64

For each target and each modality (rgb, rgbd): render views of the ground-truth graph, carve a
volumetric proxy, run structure search on the proxy, then refine geometry + per-primitive colors
with the differentiable renderer. Reports 3D IoU against the ground truth (synthetic eval only) and
held-out view metrics (mask IoU, depth MAE, PSNR).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from mlrepl import library
from mlrepl.diffrender import ImageFitConfig, make_views, orbit_cameras, render_view, target_from_views
from mlrepl.export import export_glb
from mlrepl.fit import iou
from mlrepl.pipeline import solve_from_images
from mlrepl.render import save_grid
from mlrepl.search import SearchConfig, signature
from mlrepl.targets import from_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/images")
    ap.add_argument("--only", default="mug,table,chair")
    ap.add_argument("--views", type=int, default=9)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--max_evals", type=int, default=120)
    ap.add_argument("--fit_steps", type=int, default=300)
    ap.add_argument("--modes", default="rgb,rgbd")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cams = orbit_cameras(a.views, res=a.res)
    test_cams = orbit_cameras(3, res=a.res, offset=1.1, elevations=(0.6,))
    summary = {}
    for name in a.only.split(","):
        gt = library.ALL[name]()
        T3 = from_graph(gt, n_eval=40000)
        held = make_views(gt, test_cams, with_depth=True)
        rows = [[v.rgb.reshape(a.res, a.res, 3).numpy() for v in held.views]]
        labels = ["target (held-out)"]
        for mode in a.modes.split(","):
            t0 = time.time()
            views = make_views(gt, cams, with_depth=(mode == "rgbd"))
            res = solve_from_images(views, SearchConfig(max_evals=a.max_evals, seed=0),
                                    ImageFitConfig(steps=a.fit_steps), held_out=held,
                                    log=lambda s: print(f"[{name}/{mode}] {s}", flush=True))
            g = res.graph
            row = {
                "iou3d_proxy_search": round(iou(res.proxy_graph, T3), 4),
                "iou3d_final": round(iou(g, T3), 4),
                "held_out_before_render_fit": {k: round(v, 4) for k, v in res.metrics_proxy.items()},
                "held_out_final": {k: round(v, 4) for k, v in res.metrics_final.items()},
                "structure": signature(g),
                "seconds": round(time.time() - t0, 1),
            }
            summary[f"{name}/{mode}"] = row
            print(f"[{name}/{mode}] DONE {json.dumps(row)}", flush=True)
            g.to_json(os.path.join(a.out, f"{name}.{mode}.json"))
            export_glb(g, os.path.join(a.out, f"{name}.{mode}.glb"))
            bg = torch.ones(3)
            rows.append([torch.where(r["hit"][:, None], r["rgb"], bg).reshape(a.res, a.res, 3).numpy()
                         for r in (render_view(g, v.cam) for v in held.views)])
            labels.append(mode)
            with open(os.path.join(a.out, "summary.json"), "w") as f:
                json.dump(summary, f, indent=1)
        save_grid(rows, os.path.join(a.out, f"{name}.png"), row_labels=labels)


if __name__ == "__main__":
    main()
