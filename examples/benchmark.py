"""Synthetic recovery benchmark: for each hand-crafted graph, sample a target, search from scratch,
then report SDF IoU, *mesh* IoU (exact CSG realization vs target), structure, and mutation stats.

    python examples/benchmark.py --out results --budget 300 [--only table,mug]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from mlrepl import library
from mlrepl.export import export_glb
from mlrepl.realize import mesh_contains, realize
from mlrepl.render import preview, save_grid
from mlrepl.search import SearchConfig, canonicalize, search, signature
from mlrepl.targets import from_graph


def mesh_iou(g, target, n=6000) -> float:
    P = target.eval_points[:n].numpy().astype(np.float64)
    gt = (target.eval_occ[:n] > 0.5).numpy()
    m = realize(g)
    pred = mesh_contains(m, P) if not m.is_empty() else np.zeros(len(P), bool)
    return float((pred & gt).sum() / max((pred | gt).sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(a.out, exist_ok=True)
    names = a.only.split(",") if a.only else list(library.ALL)
    summary = {}
    for name in names:
        gt = library.ALL[name]()
        T = from_graph(gt, n_eval=40000, seed=a.seed)
        t0 = time.time()
        r = search(T, SearchConfig(generations=a.generations, time_budget=a.budget, seed=a.seed),
                   log=lambda s: print(f"[{name}] {s}", flush=True))
        g = r.best.graph
        g.to_json(os.path.join(a.out, f"{name}.found.json"))
        canonicalize(gt).to_json(os.path.join(a.out, f"{name}.gt.json"))
        export_glb(g, os.path.join(a.out, f"{name}.found.glb"))
        row = {
            "iou_sdf": round(r.best.iou, 4),
            "iou_mesh": round(mesh_iou(g, T), 4),
            "score": round(r.best.score, 4),
            "complexity_found": round(g.complexity(), 1),
            "complexity_gt": round(canonicalize(gt).complexity(), 1),
            "structure_found": signature(g),
            "structure_gt": signature(canonicalize(gt)),
            "evaluated": r.evaluated,
            "seconds": round(time.time() - t0, 1),
            "mutation_stats": r.mutation_stats,
            "history": r.history,
        }
        summary[name] = row
        print(f"[{name}] DONE iou_sdf={row['iou_sdf']} iou_mesh={row['iou_mesh']} {row['structure_found']}", flush=True)
        save_grid([[preview(gt, 128), preview(g, 128, color=(0.4, 0.6, 0.85))]],
                  os.path.join(a.out, f"{name}.png"), titles=["target", "found"])
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
    rows = [[preview(library.ALL[n](), 112) for n in names],
            [preview(__import__("mlrepl.graph", fromlist=["Graph"]).Graph.from_json(os.path.join(a.out, f"{n}.found.json")),
                     112, color=(0.4, 0.6, 0.85)) for n in names]]
    save_grid(rows, os.path.join(a.out, "overview.png"), titles=names, row_labels=["target", "found"])


if __name__ == "__main__":
    main()
