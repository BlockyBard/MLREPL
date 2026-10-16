"""Does description length recover regularity without a domain switch?

Same search, same budget, same seed, with and without the detector moves (array / grid / radial /
mirror / symmetrize / plane / count). Regular targets should get shorter programs that use pattern
ops at equal-or-better fit; irregular controls (rubble, cairn) should get *no* pattern ops and no
loss of fit. No target is labelled regular or irregular anywhere in the search.

    python examples/regularity_experiment.py --out results/regularity --max-evals 150 --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import time

from mlrepl import library
from mlrepl.export import export_glb
from mlrepl.parallel import ProcessEvaluator
from mlrepl.render import preview, save_grid
from mlrepl.search import SearchConfig, search, signature
from mlrepl.targets import from_graph

REGULAR = ["fence", "gear", "table", "colonnade", "bracket", "stool", "chair", "bookshelf"]
IRREGULAR = ["rubble", "cairn"]


def pattern_ops(g):
    return {op: c for op, c in g.counts().items() if op in ("array", "radial", "mirror")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/regularity")
    ap.add_argument("--max-evals", type=int, default=150)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = a.only.split(",") if a.only else REGULAR + IRREGULAR
    path = os.path.join(a.out, "summary.json")
    summary = json.load(open(path)) if os.path.exists(path) else {}
    ev = ProcessEvaluator(a.workers) if a.workers > 1 else None
    try:
        for name in names:
            gt = library.ALL[name]()
            T = from_graph(gt, n_eval=60000, n_vol=30000)
            row = summary.get(name, {"kind": "irregular" if name in IRREGULAR else "regular",
                                     "gt_complexity": gt.complexity()})
            ims = [preview(gt, 112)]
            for mode in ("baseline", "detectors"):
                if mode in row:
                    continue
                cfg = SearchConfig(max_evals=a.max_evals, generations=40, patience=4, seed=0,
                                   regularity=(mode == "detectors"))
                t0 = time.time()
                r = search(T, cfg, evaluator=ev, log=lambda s, m=mode: print(f"[{name}/{m}] {s}", flush=True),
                           checkpoint_dir=os.path.join(a.out, "runs", f"{name}.{mode}"))
                g = r.best.graph
                g.to_json(os.path.join(a.out, f"{name}.{mode}.json"))
                export_glb(g.clone().to("cpu"), os.path.join(a.out, f"{name}.{mode}.glb"))
                row[mode] = {"score": round(r.best.score, 4), "iou": round(r.best.iou, 4),
                             "complexity": round(g.complexity(), 1), "patterns": pattern_ops(g),
                             "structure": signature(g), "evaluated": r.evaluated,
                             "seconds": round(time.time() - t0, 1),
                             "detector_stats": {k: v for k, v in r.mutation_stats.items()
                                                if k in ("pattern", "mirror", "symmetrize", "plane", "count")}}
                print(f"[{name}/{mode}] DONE {json.dumps(row[mode])}", flush=True)
                summary[name] = row
                with open(path, "w") as f:
                    json.dump(summary, f, indent=1)
            for mode in ("baseline", "detectors"):
                from mlrepl.graph import Graph

                ims.append(preview(Graph.from_json(os.path.join(a.out, f"{name}.{mode}.json")), 112,
                                   color=(0.4, 0.6, 0.85) if mode == "baseline" else (0.45, 0.75, 0.5)))
            save_grid([ims], os.path.join(a.out, f"{name}.png"), titles=["target", "baseline", "detectors"])
    finally:
        if ev is not None:
            ev.close()


if __name__ == "__main__":
    main()
