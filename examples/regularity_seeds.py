"""Multi-seed version of the regularity experiment: baseline vs detectors (both with warm-start fits),
N seeds per target, reporting mean and spread so differences can be told apart from run-to-run noise.

    python examples/regularity_seeds.py --out results/regularity_v2 --seeds 0,1,2 --max-evals 100 --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from mlrepl import library
from mlrepl.parallel import ProcessEvaluator
from mlrepl.search import SearchConfig, search, signature
from mlrepl.targets import from_graph

TARGETS = ["gear", "fence", "colonnade", "table", "stool", "bracket", "rubble", "cairn"]
IRREGULAR = {"rubble", "cairn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/regularity_v2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-evals", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "runs.json")
    runs = json.load(open(path)) if os.path.exists(path) else {}
    names = a.only.split(",") if a.only else TARGETS
    seeds = [int(s) for s in a.seeds.split(",")]
    ev = ProcessEvaluator(a.workers) if a.workers > 1 else None
    try:
        for seed in seeds:
            for name in names:
                T = from_graph(library.ALL[name](), n_eval=60000, n_vol=30000, seed=seed)
                for arm in ("baseline", "detectors"):
                    key = f"{name}/{arm}/{seed}"
                    if key in runs:
                        continue
                    cfg = SearchConfig(max_evals=a.max_evals, generations=40, patience=4, seed=seed,
                                       regularity=(arm == "detectors"), warm_start=True)
                    t0 = time.time()
                    r = search(T, cfg, evaluator=ev, log=lambda *_: None)
                    g = r.best.graph
                    runs[key] = {"score": r.best.score, "iou": r.best.iou, "complexity": g.complexity(),
                                 "patterns": {k: v for k, v in g.counts().items() if k in ("array", "radial", "mirror")},
                                 "structure": signature(g), "seconds": round(time.time() - t0, 1),
                                 "detector_stats": {k: v for k, v in r.mutation_stats.items()
                                                    if k in ("pattern", "mirror", "symmetrize", "plane", "count",
                                                             "data_pattern")}}
                    g.to_json(os.path.join(a.out, f"{name}.{arm}.{seed}.json"))
                    print(f"{key} score {r.best.score:.4f} iou {r.best.iou:.3f} {signature(g)[:90]}", flush=True)
                    with open(path, "w") as f:
                        json.dump(runs, f, indent=1)
    finally:
        if ev is not None:
            ev.close()
    # aggregate
    agg = {}
    for name in names:
        row = {"kind": "irregular" if name in IRREGULAR else "regular"}
        for arm in ("baseline", "detectors"):
            xs = [runs[f"{name}/{arm}/{s}"] for s in seeds if f"{name}/{arm}/{s}" in runs]
            if not xs:
                continue
            sc = np.array([x["score"] for x in xs])
            iou = np.array([x["iou"] for x in xs])
            row[arm] = {"score_mean": float(sc.mean()), "score_std": float(sc.std()), "iou_mean": float(iou.mean()),
                        "pattern_runs": sum(bool(x["patterns"]) for x in xs), "n": len(xs)}
        agg[name] = row
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
