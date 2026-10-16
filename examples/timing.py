"""Fixed-budget timing benchmark for the search infrastructure.

    python examples/timing.py --out timing.json [--workers 2] [--targets table,mug,bracket] [--max-evals 60]

Reports wall time, candidate fits per second and final score per target, so infrastructure changes
(parallel evaluation, early stopping, compile) can be compared at equal search budget.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from mlrepl import library
from mlrepl.search import SearchConfig, search, signature
from mlrepl.targets import from_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="timing.json")
    ap.add_argument("--targets", default="table,mug,bracket")
    ap.add_argument("--max-evals", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    evaluator = None
    if a.workers > 1:
        from mlrepl.parallel import ProcessEvaluator

        evaluator = ProcessEvaluator(a.workers)
    rows = {}
    try:
        for name in a.targets.split(","):
            T = from_graph(library.ALL[name](), n_eval=40000, n_vol=30000)
            t0 = time.time()
            r = search(T, SearchConfig(max_evals=a.max_evals, generations=50, patience=50, seed=0),
                       log=lambda *_: None, evaluator=evaluator)
            dt = time.time() - t0
            rows[name] = {"seconds": round(dt, 1), "evaluated": r.evaluated, "fits_per_s": round(r.evaluated / dt, 3),
                          "score": round(r.best.score, 4), "iou": round(r.best.iou, 4), "structure": signature(r.best.graph)}
            print(name, rows[name], flush=True)
    finally:
        if evaluator is not None:
            evaluator.close()
    rows["_total_seconds"] = round(sum(v["seconds"] for v in rows.values() if isinstance(v, dict)), 1)
    rows["_label"] = a.label
    rows["_threads"] = torch.get_num_threads()
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("total", rows["_total_seconds"], "s")


if __name__ == "__main__":
    main()
