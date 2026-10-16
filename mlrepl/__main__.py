"""Command line interface.

    python -m mlrepl search library:table --out runs/table --max-evals 150
    python -m mlrepl search runs/table --resume                 # continue a paused/killed run
    python -m mlrepl search scan.obj --out runs/scan --config my_search.json --device cuda
    python -m mlrepl fit graph.json library:table --out fitted.json
    python -m mlrepl validate graph.json
    python -m mlrepl export graph.json out.glb
    python -m mlrepl preview graph.json out.png
    python -m mlrepl config search > my_search.json             # dump defaults to edit

Targets: `library:NAME`, a graph `.json`, a mesh (`.obj .ply .stl .off .glb .gltf`), or a saved `target.pt`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def load_target(spec: str, n_eval: int = 60000, seed: int = 0):
    from . import library
    from .graph import Graph
    from .targets import Target, from_graph, from_mesh

    if spec.startswith("library:"):
        return from_graph(library.ALL[spec.split(":", 1)[1]](), n_eval=n_eval, n_vol=30000, seed=seed)
    ext = os.path.splitext(spec)[1].lower()
    if ext == ".json":
        return from_graph(Graph.from_json(spec), n_eval=n_eval, n_vol=30000, seed=seed)
    if ext == ".pt":
        return torch.load(spec, weights_only=False)
    if ext in (".obj", ".ply", ".stl", ".off", ".glb", ".gltf"):
        import trimesh

        m = trimesh.load(spec, force="mesh")
        return from_mesh(m.vertices, m.faces, n_eval=n_eval, n_vol=30000, seed=seed)
    raise SystemExit(f"don't know how to load target {spec!r}")


def _device(name: str):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def cmd_search(a):
    from .config import load_config, save_config
    from .export import export_glb
    from .render import preview, save_grid
    from .search import SearchConfig, search, signature

    out = a.out or (a.target if a.resume and a.target and os.path.isdir(a.target) else None)
    if not out:
        raise SystemExit("--out is required (or pass a run directory with --resume)")
    os.makedirs(out, exist_ok=True)
    tpath = os.path.join(out, "target.pt")
    if a.resume and os.path.exists(tpath):
        target = torch.load(tpath, weights_only=False)
    else:
        target = load_target(a.target, a.n_eval, a.seed)
        torch.save(target, tpath)
    target = target.to(_device(a.device))
    cfg = None
    if a.config:
        cfg = load_config(SearchConfig, a.config)
    elif not a.resume:
        cfg = SearchConfig(seed=a.seed)
    if a.max_evals is not None or a.generations is not None:
        cfg = cfg or (load_config(SearchConfig, os.path.join(out, "config.json"))
                      if os.path.exists(os.path.join(out, "config.json")) else SearchConfig(seed=a.seed))
        if a.max_evals is not None:
            cfg.max_evals = a.max_evals
        if a.generations is not None:
            cfg.generations = a.generations
    if cfg is not None:
        save_config(cfg, os.path.join(out, "config.json"))
    evaluator = None
    if a.workers > 1:
        from .parallel import ProcessEvaluator

        evaluator = ProcessEvaluator(a.workers)
    t0 = time.time()
    try:
        r = search(target, cfg, checkpoint_dir=out, resume=a.resume, evaluator=evaluator,
                   log=lambda s: print(s, flush=True))
    finally:
        if evaluator is not None:
            evaluator.close()
    g = r.best.graph
    g.to_json(os.path.join(out, "best.json"))
    export_glb(g.clone().to("cpu"), os.path.join(out, "best.glb"))
    if not a.no_preview:
        save_grid([[preview(g.clone().to("cpu"), 160)]], os.path.join(out, "preview.png"))
    summary = {"score": r.best.score, "iou": r.best.iou, "structure": signature(g), "complexity": g.complexity(),
               "evaluated": r.evaluated, "seconds": round(time.time() - t0, 1), "mutation_stats": r.mutation_stats}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("score", "iou", "structure", "evaluated", "seconds")}))


def cmd_fit(a):
    from .fit import FitConfig, fit
    from .graph import Graph

    target = load_target(a.target, a.n_eval).to(_device(a.device))
    g = Graph.from_json(a.graph)
    g.validate()
    r = fit(g, target, FitConfig(steps=a.steps))
    r.graph.to_json(a.out)
    print(json.dumps({"iou": r.iou, "score": r.score}))


def cmd_validate(a):
    from .graph import Graph, GraphError

    try:
        g = Graph.from_json(a.graph)
        g.validate()
    except (GraphError, KeyError, ValueError) as e:
        print(f"INVALID: {e}")
        sys.exit(1)
    print(f"ok: {len(g.reachable())} nodes, complexity {g.complexity():.1f}")


def cmd_export(a):
    from .export import export_glb
    from .graph import Graph

    export_glb(Graph.from_json(a.graph), a.out)
    print(a.out)


def cmd_preview(a):
    from .graph import Graph
    from .render import preview, save_grid

    save_grid([[preview(Graph.from_json(a.graph), a.res)]], a.out)
    print(a.out)


def cmd_config(a):
    from .config import to_dict
    from .diffrender import ImageFitConfig
    from .fit import FitConfig
    from .search import SearchConfig

    cls = {"search": SearchConfig, "fit": FitConfig, "imagefit": ImageFitConfig}[a.kind]
    print(json.dumps(to_dict(cls()), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m mlrepl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="discover a graph for a target (checkpointed, resumable)")
    s.add_argument("target", nargs="?", help="target spec, or the run directory with --resume")
    s.add_argument("--out")
    s.add_argument("--config", help="SearchConfig JSON (see `config search`)")
    s.add_argument("--max-evals", type=int)
    s.add_argument("--generations", type=int)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--device", default="auto")
    s.add_argument("--workers", type=int, default=1, help="parallel candidate fits (processes)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n-eval", type=int, default=60000)
    s.add_argument("--no-preview", action="store_true")
    s.set_defaults(fn=cmd_search)

    f = sub.add_parser("fit", help="fit the parameters of a fixed graph to a target")
    f.add_argument("graph")
    f.add_argument("target")
    f.add_argument("--out", required=True)
    f.add_argument("--steps", type=int, default=300)
    f.add_argument("--device", default="auto")
    f.add_argument("--n-eval", type=int, default=40000)
    f.set_defaults(fn=cmd_fit)

    v = sub.add_parser("validate", help="check a graph file (types, arity, cycles, references, values)")
    v.add_argument("graph")
    v.set_defaults(fn=cmd_validate)

    e = sub.add_parser("export", help="graph JSON -> structured glTF/GLB")
    e.add_argument("graph")
    e.add_argument("out")
    e.set_defaults(fn=cmd_export)

    pv = sub.add_parser("preview", help="sphere-traced PNG of a graph")
    pv.add_argument("graph")
    pv.add_argument("out")
    pv.add_argument("--res", type=int, default=256)
    pv.set_defaults(fn=cmd_preview)

    c = sub.add_parser("config", help="print a default config as JSON")
    c.add_argument("kind", choices=["search", "fit", "imagefit"])
    c.set_defaults(fn=cmd_config)

    a = p.parse_args(argv)
    if a.cmd == "search" and not a.target:
        p.error("search needs a target (or a run directory with --resume)")
    a.fn(a)


if __name__ == "__main__":
    main()
