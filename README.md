# MLREPL: differentiable graph synthesis for procedural 3D

Given a target shape (SDF, dense mesh, depth maps, Gaussian splats; multi-view RGB on the roadmap),
discover a **directed procedural graph** that regenerates it, fit its parameters by gradient descent, and
export the result as **game-ready structured meshes**: exact CSG, real GPU instancing, per-instance
deformation data, PBR materials.

```
target ──► residual analysis ──► mutations (add / subtract / swap / deform / instancify / macro …)
              ▲                                   │
              │                                   ▼
        hard-CSG IoU  ◄── soft→hard annealed fit (PyTorch) ──► score = (1 − IoU) + β·complexity
                                                  │
                                    best graph ──► manifold3d exact CSG ──► glTF (instances, PBR)
```

The contract that ties it together: **every op has two semantics**, a differentiable SDF (`ops.py`)
and an exact mesh realization (`realize.py`). Fitting uses soft booleans (log-sum-exp, temperature τ)
early and anneals τ → 0, so the scored solution *is* the exact CSG mesh that gets exported. Tests
check that the two semantics agree on ≥ 99.5 % of sample points for every op.

## Quickstart

```bash
pip install -e .[dev]
pytest -q                                          # ~4 min on 2 CPU cores; test_device also runs on CUDA if present

# discover a graph (checkpointed every generation; Ctrl-C safe)
python -m mlrepl search library:table --out runs/table --max-evals 150 --workers 2
python -m mlrepl search runs/table --resume --max-evals 300     # continue / extend a run
python -m mlrepl search scan.obj --out runs/scan --device cuda   # mesh target on GPU

python -m mlrepl config search > my_search.json    # edit, then: --config my_search.json
python -m mlrepl validate graph.json               # check a hand-written / LLM-written graph
python -m mlrepl export runs/table/best.json table.glb
python -m mlrepl preview runs/table/best.json table.png
```

A run directory holds `best.json`, `best.glb`, `preview.png`, `summary.json`, `state.json`
(resume point), `log.jsonl` (one line per generation), `config.json` and `target.pt`.

```python
from mlrepl import library
from mlrepl.targets import from_graph, from_mesh
from mlrepl.search import search, SearchConfig
from mlrepl.parallel import ProcessEvaluator
from mlrepl.export import export_glb

target = from_graph(library.table()).to("cuda")   # or from_mesh / from_depth / from_gaussians
result = search(target, SearchConfig(max_evals=150), checkpoint_dir="runs/table")
print(result.best.graph.pretty())
export_glb(result.best.graph.to("cpu"), "table.glb")
```

Images: `pipeline.solve_from_images(views)` (carve → search → differentiable-render refinement);
see `examples/image_benchmark.py`.

## Layout

| module | what |
|---|---|
| `graph.py` | typed DAG IR, raw/constrained params, JSON, complexity (description length) |
| `ops.py` | op registry with SDF semantics: primitives, transform, taper/twist/bend, union/intersect/subtract, instance |
| `realize.py` | exact mesh semantics via manifold3d; deformers as vertex warps; winding-number containment |
| `export.py` | structured glTF: parts as nodes, shared instance meshes, per-instance deform extras, materials per source primitive |
| `targets.py` | SDF / graph / mesh / depth / Gaussian targets → labelled volume + surface samples |
| `fit.py` | annealed fitting, robust losses, splat-style instance densify + leave-one-out pruning |
| `search.py` | residuals, moment-matched primitive proposals, mutations, beam search, canonicalization |
| `abstraction.py` | library learning: mine frequent subtrees from solved programs, use them as proposals |
| `library.py` | hand-crafted bootstrap graphs (12: table, mug, … chair, bookshelf, gear, lamp, colonnade) |
| `render.py` | sphere-traced previews and depth maps |
| `diffrender.py` | differentiable renderer (mask/depth/normal/albedo), multi-view targets, carving, image fitting |
| `pipeline.py` | images → proxy → search → render refinement |
| `parallel.py` | process-pool candidate evaluation, vmap-batched multi-start fitting |
| `config.py` | JSON configs |
| `__main__.py` | CLI |

See [DESIGN.md](DESIGN.md) for decisions, findings and the roadmap.
