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

## Quickstart (CPU)

```bash
pip install -e .[dev]
pytest -q                                    # 22 tests, ~4 min on 2 CPU cores
python examples/benchmark.py --out results   # recover hand-crafted graphs from their shapes
```

```python
from mlrepl import library
from mlrepl.targets import from_graph, from_mesh
from mlrepl.search import search, SearchConfig
from mlrepl.export import export_glb

target = from_graph(library.table())          # or from_mesh(V, F), from_depth(...), from_gaussians(...)
result = search(target, SearchConfig(time_budget=300))
print(result.best.graph.pretty())
export_glb(result.best.graph, "table.glb")
```

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
| `library.py` | hand-crafted bootstrap graphs (table, mug, twisted column, fence, bent pipe, bracket, stool) |
| `render.py` | sphere-traced previews and depth maps |

See [DESIGN.md](DESIGN.md) for decisions, findings and the roadmap.
