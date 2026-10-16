# Design notes

## Problem

Inverse procedural modeling: given an observation of a shape, find a program, here a directed acyclic graph
of shape operators, whose execution reproduces it, and whose output is directly usable in a game engine.
Two coupled problems:

* **structure**: which nodes, which ops, how wired: discrete, searched
* **parameters**: poses, sizes, deformation coefficients, instance sets: continuous, fitted by gradient

## The realizability contract

Each op has an SDF semantics (differentiable, `ops.py`) and a mesh semantics (exact, `realize.py`):

| op | SDF (fitting) | mesh (export) |
|---|---|---|
| primitives | analytic SDFs, +Y long axis | manifold3d constructors / hulls |
| transform | `s·child(Rᵀ(p−t)/s)` | affine transform |
| taper / twist / bend | child at inverse-warped point | **forward vertex warp** (= vertex shader) |
| union / intersect / subtract | log-sum-exp min/max at temperature τ; exact at τ = 0 | manifold3d booleans |
| instance | min over K posed (+ per-instance deformed) copies | shared mesh + K transforms in glTF |

Deformers are written as closed-form forward/inverse pairs (numerically stable at k → 0, tested to 1e-6).
Fitting anneals τ → 0, and every *score* is computed at τ = 0 with hard instance gates, so the
optimized objective is evaluated on exactly the solid that gets exported. Test: SDF vs mesh inside/outside
agreement ≥ 99.5 % for all ops; measured 99.97–100 %.

## Losses (target-agnostic)

Every modality is reduced to labelled volume samples + surface samples (`targets.py`):
SDF/graph (sign), dense mesh (generalized winding number), depth maps (ray carving: free space before the
hit, thin inside band after), Gaussian splats (density shell → cavity fill → solid). Multi-view RGB comes
with the differentiable renderer (roadmap).

Loss = bounded L2 on soft occupancy σ(−d/ε) (+ BCE warm-up that decays to 0) + Geman–McClure surface
term, with ε and the robust scale annealed.

**Finding: unbounded losses (BCE, L1 surface) make partial explanations impossible.** A single box
fitted to a table gets dragged toward the legs, because far-away misclassified points dominate the
gradient; it settles on a compromise (IoU 0.14). With the bounded loss, a box seeded on the top locks
onto the top (IoU 0.82 = the best a single box can do). The leftover volume is then explained by
residual-driven mutations instead of being averaged in.

## Instances with unknown count

Splat-style: the fitter **densifies** by k-means clustering the false-negative volume and cloning the
nearest instance's pose/shape data there. It **prunes** by greedy leave-one-out: an instance is dropped
if removing it changes the mismatch by less than `prune_gain` of the target volume.

**Finding: a differentiable presence gate is degenerate here.** The gate shifted the instance's SDF
outward by `−s·log σ(logit)` (soft erosion). The optimizer lowered presence and grew primitive radii to
compensate (leg radius 0.06 → 0.12), then the instances were pruned. The gate is kept behind
`FitConfig.soft_gates` for experiments, but discrete ablation is the default. Result: a table seeded
with 1 leg converges to exactly 4 legs (IoU 0.989).

## Structure search

`score = (1 − IoU_hard) + β · complexity`, where complexity is a description length:
4 per node + 0.5 per parameter scalar. Instancing pays per-instance pose data but saves the prototype
and node overhead, so repeated parts become cheaper as instances. Presence logits are flags, not
counted.

Proposals are data-driven (residual = connected components of false-negative / false-positive volume):

| mutation | effect |
|---|---|
| add | union a new primitive, moment-matched (2nd moments → size, PCA → orientation) to a residual part |
| subtract | cut a primitive from the top-level part that covers the spurious region |
| swap / remove / deform | primitive kind change (extent-preserving), drop/bypass node, wrap a deformer |
| to_instance | placed primitive → 1-instance node, so the fitter can densify it |
| instancify | merge ≥ 2 same-kind placed primitives into one instance node (MDL compression) |
| instance_deform | enable per-instance taper/twist/bend coefficients |
| macro | inline a mined library subgraph onto a residual part |

**Finding: proposals need locality (RANSAC-style).** Moment-matching a whole region (table = top + legs)
gives a bad init. Cropping a random ball inside the region gives sub-part-sized seeds, which the
bounded loss then refines.

Beam search with (μ+λ) selection, one candidate per structural signature (diversity), and
canonicalization after each fit (K = 1 instances → transforms, flattened unions, scale gauge folded into
primitives), so that equivalent programs dedupe. Mutation weights adapt online (bandit on improvement
rate). Per-mutation stats are reported to compare mutation strategies.

## Library learning (v1)

`abstraction.mine` enumerates complete subtrees (3–10 nodes) of solved programs and ranks skeletons by
compression gain `(count − 1) · node_cost · size`. Macros are used as search proposals and as whole-shape
seeds; parameters of a typical occurrence serve as the init prior.

v1 limits:
- macros are inlined, so the MDL score does not yet reward reuse
- no anti-unification over op choices (holes)

Both are on the roadmap.

## Roadmap

1. **Image / depth / splat supervision at scale.** Differentiable sphere tracing of the graph (implicit
   gradients), plus nvdiffrast on the realized mesh for PBR appearance. Material parameters per primitive
   are already carried through to export.
2. **Library learning v2.** Anti-unification with typed holes (any-primitive, any-deformer); macro call
   nodes with MDL credit; wake–sleep iterations over a corpus; macro parameter priors as distributions.
3. **Learned deformation fields.** Low-dimensional per-instance latent codes driving a small MLP
   displacement field, constrained to be realizable as vertex displacement (bake to a WPO texture or
   vertex colors) alongside the analytic deformers.
4. **Mutation study.** Random vs residual-guided vs bandit vs learned proposal distribution (trained on
   search traces); Gumbel-softmax op relaxation (`union`/`subtract`/`intersect` superposition) as an
   alternative to discrete mutation for op choice.
5. **LLM bootstrap.** JSON-schema prompts that emit initial graphs; induction only (seeds + macros),
   never in the inner loop.
6. **GPU.** Batched evaluation of whole populations (`torch.vmap` over candidate parameter sets of a
   shared structure), CUDA kernels for SDF evaluation, larger dense-mesh targets (winding number on GPU).

## Benchmark v0 (CPU, 2 cores, ~5 min per target, search from scratch)

`results/bench_v0/`. Target = shape sampled from a hand-crafted graph; search sees only samples.
IoU on 40k uniform points (SDF) / 6k points against the exported exact-CSG mesh.
(Run predates the box-rounding fix and adaptive mutation weights.)

| target | IoU sdf | IoU mesh | found | ground truth |
|---|---|---|---|---|
| table | 0.984 | 0.974 | `union(instance(box), transform(box)×2, transform(cylinder))` | `union(instance(cylinder), transform(box))` |
| mug | 0.989 | 0.994 | `union(subtract(cyl, cyl), torus)` (**exact structure**) | same |
| twisted column | 0.915 | 0.917 | `union(instance+dk(box), transform(cylinder))` | `union(box, box, twist(box))` |
| fence | 0.918 | 0.874 | `union(instance(capsule), subtract(box, instance(cylinder)), box)` | `union(instance(box), instance(box))` |
| bent pipe | 0.996 | 0.839* | `transform(bend(box))` (rounded box ≈ capsule) | `transform(bend(capsule))` |
| bracket | 0.993 | 1.000 | `subtract(box, instance(capsule))` (structure match) | `subtract(box, instance(cylinder))` |
| stool | 0.976 | 0.951 | `union(instance+dk(cylinder), subtract(bend(cylinder), box))` | `union(instance+dk(cylinder), cylinder)` |

\* the SDF/mesh gap on bent pipe exposed a realizability bug (box rounding > half-extent allowed by the
SDF, clamped by the mesh). Fixed; regression test added; the found graph now agrees 100 %.

Mutation effectiveness, summed over targets (fitted → improved its parent / entered the beam):

| mutation | fitted | improved | rate | entered beam |
|---|---|---|---|---|
| subtract | 147 | 63 | 43 % | 16 |
| to_instance | 117 | 47 | 40 % | 16 |
| add | 210 | 78 | 37 % | 41 |
| instance_deform | 32 | 10 | 31 % | 4 |
| remove | 139 | 26 | 19 % | 8 |
| deform | 128 | 22 | 17 % | 18 |
| swap | 136 | 16 | 12 % | 10 |
| instancify | 9 | 4 | 44 % | 1 (rarely applicable) |

Observations:
- Residual-guided add/subtract and `to_instance` + densification do most of the work.
- Missing structure is mostly a **search-budget / MDL-calibration** issue, not expressivity. The table's
  legs were found as instanced boxes plus two stray boxes, and the twist on the column was approximated by
  a cylinder: a twisted square shaft and a cylinder differ by only ~8 % IoU, and β makes the extra
  deformer + box not worth it within budget.
- Next: longer budgets / GPU population batching, a "simplify" phase that tries removals and
  instancify at the end, β sweeps, and macro-seeded runs (mine on these, re-solve) to measure library
  learning gains.
