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

## Differentiable rendering (`diffrender.py`)

- **Tracing:** sphere tracing without gradients, then implicit differentiation at the hit point
  (`t = t* − (f − sg f)/sg(∇f·d)`), giving exact first-order depth gradients (checked against finite
  differences).
- **Color:** fields carry color through every op (`EvalCtx(color=True)`: channel 0 is distance, 1–3 are
  albedo; booleans pick or blend color by the winning child, and cut walls take the cutter's color,
  matching export). Shading is Lambert + ambient.
- **Silhouettes:** a soft silhouette from the minimum SDF along the ray (sampled inside the solid for
  hit rays).
  - **Finding: a soft-mask loss on all rays is biased.** Thin parts never reach `σ(−f/σ) = 1`, so
    fitting inflates them. Starting at the exact ground truth it drifted to IoU 0.74–0.84. The loss
    now acts only on rays whose hit/miss disagrees with the mask (IDR-style), and stays at IoU 1.0 at
    the ground truth.
  - **Finding: a training-ray guard is not enough.** On the chair, refinement improved the training
    views while held-out views got much worse. Geometry is now early-stopped on held-out views (one
    view is withheld from gradients). The best-validated geometry is kept, along with the final
    colors.
- **Image pipeline (`pipeline.py`):** views → carved volumetric proxy → structure search →
  render-based refinement.
  - Carving with depth rejects only points *in front of* the observed surface, so concavities visible
    in depth are carved.
  - Masks alone give the visual hull: the mug's cavity and the space under a tabletop are filled.
  - **Cameras below the horizon are required**: upper-hemisphere orbits never see under furniture.
- **Gaussian splats:** splat centers sit on the surface, so density thresholding gives a leaky shell
  (IoU ≤ 0.47 even with tuning). `from_gaussians` therefore needs **oriented normals** (smallest-scale
  axis, oriented toward a viewing camera) and builds a point-set signed distance from the k nearest
  splats. Mug IoU > 0.85.

## Benchmark v1 (`results/bench_v1`, 150 candidate fits per target, volume-IoU score)

| target | mesh IoU | found |
|---|---|---|
| bracket | 1.000 | `subtract(box, instance(capsule))` (structure match) |
| mug | 0.991 | **exact structure** |
| bent pipe | 0.979 | `instance+dk(capsule)` ≡ `bend(capsule)` |
| table | 0.977 | **exact structure** (`union(instance(legs), box)`) |
| gear | 0.960 | disc minus instanced holes; **teeth missed** |
| colonnade | 0.915 | columns instanced; arches missed |
| twisted column | 0.911 | twist missed (box shaft) |
| chair | 0.880 | seat + instanced legs/posts; slats missed |
| stool | 0.842 | blob-and-carve |
| fence | 0.835 | rails wrong |
| bookshelf | 0.735 | frame `box − box` found; shelves/books missed |
| lamp | 0.667 | blob-and-carve |

Failure modes:
1. **Thin parts** (slats, rails, teeth, shelves) barely change volume IoU, so under the complexity
   penalty they are never worth adding.
2. **Blob-and-carve**: one large primitive plus a chain of subtractions is a deep local optimum.
3. **Deformers initialised at k ≈ 0** never reach a large twist.

Fixes (not yet re-benchmarked across all 12 targets; the rerun was paused at 2/12):
- the score mixes volume IoU with **surface recall** (fraction of target surface reproduced), which
  weighs a part by its area;
- deformer strength is **scanned** (cheap, no fitting) before a deform proposal is fitted.

Image pipeline v1 (`results/images_v1`, 9 views, 128², 120 fits): 3D IoU vs ground truth

| target | RGB only | RGB-D |
|---|---|---|
| table | 0.754 | **0.990** (exact structure, colors recovered, held-out depth MAE 0.001) |
| mug | 0.414 (cavity invisible) | 0.704 (cavity found) |
| chair | 0.298* | 0.830 (slats merged into a panel) |

\* run before the held-out early-stop fix; refinement then made the geometry worse.

## Infrastructure

| piece | where | notes |
|---|---|---|
| CPU/CUDA | everywhere | `Graph.to`, `Target.to`, `ImageTarget.to`; fits move graphs to the target's device; `tests/test_device.py` runs on CUDA when available |
| checkpoint / resume | `search(checkpoint_dir=…, resume=True)` | atomic `state.json` after every generation (beam, best, stats, RNG state, config, target fingerprint); a resumed run reproduces the uninterrupted one; resuming against a different target is refused |
| run logs | `log.jsonl` | one line per generation (+ final): best score/IoU/complexity/structure, all children's scores |
| configs | `config.py` | JSON round-trip of nested config dataclasses; unknown keys are errors |
| CLI | `python -m mlrepl` | `search` (resumable, `--workers`, `--device`), `fit`, `validate`, `export`, `preview`, `config` |
| validation | `Graph.validate()` | missing refs, cycles (iterative DFS), arity, unknown ops/params, param shapes, non-finite values; bad JSON fails at load |
| robust fitting | `fit.py` | per-group learning rates (`lr_mult`: translation / rotation / size / deform / color), grad-norm clipping, NaN/Inf recovery (skip step or roll back + reset moments + halve LR), parameter clamps with **rotation wrapping** (`|w| ≤ π`, same rotation) |
| parallel candidates | `parallel.ProcessEvaluator` | fits a generation's candidates in worker processes; proposals stay sequential, so results are identical to serial |
| batched multi-start | `parallel.fit_multistart` | B jittered copies of one structure fitted in one pass via `torch.func.vmap`; best by exact-CSG score |
| pluggable evaluation | `search(evaluator=…)` | serial (default), process pool, or anything mapping `(graph, origin)` jobs → candidates |

**Timing** (`examples/timing.py`, 3 targets × 60 candidate fits, 2-core CPU, `results/timing/`)

| run | total | fits/s | scores (table / mug / bracket) |
|---|---|---|---|
| before (serial) | 252 s | 0.75 | 0.059 / 0.149 / 0.079 |
| after, serial | 266 s | 0.71 | 0.057 / 0.155 / 0.054 |
| after, 2 workers | **167 s** | 1.13 | identical to serial |

- **Parallel evaluation: 1.6×** on 2 cores, with no change in results.
- **Robustness costs ~5 %** per fit (snapshot, finite checks, clamps).
- **Fit changes are mixed:** per-group learning rates + clipping improved bracket and table, and the
  mug was slightly worse at this budget.

Negative results, kept as options but **off by default**:
- **Plateau early stop**: the final low-learning-rate steps keep sharpening the shape after the
  minibatch loss looks flat. Bracket IoU dropped 0.993 → 0.96 for a ≤ 25 % step saving.
- **`torch.compile` on CPU**: 5× slower (17.7 s vs 3.3 s per fit). The annealed scalars trigger
  recompiles, and small batches can't amortize compilation. Worth re-measuring on GPU with long fits.
- **On a single GPU**, prefer the serial evaluator plus larger batches and `fit_multistart`; several
  processes contending for one GPU rarely helps.

## Regularity experiment v1 (`results/regularity_v1`)

**Question:** does description length recover regularity without a man-made/organic switch?

**Setup:**
- Same search with and without the detector moves: `pattern` (line/grid/circle fits on instance
  poses → array/radial), `mirror` (size-checked mirror pairs), `symmetrize` (mirror a part across a
  detected target symmetry plane), `plane` (RANSAC slab), `count` (n ± 1).
- 150 candidate fits per run, seed 0, a single seed.
- Score = 0.5 (1 − IoU) + 0.5 (1 − surface recall) + 0.001 · complexity. Lower is better.
- No target is labelled regular or irregular anywhere.

| target | kind | baseline score / IoU | detectors score / IoU | Δ score |
|---|---|---|---|---|
| gear | regular | 0.291 / 0.626 | **0.128 / 0.902** | −0.163 |
| colonnade | regular | 0.234 / 0.823 | **0.167 / 0.916** | −0.067 |
| fence | regular | 0.131 / 0.925 | **0.094 / 0.966** | −0.037 |
| bookshelf | regular | 0.290 / 0.723 | 0.288 / 0.768 | −0.002 |
| table | regular | 0.046 / 0.988 | 0.047 / 0.987 | ≈ 0 |
| chair | regular | 0.115 / 0.878 | 0.115 / 0.877 | ≈ 0 |
| bracket | regular | **0.041** / 0.995 | 0.066 / 0.994 | +0.025 |
| stool | regular | **0.072 / 0.970** | 0.161 / 0.869 | +0.089 |
| rubble | irregular | 0.123 / 0.947 | **0.086 / 0.976** | −0.037 |
| cairn | irregular | **0.164 / 0.930** | 0.217 / 0.895 | +0.053 |

Detector usage across all runs (proposed / fitted / improved parent / entered beam):

| detector | proposed | fitted | improved parent | entered beam |
|---|---|---|---|---|
| plane | 302 | 224 | 97 | 23 |
| symmetrize | 116 | 56 | 21 | 11 |
| pattern | 240 | 17 | 0 | 2 |
| mirror | 101 | 1 | 1 | 0 |
| count | 50 | 0 | 0 | 0 |

What this shows:

1. **The detectors help overall:** 5 of 10 targets better, 2 equal, 3 worse. The big wins are gear
   (IoU 0.63 → 0.90, via symmetrize → `mirror(mirror(instance))`), colonnade and fence. On the
   irregular controls, **no pattern ops were imposed**: rubble improved, cairn got worse. With one
   seed, differences under ~0.03 are within run-to-run variance.
2. **The core claim is supported only half-way.** The gains came from *data-side* detectors (plane,
   symmetry) and symmetrize-mirrors, not from the array/radial rewrites. The pattern detector was
   applicable 17 times in 240 tries and never improved its parent.
   - Applied to the ground-truth programs, the same rewrites are exact and shorter (gear 97 → 70.5,
     fence 52 → 40.5), so the *objective* does prefer patterns when the geometry is right.
   - The search doesn't reach that state, for two reasons:
     - (a) intermediate instance nodes rarely have pattern-exact poses (densify copies poses, counts
       are off), so the line/circle fits fail tolerance;
     - (b) the compression gain (≈ 0.01–0.03 in score) is the same size as the noise of
       re-fitting a candidate from scratch. Re-fitting even the exact ground-truth rewrite shifts its
       score by 0.005–0.04.
3. **Losses** (stool, bracket, cairn) come from detector proposals displacing better lineages from
   the 3-wide beam at a fixed budget, not from wrong regularity.

Next, to test the claim properly:
- **(i) Detect patterns from the data**, not from the program: cluster missing-volume components by
  shape, fit line/circle/grid to their centroids, and propose a pattern op with one fitted prototype.
- **(ii) Cut evaluation noise:** warm-start fits for children that inherit parameters (low τ₀ and
  learning rate), and score with a fixed, converged fit.
- **(iii) Repeat over 3–5 seeds** and report mean ± spread.

## Regularity experiment v2: data-side detection + warm start, multiple seeds (`results/regularity_v2`)

**Changes since v1:**
- **Warm-start fits:** nodes a child inherits unchanged from its fitted parent get 0.25× learning
  rate and low initial boolean softness. Refit spread halved, drift about 30 % lower.
- **Instance scale gauge** is pinned during fitting.
- **Data-side pattern detector:**
  - groups similar missing-volume pieces and fits line / grid / circle to their centroids;
  - first splits joined regions by density periodicity, linearly (fence posts on a rail) or
    angularly (gear teeth merged by the residual grid);
  - proposes `pattern(prototype)` directly from the data, so the search no longer has to reach a
    pattern-exact instance node first.

**Run:** 100 candidate fits per search, 8 targets × {baseline, detectors}. The planned 3 seeds were
cut by the 2 h job limit: **seeds 0 and 1 complete**, seed 2 only partly done (gear baseline).
Mean ± std over completed seeds; "pattern" = runs whose final program contains array/radial/mirror.

| target | baseline score | detectors score | detectors IoU | pattern in final |
|---|---|---|---|---|
| gear | 0.239 ± 0.053 | **0.091 ± 0.015** | 0.945 | 2/2 (seed 1: `radial[12]`) |
| fence | 0.340 ± 0.127 | **0.109 ± 0.024** | 0.913 | 2/2 (`array[6]`, `array[5]`) |
| stool | 0.157 ± 0.032 | **0.109 ± 0.004** | 0.920 | 1/2 (`radial[3]`) |
| cairn (irregular) | 0.230 ± 0.004 | **0.186 ± 0.016** | 0.905 | 0/2 |
| bracket | 0.040 ± 0.000 | 0.040 ± 0.001 | 0.993 | 0/2 |
| rubble (irregular) | 0.142 ± 0.012 | 0.147 ± 0.016 | 0.932 | 1/2 (`array[2]` of two capsules) |
| table | **0.045 ± 0.001** | 0.050 ± 0.003 | 0.990 | 1/2 (2×2 grid) |
| colonnade | **0.271 ± 0.002** | 0.302 ± 0.017 | 0.748 | 2/2 |

Detector usage across detector runs (proposed / fitted / improved parent / entered beam):

| detector | proposed | fitted | improved parent | entered beam |
|---|---|---|---|---|
| **data_pattern** | 525 | 158 | **77 (49 %)** | 21 |
| plane | 339 | 156 | 97 | 22 |
| symmetrize | 78 | 36 | 11 | 8 |
| count | 68 | 24 | 3 | 0 |
| pattern (program-side) | 278 | 23 | 0 | 2 |
| mirror (program-side) | 118 | 0 | 0 | 0 |

Reading:
- **Detecting patterns in the data is what makes regularity reachable.** Program-side pattern
  detection still never improves a parent; data-side detection improves it about half the time it
  is fitted.
- Pattern ops now survive into the final program on the shapes where repetition dominates: gear,
  fence, stool. Those are exactly the large wins, with differences far outside the seed spread.
- **Losses:**
  - *Table:* four legs are short enough that a 2×2 grid costs slightly more than a 4-copy instance
    under the parameter-count complexity. The objective is behaving as specified; the cost model
    is crude.
  - *Colonnade:* the arch/lintel structure is still not found, and detector proposals crowd the
    beam.
- **Irregular controls:** cairn improved without patterns.
  - *Rubble:* one run of two kept `array[2]` over two similar capsules. Two similar parts are a
    legitimate (if weak) "pattern" under a parameter-count cost, since an array of 2 costs about
    the same as two placements.
  - This is the case where a **precision-aware description length** (cost of the offsets at the
    precision they are needed) would be the principled tie-breaker. With plain parameter counts,
    exact and approximate repetition cost the same.

Next:
- a precision-aware complexity term (offsets/params coded at fitted precision, so near-duplicates
  aren't free);
- reduce the occupancy-loss bias on thin parts;
- finish seed 2 (the script resumes from `runs.json`).
