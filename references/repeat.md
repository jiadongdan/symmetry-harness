# Repeat Loops (Stability Testing)

A repeat loop answers one question: **if the same k-shot fine-tuning procedure is
executed N times, does it produce the same answer?**

It is not an active-learning loop and it is not a way to improve a model. Every
round trains a fresh adapter from the selected backbone with exactly
`--points-per-class` newly chosen support points. The support set never
accumulates, so round 40 is still a 5-shot experiment, not a 205-shot one.

```
anchor    the user's first fine-tune -> prediction + confidence  [FROZEN]
          +-> pool A, pool B         (the only place later rounds may sample)
round 1   sample k/class from pool   ->  adapter -> prediction -> accuracy
round 2   sample k/class from pool   ->  ...
```

The anchor is trained once and then locked. Its prediction grid decides both
**where** later rounds may sample and **what label** a sampled cell carries;
nothing a later round predicts can change either. That is what makes the rounds
independent replicates instead of one self-training trajectory.

## Why the support set must not accumulate

Accumulating turns the protocol into a different experiment after the first few
rounds: a 5-shot result and a 50-shot result are not the same measurement, and a
drop in round 30 would be unattributable. Keeping every round at k points means
the only thing that varies is **which** points were drawn, so the spread of the
accuracies is a clean measure of procedural reproducibility.

Because the support set never accumulates, the configured
`maximum_shots_per_class` never becomes a limit: 100 rounds at one point per
class stays at one point per class.

## What the main parameters mean

| Parameter | Meaning | Recommended |
| --- | --- | --- |
| `--mode` | Which flow to measure. `user` pools the anchor's confident cells; `stability` pools the cells that agree with the ground truth and clear the phase boundary. | `user` for the product path, `stability` for internal numbers |
| `--points-per-class` | Shots per class in every round. Match the experiment you are replicating (1, 3, or 5). | `5` |
| `--rounds` | Replicates after the anchor. `--rounds 100` trains the anchor plus 100 rounds. | `5` to shake out bugs, `50`–`100` for a real measurement |
| `--min-point-distance` | Minimum pixel distance between any two support points of a round (Poisson-disk spacing, both classes). | leave at the default (one classifier patch); `0` disables it |

`minimum_shots_per_class` defaults to `1`, so a 1-shot replication needs no
configuration change. Values below the configured minimum are rejected by the
annotation validator, and the loop still raises the minimum to
`--points-per-class` if a config sets it higher.

## The two modes

Both modes freeze a pool, then sample from it for every replicate. They differ in
how the pool is drawn and where the labels come from.

| | `user` | `stability` |
| --- | --- | --- |
| Pool cell | anchor confidence `>= --tau` and anchor label `c` | anchor label `== ground truth` and `>= patch/2` px from a phase boundary |
| Pseudo-label | the anchor label | the ground-truth label |
| Ground truth | optional, adds accuracy scoring only | required |
| Measures | how reproducible the fit is from one user annotation | training variance with annotation noise removed |

`stability` does not also apply `tau`: agreement with the ground truth already
implies a correct label, and a confidence floor would only shrink the pool
without changing which labels are right.

Do not subtract one mode's numbers from the other. They differ in both the pool
and the label source, so the difference mixes two effects.

A pool that cannot supply `--points-per-class` cells for both classes is a loop
error, deliberately. In `user` mode it means the first fine-tune was not good
enough to anchor anything: the guidance is to relabel a few points and fine-tune
again, **not** to lower `--tau`. Lowering the threshold just builds a dirtier
pool and hides the problem.

### Phase-boundary clearance

`stability` mode needs the distance from each grid cell to the nearest phase
change. It is computed exactly on the prediction lattice and scaled to pixels by
the grid step. Both cells either side of an interface count as boundary cells; a
patch centred on either of them straddles the phase change, so its clearance is
zero. That is at most half a grid step more conservative than the geometric
interface, which is the safe direction.

## Sampling rule

Each replicate draws `--points-per-class` cells per class from the frozen pool,
uniformly and without replacement, seeded with `base_seed * 1000 + round`. The
pool is never rewritten, so the only thing that varies between rounds is the
draw.

The points of one round also obey a Poisson-disk constraint: no two support
points are closer than `--min-point-distance` pixels. The default is the
classifier patch size (64 px), which is the distance at which two support patches
stop overlapping; `0` restores unconstrained uniform sampling. The constraint
spans **both classes** — two nearly co-located points with different labels force
the classifier to memorise one crop under two contradictory labels, which is
worse than a wasted support slot.

The spacing is a ceiling, not a promise. When the pool cannot fit the requested
radius (a grain-boundary band narrower than the patch, or many points per class
in a small phase), the radius is shrunk geometrically — 15% per attempt, floored
at two grid steps — until a draw succeeds. Every round records both numbers:
`spread_req` is what was requested, `spread` is what fit. A `spread` below
`spread_req` is a property of the data, not a silent detail, and the CLI prints a
line when it happens. If even the floor radius cannot be satisfied the round
stops with an error rather than drawing overlapping points.

Two columns track how hard the loop leans on one pool: `pool_a` / `pool_b` are
the pool sizes, and `reuse_a` / `reuse_b` are the average number of draws per
pool cell so far. A large reuse number on a small pool means the rounds are
revisiting the same cells, which weakens their independence.

## The anchor round

Round 0 is the anchor. It runs one fine-tune, writes its prediction to
`anchor/`, and freezes the pool. **It is not a replicate**: it does not appear in
`loop_rounds.csv` and it is excluded from the mean and standard deviation.

Pass `--annotations SESSION.json` to anchor on a real human annotation session —
this is the normal path for the `user` flow and the only way to run without
ground truth. The session must have been drawn on the same image file, because
the harness compares the file SHA-256.

Without `--annotations`, the anchor draws its points from the interior of each
ground-truth phase (`--interior-margin`, default 32 px, so a 64 px patch stays
inside one phase), which requires `--gt`. It applies the same
`--min-point-distance` rule, so a five-shot anchor does not spend four clicks on
the same grain.

The interior margin applies to the anchor only. Later rounds ignore it.

## Accuracy

Accuracy is scored on the dense prediction grid against a ground-truth label
raster (`.npy`, integer, two codes). The two codes are mapped positionally onto
class A and class B by `--label-codes`, default `14,17`.

Without `--gt` (allowed in `user` mode only) accuracy is `nan` and the stop rules
are inert; the loop still reports `anchor_agree` and `agree_prev`, which measure
how far each round's map moved from the anchor and from the round before.

Score on the whole grid, never on a high-confidence subset: restricting to
confident cells inflates the number and makes the alarm threshold meaningless.

## Stop rules

Checked after every round, in this order:

| Condition | Result |
| --- | --- |
| accuracy below `--alarm-floor` (default 0.90) | alarm |
| at least 6 rounds and accuracy below `mean - alarm_drop x std` (default 3 sigma) | alarm |
| either class IoU below 0.10 | alarm |
| candidate pool for a class smaller than `--points-per-class` | loop error |
| pool too small for `--points-per-class` points at the minimum spacing (two grid steps) | loop error; the spacing is shrunk first, and only the floor is fatal |
| `--rounds` reached | completed |

An alarm stops the loop by default. Use `--no-stop-on-alarm` to keep running and
record the outlier instead.

The comparison baseline is the **whole history**, not the previous round. In a
replication loop the previous round is itself a random draw, so tracking it would
flag ordinary fluctuation as a collapse.

The sigma rule needs at least 6 replicates. With four rounds the sample standard
deviation is small enough that `mean - 3 sigma` fires on ordinary spread; shorter
loops are judged on the absolute floor alone.

## One image or many

```powershell
# internal stability numbers: pool = anchor cells that agree with the truth
symmetry repeat --config symmetry-harness.json --mode stability `
  --image "D:\data\simu img1.npy" --gt "D:\data\simu lbs1.npy" `
  --rounds 100 --points-per-class 5

# product path: pool = the user's own confident region, no ground truth needed
symmetry repeat --config symmetry-harness.json --mode user `
  --image "D:\data\simu img1.npy" --annotations "D:\data\session.json" `
  --rounds 20 --points-per-class 5

# several images, one after another
symmetry repeat --config symmetry-harness.json --mode stability `
  --image "D:\data\a.npy" --gt "D:\data\a_lbs.npy" `
  --image "D:\data\b.npy" --gt "D:\data\b_lbs.npy" `
  --rounds 100 --points-per-class 5

# or a manifest
symmetry repeat --config symmetry-harness.json --dataset images.json --mode stability --rounds 100
```

`images.json` is either a list or an object with an `images` list:

```json
{"images": [
  {"image": "simu img1.npy", "gt": "simu lbs1.npy"},
  {"image": "a.npy", "gt": "a_lbs.npy"}
]}
```

Relative paths resolve against the manifest's own directory. A batch writes one
loop directory per image plus `repeat_batch.json`.

## Resume

A 100-round loop takes roughly 20 minutes, so an interruption should not cost
everything:

```powershell
symmetry repeat --config symmetry-harness.json --image ... --gt ... `
  --rounds 100 --resume "D:\runs\repeat-simu-img1-20261102T051328Z-475ee7"
```

Resume reloads the anchor from `anchor/prediction.npz` (or `rounds/round-000/`)
and rebuilds the frozen pool from it, then continues at the next replicate. Only
the horizon may grow: changing `--mode`, `--points-per-class`, `--tau`, or
`--base-seed` is refused, because mixing two protocols would corrupt the
distribution.

Loops written before schema `symmetry-repeat-loop-v2` cannot be resumed; their
pool was accumulated across rounds and no longer exists.

## Artifacts

```
<output-root>/repeat-<image>-<timestamp>-<id>/
├── loop_manifest.json      options, input SHA-256, GT SHA-256, anchor pools
├── anchor/                 anchor.json, metrics.json, prediction.npz
├── loop_rounds.csv         one row per replicate (the anchor is not a row)
├── loop_summary.md         distribution, per-round table, embedded figures
├── features/               reusable feature stack (one per image)
├── figures/                acc_vs_round, iou_vs_round, area_frac_vs_round,
│                           points_map, round_sheet
└── rounds/round-00N/
    ├── session_round00N.json   the exact support points used
    ├── metrics.json
    ├── stop_decision.json
    └── kept/                   prediction.npz, run_record.json, confidence.png,
                                prediction_overlay.png, fine_tuned_model.symmodel
```

`figures/round_sheet.png` is one row per round and three columns: the source
image with that round's support points (with a `classifier_patch_size` box per
point), the predicted mask, and the cells that disagree with the ground truth.
It caps at 12 rows, showing evenly spaced rounds including the first and the
last, so a 100-round loop still produces a readable sheet.

The harness run directory itself, its training-only artifacts, and the duplicate
`-full-run.zip` are deleted after each round. Keeping every
`fine_tuned_model.symmodel` costs about **45 MB per round**: 5 rounds ≈ 230 MB,
100 rounds ≈ 4.5 GB. Use `--no-keep-checkpoint` to keep only the maps, which is
about 0.2 MB per round plus the one-time feature stack (≈4 MB at 512x512).

## Feature reuse

The eight-channel feature stack depends only on the image and the feature
options, not on the support points, so the loop computes it once per image and
passes it to every round (`cache_reused: true` in `run_record.json`). Measured on
a 512x512 image at `symmetry_patch_size=51`: about 8 s of feature work saved per
round. If the Provider cannot accept precomputed features the loop falls back to
recomputing them and says so on stderr; results are unchanged.

## Reading a loop honestly

* A flat accuracy curve means **reproducibility under one frozen pool**. It does
  not mean the model is robust to points outside that pool: in `user` mode the
  pool is only the anchor's confident region, and in `stability` mode it excludes
  the phase boundary by construction.
* Because the pool never changes, a round that collapses cannot drag the next
  round down with it. A collapsing round is a data point about training
  variance, not a contamination event.
* On a clean simulated image accuracy can saturate near 99%, which compresses the
  spread and makes any procedure look stable. Check the absolute level before
  concluding anything about stability.
* A round that collapses under a per-round training seed (`base_seed + round`)
  cannot be attributed to the points alone. Re-run that round with a fixed seed
  to separate the two causes.

## Cost

Measured on an RTX 4070 Laptop, 512x512, `classifier_patch_size=64`,
`symmetry_patch_size=51`, 150 epochs, with feature reuse: **~12 s per round**.
100 rounds ≈ 20 minutes. Without feature reuse, add ~8 s per round.
