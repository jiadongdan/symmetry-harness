# Repeat Loops (Stability Testing)

A repeat loop answers one question: **if the same k-shot fine-tuning procedure is
executed N times, does it produce the same answer?**

It is not an active-learning loop and it is not a way to improve a model. Every
round trains a fresh adapter from the selected backbone with exactly
`--points-per-class` newly chosen support points. The support set never
accumulates, so round 40 is still a 5-shot experiment, not a 205-shot one.

```
round 0   seed support set  ->  adapter -> prediction -> accuracy vs ground truth
round 1   sample k/class    ->  adapter -> prediction -> accuracy
round 2   sample k/class    ->  ...
```

## Why the support set must not accumulate

Accumulating turns the protocol into a different experiment after the first few
rounds: a 5-shot result and a 50-shot result are not the same measurement, and a
drop in round 30 would be unattributable. Keeping every round at k points means
the only thing that varies is **which** points were drawn, so the spread of the
accuracies is a clean measure of procedural reproducibility.

Because the support set never accumulates, the configured
`maximum_shots_per_class` never becomes a limit: 100 rounds at one point per
class stays at one point per class.

## What the two parameters mean

| Parameter | Meaning | Recommended |
| --- | --- | --- |
| `--points-per-class` | Shots per class in every round. Match the experiment you are replicating (1, 3, or 5). | `5` |
| `--rounds` | Rounds after the seed round. `0` runs only round 0. | `5` to shake out bugs, `50`–`100` for a real stability measurement |

## Sampling rule

Two fields are accumulated across rounds and used only to choose the next
support set:

* **confidence** — the arithmetic mean of every round's confidence grid;
* **votes** — how often each grid cell was predicted as each class, which
  supplies the pseudo-label of a sampled cell.

A cell is a candidate when its accumulated confidence is at least `--tau`
(default `0.95`). Sampling is uniform, without replacement, and seeded with
`base_seed * 1000 + round`. There is no minimum-distance rule and no cross-round
de-duplication: the confidence field is the only filter.

`tau` is deliberately not the main constraint. On a typical phase map the model
is already above 0.95 on roughly 96% of the grid, so `tau` filters very little.
What varies between rounds is the draw itself.

## The seed round

Round 0 is the reference: it produces the first confidence map and the first
accuracy value. It draws its points from the interior of each ground-truth phase
(`--interior-margin`, default 32 px, so a 64 px patch stays inside one phase).

Pass `--annotations SESSION.json` to use a human annotation session instead. The
session must have been drawn on the same image file, because the harness compares
the file SHA-256.

The interior margin applies to round 0 only. Later rounds ignore it, exactly as
specified.

## Accuracy

Accuracy is scored on the dense prediction grid against a ground-truth label
raster (`.npy`, integer, two codes). The two codes are mapped positionally onto
class A and class B by `--label-codes`, default `14,17`.

Score on the whole grid, never on a high-confidence subset: restricting to
confident cells inflates the number and makes the alarm threshold meaningless.

## Stop rules

Checked after every round, in this order:

| Condition | Result |
| --- | --- |
| accuracy below `--alarm-floor` (default 0.90) | alarm |
| at least 3 rounds and accuracy below `mean - alarm_drop x std` (default 3 sigma) | alarm |
| either class IoU below 0.10 | alarm |
| candidate pool for a class smaller than `--points-per-class` | loop error |
| `--rounds` reached | completed |

An alarm stops the loop by default. Use `--no-stop-on-alarm` to keep running and
record the outlier instead.

The comparison baseline is the **whole history**, not the previous round. In a
replication loop the previous round is itself a random draw, so tracking it would
flag ordinary fluctuation as a collapse.

## One image or many

```powershell
# one image
symmetry repeat --config symmetry-harness.json `
  --image "D:\data\simu img1.npy" --gt "D:\data\simu lbs1.npy" `
  --rounds 5 --points-per-class 5

# several images, one after another
symmetry repeat --config symmetry-harness.json `
  --image "D:\data\a.npy" --gt "D:\data\a_lbs.npy" `
  --image "D:\data\b.npy" --gt "D:\data\b_lbs.npy" `
  --rounds 100 --points-per-class 5

# or a manifest
symmetry repeat --config symmetry-harness.json --dataset images.json --rounds 100
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

Resume replays each `round-*/kept/prediction.npz` to rebuild the confidence field
and the vote counts, then continues at the next round. Only the horizon may grow:
changing `--points-per-class`, `--tau`, or `--base-seed` is refused, because
mixing two protocols would corrupt the distribution.

## Artifacts

```
<output-root>/repeat-<image>-<timestamp>-<id>/
├── loop_manifest.json      options, input SHA-256, GT SHA-256, backbone identity
├── loop_rounds.csv         one row per round
├── loop_summary.md         distribution, per-round table, embedded figures
├── features/               reusable feature stack (one per image)
├── figures/                acc_vs_round, iou_vs_round, area_frac_vs_round, points_map
└── rounds/round-00N/
    ├── session_round00N.json   the exact support points used
    ├── metrics.json
    ├── stop_decision.json
    └── kept/                   prediction.npz, run_record.json, confidence.png,
                                prediction_overlay.png, fine_tuned_model.symmodel
```

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

* With `tau=0.95` and interior-geometry sampling, all rounds draw from a narrow
  band of "easy, confident" patches. A flat accuracy curve therefore means
  **reproducibility under one sampling rule**.
* It does **not** mean the model is robust to a human who clicks near a phase
  boundary, or to a patch that straddles two phases.
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
