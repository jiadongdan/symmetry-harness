# Run Reports

`symmetry report` converts a completed run directory into a fixed-template
Markdown report plus the PNG figures it embeds. Use it whenever the user asks
for a report, a summary, or a write-up of a run that has already finished.

It is a presentation layer over artifacts that already exist on disk. It never
recomputes features, never re-runs fine-tuning, and never runs dense prediction,
so it cannot change the scientific result — only how the result is presented.
It also never imports `gradio`, `torch`, or `symmlearn`, so it runs without a
GPU or a live Provider.

## Command

```bash
symmetry report --run latest
symmetry report --run latest --prefix prediction
symmetry report --run symmetry-runs/<run-id> --notes session-notes.md
symmetry report --run <run-id> --notes-text "Two classes, five points each."
symmetry report --run <run-dir> --no-figures
```

| Option | Meaning |
| --- | --- |
| `--run` | `latest` (default), a run id, or a run directory path. |
| `--output-root` | Directory holding run folders. Defaults to the configured output root. |
| `--prefix` | Which family `latest` considers: `symmetry` (default), `prediction`, or `any`. |
| `--notes` | File whose contents fill the `Session Notes` section. |
| `--notes-text` | Inline `Session Notes` text. Wins over `--notes`. |
| `--output` | Report path. Defaults to `<run>/report_summary.md`. |
| `--figures` / `--no-figures` | Render figures, or emit a text-only report. |

The command prints JSON:

```json
{
  "status": "ok",
  "run_id": "symmetry-<UTC>-<hash>",
  "kind": "fine-tune",
  "report": "<run>/report_summary.md",
  "figures": ["<run>/figures/01_input.png", "..."],
  "item_reports": []
}
```

Return the `report` path to the user. Do not copy, rename, or reshape the file:
the figures are referenced by relative path, so moving the Markdown away from
its `figures/` directory breaks the images.

## Which run

`--run latest` resolves by directory name, newest first, filtered by `--prefix`.
Run names embed their UTC creation timestamp, so a name sort is a time sort and
no clock comparison is needed.

- Fine-tune runs are named `symmetry-<UTC>-<hash>`.
- Prediction batches are named `prediction-<UTC>-<hash>`.

For the ordinary "I just fine-tuned something, write it up" request, the default
`--prefix symmetry` is what you want. Use `--prefix prediction` only when the
user is reporting on the prediction workspace.

If no matching run exists, the command fails with a `ReportError` naming the
searched root. Report that back; do not guess a directory.

## Sections

Every report uses the same order so reports stay comparable across runs:

| # | Section | Source |
| --- | --- | --- |
| 1 | Summary | run record: classes, points, grid, stride, diagnostics |
| 2 | Workflow Overview | resolved run options |
| 3 | Session Notes | `--notes` / `--notes-text` (free-form) |
| 4 | Input & Annotation | `input.npy`, `annotation_session.json` |
| 5 | Symmetry Features | `features.npz` |
| 6 | Fine-tuning | `training_history.json`, run options |
| 7 | Dense Prediction | `prediction.npz`, run record |
| 8 | Configuration & Provenance | run record: options, checksums, versions |
| 9 | Caveats | run warnings plus a fixed presentation addendum |
| 10 | Artifacts | run record artifact map |

Sections 4 and 6 are fine-tune only. Prediction batches use a shorter variant
with Sections 1–3, Classes, Items, Configuration, Caveats, and Artifacts; each
item also gets its own report under `items/<item-id>/report_summary.md`.

## Figures

Written into `figures/` inside the run directory. Numbering is stable across run
kinds, so a missing figure leaves a gap rather than renumbering:

| File | Content |
| --- | --- |
| `01_input.png` | normalized input image |
| `02_support_points.png` | support points on the source image, legend below |
| `03_symmetry_maps.png` | 2x4 montage of the eight channels, each with its fixed range |
| `04_training.png` | support loss (log10) and support accuracy |
| `05_prediction_mask.png` | categorical class mask with legend |
| `06_prediction_overlay.png` | mask blended onto the source image with legend |
| `07_confidence.png` | confidence, viridis, fixed `0..1`, with colorbar |
| `08_entropy.png` | entropy, magma, fixed `0..log(N)`, with colorbar |

The symmetry montage and the feature table use the same fixed per-channel
display range as the annotation UI: `[-1, 1]` for the two signed reflection
channels and `[0, 1]` for everything else. Tiles are therefore comparable across
runs and are never renormalized per image. The prediction figures reuse the same
renderers the UI uses for its download variants.

## Degradation

A report never fails because an optional artifact is missing:

- No `features.npz` -> the feature table says `feature bundle not persisted` and
  `03_symmetry_maps.png` is not written.
- No `prediction.npz` -> the prediction figures are not written; the tables stay.
- No `training_history.json` -> curves fall back to the run record's history, and
  if neither exists `04_training.png` is skipped.
- No notes -> `Session Notes` reads `_Not provided._`.

Only a missing `run_record.json`, `batch_record.json`, or `annotation_session.json`
is fatal.

## Caveats

The generated report repeats the Harness position on interpretation and so
should the agent:

- Support-set loss and accuracy are training diagnostics, not estimates of
  generalization accuracy.
- Confidence and entropy describe model output, not physical correctness.
- Dense prediction is sampled on a grid at the configured stride; cells between
  samples are filled for display only.

Never present a generated report as evidence that the model is scientifically
correct. It is a record of what the run produced.
