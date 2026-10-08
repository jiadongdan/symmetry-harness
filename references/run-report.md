# Run Reports

`symmetry report` converts a completed run directory into a fixed-template
Markdown report plus the PNG figures it embeds. Use it whenever the user asks
for a report, a summary, or a write-up of a run that has already finished.

It is a presentation layer over artifacts that already exist on disk. It never
recomputes features, never re-runs fine-tuning, and never runs dense prediction,
so it cannot change the scientific result — only how the result is presented.
It also never imports `gradio`, `torch`, or `symmlearn`, so it runs without a
GPU or a live Provider.

Report generation is intentionally lightweight: one command reads the existing
arrays and JSON records, renders figures with NumPy/Pillow, and writes the fixed
template. An agent should use context already present in the session, mark
missing scientific context as not recorded, and never delay the report by
asking follow-up questions solely to fill the free-form section.

## Command

```bash
symmetry report --session latest
symmetry report --run latest
symmetry report --run latest --prefix prediction
symmetry report --run symmetry-runs/<run-id> --notes session-notes.md
symmetry report --run <run-id> --notes-text "Two classes, five points each."
symmetry report --run <run-dir> --no-figures
```

| Option | Meaning |
| --- | --- |
| `--session` | Fine-tuning UI session id or `latest`; returns a single image report or a multi-run session index as appropriate. |
| `--run` | One run: `latest`, a run id, or a run directory path. If neither selector is supplied, the latest fine-tune run is used. |
| `--output-root` | Directory holding run folders. Defaults to the configured output root. |
| `--prefix` | Which family `latest` considers: `symmetry` (default), `prediction`, or `any`. |
| `--notes` | File whose contents fill the `Analysis Context` section. |
| `--notes-text` | Inline `Analysis Context` text. Wins over `--notes`. |
| `--output` | Report path. Defaults to `<run>/report_summary.md`. |
| `--figures` / `--no-figures` | Render figures, or emit a text-only report. |

The command prints JSON:

```json
{
  "status": "ok",
  "run_id": "symmetry-<UTC>-<hash>",
  "kind": "fine-tune",
  "report": "<run>/report_summary.md",
  "figures": ["<run>/figures/00_results_overview.png", "..."],
  "item_reports": []
}
```

For a multi-run fine-tuning session, `kind` is `fine-tune-session`, `report`
points to the compact session index, and `item_reports` lists every independent
image report.

Return the `report` path to the user. Do not copy, rename, or reshape the file:
the figures are referenced by relative path, so moving the Markdown away from
its `figures/` directory breaks the images.

## Which run

`--run latest` resolves by directory name, newest first, filtered by `--prefix`.
Run names embed their UTC creation timestamp, so a name sort is a time sort and
no clock comparison is needed.

- Fine-tune runs are named `symmetry-<UTC>-<hash>`.
- Prediction batches are named `prediction-<UTC>-<hash>`.

For the ordinary "write up what I just fine-tuned" request, use `--session
latest`. The UI records one session id for all fine-tune runs produced while
that workspace instance is open. A single run returns its normal report; two or
more runs receive a compact index without merging their models or training data.

Use `--run` when the user identifies one run. Use `--prefix prediction` only
when reporting on the saved-model prediction workspace.

If no matching run exists, the command fails with a `ReportError` naming the
searched root. Report that back; do not guess a directory.

## Sections

The fine-tune report is organized for a human expert first and reproducibility
second. Every report uses the same order so runs stay comparable:

| # | Section | Source |
| --- | --- | --- |
| 1 | Result at a Glance | task, class distribution, evidence status |
| 2 | Main Visual Evidence | compact annotation/prediction/uncertainty overview |
| 3 | Analysis Context | `--notes` / `--notes-text` (free-form) |
| 4 | Input and Support Annotations | `input.npy`, `annotation_session.json` |
| 5 | Prediction Certainty and Fine-tuning | plain-language certainty checks and support-fit history |
| 6 | Symmetry Representation | `features.npz` |
| 7 | Interpretation Boundaries | run warnings plus fixed scientific caveats |
| 8 | Reproducibility | workflow, options, checksums, versions |
| A | Artifacts | portable relative links to persisted files |

The annotation and fine-tuning sections are fine-tune only. Prediction batches
use a shorter index plus one expert-facing report per item under
`items/<item-id>/report_summary.md`.

## Multi-image Fine-tuning Sessions

Every fine-tuning action remains an independent, reproducible run. The session
report provides a compact comparison table, collapsible overview for each image,
and links to complete image reports. It explicitly prevents users from mistaking
sequential fine-tunes for joint multi-image training. If class names or core
analysis settings differ, the report warns against direct comparison.

Session indexes are written to:

```text
<output-root>/sessions/<session-id>/report_summary.md
```

Runs created before session ids were introduced degrade to a one-run report.

## Figures

Written into `figures/` inside the run directory. Numbering is stable across run
kinds, so a missing figure leaves a gap rather than renumbering:

| File | Content |
| --- | --- |
| `00_results_overview.png` | 2x2 overview of annotation/input, overlay, confidence and entropy |
| `01_input.png` | normalized input image |
| `02_support_points.png` | support points on the source image, legend below |
| `03_symmetry_maps.png` | 2x4 montage of the eight channels, each with its fixed range |
| `04_training.png` | support loss (log10) and support accuracy |
| `05_prediction_mask.png` | categorical class mask with legend |
| `06_prediction_overlay.png` | mask blended onto the source image with legend |
| `07_confidence.png` | confidence, viridis, fixed `0..1`, with colorbar |
| `08_entropy.png` | entropy, magma, fixed `0..log(N)`, with colorbar |

The symmetry montage and the feature table use the same fixed per-channel
display range as the annotation UI: `[0, 1]` for source intensity and reflection
strength, and `[-1, 1]` for reflection orientation and every rotation-response
channel. Tiles are therefore comparable across runs and are never renormalized
per image. The prediction figures reuse the same mask/overlay/confidence/entropy
renderers the UI uses for its download variants.

## Degradation

A report never fails because an optional artifact is missing:

- No `features.npz` -> the feature table says `feature bundle not persisted` and
  `03_symmetry_maps.png` is not written.
- No `prediction.npz` -> the prediction and overview figures are not written;
  the tables state that the prediction bundle was not persisted.
- No `training_history.json` -> curves fall back to the run record's history, and
  if neither exists `04_training.png` is skipped.
- No notes -> `Analysis Context` reads `_Not provided._`.

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
