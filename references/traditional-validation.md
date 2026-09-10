# Traditional ML Validation (explicit request only)

This page is **not** part of the normal launch path. It is a separate,
single-purpose workspace that only runs when the user explicitly asks for
traditional machine-learning validation. The Fast Launch path in `SKILL.md`
still starts the two-workspace application and is unchanged.

## What it is

An exploratory comparison surface: it trains one conventional scikit-learn
classifier from scratch on patches the user selects, then densely predicts the
whole image so the result can be reviewed next to the source image and the
symmetry maps.

## What it is not

- Not a formal paired benchmark against the pretrained network.
- Not proof that the pretrained model is necessary or unnecessary.
- Not a third daily-use workspace, and not a replacement for fine-tuning.
- It never loads, probes, requires, fine-tunes, or silently installs a
  pretrained checkpoint, adapter, or saved model. An absent pretrained weight
  must not block the page.

## Launch

Only when the user explicitly requests traditional ML validation, run exactly
one bundled script relative to this `SKILL.md` file. On Windows:

```powershell
& "<skill-directory>\scripts\launch-traditional.ps1"
```

On macOS or Linux:

```bash
"<skill-directory>/scripts/launch-traditional.sh"
```

Append `--input <path>` only when the user explicitly asks to preload an image.
Wait for the first JSON object whose `status` is `ready` or `blocked`, report it,
and stop using tools. On `blocked`, report the phase, issues, and
recommendations; do not troubleshoot unless asked.

The equivalent explicit command, without the script wrapper, is:

```powershell
& "<harness-python>" -m symmetry_harness.cli validate-traditional --config "<config-path>" --server-port 0 --no-inbrowser
```

The page binds `127.0.0.1` (or `localhost`) only. Any other `--server-name`
is refused and reported as a `blocked` payload with phase `server_name`.

## Requirements

- Provider must advertise both `compute_features` and `traditional_ml_analyze`.
- The Provider runtime needs `scikit-learn`.
- The Harness runtime needs Gradio (the `[ui]` extra) to render the page.
- No pretrained weight, checkpoint, or `.symmodel` package is required.

`validate-traditional` reports readiness without ever resolving a pretrained
checkpoint.

## Inputs

- One single-channel image, normalized to the shared `[0, 1]` space, with both
  dimensions at least the classifier patch size.
- At least two user-defined classes.
- Between 2 and 5000 support patches per class, all distinct and all far enough
  from the border to yield a full patch.

Feature modes:

- `raw_image` — uses exactly channel 0 of the image.
- `image_plus_symmetry_maps` — uses exactly all eight channels, in the
  Provider-declared order.

## Classifiers

- `logistic_regression` — always a training-only `StandardScaler` pipeline fed
  into logistic regression (lbfgs).
- `random_forest` — always receives the configured seed and runs with
  `n_jobs=1` for strict reproducibility.

Only the selected classifier's parameter controls are shown, and only those
parameters are sent to the Provider.

## Outputs

Each run writes a self-contained directory containing the exact input, the
annotation session, the feature artifact, the Provider record, the dense
prediction arrays, a prediction overlay, confidence and entropy maps,
`training_summary.json`, `report.md`, `run_record.json`, and a `results.zip`
archive.

The same image, annotations, classifier, and seed reproduce the same
prediction grid and probabilities.

## Limits

- `support_training_accuracy` is measured on the user-selected support patches
  themselves. It is **not** held-out validation accuracy and must never be
  reported as such.
- Confidence and entropy describe classifier output, not physical correctness.
- Conclusions are exploratory. Review important boundaries and rare structures
  against the source image and relevant domain evidence.
