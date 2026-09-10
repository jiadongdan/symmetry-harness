# Saved-Model Prediction

Read this reference when the user wants to apply a previously fine-tuned
`fine_tuned_model.symmodel` package to new images. This workflow never
fine-tunes and never collects support points.

## When to Use

Use prediction when a user already has a `.symmodel` file produced by a previous
fine-tuning run and wants to classify new, similar images. Do not ask for
support points, class definitions, or a base checkpoint.

## Non-UI Path (preferred for automation)

```bash
symmetry predict \
  --model fine_tuned_model.symmodel \
  --input image.npy \
  --output-root symmetry-prediction-runs
```

Repeat `--input` for a batch. The Provider restores the saved model once for the
whole batch. Supported inputs are `npy`, `npz`, `tif`, `tiff`, `png`, `jpg`,
`jpeg`, and `bmp`. All output is machine-readable JSON.

## Allowed Overrides

Only these three are runtime-safe:

- `--device` (`auto`, `cpu`, or a CUDA device)
- `--stride`
- `--batch-size`

Feature parameters (`n_max`, symmetry patch size, rotation folds, reflection
parameters, channel order, normalization, classifier patch size) are restored
from the package and are read-only. Never offer to change them, and never
substitute current defaults for the saved values.

## UI Path

Open the **Predict with a saved model** tab, or launch directly into it:

```bash
symmetry launch --mode predict
```

The Predict button stays disabled until a package passes validation, the Provider
supports saved-model prediction, at least one image passes validation, and the
runtime overrides are valid.

## Artifacts

```text
symmetry-prediction-runs/
└── prediction-<UTC>-<id>/
    ├── batch_record.json
    ├── report.md
    ├── results.zip
    └── items/
        └── image-0001/
            ├── input.npy
            ├── input_preview.png
            ├── provider_record.json
            ├── prediction.npz
            ├── prediction_overlay.png
            ├── confidence.png
            ├── entropy.png
            └── prediction_record.json
```

The source `.symmodel` file is read-only. It is never modified, renamed, or
copied into per-image directories.

## Blocking Conditions

Prediction stops when:

- the package or model-state schema is unsupported;
- required ZIP members are missing or unsafe;
- the model-state checksum does not match the manifest;
- the saved model identifier is not registered in the installed Provider;
- the Provider does not advertise `predict_with_fine_tuned_model`;
- the saved feature pipeline or channel order does not match the Provider;
- an image is smaller than the saved classifier patch.

An image-specific failure is recorded against that image and the batch
continues. A model-state or contract failure stops the batch immediately.

## Interpretation

- Class counts and fractions are measured over the predicted grid, not over
  ground truth.
- Confidence and entropy describe model output, not physical correctness.
- Prediction reuses the feature contract saved during fine-tuning. It does not
  re-validate the model against new labels.
