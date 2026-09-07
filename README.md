# symmetry-harness

`symmetry-harness` is a cross-agent, human-in-the-loop harness for few-shot
symmetry analysis of single-channel scientific images. It provides a local
point-annotation interface, adapter-plus-head fine-tuning, dense prediction,
versioned run records, and a machine-readable CLI for Codex, Claude Code,
WorkBuddy, Cursor, and other shell-capable agents.

The intended workflow is interactive. A user defines local classes and clicks
three to five representative points per class. The Harness records that intent
and delegates feature extraction, `64 x 64` support-patch construction,
adapter-plus-head fine-tuning, and dense inference to the versioned
`symmetry-learn` Provider.

## Architecture

```text
Host general agent
        |
        v
symmetry-harness Skill + JSON CLI + local UI
        |
        +-- annotation session and reproducible run record
        |
        v
symmetry-learn Provider: feature maps + checkpoint-compatible CNN
        |
        v
user-managed trusted pretrained checkpoint
```

The repository contains no model weights, training datasets, or private image
data. It does not silently install software, download checkpoints, or upload
images.

## Scientific Workflow

For the current eight-channel model, the Provider builds the following channels
in this exact order and the Harness validates the returned contract:

1. normalized source image;
2. reflection-strength map;
3. sine of twice the reflection orientation;
4. cosine of twice the reflection orientation;
5. two-fold rotational-symmetry map;
6. three-fold rotational-symmetry map;
7. four-fold rotational-symmetry map;
8. six-fold rotational-symmetry map.

The pretrained 17-class head is used only to load the learned representation.
For a new image, the Provider freezes the pretrained network, inserts residual
bottleneck adapters, replaces the original head with an `N`-class local head,
and trains only the adapters and local head on the clicked support patches.

## Requirements

- Python 3.10 or newer
- NumPy and Pillow
- [symmetry-learn](https://github.com/jiadongdan/symmetry-learn) with its
  Provider runtime; its environment owns PyTorch and all numerical execution
- Gradio for the local annotation interface
- A trusted, user-supplied eight-channel 17-class checkpoint

Install the Harness from a cloned checkout:

```bash
python -m pip install -e ".[ui,dev]"
```

Install the external numerical Provider in the Python runtime that will execute
the model:

```bash
python -m pip install "symmetry-learn[provider] @ git+https://github.com/jiadongdan/symmetry-learn.git@main"
```

PyTorch installation depends on the required CPU or CUDA platform and belongs
to the Provider environment, not the Harness package.

## Initialize

Create a portable configuration and register a trusted checkpoint:

```bash
symmetry init --provider-python /path/to/python --checkpoint /path/to/best_model.pth
symmetry doctor
```

The initialization command discovers the Provider, records its version and
Python executable, and records the checkpoint path and SHA-256 without copying
or modifying the checkpoint. For a development checkout, add
`--provider-source-root /path/to/symmetry-learn`.

## Interactive Use

Launch the local interface:

```bash
symmetry ui
```

Then:

1. load a single-channel image;
2. enter comma-separated local class names;
3. choose the active class and click support points;
4. review the patch outlines and support counts;
5. select at least three points per class, with five recommended;
6. run fine-tuning and dense prediction;
7. inspect the overlay, confidence, entropy, and saved run directory.

Clicks too close to the border are rejected because a complete classifier patch
cannot be extracted. The pretrained checkpoint is never overwritten.

## Agent-Friendly CLI

```text
symmetry init        discover the Provider and create configuration
symmetry doctor      validate Provider, model, device, and checkpoint
symmetry models      report the configured model and scientific contract
symmetry inspect     inspect and normalize an input without running the model
symmetry run         execute a saved annotation session
symmetry reproduce   repeat a prior completed run
symmetry ui          launch the local point-annotation interface
```

All non-UI commands emit JSON. Errors use nonzero exit codes.

Examples:

```bash
symmetry inspect --input image.tif
symmetry run --input image.tif --annotations annotation_session.json
symmetry reproduce --record symmetry-runs/<run-id>/run_record.json
```

Configuration defaults to `SYMMETRY_HARNESS_CONFIG` or
`./symmetry-harness.json`.

## Run Artifacts

Every run is isolated under a unique identifier and includes:

```text
annotation_session.json
input.npy
input_preview.png
features.npz
support_patches.npz
adapter_head.pt
provider_record.json
training_history.json
prediction.npz
prediction_overlay.png
confidence.png
entropy.png
run_record.json
report.md
```

The Provider record preserves its numerical runtime and contract identity. The
Harness run record preserves image and checkpoint checksums, exact click
coordinates, local class mapping, feature parameters, fine-tuning options,
device details, software versions, and artifact paths. Each run-local
annotation session points to the bundled normalized `input.npy`, so a run does
not depend on a temporary UI upload after it completes. Reproduction prefers
that checksum-verified bundled input and remains compatible with records that
refer to the original source file.

Adapter and local-head initialization is seeded before model construction.
Repeating a run on the same recorded runtime is therefore deterministic;
bitwise identity across different PyTorch, CUDA, driver, or hardware versions
is not guaranteed. Confidence is rendered against the fixed interval `0..1`,
and entropy is rendered against `0..log(N)` for `N` local classes. Both use a
blue-to-green-to-red scale from low to high.

## Interpretation

The clicked points are training support data. Support accuracy and training loss
must not be reported as independent accuracy. Confidence and entropy describe
the model output but do not establish physical correctness. Important regions
should be reviewed against the source image and domain knowledge.

## Development

```bash
python -m pip install -e ".[ui,dev]"
python -m pytest -q
```

Unit tests do not require symmetry-learn, a checkpoint, or a GPU. Optional
integration checks can use a locally configured runtime.

All repository content, including documentation, source code, comments, UI
messages, configuration, and tests, is written in English.

## License

This project is released under the [MIT License](LICENSE).
