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
        +-- model and weight selection
        +-- image selection and point annotation
        +-- reproducible run record
        |
        v
symmetry-learn Provider: model registry + numerical workflows
        |
        v
bundled, optional, or user-supplied model weight
```

The repository contains no model weights, training datasets, or private image
data. The default runnable weight is distributed by `symmetry-learn`, while
additional registered weights require an explicit user installation action.
The Harness never silently installs software or uploads images.

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

## One-time Codex Installation

Clone `symmetry-harness` and `symmetry-learn` as sibling directories. Run the
installer with the Python runtime that should own both packages:

```bash
python scripts/install.py
```

The installer performs the editable package installation, creates and validates
the default Harness configuration, records the exact Python and configuration
paths under `~/.symmetry-harness`, and installs the maintained Skill resources
under `~/.codex/skills/symmetry-harness`. It does not require `conda` or the
`symmetry` entry point to be available on PATH after installation.

To use an existing package installation without reinstalling dependencies:

```bash
python scripts/install.py --skip-package-install
```

After installation, start a new Codex task and invoke:

```text
$symmetry-harness launch
```

The Skill runs its bundled launcher once and returns the first ready URL. The
launcher uses the recorded absolute Python executable and configuration path, so
the command works from any project directory.

Install the Harness from a cloned checkout:

```bash
python -m pip install -e ".[ui,dev]"
```

The Provider is not currently published on PyPI. Clone its repository, then
install its model package and main library in the Python runtime that will
execute the model:

```bash
git clone https://github.com/jiadongdan/symmetry-learn.git
python -m pip install -e ./symmetry-learn/model_packages/symmetry_learn_default_model
python -m pip install -e "./symmetry-learn[provider]"
```

The default model-weight package comes from the same source checkout. Optional
registered weight packages can be installed later from the Gradio catalog when
their distribution channels are available. PyTorch belongs to the Provider
environment, not the Harness package.

## Initialize

Create a portable configuration using the Provider default model and weight:

```bash
symmetry init --provider-python /path/to/python
symmetry doctor
```

The initialization command discovers the Provider, records its version and
Python executable, and selects the sole available model plus its default weight.
When the Provider exposes multiple models, select one explicitly:

```bash
symmetry init --model cnn_8ch_pg17 --weight pg17-symmetry-v1
```

A trusted custom checkpoint remains supported. The Harness records its path and
SHA-256 without copying or modifying it:

```bash
symmetry init --model cnn_8ch_pg17 --checkpoint /path/to/best_model.pth
```

For a development checkout, add
`--provider-source-root /path/to/symmetry-learn`.

## Interactive Use

Start the local interface without choosing an image on the command line:

```bash
symmetry launch --config symmetry-harness.json --server-port 0 --no-inbrowser
```

The command performs the UI-aware readiness check, records the model contract,
selects an available local port, and emits one ready JSON object before it blocks
for interaction. The ready payload reports `awaiting_user_selection`; choose or
drag an image into Gradio after opening the URL.

An input path remains an optional preload shortcut:

```bash
symmetry launch --config symmetry-harness.json --input image.npy --server-port 0
```

Then:

1. choose a model and registered weight, or select a trusted custom checkpoint;
2. explicitly install a selected optional weight if its status is not installed;
3. choose or drag in a single-channel image;
4. review its shape, dtype, normalization, and checksum status;
5. enter comma-separated local class names;
6. choose the active class and click support points;
7. review the patch outlines and support counts;
8. select at least three points per class, with five recommended;
9. run fine-tuning and dense prediction;
10. inspect the overlay, confidence, entropy, and saved run directory.

Clicks too close to the border are rejected because a complete classifier patch
cannot be extracted. Choosing another image clears support points and prediction
state while retaining class names and colors. Registered and custom pretrained
weights are always read-only.

The `symmetry ui` compatibility command launches the same browser-first workflow.

## Agent-Friendly CLI

```text
symmetry init        discover the Provider and create configuration
symmetry doctor      validate Provider, model, selected weight, and device
symmetry models      report the complete model and weight catalog
symmetry inspect     inspect and normalize an input without running the model
symmetry launch      preflight the runtime and launch the local UI
symmetry run         execute a saved annotation session
symmetry reproduce   repeat a prior completed run
symmetry ui          launch the local point-annotation interface
```

All non-UI commands emit JSON. `launch` emits ready JSON before blocking and exits
with code 2 when its preflight is blocked. Errors use nonzero exit codes.

Examples:

```bash
symmetry inspect --input image.tif
symmetry launch --config symmetry-harness.json --server-port 0
symmetry launch --config symmetry-harness.json --input image.tif --server-port 0
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
Harness run record preserves image and resolved weight checksums, exact click
coordinates, local class mapping, feature parameters, fine-tuning options,
device details, software versions, and artifact paths. Each run-local
annotation session points to the bundled normalized `input.npy`, so a run does
not depend on a temporary UI upload after it completes. Reproduction prefers
that checksum-verified bundled input and remains compatible with records that
refer to the original source file. Registered-weight reproduction asks the
Provider to resolve the selected weight and verifies its checksum; custom
checkpoint reproduction applies the same checksum rule.

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
