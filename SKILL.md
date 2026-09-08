---
name: symmetry-harness
description: Run interactive few-shot symmetry analysis on single-channel scientific or microscopy images. Use when a user wants to choose an installed local symmetry model, select three to five representative points per class, fine-tune adapters and a task-specific head, create a dense classification map, inspect confidence or entropy, or reproduce a prior run. Requires the local symmetry-harness CLI and symmetry-learn Provider runtime. Do not use for foundation-model training, silent weight installation, or unattended label selection.
metadata:
  version: "0.2.0"
  product: "symmetry-harness"
---

# Symmetry Harness

Use the local `symmetry` CLI as the deterministic execution surface. The host
agent resolves user intent, launches the local annotation interface, waits for
the user to choose support points, invokes the recorded workflow, and explains
the resulting JSON and artifacts. Numerical feature extraction and model
execution remain local. The Harness owns interaction and orchestration;
symmetry-learn owns feature extraction, checkpoint loading, fine-tuning, and
inference through its versioned Provider contract.

## Warm Start

Prefer one deterministic browser-first invocation:

```bash
symmetry launch --config <path> --server-port 0 --no-inbrowser
```

`launch` performs the required UI-aware doctor check, reports the configured model
and weight contract, chooses an available local port, and emits one ready JSON
object before blocking. Open or report its URL immediately. The user then chooses
the image in Gradio. When the input is already known, `--input <path>` remains an
optional preload shortcut.

Do not separately repeat environment discovery, README or source inspection,
`doctor`, `models`, `inspect`, or port probing after `launch` reports `ready`. If
`launch` reports `blocked`, report its phase, issues, and recommendations and stop.
If the installed Harness does not expose `launch`, use the compatible cold-start
and interactive workflow below. When the host can choose task reasoning effort,
prefer low or medium effort for this deterministic warm-start orchestration.

## Cold Start

1. Run `symmetry doctor`. If the command is missing, explain that the Harness
   package must be installed from this repository. Do not install software
   without permission.
2. If the configuration is missing, run `symmetry init`. It selects the sole
   Provider model and bundled default weight. When several models exist, ask the
   user to select one and pass `--model` and optionally `--weight`.
3. If `doctor` reports a missing or incompatible symmetry-learn Provider,
   Provider dependency, Gradio installation, or selected weight, present its
   recommendations and stop when the status is `blocked`. Install an optional
   weight only after the user explicitly clicks the UI installation action or
   explicitly asks for the reported installation command.
4. Run `symmetry models` and use only the reported model, weight, device,
   feature, and fine-tuning contracts.

## Interactive Workflow

1. Launch `symmetry launch` or the compatible `symmetry ui` command without
   requiring an input path.
2. Let the user choose the model, weight, and local image in Gradio. Do not
   upload the image to a remote service.
3. Ask the user to define the local classes and select at least three,
   preferably five, representative support points per class.
4. Do not choose points on the user's behalf. The interface rejects points
   whose full classifier patch would cross the image boundary.
5. Ask the user to review class names, colors, counts, patch outlines, and patch
   previews before starting fine-tuning.
6. Let the interface run adapter-plus-head fine-tuning and dense prediction.
   Report the run directory, support counts, configuration, artifacts, and
   warnings.
7. If the map is not satisfactory, invite the user to add representative
   points and create a new run. Never overwrite an earlier run.

## Headless Reproduction

Use `symmetry run --input <path> --annotations <session.json>` to execute a
saved annotation session. Use `symmetry reproduce --record <run_record.json>`
to repeat a completed run with the recorded scientific and training options.

## Scientific Invariants

- Treat the selected points as support data, not independent validation data.
- Never describe support accuracy or training loss as generalization accuracy.
- Keep the pretrained checkpoint read-only and save adapter-plus-head weights
  separately with the base checkpoint checksum.
- Record image checksum, coordinates, class mapping, feature parameters,
  classifier patch size, stride, seed, runtime, and software versions.
- Preserve exact source-image coordinates. Display coordinates must never be
  substituted for array coordinates without validation.
- Confidence and entropy describe model output, not physical correctness.
- Keep unpublished images local unless the user explicitly authorizes upload.

## Compatibility

This Skill uses no host-specific tool names, multi-agent primitives, fixed
workspace paths, or fixed Python environments. A host that can read local
files, execute shell commands, parse JSON, and open a local browser can use it.
Report a missing capability instead of inventing a fallback.
