---
name: symmetry-harness
description: Run interactive few-shot symmetry analysis on single-channel scientific or microscopy images. Use when a user wants to label three to five representative points per class, fine-tune the installed local symmetry model with adapters and a task-specific head, create a dense classification map, inspect confidence or entropy, or reproduce a prior run. Requires the local symmetry-harness CLI, the symmetry-learn Provider runtime, and a user-supplied trusted checkpoint. Do not use for foundation-model training, checkpoint downloads, or unattended label selection.
metadata:
  version: "0.1.0"
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

## Cold Start

1. Run `symmetry doctor`. If the command is missing, explain that the Harness
   package must be installed from this repository. Do not install software
   without permission.
2. If the configuration is missing, run `symmetry init --checkpoint <path>`
   using a checkpoint path supplied by the user.
3. If `doctor` reports a missing or incompatible symmetry-learn Provider,
   Provider dependency, Gradio installation, or checkpoint, present its
   recommendations and stop when the status is `blocked`. Never search for or
   download model weights.
4. Run `symmetry models` and use only the reported model, device, feature, and
   fine-tuning contracts.

## Interactive Workflow

1. Run `symmetry inspect --input <path>` before analysis.
2. Launch `symmetry ui`. Ask the user to define the local classes and select at
   least three, preferably five, representative support points per class.
3. Do not choose points on the user's behalf. The interface rejects points
   whose full classifier patch would cross the image boundary.
4. Ask the user to review class names, colors, counts, patch outlines, and patch
   previews before starting fine-tuning.
5. Let the interface run adapter-plus-head fine-tuning and dense prediction.
   Report the run directory, support counts, configuration, artifacts, and
   warnings.
6. If the map is not satisfactory, invite the user to add representative
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
