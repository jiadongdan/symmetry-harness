---
name: symmetry-harness
description: Run local few-shot symmetry analysis on single-channel scientific or microscopy images. Use when a user wants to launch the local interface, fine-tune adapters on representative points, inspect confidence or entropy, or apply a saved .symmodel package to new images without re-training. Requires the local symmetry-harness runtime and the symmetry-learn Provider. Do not use for foundation-model training, silent weight installation, or unattended label selection.
metadata:
  version: "0.2.2"
  product: "symmetry-harness"
---

# Symmetry Harness

Use the bundled launcher as the deterministic execution surface. The one-time
installer records the exact Harness Python executable and configuration in the
user's local runtime state. The Harness owns interaction and orchestration.
`symmetry-learn` owns feature extraction, checkpoint loading, fine-tuning, and
inference through its versioned Provider contract.

There are two peer workspaces in one application:

```text
Fine-tune a model                    Predict with a saved model
-----------------                    --------------------------
Select base model                    Import .symmodel package
Load reference image                 Load one or more images
Compute symmetry features            Validate package and inputs
Define classes                       Compute matching features
Select support points                Restore fine-tuned model
Fine-tune adapters and head          Run dense prediction only
Review prediction                    Review/export results
Export .symmodel
```

## Fast Launch

For a request to start or open the interactive interface, follow only this path:

1. Run exactly one bundled script relative to this `SKILL.md` file. On Windows,
   run:

   ```powershell
   & "<skill-directory>\scripts\launch.ps1"
   ```

   On macOS or Linux, run:

   ```bash
   "<skill-directory>/scripts/launch.sh"
   ```

   Resolve `<skill-directory>` only from the location of this loaded Skill. Do not
   search for it. Append `--input <path>` only when the user explicitly asks to
   preload an image. Append `--mode predict` only when the user explicitly asks
   for the saved-model prediction workspace. Waiting for or reading more output
   from this same process remains part of the single launch invocation.

2. Read stdout until the first JSON object whose `status` is `ready` or `blocked`.

3. If `status` is `ready`, return the reported URL immediately and stop using
   tools. The ready result is authoritative evidence that startup succeeded. Do
   not inspect, open, screenshot, scan, or verify the browser page unless the user
   explicitly asks. Do not run environment discovery, `doctor`, `models`,
   `inspect`, package checks, repository searches, source or README inspection,
   port checks, Conda commands, PATH checks, or another launch command.

4. If `status` is `blocked`, report the returned phase, issues, and
   recommendations, then stop using tools. Do not troubleshoot unless the user
   explicitly asks for troubleshooting.

5. If the bundled launcher is missing or reports that the runtime has not been
   installed, report its recommendation and stop. Do not search Conda
   environments, Python installations, repositories, or PATH entries.

For a startup-only request, do not read any supporting reference.

## Traditional ML Validation (explicit request only)

This is a separate page, not a normal workspace. Launch it only when the user
explicitly asks for traditional machine-learning validation. Fast Launch is
unchanged.

On Windows:

```powershell
& "<skill-directory>\scripts\launch-traditional.ps1"
```

On macOS or Linux:

```bash
"<skill-directory>/scripts/launch-traditional.sh"
```

Scientific scope: the page trains a conventional scikit-learn classifier from
scratch on user-selected patches and densely predicts the image for exploratory
comparison. It is **not** proof that the pretrained model is necessary or
unnecessary, and it never loads, probes, or requires a pretrained checkpoint.
See [references/traditional-validation.md](references/traditional-validation.md)
for inputs, classifiers, outputs, and limits.

## Stopping

For a request to stop, close, or shut down the interface, run exactly one
bundled stop script. It terminates every Harness instance regardless of which
tool or terminal started it, and then verifies the count is zero.

On Windows:

```powershell
& "<skill-directory>\scripts\stop.ps1"
```

On macOS or Linux:

```bash
"<skill-directory>/scripts/stop.sh"
```

The scripts match on the whole command line (`symmetry_harness.cli` together
with `launch`), so they work whether the interface was started via
`python -m symmetry_harness.cli` or a console script. Do not hand-roll a
kill command that matches a single argument position; `-m` shifts the module
name and such a match silently finds nothing.

## Other Modes

- When the user explicitly asks for traditional machine-learning validation — a
  conventional scikit-learn classifier trained on user-selected patches and
  compared against the source image — read
  [references/traditional-validation.md](references/traditional-validation.md).
  This is a separate, explicit-request-only page. It is **not** part of Fast
  Launch and does not change the two normal workspaces.
- When the user explicitly asks to diagnose or repair a blocked launch, read
  [references/troubleshooting.md](references/troubleshooting.md).
- When the interface is already available and the user asks for fine-tuning
  workflow help or result interpretation, read
  [references/interactive-workflow.md](references/interactive-workflow.md).
- When the user wants to apply a saved `.symmodel` package to new images, read
  [references/prediction.md](references/prediction.md).
- For a saved annotation session or completed-run reproduction, read
  [references/reproduction.md](references/reproduction.md).

## Compatibility

The launchers contain no fixed workspace paths or Python environments. The
installer writes host-specific paths only to user-local runtime state.
