---
name: symmetry-harness
description: Run local few-shot symmetry analysis on single-channel scientific or microscopy images. Use when a user wants to launch the local interface, fine-tune adapters on representative points, inspect confidence or entropy, apply a saved .symmodel package to new images without re-training, or check whether repeated k-shot runs are reproducible. Requires the local symmetry-harness runtime and the symmetry-learn Provider. Do not use for foundation-model training, silent weight installation, or unattended label selection.
metadata:
  version: "0.3.1"
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

1. Run exactly one bundled fast-open script relative to this `SKILL.md` file.
   It starts the long-lived UI process independently, waits for an
   identity-checked ready state on the fixed local port, and reuses an existing
   matching instance. On Windows, run:

   ```powershell
   & "<skill-directory>\scripts\open.ps1"
   ```

   On macOS or Linux, run:

   ```bash
   "<skill-directory>/scripts/open.sh"
   ```

   Resolve `<skill-directory>` only from the location of this loaded Skill. Do not
   search for it. Append `--input <path>` only when the user explicitly asks to
   preload an image. Append `--mode predict` only when the user explicitly asks
   for the saved-model prediction workspace.

2. Read the script's single JSON result. `ready` means the Harness-owned process
   and socket both passed the identity check; `blocked` contains the authoritative
   failure guidance.

3. If `status` is `ready`, return the reported URL immediately and stop using
   tools. The ready result is authoritative evidence that startup succeeded. Do
   not inspect, open, screenshot, scan, or verify the browser page unless the user
   explicitly asks. Do not run environment discovery, `doctor`, `models`,
   `inspect`, package checks, repository searches, source or README inspection,
   port checks, Conda commands, PATH checks, or another launch command.

4. If `status` is `blocked`, report the returned phase, issues, and
   recommendations, then stop using tools. Do not troubleshoot unless the user
   explicitly asks for troubleshooting.

5. If the bundled script is missing or reports that the runtime has not been
   installed, report its recommendation and stop. Do not search Conda
   environments, Python installations, repositories, or PATH entries.

For a startup-only request, do not read any supporting reference.

Do not start the blocking `launch.ps1` or `launch.sh` scripts for an ordinary
agent request. They remain low-level debugging entry points. Do not launch a
background task and then wait for that permanent task to finish; `open.*`
already owns process detachment and readiness waiting. The installer also writes
the same machine-readable protocol to
`~/.symmetry-harness/agent-launch.json` so another agent platform can cache the
exact command in its cross-project memory without encoding a repository path.

Startup is not instantaneous. The script waits for the Harness itself, so
silence before its final JSON is not a reason to add checks or restart it. The
installer primes a fingerprinted strict-model-probe cache. A launch reuses that
result only while the Provider identity and source, selected model and device,
configuration, and weight-file signature still match. A cache miss runs in
parallel with UI construction. The ready payload reports the outcome and cache
status under `model_probe` when a new instance was started:

```json
"model_probe": {
  "status": "ready",
  "cache": {"status": "hit", "schema_version": 1},
  "issues": [],
  "recommendations": []
}
```

Any model-probe status other than `ready` means the configured weight could not
be strictly loaded. The interface still opened, because the user can select
another weight there, so report the issue and do not treat it as a launch
failure. Do not run `doctor` yourself to re-check it; the payload already
carries the answer.

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

## Run Reports

When the user asks for a report, a summary, or a write-up of a completed run,
generate it with the built-in command instead of writing the document yourself.
Run exactly one command, then return the reported `report` path:

```bash
symmetry report --session latest --notes-text "<sample and purpose; class meanings; annotation rationale; notable observations; unresolved questions>"
```

- Use `--session latest` after an interactive fine-tuning session. One completed
  fine-tune returns its ordinary image report; two or more runs from the same UI
  session receive a compact session index plus one complete report per image.
  This prevents the default workflow from silently reporting only the last image.
- Use `--run` only for one explicitly requested run or prediction batch. It
  accepts `latest`, a run id, or a run directory; use `--prefix prediction` to
  target a prediction batch.
- Always pass useful scientific context through `--notes-text` or `--notes
  <file>`: the sample and analysis purpose, user-defined class meanings, support
  point rationale, notable observations, and unresolved questions. Record only
  facts supplied or confirmed by the user; never invent physical meaning. Avoid
  operational filler such as merely saying that the UI was launched or a report
  was requested. This is the only free-form section; every other section is
  derived from persisted run artifacts. Use context already available in the
  conversation and run record. Do not pause to ask the user extra questions only
  to complete a report; mark unavailable context as not recorded instead.
- Image reports are written inside their run directories. A multi-run session
  index is written under `<output-root>/sessions/<session-id>/report_summary.md`.
  The command prints JSON containing `report`, `figures`, and `item_reports`.
  Return the reported path; do not copy or reshape the files.
- Add `--no-figures` only when the user explicitly asks for text without images.
- Do not summarize a run from `report.md` alone: that file is the minimal
  automatic record and carries no figures.

See [references/run-report.md](references/run-report.md) for the section list,
the figure set, and the degradation rules.

## Reproducibility Loops

This is the second conversation flow, and it follows a report naturally. It
answers one question: **if the same k-shot fine-tuning runs N times, does it give
the same answer?** It is not active learning and it never improves the model.

Trigger it when the user asks whether the result is stable or reproducible, or
asks for repeated trials — "run this twenty times", "repeat the 5-shot
experiment", "check reproducibility", or a report and repeated trials in one
breath.

1. Anchor on the run that was just finished. Do **not** ask which one: `--run
   latest` is that run. Print its id and image name in one line so a wrong pick
   is visible immediately, then continue.
2. If the user also asked for a report, generate it first (it takes seconds) and
   relay one line of it before starting the loop.
3. Ask once, not per item: how many replicates, how many shots per class, and
   whether a ground-truth raster exists. Defaults: 20 replicates, 5 shots. Then
   start the loop in the background.
4. Relay every `round i/N acc=... spread=... pool=...` line as it arrives. Four
   minutes of silence is not an acceptable user experience.
5. Finish with mean ± sd, pool sizes, draws per pool cell, and the list of any
   round outside the band. State the measurement mode in the first sentence.

```powershell
symmetry repeat --run latest --rounds 20 --points-per-class 5 `
  --no-keep-checkpoint --no-stop-on-alarm
```

- `--run` takes a run directory, a run id, or `latest`. It reuses that run's
  annotation session as the anchor support set and its stored `input.npy` as the
  image, and inherits the classifier patch, the symmetry patch and the epoch
  count, so the loop cannot drift from its anchor. Add `--gt` for ground-truth
  scoring.
- **The data decides the mode, not preference.** With a label raster, pass
  `--gt --mode stability`; the pool is then limited to anchor cells that agree
  with the truth and clear the phase boundary. Without one there is nothing to
  pool against: the loop is `user` mode, accuracy is `nan`, and the only honest
  summary is drift from the anchor — say "agreement with the anchor, not
  accuracy" and never present it as a correct rate.
- Never lower `--tau` to make a small pool fit. A pool that cannot supply the
  requested points means the first fine-tune was too weak: report that and ask
  for a few more labels plus another fine-tune.
- Do not re-draw a round that fell outside the band. Report its number and value.
  Re-drawing makes the replicate count incomparable.
- Default to `--no-keep-checkpoint` (about 8 MB per round instead of 45) and
  `--no-stop-on-alarm` (the distribution is the point). Continue an interrupted
  loop with `--resume <loop directory>`; never restart it from scratch.

The interface's **Finish this run** button does the report half of this flow
inside the page: it writes `report_summary.md`, a self-contained
`report_view.html`, and `ui_handoff.json` into the run directory. When the user
says "pick up the run I just finished", read that handoff instead of guessing —
it records the run id, image, classes and a ready command. The button cannot
notify the assistant; it only leaves the record. So when the conversation starts
after a finished run, say what is available and offer the loop rather than
waiting to be asked.

See [references/repeat.md](references/repeat.md) for the pool definitions, the
stop rules, the resume rules, and the artifact layout.

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
- When the user asks whether the fine-tuning pipeline is **stable across repeated
  runs** — a reproducibility test with a fixed number of shots per round — follow
  the Reproducibility Loops flow above.
- When the user asks for a report or summary of a completed run, read
  [references/run-report.md](references/run-report.md).

## Compatibility

The launchers contain no fixed workspace paths or Python environments. The
installer writes host-specific paths only to user-local runtime state.

Runtime state is written with LF endings, and the POSIX launchers strip a stray
CR from state written by older installers, so `launch.sh` works on Windows as
well as on macOS and Linux. Prefer the `.ps1` scripts on Windows anyway:
`stop.sh` deliberately refuses to run there because MSYS cannot list native
Windows processes and would otherwise report a false "nothing running".
