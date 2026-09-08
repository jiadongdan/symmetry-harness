---
name: symmetry-harness
description: Run interactive few-shot symmetry analysis on single-channel scientific or microscopy images. Use when a user wants to launch the local interface, choose an installed symmetry model, select three to five representative points per class, fine-tune adapters and a task-specific head, create a dense classification map, inspect confidence or entropy, or reproduce a prior run. Requires the local symmetry-harness CLI and symmetry-learn Provider runtime. Do not use for foundation-model training, silent weight installation, or unattended label selection.
metadata:
  version: "0.2.0"
  product: "symmetry-harness"
---

# Symmetry Harness

Use the bundled launcher as the deterministic execution surface. The one-time
installer records the exact Harness Python executable and configuration in the
user's local runtime state. The Harness owns interaction and orchestration.
`symmetry-learn` owns feature extraction, checkpoint loading, fine-tuning, and
inference through its versioned Provider contract.

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
   preload an image. Waiting for or reading more output from this same process
   remains part of the single launch invocation.

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

## Other Modes

- When the user explicitly asks to diagnose or repair a blocked launch, read
  [references/troubleshooting.md](references/troubleshooting.md).
- When the interface is already available and the user asks for workflow help or
  result interpretation, read
  [references/interactive-workflow.md](references/interactive-workflow.md).
- For a saved annotation session or completed-run reproduction, read
  [references/reproduction.md](references/reproduction.md).

## Compatibility

The launchers contain no fixed workspace paths or Python environments. The
installer writes host-specific paths only to user-local runtime state.
