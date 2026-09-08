# Troubleshooting a Blocked Launch

Read this reference only when the user explicitly asks to diagnose or repair a
blocked launch. Do not use it for an ordinary startup request.

1. Use the phase, issues, and recommendations returned by the bundled launcher as
   the starting evidence. Do not repeat checks whose results are already present.
2. If the phase is `runtime`, run `scripts/install.py` once with the Python runtime
   chosen for symmetry-harness. The installer records the absolute executable and
   configuration paths, validates the runtime, and updates the Codex Skill.
3. For a Harness preflight failure, run `symmetry doctor` through the exact Python
   executable recorded in `~/.symmetry-harness/runtime.json`. Use the same
   configuration path recorded there. Do not search Conda environments or PATH.
4. If the configuration is missing, run `symmetry init`. It selects the sole
   Provider model and bundled default weight. When several models exist, ask the
   user to select one and pass `--model` and optionally `--weight`.
5. If `doctor` reports a missing or incompatible symmetry-learn Provider,
   Provider dependency, Gradio installation, or selected weight, report its
   recommendations. Stop when the status remains `blocked`.
6. Install an optional weight only after the user explicitly asks for the reported
   installation command or selects the installation action in the interface.
7. Run `symmetry models` only when model or weight selection is part of the
   reported problem.

Report a missing local capability instead of inventing a fallback.
