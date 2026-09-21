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

## The interface reports `ready`, then disappears on the first interaction

Symptom: the launcher prints `status: ready`, but the interface dies within a
minute or two — reliably on the first page load, upload, or click. The final
launcher output is a `blocked` result whose issue text quotes a library warning,
usually followed by the recommendation "Run the symmetry-harness installer
once." That recommendation is a fallback string, not a diagnosis; nothing is
wrong with the installation.

Cause: the launcher runs the host as a native child process, and the host writes
progress and library deprecation notices to standard error. When the launcher is
invoked with its streams merged or redirected (`& launch.ps1 *>&1 | ...`,
`2>&1 | Tee-Object ...`, a CI or task-runner capture), PowerShell converts every
native stderr line into an `ErrorRecord`. Under the launcher's `Stop` preference
that becomes a terminating error: the script aborts mid-run, its `catch` emits
the misleading `blocked` result, and the host process dies with it.

Because the host imports its web layer lazily, the first stderr line usually
appears only when the first HTTP request is served — which is why the failure
presents as "it breaks when I give it data" rather than as a startup failure.

Both Windows launchers now set a non-terminating preference before invoking the
host, so host stderr can never abort a launch. If a launcher predating that
change is in use, invoke it without merging or redirecting its streams:

```powershell
& "<skill-directory>\scripts\launch.ps1" --server-port 7860
```

A notice such as `StarletteDeprecationWarning: 'HTTP_422_UNPROCESSABLE_ENTITY'
is deprecated` on Gradio 6 is expected and is not a launch failure.

Passing `--server-port <port>` pins the port: the launcher appends its arguments
after its own `--server-port 0`, so the later value wins and the reported URL is
predictable.

