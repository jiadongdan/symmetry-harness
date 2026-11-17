# Contributor guidance for coding agents

This file describes how to modify `symmetry-harness`. Keep changes portable,
reviewable, and suitable for a public repository.

## Start here

- Read `README.md` for the architecture, installation flow, CLI, and artifact
  formats.
- Read `SKILL.md` and the relevant file under `references/` when changing the
  agent-facing workflow.
- Treat `pyproject.toml` and `.github/workflows/ci.yml` as the authoritative
  test configuration.

## Repository boundaries

- `src/symmetry_harness/` owns orchestration, validation, UI, reporting, and
  communication with the Provider.
- Model definitions, feature extraction, model restoration, training, and
  numerical inference belong to the `symmetry-learn` Provider, not this
  repository.
- Keep the Provider boundary explicit and subprocess-friendly. Do not bypass
  its public contract by importing private Provider implementation details.
- Do not add model weights, training datasets, private images, or machine-local
  configuration to this repository.
- Never silently install software, download optional weights, upload images, or
  perform network operations. Actions that change the environment require an
  explicit user request.

## Important paths

- `src/symmetry_harness/`: package implementation.
- `tests/`: unit, UI, workflow, and real-Provider integration tests.
- `references/`: focused workflow and troubleshooting documentation used by
  the Skill.
- `src/symmetry_harness/templates/`: packaged report templates.
- `scripts/`: cross-platform installation and launch helpers.
- `configs/config.example.json`: public configuration example; keep it free of
  machine-specific paths and credentials.

## Development and testing

Use the active Python environment; do not encode a contributor's interpreter
path or environment name in repository files.

```bash
python -m pip install -e ".[ui,dev]"
python -m pytest -q
```

The default pytest configuration runs the fast suite and excludes two real
Provider integration suites. Run the relevant integration marker whenever a
change touches the Provider contract, feature workflow, model package, or saved
prediction behavior:

```bash
python -m pytest -q -m traditional_integration
python -m pytest -q -m prediction_integration
```

Run every test, including both integration suites, with:

```bash
python -m pytest -q -m ""
```

The integration suites require a working `symmetry-learn` Provider and may take
several minutes. A missing Provider is not evidence that the integration path
works; report the missing prerequisite clearly.

## Change rules

- Preserve public CLI behavior and serialized artifact compatibility unless a
  breaking change is explicitly requested and documented.
- Keep validation at external boundaries. Reject malformed shapes, non-finite
  numeric values, invalid paths, incompatible model contracts, and corrupted
  model packages with actionable errors.
- Keep registered weights and user-supplied checkpoints read-only.
- Preserve provenance fields, checksums, seeds, and resolved configuration in
  run artifacts. Reproducibility metadata is part of the product behavior.
- When changing packaged templates, Skill instructions, or references, verify
  that installation still includes the updated resources.
- Use temporary directories and synthetic data in tests. Never depend on a
  contributor's home directory, local repository layout, credentials, or
  unpublished data.
- Add or update regression tests for behavior changes. Prefer focused tests in
  the existing test module closest to the changed code.

## Completion checklist

Before declaring a change complete:

1. Run the focused tests for the changed behavior.
2. Run the default fast suite.
3. Run the applicable real-Provider marker for integration-sensitive changes.
4. Check documentation and examples when CLI, configuration, artifacts, or
   agent workflow behavior changes.
5. Review the diff for accidental generated files, local paths, credentials,
   datasets, checkpoints, and unrelated edits.

