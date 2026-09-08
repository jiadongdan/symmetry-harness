#!/usr/bin/env python3
"""Install the local runtime and register the symmetry-harness Codex skill."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence


RUNTIME_SCHEMA_VERSION = "symmetry-harness-runtime-v1"
STATE_DIRECTORY_NAME = ".symmetry-harness"
CONFIG_NAME = "symmetry-harness.json"
RUNTIME_NAME = "runtime.json"
PYTHON_PATH_NAME = "python-path.txt"
CONFIG_PATH_NAME = "config-path.txt"


def _run(command: Sequence[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        text=True,
        capture_output=capture,
        check=False,
    )
    if completed.returncode != 0:
        detail = ""
        if capture:
            detail = completed.stderr.strip() or completed.stdout.strip()
        suffix = f"\n{detail}" if detail else ""
        raise RuntimeError(f"Command failed: {subprocess.list2cmdline(list(command))}{suffix}")
    return completed


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_runtime_state(
    state_directory: Path,
    *,
    harness_python: Path,
    config_path: Path,
) -> Path:
    """Persist exact launch paths without depending on PATH or shell activation."""
    state_directory = state_directory.expanduser().resolve()
    python_path = harness_python.expanduser().resolve()
    resolved_config = config_path.expanduser().resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Harness Python does not exist: {python_path}")
    if not resolved_config.is_file():
        raise FileNotFoundError(f"Harness configuration does not exist: {resolved_config}")

    payload = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "harness_python": str(python_path),
        "config_path": str(resolved_config),
    }
    runtime_path = state_directory / RUNTIME_NAME
    _write_text_atomic(
        runtime_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(state_directory / PYTHON_PATH_NAME, f"{python_path}\n")
    _write_text_atomic(state_directory / CONFIG_PATH_NAME, f"{resolved_config}\n")
    return runtime_path


def install_codex_skill(repository_root: Path, codex_home: Path) -> Path:
    """Copy only the maintained Skill resources into the user's Codex directory."""
    source_root = repository_root.expanduser().resolve()
    destination = codex_home.expanduser().resolve() / "skills" / "symmetry-harness"
    if destination.exists() and source_root.samefile(destination):
        return destination
    destination.mkdir(parents=True, exist_ok=True)

    for relative in (Path("SKILL.md"), Path("agents/openai.yaml")):
        source = source_root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for directory_name in ("references",):
        source = source_root / directory_name
        if source.is_dir():
            shutil.copytree(source, destination / directory_name, dirs_exist_ok=True)

    script_destination = destination / "scripts"
    script_destination.mkdir(parents=True, exist_ok=True)
    for name in ("launch.ps1", "launch.sh"):
        target = script_destination / name
        shutil.copy2(source_root / "scripts" / name, target)
        if name.endswith(".sh"):
            target.chmod(target.stat().st_mode | 0o111)
    return destination


def _json_result(command: Sequence[str]) -> dict[str, Any]:
    completed = _run(command, capture=True)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The symmetry-harness command returned invalid JSON: "
            f"{completed.stdout.strip()}"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symmetry-learn-source",
        type=Path,
        help="Source checkout. Defaults to the symmetry-learn sibling directory.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help=f"Runtime state directory. Defaults to ~/{STATE_DIRECTORY_NAME}.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Existing or new Harness configuration path.",
    )
    parser.add_argument(
        "--codex-home",
        type=Path,
        help="Codex home directory. Defaults to CODEX_HOME or ~/.codex.",
    )
    parser.add_argument(
        "--skip-package-install",
        action="store_true",
        help="Reuse packages already installed in this Python runtime.",
    )
    parser.add_argument(
        "--skip-codex-skill",
        action="store_true",
        help="Configure the runtime without installing the Codex skill.",
    )
    parser.add_argument(
        "--reset-config",
        action="store_true",
        help="Replace an existing generated Harness configuration.",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    learn_root = (
        repository_root.parent / "symmetry-learn"
        if arguments.symmetry_learn_source is None
        else arguments.symmetry_learn_source
    ).expanduser().resolve()
    state_directory = (
        Path.home() / STATE_DIRECTORY_NAME
        if arguments.state_dir is None
        else arguments.state_dir
    ).expanduser().resolve()
    config_path = (
        state_directory / CONFIG_NAME
        if arguments.config is None
        else arguments.config
    ).expanduser().resolve()
    harness_python = Path(sys.executable).resolve()

    if not learn_root.is_dir():
        raise FileNotFoundError(
            "symmetry-learn source checkout does not exist: " f"{learn_root}"
        )
    model_package = learn_root / "model_packages" / "symmetry_learn_default_model"
    if not model_package.is_dir():
        raise FileNotFoundError(f"Default model package does not exist: {model_package}")

    if not arguments.skip_package_install:
        for requirement in (
            str(model_package),
            str(learn_root),
            f"{repository_root}[ui]",
        ):
            _run(
                [
                    str(harness_python),
                    "-m",
                    "pip",
                    "install",
                    "--editable",
                    requirement,
                ]
            )

    if arguments.reset_config or not config_path.is_file():
        command = [
            str(harness_python),
            "-m",
            "symmetry_harness.cli",
            "init",
            "--output",
            str(config_path),
            "--provider-python",
            str(harness_python),
            "--provider-source-root",
            str(learn_root),
        ]
        if config_path.exists():
            command.append("--force")
        initialized = _json_result(command)
        if initialized.get("status") != "configured":
            raise RuntimeError("Harness configuration was not created successfully.")

    readiness = _json_result(
        [
            str(harness_python),
            "-m",
            "symmetry_harness.cli",
            "doctor",
            "--config",
            str(config_path),
            "--require-ui",
        ]
    )
    if readiness.get("status") != "ready":
        issues = "; ".join(str(item) for item in readiness.get("issues", []))
        raise RuntimeError(f"The installed runtime is blocked: {issues}")

    runtime_path = write_runtime_state(
        state_directory,
        harness_python=harness_python,
        config_path=config_path,
    )

    skill_path = None
    if not arguments.skip_codex_skill:
        configured_codex_home = arguments.codex_home
        if configured_codex_home is None:
            configured_codex_home = Path(
                os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
            )
        skill_path = install_codex_skill(repository_root, configured_codex_home)

    print(
        json.dumps(
            {
                "status": "installed",
                "harness_python": str(harness_python),
                "config_path": str(config_path),
                "runtime_path": str(runtime_path),
                "codex_skill_path": None if skill_path is None else str(skill_path),
                "next_prompt": "$symmetry-harness launch",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from error
