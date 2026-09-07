"""Create a portable configuration from explicit user-supplied paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from .config import CONFIG_SCHEMA_VERSION
from .image_io import file_sha256
from .provider import (
    PROVIDER_CONTRACT_VERSION,
    default_provider_install_command,
)


def _resolve_python(value: str | Path | None) -> Path:
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return Path(sys.executable).resolve()
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    discovered = shutil.which(str(value))
    if discovered is not None:
        return Path(discovered).resolve()
    return candidate.resolve()


def _provider_environment(source_root: Path | None) -> dict[str, str]:
    environment = os.environ.copy()
    if source_root is not None:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(source_root) if not existing else os.pathsep.join(
            (str(source_root), existing)
        )
    return environment


def discover_provider(
    python_executable: str | Path | None = None,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Query symmetry-learn through its versioned subprocess interface."""
    executable = _resolve_python(python_executable)
    root = None if source_root is None else Path(source_root).expanduser().resolve()
    completed = subprocess.run(
        [str(executable), "-m", "symmlearn.provider.worker", "--capabilities"],
        cwd=None if root is None else root,
        env=_provider_environment(root),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "A compatible symmetry-learn provider was not discovered in "
            f"{executable}. Install it with: "
            f"{default_provider_install_command(str(executable))}. "
            f"Provider detail: {detail}"
        )
    try:
        capabilities = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"symmetry-learn returned invalid capability JSON: {error}"
        ) from error
    if capabilities.get("contract_version") != PROVIDER_CONTRACT_VERSION:
        raise RuntimeError(
            "The installed symmetry-learn provider uses an incompatible contract."
        )
    if capabilities.get("provider") != "symmetry-learn":
        raise RuntimeError("Provider discovery returned an unexpected identity.")
    return capabilities


def default_configuration(
    checkpoint: Path | None,
    *,
    provider_python: Path,
    provider_source_root: Path | None,
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    """Build the v1 configuration from discovered provider capabilities."""
    if capabilities.get("contract_version") != PROVIDER_CONTRACT_VERSION:
        raise RuntimeError("Provider capabilities use an incompatible contract.")
    if capabilities.get("provider") != "symmetry-learn":
        raise RuntimeError("Provider capabilities use an unexpected identity.")
    checkpoint_path = None
    checkpoint_sha256 = None
    if checkpoint is not None:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        checkpoint_path = str(checkpoint)
        checkpoint_sha256 = file_sha256(checkpoint)
    models = list(capabilities.get("models", []))
    capability = next(
        (item for item in models if item.get("identifier") == "cnn_8ch_pg17"),
        None,
    )
    if capability is None or not bool(capability.get("available", False)):
        raise RuntimeError(
            "The symmetry-learn provider does not expose the required cnn_8ch_pg17 model."
        )
    defaults = dict(capability.get("defaults", {}))
    provider_version = str(capabilities.get("provider_version", "")).strip()
    if not provider_version:
        raise RuntimeError("Provider discovery did not report a provider version.")
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "project_root": ".",
        "output_root": "symmetry-runs",
        "provider": {
            "name": str(capabilities.get("provider", "symmetry-learn")),
            "distribution": "symmetry-learn",
            "module": "symmlearn.provider.worker",
            "minimum_version": provider_version,
            "source_root": (
                None
                if provider_source_root is None
                else str(provider_source_root)
            ),
            "install_hint": default_provider_install_command("{python}"),
            "timeout_seconds": 3600,
        },
        "model": {
            "identifier": "cnn_8ch_pg17",
            "python_executable": str(provider_python),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "device": "auto",
            "input_channels": int(capability.get("input_channels", 8)),
            "pretrained_classes": int(capability.get("pretrained_classes", 17)),
            "classifier_patch_size": int(
                capability.get("classifier_patch_size", 64)
            ),
        },
        "features": {
            "n_max": int(defaults.get("n_max", 20)),
            "symmetry_patch_size": int(defaults.get("symmetry_patch_size", 51)),
            "rotation_folds": list(defaults.get("rotation_folds", [2, 3, 4, 6])),
            "reflection_p": float(defaults.get("reflection_p", 2)),
            "normalize_rotation_maps": bool(
                defaults.get("normalize_rotation_maps", False)
            ),
            "input_normalization": "minmax_0_1",
        },
        "fine_tuning": {
            "minimum_shots_per_class": int(
                defaults.get("minimum_shots_per_class", 3)
            ),
            "recommended_shots_per_class": 5,
            "maximum_shots_per_class": int(
                defaults.get("maximum_shots_per_class", 50)
            ),
            "adapter_bottleneck": int(defaults.get("adapter_bottleneck", 16)),
            "epochs": int(defaults.get("epochs", 150)),
            "learning_rate": float(defaults.get("learning_rate", 0.0005)),
            "weight_decay": float(defaults.get("weight_decay", 0.0)),
            "seed": int(defaults.get("seed", 42)),
        },
        "prediction": {
            "stride": int(defaults.get("stride", 4)),
            "batch_size": int(defaults.get("batch_size", 512)),
        },
    }


def initialize_config(
    output: str | Path,
    *,
    checkpoint: str | Path | None,
    provider_python: str | Path | None = None,
    provider_source_root: str | Path | None = None,
    force: bool,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Discover the provider and write one portable configuration."""
    destination = Path(output).expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(
            f"Configuration already exists: {destination}. Use --force to replace it."
    )
    resolved_checkpoint = None if checkpoint is None else Path(checkpoint)
    python_path = _resolve_python(provider_python)
    source_root = (
        None
        if provider_source_root is None
        else Path(provider_source_root).expanduser().resolve()
    )
    discovered = capabilities or discover_provider(python_path, source_root)
    payload = default_configuration(
        resolved_checkpoint,
        provider_python=python_path,
        provider_source_root=source_root,
        capabilities=discovered,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    guidance = []
    if resolved_checkpoint is None:
        guidance.append(
            "Register a trusted pretrained checkpoint with "
            "symmetry init --force --checkpoint <path>."
        )
    return {
        "status": "configured",
        "config_path": str(destination),
        "checkpoint_configured": resolved_checkpoint is not None,
        "checkpoint_sha256": payload["model"]["checkpoint_sha256"],
        "provider": payload["provider"]["name"],
        "provider_version": payload["provider"]["minimum_version"],
        "guidance": guidance,
        "next_command": f'symmetry doctor --config "{destination}"',
        "python_executable": str(python_path),
    }
