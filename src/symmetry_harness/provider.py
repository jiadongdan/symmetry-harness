"""Client helpers and readiness checks for an external symmetry provider."""

from __future__ import annotations

from importlib.util import find_spec
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from .config import HarnessConfig


WORKER_SCHEMA_VERSION = "scientific-symmetry-worker-v1"
PROVIDER_CONTRACT_VERSION = "symmetry-learn-provider-v1"
PROVIDER_REPOSITORY_URL = "https://github.com/jiadongdan/symmetry-learn"
PROVIDER_INSTALL_REQUIREMENT = (
    "symmetry-learn[provider] @ "
    "git+https://github.com/jiadongdan/symmetry-learn.git@main"
)


@dataclass(frozen=True)
class ProviderAnalysis:
    """Materialized results returned by one symmetry-learn worker job."""

    arrays: dict[str, np.ndarray]
    adapter_checkpoint: bytes
    record: dict[str, Any]


def default_provider_install_command(python_executable: str) -> str:
    """Return the public installation command for the numerical provider."""
    return f'"{python_executable}" -m pip install "{PROVIDER_INSTALL_REQUIREMENT}"'


def _version_tuple(value: str) -> tuple[int, ...] | None:
    parts = value.split("+", 1)[0].split("-", 1)[0].split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _version_is_older(actual: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    width = max(len(actual), len(minimum))
    return actual + (0,) * (width - len(actual)) < minimum + (0,) * (
        width - len(minimum)
    )


def validate_provider_capabilities(
    config: HarnessConfig, capabilities: dict[str, Any]
) -> list[str]:
    """Return compatibility issues for one provider capability document."""
    issues: list[str] = []
    if capabilities.get("contract_version") != PROVIDER_CONTRACT_VERSION:
        issues.append("provider contract version mismatch")
    if capabilities.get("provider") != config.provider.name:
        issues.append("unexpected provider identity")
    actual = str(capabilities.get("provider_version", "unknown"))
    minimum = config.provider.minimum_version
    if minimum is not None:
        actual_tuple = _version_tuple(actual)
        minimum_tuple = _version_tuple(minimum)
        if actual_tuple is None:
            issues.append("provider version is unavailable")
        elif minimum_tuple is not None and _version_is_older(actual_tuple, minimum_tuple):
            issues.append(f"provider version {actual} is older than required {minimum}")
    models = capabilities.get("models", [])
    model = next(
        (
            entry
            for entry in models
            if isinstance(entry, dict)
            and entry.get("identifier") == config.model.identifier
        ),
        None,
    )
    if model is None:
        issues.append("configured model is missing from provider capabilities")
    elif not bool(model.get("available", False)):
        issues.append("configured model dependency is unavailable")
    elif (
        int(model.get("input_channels", -1)) != config.model.input_channels
        or int(model.get("pretrained_classes", -1))
        != config.model.pretrained_classes
        or int(model.get("classifier_patch_size", -1))
        != config.model.classifier_patch_size
    ):
        issues.append("configured model contract does not match the provider")
    return issues


def provider_environment(config: HarnessConfig) -> dict[str, str]:
    """Build a subprocess environment with an optional development provider root."""
    environment = os.environ.copy()
    entries: list[str] = []
    if config.provider.source_root is not None:
        entries.append(str(config.provider.source_root))
    existing = environment.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    if entries:
        environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def provider_working_directory(config: HarnessConfig) -> Path:
    """Return a stable provider working directory without repository discovery."""
    if config.provider.source_root is not None:
        return config.provider.source_root
    return config.source_path.parent


def provider_command(
    config: HarnessConfig, action: str, path: Path | None = None
) -> list[str]:
    """Build one configured provider module invocation."""
    command = [
        config.model.python_executable,
        "-m",
        config.provider.module,
        action,
    ]
    if path is not None:
        command.append(str(path))
    return command


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_command(config: HarnessConfig) -> str:
    python_executable = config.model.python_executable
    if config.provider.install_hint:
        return config.provider.install_hint.replace("{python}", python_executable)
    if config.provider.source_root is not None:
        return (
            f'"{python_executable}" -m pip install -e '
            f'"{config.provider.source_root}[provider]"'
        )
    return default_provider_install_command(python_executable)


def _invoke_provider_json(
    config: HarnessConfig,
    action: str,
    path: Path | None = None,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        provider_command(config, action, path),
        cwd=provider_working_directory(config),
        env=provider_environment(config),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"symmetry-learn provider failed: {detail}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"symmetry-learn returned invalid JSON: {error}") from error


def _method_payload(config: HarnessConfig) -> dict[str, Any]:
    checkpoint = config.model.checkpoint_path
    if checkpoint is None:
        raise RuntimeError("No pretrained checkpoint is configured.")
    return {
        "identifier": config.model.identifier,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": config.model.checkpoint_sha256,
    }


def probe_model(config: HarnessConfig) -> dict[str, Any]:
    """Ask the provider to strictly load the configured checkpoint and device."""
    with tempfile.TemporaryDirectory(prefix="symmetry-harness-probe-") as temporary:
        job_path = Path(temporary) / "probe.json"
        payload = {
            "schema_version": WORKER_SCHEMA_VERSION,
            "method": _method_payload(config),
            "options": {"device": config.model.device},
        }
        job_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result = _invoke_provider_json(
            config,
            "--probe",
            job_path,
            timeout_seconds=min(config.provider.timeout_seconds, 300),
        )
    if result.get("provider_contract_version") != PROVIDER_CONTRACT_VERSION:
        raise RuntimeError("Provider probe contract version mismatch.")
    if result.get("provider") != config.provider.name:
        raise RuntimeError("Provider probe identity mismatch.")
    if result.get("identifier") != config.model.identifier:
        raise RuntimeError("Provider probe model identity mismatch.")
    return result


def run_provider_analysis(
    config: HarnessConfig,
    image: np.ndarray,
    *,
    coordinates_xy: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    options: dict[str, Any],
) -> ProviderAnalysis:
    """Execute the complete numerical workflow in the configured provider."""
    values = np.asarray(image, dtype=np.float32)
    coordinates = np.asarray(coordinates_xy, dtype=np.int32)
    support_labels = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Provider input must be a finite two-dimensional image.")
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise ValueError("Support coordinates must have shape (samples, 2).")
    if support_labels.shape != (len(coordinates),):
        raise ValueError("Support labels must align with support coordinates.")
    if len(class_names) < 2 or not all(str(name).strip() for name in class_names):
        raise ValueError("At least two non-empty class names are required.")
    provider_options = dict(options)
    provider_options.pop("input_normalization", None)
    if isinstance(provider_options.get("rotation_folds"), tuple):
        provider_options["rotation_folds"] = list(provider_options["rotation_folds"])

    with tempfile.TemporaryDirectory(prefix="symmetry-harness-provider-") as temporary:
        directory = Path(temporary)
        input_path = directory / "input.npy"
        output_path = directory / "output.npz"
        adapter_path = directory / "adapter_head.pt"
        record_path = directory / "provider_record.json"
        job_path = directory / "job.json"
        np.save(input_path, values)
        payload = {
            "schema_version": WORKER_SCHEMA_VERSION,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "adapter_path": str(adapter_path),
            "record_path": str(record_path),
            "support": {
                "coordinates_xy": coordinates.tolist(),
                "labels": support_labels.tolist(),
                "class_names": list(class_names),
            },
            "options": provider_options,
            "method": _method_payload(config),
        }
        job_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _invoke_provider_json(
            config,
            "--job",
            job_path,
            timeout_seconds=config.provider.timeout_seconds,
        )
        if not output_path.is_file() or not adapter_path.is_file() or not record_path.is_file():
            raise RuntimeError("Provider completed without all required output artifacts.")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("schema_version") != WORKER_SCHEMA_VERSION:
            raise RuntimeError("Provider result worker schema mismatch.")
        if record.get("provider_contract_version") != PROVIDER_CONTRACT_VERSION:
            raise RuntimeError("Provider result contract version mismatch.")
        if record.get("provider") != config.provider.name:
            raise RuntimeError("Provider result identity mismatch.")
        if record.get("identifier") != config.model.identifier:
            raise RuntimeError("Provider result model identity mismatch.")
        with np.load(output_path, allow_pickle=False) as payload_arrays:
            arrays = {
                name: np.asarray(payload_arrays[name]).copy()
                for name in payload_arrays.files
            }
        adapter_checkpoint = adapter_path.read_bytes()
    for temporary_key in ("output_path", "adapter_path", "record_path"):
        record.pop(temporary_key, None)
    record["transport"] = {
        "kind": "subprocess_json_npy_npz",
        "python_executable": config.model.python_executable,
        "module": config.provider.module,
    }
    return ProviderAnalysis(
        arrays=arrays,
        adapter_checkpoint=adapter_checkpoint,
        record=record,
    )


def doctor(config: HarnessConfig, *, require_ui: bool = False) -> dict[str, Any]:
    """Check provider compatibility, checkpoint identity, device, and optional UI."""
    issues: list[str] = []
    recommendations: list[str] = []
    executable = Path(config.model.python_executable)
    environment: dict[str, Any] = {
        "python_executable": config.model.python_executable,
        "status": "blocked",
    }
    capabilities: dict[str, Any] | None = None
    if not executable.is_file():
        issues.append("The configured provider Python executable does not exist.")
        recommendations.append(
            f"Configure an existing Python executable: {config.model.python_executable}"
        )
    else:
        try:
            capabilities = _invoke_provider_json(
                config, "--capabilities", timeout_seconds=120
            )
            compatibility_issues = validate_provider_capabilities(config, capabilities)
            if compatibility_issues:
                issues.extend(compatibility_issues)
                recommendations.append(
                    "Install a compatible symmetry provider: " + _install_command(config)
                )
            else:
                environment["status"] = "ready"
                environment["capabilities"] = capabilities
        except Exception as error:
            issues.append(str(error))
            recommendations.append(
                "Install the symmetry provider in this runtime: "
                + _install_command(config)
            )

    checkpoint = config.model.checkpoint_path
    checkpoint_record: dict[str, Any] = {
        "path": None if checkpoint is None else str(checkpoint),
        "configured_sha256": config.model.checkpoint_sha256,
        "exists": bool(checkpoint is not None and checkpoint.is_file()),
    }
    if checkpoint is None:
        issues.append("No pretrained checkpoint is configured.")
        recommendations.append(
            "symmetry init --force --checkpoint \"<trusted-checkpoint-path>\" "
            f'--output "{config.source_path}"'
        )
    elif not checkpoint.is_file():
        issues.append(f"The configured checkpoint does not exist: {checkpoint}")
    else:
        actual = _file_sha256(checkpoint)
        checkpoint_record["actual_sha256"] = actual
        if config.model.checkpoint_sha256 is None:
            issues.append("The configured checkpoint has no recorded SHA-256.")
        elif actual != config.model.checkpoint_sha256:
            issues.append("The configured checkpoint SHA-256 does not match the file.")

    probe: dict[str, Any] | None = None
    if not issues:
        try:
            probe = probe_model(config)
        except Exception as error:
            issues.append(f"Provider model probe failed: {error}")
    if require_ui:
        try:
            gradio_available = find_spec("gradio") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            gradio_available = False
        if not gradio_available:
            issues.append("Gradio is required for the local annotation interface.")
            recommendations.append(
                f'"{Path(sys.executable).resolve()}" -m pip install "symmetry-harness[ui]"'
            )
    return {
        "status": "ready" if not issues else "blocked",
        "config_path": str(config.source_path),
        "provider": {
            "name": config.provider.name,
            "module": config.provider.module,
            "source_root": (
                None
                if config.provider.source_root is None
                else str(config.provider.source_root)
            ),
        },
        "environment": environment,
        "model": {
            "identifier": config.model.identifier,
            "status": "ready" if not issues else "blocked",
            "probe": probe,
        },
        "checkpoint": checkpoint_record,
        "issues": issues,
        "recommendations": list(dict.fromkeys(recommendations)),
    }
