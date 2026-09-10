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
from time import monotonic, perf_counter
from typing import Any, Callable

import numpy as np

from .catalog import model_capability, weight_capability, weight_install_requirement
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


@dataclass(frozen=True)
class ProviderFeatures:
    """Reusable feature representation returned by one Provider worker job."""

    features: np.ndarray
    channel_names: np.ndarray
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
    operations = capabilities.get("operations", [])
    if "compute_features" not in operations:
        issues.append("provider does not expose reusable feature computation")
    if "few_shot_analyze_precomputed_features" not in operations:
        issues.append("provider does not accept reusable precomputed features")
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
    elif len(model.get("feature_channels", [])) != config.model.input_channels:
        issues.append("configured feature-channel contract does not match the provider")
    elif not str(model.get("feature_pipeline", "")).strip():
        issues.append("provider model does not declare a feature pipeline")
    elif not str(model.get("fine_tuning_strategy", "")).strip():
        issues.append("provider model does not declare a fine-tuning strategy")
    elif (
        config.fine_tuning.minimum_shots_per_class
        < int(model.get("minimum_shots_per_class", 1))
        or config.fine_tuning.maximum_shots_per_class
        > int(
            model.get(
                "maximum_shots_per_class",
                config.fine_tuning.maximum_shots_per_class,
            )
        )
    ):
        issues.append("configured support-point bounds exceed the provider contract")
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


def registered_weight_install_command(
    config: HarnessConfig, weight: dict[str, Any]
) -> str:
    """Return the explicit installation command for a registered weight package."""
    requirement = weight_install_requirement(weight)
    return f'"{config.model.python_executable}" -m pip install "{requirement}"'


def install_registered_weight(
    config: HarnessConfig,
    *,
    model_identifier: str,
    weight_identifier: str,
) -> dict[str, Any]:
    """Install one user-selected registered weight and refresh its Provider status."""
    capabilities = provider_capabilities(config)
    model = model_capability(capabilities, model_identifier)
    weight = weight_capability(model, weight_identifier)
    requirement = weight_install_requirement(weight)
    if weight.get("status") != "installed":
        completed = subprocess.run(
            [
                config.model.python_executable,
                "-m",
                "pip",
                "install",
                requirement,
            ],
            cwd=provider_working_directory(config),
            env=provider_environment(config),
            text=True,
            capture_output=True,
            timeout=config.provider.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Registered weight installation failed: {detail}")
    refreshed = provider_capabilities(config)
    refreshed_weight = weight_capability(
        model_capability(refreshed, model_identifier), weight_identifier
    )
    if refreshed_weight.get("status") != "installed":
        raise RuntimeError(
            "The weight package installed, but the Provider still reports it as unavailable."
        )
    return {
        "status": "installed",
        "model_identifier": model_identifier,
        "weight": refreshed_weight,
        "requirement": requirement,
        "capabilities": refreshed,
    }


def _invoke_provider_json(
    config: HarnessConfig,
    action: str,
    path: Path | None = None,
    *,
    timeout_seconds: int,
    progress_path: Path | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    if progress_path is not None and progress_callback is not None:
        process = subprocess.Popen(
            provider_command(config, action, path),
            cwd=provider_working_directory(config),
            env=provider_environment(config),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = monotonic() + timeout_seconds
        last_progress: tuple[str, int, int] | None = None
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                process.kill()
                process.communicate()
                raise RuntimeError("symmetry-learn provider timed out.")
            try:
                stdout, stderr = process.communicate(timeout=min(0.2, remaining))
                break
            except subprocess.TimeoutExpired:
                if not progress_path.is_file():
                    continue
                try:
                    payload = json.loads(progress_path.read_text(encoding="utf-8"))
                    progress = (
                        str(payload["phase"]),
                        int(payload["current"]),
                        int(payload["total"]),
                    )
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                if progress != last_progress:
                    progress_callback(*progress)
                    last_progress = progress
        if progress_path.is_file():
            try:
                payload = json.loads(progress_path.read_text(encoding="utf-8"))
                progress = (
                    str(payload["phase"]),
                    int(payload["current"]),
                    int(payload["total"]),
                )
                if progress != last_progress:
                    progress_callback(*progress)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
        if process.returncode != 0:
            detail = stderr.strip() or stdout.strip()
            raise RuntimeError(f"symmetry-learn provider failed: {detail}")
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"symmetry-learn returned invalid JSON: {error}") from error

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


def provider_capabilities(config: HarnessConfig) -> dict[str, Any]:
    """Read the current versioned Provider model and weight catalog."""
    return _invoke_provider_json(config, "--capabilities", timeout_seconds=120)


def _method_payload(config: HarnessConfig) -> dict[str, Any]:
    checkpoint = config.model.checkpoint_path
    payload: dict[str, Any] = {"identifier": config.model.identifier}
    if checkpoint is not None:
        payload.update(
            {
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": config.model.checkpoint_sha256,
            }
        )
    elif config.model.weight_identifier is not None:
        payload["weight_identifier"] = config.model.weight_identifier
    return payload


def probe_model(config: HarnessConfig) -> dict[str, Any]:
    """Ask the Provider to strictly load the selected model weight and device."""
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


def run_provider_features(
    config: HarnessConfig,
    image: np.ndarray,
    *,
    options: dict[str, Any],
) -> ProviderFeatures:
    """Compute a reusable feature representation in the configured Provider."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Provider input must be a finite two-dimensional image.")
    provider_options = dict(options)
    provider_options.pop("input_normalization", None)
    if isinstance(provider_options.get("rotation_folds"), tuple):
        provider_options["rotation_folds"] = list(provider_options["rotation_folds"])

    with tempfile.TemporaryDirectory(prefix="symmetry-harness-features-") as temporary:
        directory = Path(temporary)
        input_path = directory / "input.npy"
        output_path = directory / "features.npz"
        record_path = directory / "feature_record.json"
        job_path = directory / "job.json"
        np.save(input_path, values)
        payload = {
            "schema_version": WORKER_SCHEMA_VERSION,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "record_path": str(record_path),
            "options": provider_options,
            "method": {"identifier": config.model.identifier},
        }
        job_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _invoke_provider_json(
            config,
            "--features",
            job_path,
            timeout_seconds=config.provider.timeout_seconds,
        )
        if not output_path.is_file() or not record_path.is_file():
            raise RuntimeError("Provider completed without reusable feature artifacts.")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("schema_version") != WORKER_SCHEMA_VERSION:
            raise RuntimeError("Provider feature worker schema mismatch.")
        if record.get("provider_contract_version") != PROVIDER_CONTRACT_VERSION:
            raise RuntimeError("Provider feature contract version mismatch.")
        if record.get("provider") != config.provider.name:
            raise RuntimeError("Provider feature identity mismatch.")
        if record.get("identifier") != config.model.identifier:
            raise RuntimeError("Provider feature model identity mismatch.")
        with np.load(output_path, allow_pickle=False) as archive:
            features = np.asarray(archive["features"], dtype=np.float32).copy()
            channel_names = np.asarray(archive["channel_names"]).copy()
    for temporary_key in ("output_path", "record_path"):
        record.pop(temporary_key, None)
    record["transport"] = {
        "kind": "subprocess_json_npy_npz",
        "python_executable": config.model.python_executable,
        "module": config.provider.module,
    }
    return ProviderFeatures(
        features=features,
        channel_names=channel_names,
        record=record,
    )


def run_provider_analysis(
    config: HarnessConfig,
    image: np.ndarray,
    *,
    coordinates_xy: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    options: dict[str, Any],
    features_path: str | Path | None = None,
    features_record_path: str | Path | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
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
        progress_path = directory / "progress.json"
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
        if progress_callback is not None:
            payload["progress_path"] = str(progress_path)
        if features_path is not None or features_record_path is not None:
            if features_path is None or features_record_path is None:
                raise ValueError(
                    "Reusable features require both an array and a record path."
                )
            payload["features_path"] = str(Path(features_path).resolve())
            payload["features_record_path"] = str(
                Path(features_record_path).resolve()
            )
        job_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _invoke_provider_json(
            config,
            "--job",
            job_path,
            timeout_seconds=config.provider.timeout_seconds,
            progress_path=progress_path if progress_callback is not None else None,
            progress_callback=progress_callback,
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
    """Check Provider, selected weight, device, and optional UI readiness."""
    total_started = perf_counter()
    timings: dict[str, float] = {}
    issues: list[str] = []
    recommendations: list[str] = []
    executable = Path(config.model.python_executable)
    environment: dict[str, Any] = {
        "python_executable": config.model.python_executable,
        "status": "blocked",
    }
    capabilities: dict[str, Any] | None = None
    provider_started = perf_counter()
    if not executable.is_file():
        issues.append("The configured Provider Python executable does not exist.")
        recommendations.append(
            f"Configure an existing Python executable: {config.model.python_executable}"
        )
    else:
        try:
            capabilities = provider_capabilities(config)
            compatibility_issues = validate_provider_capabilities(config, capabilities)
            if compatibility_issues:
                issues.extend(compatibility_issues)
                if any("weight" in issue.lower() for issue in compatibility_issues):
                    recommendations.append(
                        "Select a registered weight reported by `symmetry models`."
                    )
                else:
                    recommendations.append(
                        "Install a compatible symmetry Provider: "
                        + _install_command(config)
                    )
            else:
                environment["status"] = "ready"
                environment["capabilities"] = capabilities
        except Exception as error:
            issues.append(str(error))
            recommendations.append(
                "Install the symmetry Provider in this runtime: "
                + _install_command(config)
            )
    timings["provider_capabilities"] = round(perf_counter() - provider_started, 6)

    checkpoint_started = perf_counter()
    checkpoint = config.model.checkpoint_path
    checkpoint_record: dict[str, Any] = {
        "kind": "custom" if checkpoint is not None else "registered",
        "path": None if checkpoint is None else str(checkpoint),
        "configured_sha256": config.model.checkpoint_sha256,
        "weight_identifier": config.model.weight_identifier,
        "exists": bool(checkpoint is not None and checkpoint.is_file()),
    }
    selected_weight_ready = True
    if checkpoint is not None:
        if not checkpoint.is_file():
            selected_weight_ready = False
            message = f"The configured checkpoint does not exist: {checkpoint}"
            if require_ui:
                recommendations.append(
                    "Choose an installed registered weight or another custom checkpoint "
                    "in the local interface."
                )
            else:
                issues.append(message)
        else:
            actual = _file_sha256(checkpoint)
            checkpoint_record["actual_sha256"] = actual
            if config.model.checkpoint_sha256 is None:
                selected_weight_ready = False
                message = "The configured checkpoint has no recorded SHA-256."
                if require_ui:
                    recommendations.append(
                        "Choose a weight in the local interface before running."
                    )
                else:
                    issues.append(message)
            elif actual != config.model.checkpoint_sha256:
                selected_weight_ready = False
                message = "The configured checkpoint SHA-256 does not match the file."
                if require_ui:
                    recommendations.append(
                        "Choose a weight in the local interface before running."
                    )
                else:
                    issues.append(message)
    elif capabilities is not None:
        try:
            selected_model = model_capability(capabilities, config.model.identifier)
            selected_weight = weight_capability(
                selected_model, config.model.weight_identifier
            )
            checkpoint_record.update(
                {
                    "weight_identifier": selected_weight.get("identifier"),
                    "distribution": selected_weight.get("distribution"),
                    "version": selected_weight.get("version"),
                    "status": selected_weight.get("status"),
                    "bundled": bool(selected_weight.get("bundled", False)),
                    "path": selected_weight.get("installed_path"),
                    "expected_sha256": selected_weight.get("sha256"),
                    "exists": selected_weight.get("status") == "installed",
                }
            )
            if selected_weight.get("status") != "installed":
                selected_weight_ready = False
                message = (
                    f"Registered weight {selected_weight.get('identifier')!r} "
                    "is not installed."
                )
                if not require_ui:
                    issues.append(message)
                recommendations.append(
                    "Install the selected weight explicitly: "
                    + registered_weight_install_command(config, selected_weight)
                )
        except (TypeError, ValueError) as error:
            selected_weight_ready = False
            if not require_ui and str(error) not in issues:
                issues.append(str(error))
            recommendations.append(
                "Choose a registered weight reported by `symmetry models`."
            )
    timings["checkpoint_sha256"] = round(perf_counter() - checkpoint_started, 6)

    probe: dict[str, Any] | None = None
    probe_started = perf_counter()
    if not issues and selected_weight_ready:
        try:
            probe = probe_model(config)
            resolved = dict(probe.get("details", {}).get("checkpoint", {}))
            if resolved:
                checkpoint_record.update(
                    {
                        "path": resolved.get("path"),
                        "actual_sha256": resolved.get("sha256"),
                        "weight_identifier": resolved.get("weight_identifier"),
                        "source": resolved.get("source"),
                        "bundled": bool(resolved.get("bundled", False)),
                        "strict_load": bool(resolved.get("strict_load", False)),
                        "exists": True,
                    }
                )
        except Exception as error:
            issues.append(f"Provider model probe failed: {error}")
    timings["model_probe"] = round(perf_counter() - probe_started, 6)
    model_status = (
        "ready"
        if not issues and selected_weight_ready
        else "selection_required"
        if not issues and require_ui
        else "blocked"
    )

    if require_ui:
        ui_started = perf_counter()
        try:
            gradio_available = find_spec("gradio") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            gradio_available = False
        if not gradio_available:
            issues.append("Gradio is required for the local annotation interface.")
            recommendations.append(
                f'"{Path(sys.executable).resolve()}" -m pip install "symmetry-harness[ui]"'
            )
        timings["ui_dependency"] = round(perf_counter() - ui_started, 6)
    timings["total"] = round(perf_counter() - total_started, 6)
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
            "weight_identifier": checkpoint_record.get("weight_identifier"),
            "status": model_status,
            "probe": probe,
        },
        "checkpoint": checkpoint_record,
        "issues": issues,
        "recommendations": list(dict.fromkeys(recommendations)),
        "timings_seconds": timings,
    }
