"""Stable JSON CLI and local UI launcher for symmetry-harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from . import __version__
from .config import load_harness_config
from .image_io import inspect_input
from .initialization import initialize_config
from .provider import default_provider_install_command, doctor, probe_model


DEFAULT_CONFIG_NAME = "symmetry-harness.json"


def _config_path(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    configured = os.environ.get("SYMMETRY_HARNESS_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / DEFAULT_CONFIG_NAME).resolve()


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Configuration path. Defaults to SYMMETRY_HARNESS_CONFIG or "
            f"./{DEFAULT_CONFIG_NAME}."
        ),
    )


def _add_run_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symmetry-patch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")


def _overrides(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        name: getattr(arguments, name)
        for name in (
            "symmetry_patch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "seed",
            "stride",
            "batch_size",
            "device",
        )
        if getattr(arguments, name) is not None
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, prog="symmetry")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser(
        "init", help="Create a configuration from an explicit trusted checkpoint."
    )
    initialize.add_argument("--output", type=Path, default=Path(DEFAULT_CONFIG_NAME))
    initialize.add_argument("--checkpoint", type=Path)
    initialize.add_argument("--provider-python", default=sys.executable)
    initialize.add_argument("--provider-source-root", type=Path)
    initialize.add_argument("--force", action="store_true")

    doctor_parser = commands.add_parser(
        "doctor", help="Check dependencies, device, and checkpoint identity."
    )
    _add_config_argument(doctor_parser)
    doctor_parser.add_argument("--require-ui", action="store_true")

    models = commands.add_parser(
        "models", help="Report the configured model and scientific contract."
    )
    _add_config_argument(models)
    models.add_argument("--probe", action="store_true")

    inspect = commands.add_parser(
        "inspect", help="Inspect and normalize one single-channel input."
    )
    _add_config_argument(inspect)
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--normalization")

    run = commands.add_parser(
        "run", help="Run a saved annotation session and persist all artifacts."
    )
    _add_config_argument(run)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--annotations", type=Path, required=True)
    run.add_argument("--output-root", type=Path)
    _add_run_overrides(run)

    reproduce = commands.add_parser(
        "reproduce", help="Repeat a completed run from its versioned record."
    )
    _add_config_argument(reproduce)
    reproduce.add_argument("--record", type=Path, required=True)
    reproduce.add_argument("--output-root", type=Path)

    ui = commands.add_parser("ui", help="Launch the local point-annotation interface.")
    _add_config_argument(ui)
    ui.add_argument("--server-name", default="127.0.0.1")
    ui.add_argument("--server-port", type=int)
    ui.add_argument(
        "--inbrowser", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _missing_config_report(path: Path) -> dict[str, Any]:
    return {
        "status": "blocked",
        "config_path": str(path),
        "issues": ["The symmetry-harness configuration does not exist."],
        "recommendations": [
            f'symmetry init --output "{path}" --checkpoint "<trusted-checkpoint-path>"',
            "If symmetry-learn is missing, install its Provider: "
            + default_provider_install_command("python"),
        ],
    }


def _model_report(config, *, probe: bool) -> dict[str, Any]:
    readiness = doctor(config)
    capabilities = readiness.get("environment", {}).get("capabilities", {})
    capability = next(
        (
            item
            for item in capabilities.get("models", [])
            if item.get("identifier") == config.model.identifier
        ),
        {},
    )
    report: dict[str, Any] = {
        "status": readiness["status"],
        "model": {
            "identifier": config.model.identifier,
            "input_channels": config.model.input_channels,
            "pretrained_classes": config.model.pretrained_classes,
            "classifier_patch_size": config.model.classifier_patch_size,
            "checkpoint_path": (
                None
                if config.model.checkpoint_path is None
                else str(config.model.checkpoint_path)
            ),
            "checkpoint_sha256": config.model.checkpoint_sha256,
            "device": config.model.device,
        },
        "feature_channels": list(capability.get("feature_channels", [])),
        "fine_tuning": {
            "strategy": capability.get("fine_tuning_strategy"),
            "minimum_shots_per_class": config.fine_tuning.minimum_shots_per_class,
            "recommended_shots_per_class": config.fine_tuning.recommended_shots_per_class,
            "support_metrics_are_generalization_metrics": False,
        },
        "readiness": readiness,
    }
    if probe:
        if readiness["status"] != "ready":
            raise RuntimeError("The runtime must be ready before probing the model.")
        report["checkpoint_probe"] = readiness["model"]["probe"] or probe_model(config)
    return report


def _execute(arguments: argparse.Namespace) -> dict[str, Any] | None:
    if arguments.command == "init":
        return initialize_config(
            arguments.output,
            checkpoint=arguments.checkpoint,
            provider_python=arguments.provider_python,
            provider_source_root=arguments.provider_source_root,
            force=bool(arguments.force),
        )
    config_path = _config_path(arguments.config)
    if arguments.command == "doctor" and not config_path.is_file():
        return _missing_config_report(config_path)
    config = load_harness_config(config_path)
    if arguments.command == "doctor":
        return doctor(config, require_ui=bool(arguments.require_ui))
    if arguments.command == "models":
        return _model_report(config, probe=bool(arguments.probe))
    if arguments.command == "inspect":
        normalization = arguments.normalization or config.features.input_normalization
        _, record = inspect_input(arguments.input, normalization)
        return {"status": "ok", "input": record}
    if arguments.command == "run":
        from .workflow import run_analysis

        return run_analysis(
            config,
            image_path=arguments.input,
            annotation_session=arguments.annotations,
            overrides=_overrides(arguments),
            output_root=arguments.output_root,
        )
    if arguments.command == "reproduce":
        from .workflow import reproduce_analysis

        return reproduce_analysis(
            config, arguments.record, output_root=arguments.output_root
        )
    from .ui import launch_ui

    readiness = doctor(config, require_ui=True)
    if readiness["status"] != "ready":
        raise RuntimeError(
            "The local interface is blocked: " + "; ".join(readiness["issues"])
        )
    launch_ui(
        config,
        server_name=arguments.server_name,
        server_port=arguments.server_port,
        inbrowser=bool(arguments.inbrowser),
    )
    return None


def main() -> None:
    arguments = _parser().parse_args()
    try:
        result = _execute(arguments)
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
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
