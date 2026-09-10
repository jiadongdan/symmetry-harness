"""Stable JSON CLI and local UI launcher for symmetry-harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from . import __version__
from .catalog import model_catalog_report
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
        "init", help="Discover models and create a portable Harness configuration."
    )
    initialize.add_argument("--output", type=Path, default=Path(DEFAULT_CONFIG_NAME))
    initialize.add_argument("--model")
    initialize.add_argument("--weight")
    initialize.add_argument("--checkpoint", type=Path)
    initialize.add_argument("--provider-python", default=sys.executable)
    initialize.add_argument("--provider-source-root", type=Path)
    initialize.add_argument("--force", action="store_true")

    doctor_parser = commands.add_parser(
        "doctor", help="Check dependencies, device, and selected model weight."
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

    launch = commands.add_parser(
        "launch",
        help="Preflight the runtime and launch the local annotation UI.",
    )
    _add_config_argument(launch)
    launch.add_argument(
        "--input", type=Path, help="Optional image to preload before the UI opens."
    )
    launch.add_argument("--server-name", default="127.0.0.1")
    launch.add_argument(
        "--server-port",
        type=int,
        default=0,
        help="Local port. Use 0 (the default) to choose an available port.",
    )
    launch.add_argument(
        "--inbrowser", action=argparse.BooleanOptionalAction, default=True
    )
    launch.add_argument(
        "--mode",
        choices=["fine-tune", "predict"],
        default="fine-tune",
        help="Workspace opened when the interface starts.",
    )

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

    predict = commands.add_parser(
        "predict",
        help="Predict one or more images with a saved fine-tuned model package.",
    )
    _add_config_argument(predict)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Input image. Repeat the option to predict several images.",
    )
    predict.add_argument("--output-root", type=Path)
    predict.add_argument("--device")
    predict.add_argument("--stride", type=int)
    predict.add_argument("--batch-size", type=int)

    ui = commands.add_parser("ui", help="Launch the local point-annotation interface.")
    _add_config_argument(ui)
    ui.add_argument("--server-name", default="127.0.0.1")
    ui.add_argument("--server-port", type=int)
    ui.add_argument(
        "--inbrowser", action=argparse.BooleanOptionalAction, default=True
    )
    ui.add_argument(
        "--mode",
        choices=["fine-tune", "predict"],
        default="fine-tune",
        help="Workspace opened when the interface starts.",
    )
    return parser


def _missing_config_report(path: Path) -> dict[str, Any]:
    return {
        "status": "blocked",
        "config_path": str(path),
        "issues": ["The symmetry-harness configuration does not exist."],
        "recommendations": [
            f'symmetry init --output "{path}"',
            "If symmetry-learn is missing, install its Provider: "
            + default_provider_install_command("python"),
        ],
    }


def _model_report(
    config, *, probe: bool, readiness: dict[str, Any] | None = None
) -> dict[str, Any]:
    if readiness is None:
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
        "catalog": model_catalog_report(
            capabilities,
            selected_model=config.model.identifier,
            selected_weight=readiness.get("model", {}).get("weight_identifier")
            or config.model.weight_identifier,
            uses_custom_checkpoint=config.model.checkpoint_path is not None,
        ),
        "model": {
            "identifier": config.model.identifier,
            "weight_identifier": readiness.get("model", {}).get(
                "weight_identifier"
            )
            or config.model.weight_identifier,
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
            "maximum_shots_per_class": config.fine_tuning.maximum_shots_per_class,
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
            model_identifier=arguments.model,
            weight_identifier=arguments.weight,
            force=bool(arguments.force),
        )
    launch_started = perf_counter() if arguments.command == "launch" else None
    config_path = _config_path(arguments.config)
    if arguments.command in {"doctor", "launch"} and not config_path.is_file():
        report = _missing_config_report(config_path)
        if arguments.command == "launch":
            report["phase"] = "config"
            report["timings_seconds"] = {
                "total": round(perf_counter() - launch_started, 6)
            }
        return report
    config_started = perf_counter()
    config = load_harness_config(config_path)
    config_seconds = round(perf_counter() - config_started, 6)
    if arguments.command == "doctor":
        return doctor(config, require_ui=bool(arguments.require_ui))
    if arguments.command == "models":
        return _model_report(config, probe=bool(arguments.probe))
    if arguments.command == "inspect":
        normalization = arguments.normalization or config.features.input_normalization
        _, record = inspect_input(arguments.input, normalization)
        return {"status": "ok", "input": record}
    if arguments.command == "launch":
        timings = {"config_load": config_seconds}
        doctor_started = perf_counter()
        readiness = doctor(config, require_ui=True)
        timings["doctor"] = round(perf_counter() - doctor_started, 6)
        if readiness["status"] != "ready":
            timings["total"] = round(perf_counter() - launch_started, 6)
            return {
                "status": "blocked",
                "phase": "doctor",
                "config_path": str(config_path),
                "readiness": readiness,
                "timings_seconds": timings,
            }

        model_started = perf_counter()
        model_report = _model_report(config, probe=False, readiness=readiness)
        capabilities = readiness["environment"]["capabilities"]
        timings["model_contract"] = round(perf_counter() - model_started, 6)

        prepared_input = None
        input_record = None
        if arguments.input is not None:
            inspect_started = perf_counter()
            image, input_record = inspect_input(
                arguments.input, config.features.input_normalization
            )
            prepared_input = (image, input_record)
            timings["input_inspection"] = round(
                perf_counter() - inspect_started, 6
            )

        from .ui import launch_ui

        ui_started = perf_counter()

        def emit_ready(local_url: str) -> None:
            timings["ui_startup"] = round(perf_counter() - ui_started, 6)
            timings["total"] = round(perf_counter() - launch_started, 6)
            payload = {
                "status": "ready",
                "url": local_url,
                "mode": str(arguments.mode),
                "config_path": str(config_path),
                "output_root": str(config.output_root),
                "input": input_record,
                "input_status": (
                    "preloaded"
                    if input_record is not None
                    else "awaiting_user_selection"
                ),
                "model": model_report["model"],
                "feature_channels": model_report["feature_channels"],
                "fine_tuning": model_report["fine_tuning"],
                "readiness_timings_seconds": readiness.get(
                    "timings_seconds", {}
                ),
                "timings_seconds": timings,
            }
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)

        launch_ui(
            config,
            server_name=arguments.server_name,
            server_port=arguments.server_port,
            inbrowser=bool(arguments.inbrowser),
            prepared_input=prepared_input,
            capabilities=capabilities,
            mode=str(arguments.mode),
            on_ready=emit_ready,
        )
        return None
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
    if arguments.command == "predict":
        from .prediction_workflow import run_saved_model_prediction_batch

        if not arguments.input:
            raise ValueError("Provide at least one --input image.")
        return run_saved_model_prediction_batch(
            config,
            model_package=arguments.model,
            image_paths=list(arguments.input),
            output_root=arguments.output_root,
            device=arguments.device,
            stride=arguments.stride,
            batch_size=arguments.batch_size,
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
        capabilities=readiness["environment"]["capabilities"],
        mode=str(getattr(arguments, "mode", "fine-tune")),
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
        if arguments.command == "launch" and result.get("status") == "blocked":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
