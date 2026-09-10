"""Application shell and launch behavior for the local symmetry interface.

The fine-tuning workspace lives in :mod:`symmetry_harness.ui_fine_tune` and the
saved-model prediction workspace lives in :mod:`symmetry_harness.ui_predict`.
Both are peers inside one Gradio application.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
from typing import Any, Callable

import numpy as np

from .config import HarnessConfig

# Re-exported so existing imports and tests keep working unchanged.
from .ui_fine_tune import (  # noqa: F401
    _add_point,
    _annotation_display_shape,
    _configure_classes,
    _display_to_source_point,
    _feature_preview,
    _feature_request,
    _loaded_image_outputs,
    feature_gallery,
    prepare_feature_exports,
    render_annotations,
)
from .ui_shared import (  # noqa: F401
    ANNOTATION_DISPLAY_MAX_EDGE,
    APP_CSS,
    CUSTOM_CHECKPOINT_SOURCE,
    INVALID_SUPPORT_POINT_MESSAGE,
    REGISTERED_WEIGHT_SOURCE,
    parse_class_names,
    progress_bar_html,
)


FINE_TUNE_MODE = "fine-tune"
PREDICTION_MODE = "predict"
LAUNCH_MODES = (FINE_TUNE_MODE, PREDICTION_MODE)
FINE_TUNE_TAB_ID = "fine-tune"
PREDICTION_TAB_ID = "predict"


def _resolve_server_port(server_name: str, server_port: int | None) -> int | None:
    if server_port is None:
        return None
    if not 0 <= server_port <= 65535:
        raise ValueError("server_port must be between 0 and 65535.")
    if server_port:
        return server_port
    host = "127.0.0.1" if server_name == "localhost" else server_name
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def _protect_localhost_from_proxies() -> None:
    """Ensure Gradio startup checks connect directly to the local-only server."""
    required = ("127.0.0.1", "localhost")
    existing = []
    for name in ("NO_PROXY", "no_proxy"):
        existing.extend(
            item.strip()
            for item in os.environ.get(name, "").split(",")
            if item.strip()
        )
    unique = []
    lowered: set[str] = set()
    for item in (*existing, *required):
        if item.lower() not in lowered:
            unique.append(item)
            lowered.add(item.lower())
    value = ",".join(unique)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def build_app(
    config: HarnessConfig,
    *,
    prepared_input: tuple[np.ndarray, dict[str, Any]] | None = None,
    capabilities: dict[str, Any] | None = None,
    mode: str = FINE_TUNE_MODE,
):
    """Construct the local interface with two independent top-level workspaces."""
    import gradio as gr

    from .ui_fine_tune import build_fine_tune_workspace
    from .ui_predict import build_prediction_workspace

    if mode not in LAUNCH_MODES:
        raise ValueError(f"Unknown launch mode {mode!r}; expected one of {LAUNCH_MODES}.")

    catalog = capabilities
    with gr.Blocks(title="Symmetry Harness") as app:
        gr.Markdown(
            "# Symmetry Harness\n"
            "Fine-tune a local symmetry model on representative points, or apply "
            "a saved model to new images."
        )
        # Gradio selects tabs by the child TabItem id, not by index. Passing a
        # number silently falls back to the first tab.
        selected_tab = FINE_TUNE_TAB_ID if mode == FINE_TUNE_MODE else PREDICTION_TAB_ID
        with gr.Tabs(selected=selected_tab) as tabs:
            with gr.TabItem("Fine-tune a model", id=FINE_TUNE_TAB_ID):
                build_fine_tune_workspace(
                    config,
                    prepared_input=prepared_input,
                    capabilities=catalog,
                )
            with gr.TabItem("Predict with a saved model", id=PREDICTION_TAB_ID):
                build_prediction_workspace(config, capabilities=catalog)
    return app.queue(default_concurrency_limit=1)


def launch_ui(
    config: HarnessConfig,
    *,
    server_name: str,
    server_port: int | None,
    inbrowser: bool,
    prepared_input: tuple[np.ndarray, dict[str, Any]] | None = None,
    capabilities: dict[str, Any] | None = None,
    mode: str = FINE_TUNE_MODE,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    """Launch a local-only interface without public sharing."""
    if server_name not in {"127.0.0.1", "localhost"}:
        raise ValueError("The v1 interface only binds to localhost.")
    _protect_localhost_from_proxies()
    resolved_port = _resolve_server_port(server_name, server_port)
    app = build_app(
        config,
        prepared_input=prepared_input,
        capabilities=capabilities,
        mode=mode,
    )
    _, local_url, _ = app.launch(
        server_name=server_name,
        server_port=resolved_port,
        inbrowser=inbrowser,
        share=False,
        show_error=True,
        prevent_thread_lock=True,
        quiet=on_ready is not None,
        css=APP_CSS,
    )
    if on_ready is not None:
        on_ready(local_url)
    # Keep the local server alive after the ready payload is emitted.
    app.block_thread()
