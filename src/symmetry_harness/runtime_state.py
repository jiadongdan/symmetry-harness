"""Process-scoped state for fast, cross-agent Harness startup."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from time import monotonic, sleep
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4


SERVER_STATE_SCHEMA_VERSION = "symmetry-harness-server-v1"
DEFAULT_AGENT_PORT = 7860


def state_directory() -> Path:
    """Return the user-local state directory shared by launch helpers."""
    configured = os.environ.get("SYMMETRY_HARNESS_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".symmetry-harness").resolve()


def server_state_path(port: int) -> Path:
    """Return the state file for one explicit local server port."""
    return state_directory() / f"server-{int(port)}.json"


def new_instance_id() -> str:
    """Create an identity that prevents one process clearing another's state."""
    return uuid4().hex


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _url_port(url: str) -> int:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(f"Invalid local server URL: {url}")
    if parsed.port is not None:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def write_ready_server(
    *,
    url: str,
    config_path: Path,
    mode: str,
    instance_id: str,
    pid: int | None = None,
) -> dict[str, Any]:
    """Publish a ready server only after the UI has bound its socket."""
    port = _url_port(url)
    payload: dict[str, Any] = {
        "schema_version": SERVER_STATE_SCHEMA_VERSION,
        "status": "ready",
        "instance_id": instance_id,
        "pid": os.getpid() if pid is None else int(pid),
        "url": url,
        "port": port,
        "mode": mode,
        "config_path": str(config_path.expanduser().resolve()),
    }
    _write_json_atomic(server_state_path(port), payload)
    return payload


def clear_ready_server(port: int, instance_id: str) -> None:
    """Remove state only when it still belongs to this launch instance."""
    path = server_state_path(port)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return
    if payload.get("instance_id") != instance_id:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # On Windows, os.kill(pid, 0) sends CTRL_C_EVENT instead of performing
        # the POSIX existence check. Query a synchronization handle so a
        # readiness probe can never interrupt the server (or its own caller).
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        open_process.restype = ctypes.c_void_p
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        wait_for_single_object = kernel32.WaitForSingleObject
        wait_for_single_object.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        wait_for_single_object.restype = ctypes.c_ulong

        synchronize = 0x00100000
        wait_timeout = 0x00000102
        handle = open_process(synchronize, False, pid)
        if not handle:
            return False
        try:
            return wait_for_single_object(handle, 0) == wait_timeout
        finally:
            close_handle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _socket_is_ready(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = _url_port(url)
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return False
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except (OSError, ValueError):
        return False


def ready_server(
    port: int,
    *,
    config_path: Path | None = None,
) -> dict[str, Any] | None:
    """Return a verified Harness instance, never an unrelated port listener."""
    path = server_state_path(port)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") != SERVER_STATE_SCHEMA_VERSION
        or payload.get("status") != "ready"
        or payload.get("port") != int(port)
    ):
        return None
    if config_path is not None:
        expected = str(config_path.expanduser().resolve())
        if payload.get("config_path") != expected:
            return None
    try:
        pid = int(payload["pid"])
        url = str(payload["url"])
    except (KeyError, TypeError, ValueError):
        return None
    if not _process_is_alive(pid) or not _socket_is_ready(url):
        instance_id = payload.get("instance_id")
        if isinstance(instance_id, str):
            clear_ready_server(port, instance_id)
        return None
    return payload


def _launcher_exit_report(output_path: Path | None) -> dict[str, Any] | None:
    if output_path is None:
        return None
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def wait_for_ready_server(
    port: int = DEFAULT_AGENT_PORT,
    *,
    timeout_seconds: float = 180.0,
    poll_seconds: float = 0.25,
    launch_pid: int | None = None,
    launch_output: Path | None = None,
    launch_error: Path | None = None,
) -> dict[str, Any]:
    """Wait for an identity-checked server or a failed detached launcher."""
    started = monotonic()
    while True:
        payload = ready_server(port)
        if payload is not None and (
            launch_pid is None or int(payload["pid"]) == int(launch_pid)
        ):
            return {
                "status": "ready",
                "url": payload["url"],
                "port": int(port),
                "pid": payload["pid"],
                "instance_id": payload["instance_id"],
                "mode": payload["mode"],
                "config_path": payload["config_path"],
                "wait_seconds": round(monotonic() - started, 3),
            }

        if launch_pid is not None and not _process_is_alive(launch_pid):
            report = _launcher_exit_report(launch_output)
            if report is not None and report.get("status") in {
                "ready",
                "blocked",
                "error",
            }:
                report = dict(report)
                if report.get("status") != "ready":
                    report.setdefault("phase", "launch")
                report["wait_seconds"] = round(monotonic() - started, 3)
                return report
            issues = [
                "The detached Harness launcher exited before publishing a ready server."
            ]
            if launch_error is not None:
                issues.append(f"Launcher diagnostics: {launch_error.expanduser().resolve()}")
            return {
                "status": "blocked",
                "phase": "launch",
                "issues": issues,
                "recommendations": [
                    "Read the launcher diagnostics only when troubleshooting was requested."
                ],
                "wait_seconds": round(monotonic() - started, 3),
            }

        elapsed = monotonic() - started
        if elapsed >= max(0.0, timeout_seconds):
            return {
                "status": "blocked",
                "phase": "wait",
                "issues": [
                    f"No verified Symmetry Harness instance became ready on port {port}."
                ],
                "recommendations": [
                    "Run the installed fast-open script again.",
                    "Inspect launcher diagnostics only if the second attempt is blocked.",
                ],
                "wait_seconds": round(elapsed, 3),
            }
        sleep(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))
