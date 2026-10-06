from __future__ import annotations

import json
import os
from pathlib import Path
import socket

from symmetry_harness import runtime_state


def test_ready_state_requires_process_identity_and_listening_socket(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMMETRY_HARNESS_HOME", str(tmp_path))
    config_path = tmp_path / "symmetry-harness.json"
    config_path.write_text("{}\n", encoding="utf-8")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        published = runtime_state.write_ready_server(
            url=f"http://127.0.0.1:{port}/",
            config_path=config_path,
            mode="fine-tune",
            instance_id="instance-a",
        )

        verified = runtime_state.ready_server(port, config_path=config_path)
        assert verified == published
        waited = runtime_state.wait_for_ready_server(port, timeout_seconds=0)
        assert waited["status"] == "ready"
        assert waited["instance_id"] == "instance-a"

    assert runtime_state.ready_server(port) is None


def test_clear_ready_state_cannot_remove_another_instance(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMMETRY_HARNESS_HOME", str(tmp_path))
    config_path = tmp_path / "symmetry-harness.json"
    config_path.write_text("{}\n", encoding="utf-8")
    port = 47860
    runtime_state.write_ready_server(
        url=f"http://127.0.0.1:{port}/",
        config_path=config_path,
        mode="predict",
        instance_id="current",
    )

    runtime_state.clear_ready_server(port, "older")
    assert runtime_state.server_state_path(port).is_file()
    runtime_state.clear_ready_server(port, "current")
    assert not runtime_state.server_state_path(port).exists()


def test_wait_reports_a_detached_launcher_blocker(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMMETRY_HARNESS_HOME", str(tmp_path / "state"))
    output_path = tmp_path / "launch.json"
    output_path.write_text(
        json.dumps(
            {
                "status": "blocked",
                "phase": "doctor",
                "issues": ["Provider unavailable."],
                "recommendations": ["Repair the Provider."],
            }
        ),
        encoding="utf-8",
    )

    report = runtime_state.wait_for_ready_server(
        47861,
        timeout_seconds=1,
        launch_pid=max(os.getpid() + 1_000_000, 99_999_999),
        launch_output=output_path,
    )
    assert report["status"] == "blocked"
    assert report["phase"] == "doctor"
    assert report["issues"] == ["Provider unavailable."]


def test_wait_timeout_does_not_accept_an_unidentified_listener(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMMETRY_HARNESS_HOME", str(tmp_path))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        report = runtime_state.wait_for_ready_server(port, timeout_seconds=0)
    assert report["status"] == "blocked"
    assert report["phase"] == "wait"
