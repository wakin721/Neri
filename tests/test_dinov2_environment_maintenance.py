from __future__ import annotations


def test_missing_dinov2_dependencies_detects_only_optional_packages(monkeypatch):
    import system.backend.dinov2_maintenance as maintenance

    available = {"torch": True, "transformers": False, "safetensors": False}
    monkeypatch.setattr(
        maintenance.importlib.util,
        "find_spec",
        lambda name: object() if available.get(name, False) else None,
    )

    assert maintenance._missing_dinov2_dependencies() == [
        "transformers>=4.40",
        "safetensors>=0.4",
    ]


def test_dependency_install_reuses_shared_torch_and_installs_missing_packages(monkeypatch):
    import system.backend.dinov2_maintenance as maintenance

    monkeypatch.setattr(
        maintenance,
        "_missing_dinov2_dependencies",
        lambda: ["transformers>=4.40", "safetensors>=0.4"],
    )
    observed = {}
    monkeypatch.setattr(
        maintenance._maintenance,
        "toolkit_python",
        lambda: __import__("pathlib").Path("/toolkit/python.exe"),
    )
    monkeypatch.setattr(
        maintenance._maintenance,
        "resolve_package_source",
        lambda source: (
            "nju",
            "南京大学源",
            "https://mirror.nju.edu.cn/pypi/web/simple",
        ),
    )
    monkeypatch.setattr(
        maintenance._maintenance,
        "maintenance_pip_cache_dir",
        lambda: __import__("pathlib").Path("/tmp/cache"),
    )
    monkeypatch.setattr(
        maintenance._maintenance,
        "_run_commands",
        lambda commands, **kwargs: observed.update(
            commands=commands,
            kwargs=kwargs,
        ),
    )
    monkeypatch.setattr(
        maintenance._maintenance,
        "_write_status",
        lambda *args, **kwargs: None,
    )

    maintenance._install_dinov2_python_dependencies(
        "nju",
        progress_start=20,
        progress_end=50,
    )

    command = observed["commands"][0]
    assert "transformers>=4.40" in command
    assert "safetensors>=0.4" in command
    assert "torch" not in command
    assert "--upgrade" in command
    assert observed["kwargs"]["progress_start"] == 20
    assert observed["kwargs"]["progress_end"] == 50


def test_start_uses_dinov2_operation_and_module(monkeypatch, tmp_path):
    import system.backend.dinov2_maintenance as maintenance

    python_exe = tmp_path / "python.exe"
    python_exe.write_bytes(b"")
    monkeypatch.setattr(
        maintenance._maintenance,
        "toolkit_python",
        lambda: python_exe,
    )
    monkeypatch.setattr(
        maintenance._maintenance,
        "read_maintenance_status",
        lambda: {"state": "idle"},
    )
    monkeypatch.setattr(
        maintenance._maintenance,
        "resolve_package_source",
        lambda source: ("official", "官方源", "https://pypi.org/simple"),
    )
    monkeypatch.setattr(maintenance._maintenance, "_utc_now", lambda: "now")
    monkeypatch.setattr(
        maintenance._maintenance,
        "maintenance_log_path",
        lambda: tmp_path / "log",
    )
    monkeypatch.setattr(
        maintenance._maintenance,
        "maintenance_status_path",
        lambda: tmp_path / "status",
    )
    monkeypatch.setattr(
        maintenance._maintenance,
        "_write_status",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        maintenance._maintenance,
        "project_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(maintenance._maintenance, "_creation_flags", lambda: 0)
    observed = {}

    class Process:
        pid = 123

    monkeypatch.setattr(
        maintenance.subprocess,
        "Popen",
        lambda command, **kwargs: observed.update(
            command=command,
            kwargs=kwargs,
        )
        or Process(),
    )

    result = maintenance.start_dinov2_install("CPU Only", "official")

    assert result["operation"] == "install_dinov2"
    assert "system.backend.dinov2_maintenance" in observed["command"]
    assert "install_dinov2" in observed["command"]
