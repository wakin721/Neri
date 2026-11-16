from pathlib import Path
import sys

import pytest

from scripts.verify_python_isolation import verify
from system.python_runtime import python_environment


def test_python_environment_preserves_application_and_os_settings():
    parent = {
        "Path": "system-tools",
        "APPDATA": "user-data",
        "NERI_PARENT_PID": "123",
        "HTTPS_PROXY": "http://proxy.test",
        "PIP_INDEX_URL": "https://packages.test/simple",
        "PythonPath": "foreign-packages",
        "PYTHONHOME": "foreign-interpreter",
        "pythonNOUSERsite": "0",
    }
    environment = python_environment(parent)
    assert environment == {
        key: value for key, value in parent.items()
        if not key.upper().startswith("PYTHON")
    } | {"PYTHONNOUSERSITE": "1"}
    assert parent["PythonPath"] == "foreign-packages"


def test_uvicorn_imports_bundled_dependencies_despite_host_pollution():
    root = Path(__file__).resolve().parents[1]
    assert verify(root, python_executable=sys.executable)["passed"]


@pytest.mark.parametrize("incomplete_flags", [(), ("-E",)])
def test_pollution_probe_detects_missing_isolation(incomplete_flags):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(RuntimeError, match="Python isolation probe failed"):
        verify(root, python_executable=sys.executable, isolation_args=incomplete_flags)


def test_backend_restart_and_workers_propagate_clean_environment(monkeypatch):
    from system.backend import maintenance
    from system.backend import dinov3_maintenance

    calls = []
    class Child:
        pid = 42

    def start(command, **kwargs):
        calls.append((command, kwargs))
        return Child()

    monkeypatch.setenv("PYTHONHOME", "foreign-python")
    monkeypatch.setenv("PYTHONPATH", "foreign-packages")
    monkeypatch.setattr(maintenance.subprocess, "Popen", start)
    monkeypatch.setattr(maintenance, "read_maintenance_status", lambda: {})
    monkeypatch.setattr(maintenance, "_write_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(maintenance, "toolkit_python", lambda: Path(sys.executable))
    monkeypatch.setattr(maintenance, "resolve_package_source", lambda _: ("official", "official", "https://pypi.org/simple"))
    maintenance._start_backend()
    maintenance._start_maintenance("reinstall_package", [], "test")
    dinov3_maintenance._start("remove_dinov3")
    assert len(calls) == 3
    for command, options in calls:
        assert command[1:3] == ["-E", "-s"]
        environment = options["env"]
        assert "PYTHONHOME" not in environment
        assert "PYTHONPATH" not in environment
        assert environment["PYTHONNOUSERSITE"] == "1"
