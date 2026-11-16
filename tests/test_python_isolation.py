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


def test_backend_restart_and_workers_propagate_clean_environment(monkeypatch, tmp_path):
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
    monkeypatch.setattr(maintenance, "logs_dir", lambda: tmp_path)
    monkeypatch.setattr(maintenance, "_backend_is_ready", lambda: True, raising=False)
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


def test_backend_restart_waits_for_health_and_captures_logs(monkeypatch, tmp_path):
    from system.backend import maintenance
    from types import SimpleNamespace

    checks, launches = [], []
    def ready():
        checks.append(True)
        return len(checks) == 3
    def start(command, **options):
        launches.append((command, options))
        return SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(maintenance, 'logs_dir', lambda: tmp_path)
    monkeypatch.setattr(maintenance.subprocess, 'Popen', start)
    monkeypatch.setattr(maintenance, '_backend_is_ready', ready, raising=False)
    monkeypatch.setattr(maintenance.time, 'sleep', lambda _: None)
    maintenance._start_backend()
    assert len(checks) == 3
    command, options = launches[0]
    assert '-c' in command and 'uvicorn' in command
    assert options['stdin'] == maintenance.subprocess.DEVNULL
    assert options['stderr'] == maintenance.subprocess.STDOUT
    assert options['stdout'].name == str(tmp_path/'backend_restart.log')


def test_backend_restart_detects_child_exit(monkeypatch, tmp_path):
    from system.backend import maintenance
    from types import SimpleNamespace
    monkeypatch.setattr(maintenance, 'logs_dir', lambda: tmp_path)
    monkeypatch.setattr(maintenance, '_backend_is_ready', lambda: False, raising=False)
    monkeypatch.setattr(maintenance.subprocess, 'Popen', lambda *a, **kw:
                        SimpleNamespace(poll=lambda: 61))
    with pytest.raises(RuntimeError, match='61'):
        maintenance._start_backend()


def test_backend_restart_terminates_unready_child(monkeypatch, tmp_path):
    from system.backend import maintenance
    from types import SimpleNamespace
    stopped = []
    child = SimpleNamespace(poll=lambda: None,
        terminate=lambda: stopped.append(True), wait=lambda **kw: None)
    monkeypatch.setattr(maintenance, 'logs_dir', lambda: tmp_path)
    monkeypatch.setattr(maintenance, '_backend_is_ready', lambda: False, raising=False)
    monkeypatch.setattr(maintenance.subprocess, 'Popen', lambda *a, **kw: child)
    with pytest.raises(RuntimeError, match='就绪'):
        maintenance._start_backend(timeout_seconds=0)
    assert stopped == [True]
