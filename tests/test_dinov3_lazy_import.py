import subprocess
import sys
from pathlib import Path


def _run_without_torch(code: str):
    root = Path(__file__).resolve().parents[1]
    prelude = r'''
import builtins
import importlib.util
real_import = builtins.__import__
real_find_spec = importlib.util.find_spec
def blocked(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("torch intentionally unavailable")
    return real_import(name, *args, **kwargs)
def no_torch_spec(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        return None
    return real_find_spec(name, *args, **kwargs)
builtins.__import__ = blocked
importlib.util.find_spec = no_torch_spec
'''
    return subprocess.run(
        [sys.executable, "-c", prelude + code],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def test_registry_api_import_does_not_require_torch():
    result = _run_without_torch(
        "\nimport system.dinov3.api\nprint('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_backend_health_and_settings_do_not_require_torch():
    result = _run_without_torch(
        r'''
from fastapi.testclient import TestClient
from system.backend.main import app
with TestClient(app) as client:
    health = client.get("/api/health")
    settings = client.get("/api/settings")
assert health.status_code == 200, health.text
assert settings.status_code == 200, settings.text
assert "torch" in settings.json()["missing_yolo_dependencies"]
print("ok")
'''
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
