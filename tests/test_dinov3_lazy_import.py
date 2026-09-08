import subprocess
import sys
from pathlib import Path


def test_registry_api_import_does_not_require_torch():
    root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("torch intentionally unavailable")
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import system.dinov3.api
print("ok")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
