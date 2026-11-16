"""Exercise bundled Uvicorn startup with deliberately polluted host Python paths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def verify(application_root, *, python_executable=None, isolation_args=None):
    root = Path(application_root).resolve(strict=True)
    sys.path.insert(0, str(root))
    from system.python_runtime import PYTHON_ISOLATION_ARGS, python_module_command

    executable = Path(python_executable) if python_executable else root / "toolkit/python.exe"
    arguments = PYTHON_ISOLATION_ARGS if isolation_args is None else isolation_args
    with tempfile.TemporaryDirectory(prefix="neri-python-isolation-") as temporary:
        directory = Path(temporary)
        user_base = directory / "user-base"
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        user_site = (
            user_base / f"Python{version.replace('.', '')}" / "site-packages"
            if os.name == "nt"
            else user_base / "lib" / f"python{version}" / "site-packages"
        )
        python_path = directory / "python-path"
        for polluted in (user_site, python_path):
            polluted.mkdir(parents=True)
            for module in ("typing_extensions", "uvicorn", "sitecustomize"):
                (polluted / f"{module}.py").write_text(
                    "raise RuntimeError('NERI_HOST_PYTHON_POLLUTION')\n", encoding="utf-8"
                )
        probe = directory / "neri_runtime_probe.py"
        probe.write_text('''import importlib
import json
import os
from pathlib import Path
import site
import sys

assert sys.flags.no_user_site, "user site is enabled"
assert sys.flags.ignore_environment, "Python environment is enabled"
assert not site.ENABLE_USER_SITE
assert os.environ["NERI_PARENT_PID"] == "4242"
prefix = Path(sys.prefix).resolve()
origins = {}
for name in ("typing_extensions", "anyio", "fastapi", "uvicorn"):
    module = importlib.import_module(name)
    origin = Path(module.__file__).resolve()
    assert origin.is_relative_to(prefix), (name, str(origin), str(prefix))
    origins[name] = str(origin)
from anyio._core import _typedattr
print(json.dumps({"passed": True, "origins": origins}))
# Stop after Uvicorn imports the app, before opening a listening socket.
raise SystemExit(0)
''', encoding="utf-8")
        environment = dict(os.environ)
        environment.pop("PYTHONNOUSERSITE", None)
        environment.update(
            APPDATA=str(directory / "appdata"),
            PYTHONUSERBASE=str(user_base),
            PYTHONPATH=str(python_path),
            PYTHONHOME=str(directory / "nonexistent-python-home"),
            NERI_PARENT_PID="4242",
        )
        result = subprocess.run(
            [str(executable), *arguments, "-B", "-m", "uvicorn",
             "neri_runtime_probe:app", "--app-dir", str(directory)],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if result.returncode:
            raise RuntimeError(
                f"Python isolation probe failed ({result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )
        report = json.loads(result.stdout.strip())
        assert report["passed"]
        for module in ("system.backend.maintenance", "system.backend.dinov3_maintenance"):
            worker = subprocess.run(
                python_module_command(executable, root, module, "--help"),
                cwd=root, env=environment, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60,
            )
            if worker.returncode or "--operation" not in worker.stdout:
                raise RuntimeError(f"Maintenance module import failed: {module}\n{worker.stdout}\n{worker.stderr}")
        report["maintenance_modules"] = True
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().application_root), indent=2))
