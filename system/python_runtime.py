"""Keep the bundled interpreter independent of the user's Python installation."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

# Keep the working/script directory available for maintenance modules.
PYTHON_ISOLATION_ARGS = ("-E", "-s")


def python_module_command(executable, application_root, module, *arguments) -> list[str]:
    """Run app modules even when an embedded ._pth omits the working directory."""
    return [
        str(executable), *PYTHON_ISOLATION_ARGS, "-c",
        "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
        "runpy.run_module(sys.argv.pop(1), run_name='__main__', alter_sys=True)",
        str(Path(application_root).resolve()), module, *arguments,
    ]


def python_environment(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = {
        key: value
        for key, value in (os.environ if parent is None else parent).items()
        if key.upper() not in {"PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE"}
    }
    # Also protect Python descendants launched by pip/build tools.
    environment["PYTHONNOUSERSITE"] = "1"
    return environment
