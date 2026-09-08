"""Filesystem locations for DINOv3 registry state."""
from __future__ import annotations

import os
from pathlib import Path


def default_dinov3_state_root() -> Path:
    configured = os.environ.get("NERI_DINOV3_STATE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return (Path(local_app_data) / "Neri").resolve()
        return (Path.home() / "AppData" / "Local" / "Neri").resolve()
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state_home).expanduser() if xdg_state_home else Path.home() / ".local" / "state"
    return (base / "neri").resolve()
