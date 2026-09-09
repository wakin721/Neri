"""Lifecycle helpers for the optional DINOv3 runtime component."""
from __future__ import annotations

from pathlib import Path

from system.model_sync.layout import get_model_layout

DINO_COMPONENT_VERSION = 1
DINO_ARCHITECTURE = "DINOv3 ViT-B/16"
DINO_SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"


def _component_root(root: Path | None = None) -> Path:
    return Path(root).resolve() if root is not None else get_model_layout().dinov3_root


def dinov3_component_status(*, root: Path | None = None) -> dict[str, object]:
    """Return the optional component's current installation state."""

    component_root = _component_root(root)
    installed = component_root.is_dir()
    return {
        "installed": installed,
        "healthy": False,
        "architecture": DINO_ARCHITECTURE,
        "component_version": DINO_COMPONENT_VERSION,
        "source_commit": DINO_SOURCE_COMMIT,
        "message": (
            "DINOv3 安装不完整，需要修复。"
            if installed
            else "DINOv3 未安装。"
        ),
    }
