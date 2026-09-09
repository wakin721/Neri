"""Lifecycle helpers for the optional DINOv3 runtime component."""
from __future__ import annotations

from pathlib import Path

DINO_COMPONENT_VERSION = 1
DINO_ARCHITECTURE = "DINOv3 ViT-B/16"
DINO_SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"


def dinov3_component_status(*, root: Path | None = None) -> dict[str, object]:
    """Return the optional component's current installation state."""

    return {
        "installed": False,
        "healthy": False,
        "architecture": DINO_ARCHITECTURE,
        "component_version": DINO_COMPONENT_VERSION,
        "source_commit": DINO_SOURCE_COMMIT,
        "message": "DINOv3 未安装。",
    }
