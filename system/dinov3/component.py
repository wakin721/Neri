"""Lifecycle helpers for the optional DINOv3 runtime component."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from system.model_sync.layout import get_model_layout

DINO_COMPONENT_VERSION = 1
DINO_ARCHITECTURE = "DINOv3 ViT-B/16"
DINO_SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
DINO_BACKBONE_FILENAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DINO_CLASSIFIER_FILENAME = "dinov3_classifier_merged_reviewed_20260908.pt"
DINO_MODEL_MANIFEST_FILENAME = "dinov3_classifier_merged_reviewed_20260908.neri.json"


@dataclass(frozen=True)
class DinoV3ComponentPaths:
    root: Path
    source_root: Path
    source_package: Path
    backbone_dir: Path
    backbone: Path
    classifier: Path
    model_manifest: Path
    license: Path
    install_manifest: Path


def _component_root(root: Path | None = None) -> Path:
    return Path(root).resolve() if root is not None else get_model_layout().dinov3_root


def dinov3_component_paths(*, root: Path | None = None) -> DinoV3ComponentPaths:
    """Return canonical installed paths for the optional DINOv3 component."""

    component_root = _component_root(root)
    source_root = component_root / "source"
    backbone_dir = component_root / "dinov3-vitb16"
    return DinoV3ComponentPaths(
        root=component_root,
        source_root=source_root,
        source_package=source_root / "dinov3",
        backbone_dir=backbone_dir,
        backbone=backbone_dir / DINO_BACKBONE_FILENAME,
        classifier=component_root / DINO_CLASSIFIER_FILENAME,
        model_manifest=component_root / DINO_MODEL_MANIFEST_FILENAME,
        license=component_root / "LICENSE_DINOv3.md",
        install_manifest=component_root / "install.json",
    )


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
