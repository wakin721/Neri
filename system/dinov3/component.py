"""Lifecycle helpers for the optional DINOv3 runtime component."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from system.model_sync.layout import get_model_layout

DINO_COMPONENT_VERSION = 1
DINO_ARCHITECTURE = "DINOv3 ViT-B/16"
DINO_ARCHITECTURE_ID = "dinov3_vitb16"
DINO_SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
DINO_BACKBONE_FILENAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DINO_BACKBONE_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
DINO_CLASSIFIER_FILENAME = "dinov3_classifier_merged_reviewed_20260908.pt"
DINO_CLASSIFIER_SHA256 = "b2f334da61c9feee51cff51bcded0af16878dcd5d72ea4f2e3e47b7f2adab76a"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 不是 JSON 对象。")
    return value


def _component_health(paths: DinoV3ComponentPaths) -> tuple[bool, str]:
    required_files = (
        paths.source_package / "__init__.py",
        paths.source_package / "hub" / "backbones.py",
        paths.source_package / "models" / "vision_transformer.py",
        paths.backbone,
        paths.classifier,
        paths.model_manifest,
        paths.license,
        paths.install_manifest,
    )
    missing = [path.relative_to(paths.root).as_posix() for path in required_files if not path.is_file()]
    if missing:
        return False, "DINOv3 安装不完整，缺少: " + "、".join(missing[:4])

    try:
        install_manifest = _read_json_object(paths.install_manifest)
        if install_manifest.get("schema_version") != 1:
            return False, "DINOv3 install.json schema 不受支持。"
        if install_manifest.get("component") not in (None, "dinov3"):
            return False, "DINOv3 install.json component 不匹配。"
        if install_manifest.get("architecture") != DINO_ARCHITECTURE_ID:
            return False, "DINOv3 install.json architecture 不匹配。"
        source = install_manifest.get("source")
        if not isinstance(source, dict) or source.get("commit") != DINO_SOURCE_COMMIT:
            return False, "DINOv3 source commit 不匹配。"
        backbone_meta = install_manifest.get("backbone")
        if not isinstance(backbone_meta, dict) or backbone_meta.get("sha256") != DINO_BACKBONE_SHA256:
            return False, "DINOv3 backbone 元数据不匹配。"

        model_manifest = _read_json_object(paths.model_manifest)
        if model_manifest.get("schema_version") != 1 or model_manifest.get("backend") != "dinov3":
            return False, "DINOv3 模型 manifest 不受支持。"
        if model_manifest.get("architecture") != DINO_ARCHITECTURE_ID:
            return False, "DINOv3 模型 architecture 不匹配。"
        if model_manifest.get("feature_dim") != 768:
            return False, "DINOv3 模型 feature_dim 不匹配。"
        if model_manifest.get("checkpoint") != DINO_CLASSIFIER_FILENAME:
            return False, "DINOv3 classifier 文件名不匹配。"
        if model_manifest.get("encoder_sha256") != DINO_BACKBONE_SHA256:
            return False, "DINOv3 模型 backbone 校验值不匹配。"
    except ValueError as exc:
        return False, str(exc)

    if _sha256_file(paths.backbone).lower() != DINO_BACKBONE_SHA256:
        return False, "DINOv3 backbone SHA-256 校验失败。"
    if _sha256_file(paths.classifier).lower() != DINO_CLASSIFIER_SHA256:
        return False, "DINOv3 classifier SHA-256 校验失败。"
    return True, "DINOv3 ViT-B/16 已安装。"


def dinov3_component_status(*, root: Path | None = None) -> dict[str, object]:
    """Return the optional component's current installation state."""

    paths = dinov3_component_paths(root=root)
    installed = paths.root.is_dir()
    healthy = False
    message = "DINOv3 未安装。"
    if installed:
        try:
            healthy, message = _component_health(paths)
        except OSError as exc:
            healthy = False
            message = f"DINOv3 健康检查失败: {exc}"
    return {
        "installed": installed,
        "healthy": healthy,
        "architecture": DINO_ARCHITECTURE,
        "component_version": DINO_COMPONENT_VERSION,
        "source_commit": DINO_SOURCE_COMMIT,
        "message": message,
    }


def remove_dinov3_component(*, root: Path | None = None) -> None:
    """Remove DINOv3-specific runtime assets without touching shared packages."""

    component_root = _component_root(root)
    if not component_root.exists():
        return
    if component_root.is_symlink() or component_root.is_file():
        component_root.unlink()
        return
    shutil.rmtree(component_root)
