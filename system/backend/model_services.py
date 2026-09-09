"""Backend-facing model catalog helpers backed by the canonical Model layout."""
from __future__ import annotations

import json
from pathlib import Path

from system.dinov3.checkpoint import load_checkpoint
from system.model_sync.catalog import discover_models
from system.model_sync.layout import get_model_layout

from .models import ModelClassInfo, ModelInfo


def model_directory() -> Path:
    return get_model_layout().root / "detect"


def classification_model_directory() -> Path:
    return get_model_layout().root / "cls"


def _to_model_info(item) -> ModelInfo:
    return ModelInfo(
        name=item.name,
        path=item.path,
        size_bytes=item.size_bytes,
        source=item.source,
        kind=item.kind,
        backend=item.backend,
        architecture=item.architecture,
        feature_dim=item.feature_dim,
        requires_detector=item.requires_detector,
        supports_video_fast=item.supports_video_fast,
        supports_video_all=item.supports_video_all,
        checkpoint_path=item.checkpoint_path,
    )


def list_available_models() -> list[ModelInfo]:
    layout = get_model_layout()
    return [_to_model_info(item) for item in discover_models(layout, "detect")]


def list_available_classification_models() -> list[ModelInfo]:
    layout = get_model_layout()
    return [_to_model_info(item) for item in discover_models(layout, "cls")]


def _read_dinov3_manifest(path: Path) -> dict[str, object] | None:
    if not path.name.endswith(".neri.json"):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"DINOv3 模型 manifest 无法读取: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("backend") != "dinov3":
        return None
    return payload


def _dinov3_checkpoint_path(manifest_path: Path, manifest: dict[str, object]) -> Path:
    raw_checkpoint = manifest.get("checkpoint")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
        raise ValueError("DINOv3 模型 manifest 缺少 checkpoint。")
    raw_checkpoint = raw_checkpoint.strip()
    candidate = Path(raw_checkpoint)
    if candidate.is_absolute() or "\\" in raw_checkpoint or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ValueError("DINOv3 模型 checkpoint 路径不安全。")
    checkpoint = (manifest_path.parent / candidate).resolve()
    try:
        checkpoint.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("DINOv3 模型 checkpoint 必须位于 manifest 目录内。") from exc
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint 不存在: {checkpoint}")
    return checkpoint


def _load_dinov3_model_classes(
    manifest_path: Path,
    manifest: dict[str, object],
) -> list[ModelClassInfo]:
    checkpoint = load_checkpoint(_dinov3_checkpoint_path(manifest_path, manifest))
    return [
        ModelClassInfo(id=index, name=name, display_name=name)
        for index, name in enumerate(checkpoint.classes)
    ]


def list_model_classes(model_path: str | None) -> list[ModelClassInfo]:
    """Expose classes for DINOv3 manifests or legacy YOLO model files."""

    # Keep path resolution and YOLO loading in the historical service module,
    # but branch before checking any Ultralytics dependency for DINOv3.
    from . import services as legacy_services

    resolved_model_path = legacy_services._resolve_model_path(model_path)
    if not resolved_model_path.exists() or not resolved_model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {resolved_model_path}")

    manifest = _read_dinov3_manifest(resolved_model_path)
    if manifest is not None:
        return _load_dinov3_model_classes(resolved_model_path, manifest)

    missing = legacy_services.missing_yolo_dependencies()
    if missing:
        raise RuntimeError(f"缺少 YOLO 处理依赖: {', '.join(missing)}")
    return list(legacy_services._cached_model_classes(str(resolved_model_path)))
