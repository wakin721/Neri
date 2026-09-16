"""Backend-facing model helpers with DINOv2 manifest support."""
from __future__ import annotations

import json
from pathlib import Path

from system.dinov2.checkpoint import load_checkpoint

from .models import ModelClassInfo, ModelInfo


def model_directory() -> Path:
    from system.model_sync.layout import get_model_layout
    return get_model_layout().root / "detect"


def classification_model_directory() -> Path:
    from system.model_sync.layout import get_model_layout
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
    from system.model_sync.catalog import discover_models
    from system.model_sync.layout import get_model_layout
    return [_to_model_info(item) for item in discover_models(get_model_layout(), "detect")]


def list_available_classification_models() -> list[ModelInfo]:
    from system.model_sync.catalog import discover_models
    from system.model_sync.layout import get_model_layout
    return [_to_model_info(item) for item in discover_models(get_model_layout(), "cls")]


def _read_dinov2_manifest(path: Path) -> dict[str, object] | None:
    if not path.name.endswith(".neri.json"):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"DINOv2 模型 manifest 无法读取: {exc}") from exc
    if not isinstance(payload, dict):
        return None
    if payload.get("backend") == "dinov3":
        raise ValueError("DINOv3 模型已不再支持；请安装或选择 DINOv2 模型。")
    return payload if payload.get("backend") == "dinov2" else None


def _dinov2_checkpoint_path(manifest_path: Path, manifest: dict[str, object]) -> Path:
    raw = manifest.get("checkpoint")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("DINOv2 模型 manifest 缺少 checkpoint。")
    raw = raw.strip()
    candidate = Path(raw)
    if candidate.is_absolute() or "\\" in raw or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ValueError("DINOv2 模型 checkpoint 路径不安全。")
    checkpoint = (manifest_path.parent / candidate).resolve()
    try:
        checkpoint.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("DINOv2 模型 checkpoint 必须位于 manifest 目录内。") from exc
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv2 checkpoint 不存在: {checkpoint}")
    return checkpoint


def _load_dinov2_model_classes(
    manifest_path: Path,
    manifest: dict[str, object],
) -> list[ModelClassInfo]:
    checkpoint = load_checkpoint(_dinov2_checkpoint_path(manifest_path, manifest))
    return [
        ModelClassInfo(id=index, name=name, display_name=name)
        for index, name in enumerate(checkpoint.classes)
    ]


def list_model_classes(model_path: str | None) -> list[ModelClassInfo]:
    """Expose classes for DINOv2 manifests or legacy YOLO model files."""
    from . import services

    resolved_model_path = services._resolve_model_path(model_path)
    if not resolved_model_path.exists() or not resolved_model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {resolved_model_path}")

    manifest = _read_dinov2_manifest(resolved_model_path)
    if manifest is not None:
        return _load_dinov2_model_classes(resolved_model_path, manifest)

    missing = services.missing_yolo_dependencies()
    if missing:
        raise RuntimeError(f"缺少 YOLO 处理依赖: {', '.join(missing)}")
    return list(services._cached_model_classes(str(resolved_model_path)))
