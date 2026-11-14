"""Backend model discovery and class lists for YOLO and DINO manifests."""
from __future__ import annotations

import json
from pathlib import Path

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
    from system.model_sync.catalog import discover_models
    return [_to_model_info(item) for item in discover_models(get_model_layout(), "detect")]


def list_available_classification_models() -> list[ModelInfo]:
    from system.model_sync.catalog import discover_models
    return [_to_model_info(item) for item in discover_models(get_model_layout(), "cls")]


def _read_classification_manifest(path: Path) -> dict[str, object] | None:
    if not path.name.endswith(".neri.json"):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"模型 manifest 无法读取: {exc}") from exc
    if not isinstance(payload, dict):
        return None
    if payload.get("backend") != "dinov3":
        return None
    try:
        from system.dinov3.runtime import read_manifest

        return read_manifest(path)
    except ValueError as exc:
        raise ValueError(f"DINOv3联合模型无效: {exc}") from exc
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"DINOv3联合模型无效: {exc}") from exc


def _load_dinov3_model_classes(
    manifest_path: Path,
    manifest: dict[str, object],
) -> list[ModelClassInfo]:
    from system.dinov3.runtime import resolve_manifest_asset
    from system.dinov3.model import DinoV3RidgeWithinModel
    from system.dinov3.encoder import sha256

    checkpoint = resolve_manifest_asset(manifest_path, manifest["checkpoint"])
    if sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("DINOv3 classifier SHA-256 mismatch")
    model = DinoV3RidgeWithinModel.load(checkpoint)
    return [
        ModelClassInfo(id=index, name=str(name), display_name=str(name))
        for index, name in enumerate(model.classes)
    ]


def list_model_classes(model_path: str | None) -> list[ModelClassInfo]:
    """Expose classes for versioned DINO manifests or YOLO model files."""
    from . import services

    resolved_model_path = services._resolve_model_path(model_path)
    if not resolved_model_path.exists() or not resolved_model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {resolved_model_path}")

    manifest = _read_classification_manifest(resolved_model_path)
    if manifest is not None:
        return _load_dinov3_model_classes(resolved_model_path, manifest)

    if resolved_model_path.suffix.lower() in {".tflite", ".litert"}:
        from system.dino.litert_detector import read_model_classes

        translations = {"animal": "动物", "person": "人", "vehicle": "车辆"}
        return [
            ModelClassInfo(
                id=index,
                name=name,
                display_name=translations.get(name, name),
            )
            for index, name in sorted(
                read_model_classes(resolved_model_path).items()
            )
        ]

    missing = services.missing_yolo_dependencies()
    if missing:
        raise RuntimeError(f"缺少 YOLO 处理依赖: {', '.join(missing)}")
    return list(services._cached_model_classes(str(resolved_model_path)))
