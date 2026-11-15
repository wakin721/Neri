"""Discovery and compatibility resolution for user and synchronized models."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal, Sequence

from .layout import ModelLayout

from system.dinov3.component import dinov3_component_status, installed_dinov3_manifests


ModelKind = Literal["detect", "cls"]
ModelSource = Literal["bundled", "user", "sync"]
ModelBackend = Literal["yolo", "dinov3", "litert"]
_DETECT_EXTENSIONS = frozenset({".pt", ".tflite", ".litert"})
_CLS_EXTENSIONS = frozenset({".pt", ".onnx", ".engine"})
_MANIFEST_SUFFIX = ".neri.json"
_SOURCE_ORDER: dict[ModelSource, int] = {"bundled": 0, "user": 1, "sync": 2}


@dataclass(frozen=True)
class DiscoveredModel:
    name: str
    path: str
    size_bytes: int | None
    source: ModelSource
    kind: ModelKind
    backend: ModelBackend = "yolo"
    architecture: str | None = None
    feature_dim: int | None = None
    requires_detector: bool = False
    supports_video_fast: bool = True
    supports_video_all: bool = True
    checkpoint_path: str | None = None


def _size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _read_manifest(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_reference(path: Path) -> tuple[dict | None, Path | None]:
    payload = _read_manifest(path)
    if not payload or payload.get("backend") != "dinov3":
        return payload, None
    name = payload.get("checkpoint")
    if not isinstance(name, str) or not name.strip():
        return payload, None
    checkpoint = (path.parent / name).resolve()
    return payload, checkpoint if checkpoint.is_file() else None


def _manifest_model(path: Path, source: ModelSource):
    payload, checkpoint = _manifest_reference(path)
    if not payload or checkpoint is None:
        return None
    backend = payload.get("backend")
    if backend == "dinov3":
        try:
            from system.dinov3.runtime import read_manifest
            valid_payload = read_manifest(path)
        except Exception:
            return None
        dim = valid_payload.get("feature_dim")
        dim = dim if isinstance(dim, int) else 768
        name = str(valid_payload.get("display_name") or path.name)
        model = DiscoveredModel(
            name,
            str(path.resolve()),
            _size(checkpoint),
            source,
            "cls",
            "dinov3",
            str(valid_payload.get("architecture") or "") or None,
            dim,
            bool(valid_payload.get("requires_detector", True)),
            bool(valid_payload.get("supports_video_fast", False)),
            bool(valid_payload.get("supports_video_all", False)),
            str(checkpoint),
        )
        return model, checkpoint
    return None


def _scan(
    directory: Path,
    source: ModelSource,
    kind: ModelKind,
) -> list[DiscoveredModel]:
    extensions = _DETECT_EXTENSIONS if kind == "detect" else _CLS_EXTENSIONS
    if not directory.is_dir():
        return []
    items: list[DiscoveredModel] = []
    referenced: set[Path] = set()
    if kind == "cls":
        for manifest in sorted(
            directory.glob(f"*{_MANIFEST_SUFFIX}"),
            key=lambda path: path.name.casefold(),
        ):
            _payload, checkpoint = _manifest_reference(manifest)
            if checkpoint is not None:
                # DINOv3 model checkpoints must remain hidden so they cannot
                # fall through as YOLO models.
                referenced.add(checkpoint)
            if _payload and _payload.get("backend") == "dinov3":
                weights_name = _payload.get("encoder_weights")
                if isinstance(weights_name, str) and weights_name.strip():
                    weights_path = (manifest.parent / weights_name).resolve()
                    if weights_path.is_file():
                        referenced.add(weights_path)
            resolved = _manifest_model(manifest, source)
            if resolved:
                items.append(resolved[0])
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if (
            path.is_file()
            and path.suffix.lower() in extensions
            and path.resolve() not in referenced
        ):
            backend: ModelBackend = (
                "litert"
                if kind == "detect"
                and path.suffix.lower() in {".tflite", ".litert"}
                else "yolo"
            )
            items.append(
                DiscoveredModel(
                    path.name,
                    str(path.resolve()),
                    _size(path),
                    source,
                    kind,
                    backend,
                    None,
                    None,
                    False,
                    True,
                    backend != "litert",
                )
            )
    return sorted(items, key=lambda item: item.name.casefold())


def discover_models(layout: ModelLayout, kind: ModelKind) -> list[DiscoveredModel]:
    if kind == "detect":
        directories: tuple[tuple[Path, ModelSource], ...] = (
            (layout.detect_bundled, 'bundled'),
            (layout.detect_user, "user"),
            (layout.detect_sync, "sync"),
        )
    elif kind == "cls":
        directories = (
            (layout.cls_user, "user"),
            (layout.cls_sync, "sync"),
        )
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    models: list[DiscoveredModel] = []
    for directory, source in directories:
        models.extend(_scan(directory, source, kind))

    if kind == "cls":
        status = dinov3_component_status(root=layout.dinov3_root)
        if status.get("healthy") is True:
            known_paths = {model.path for model in models}
            known_names = {model.name.casefold() for model in models}
            for manifest in installed_dinov3_manifests(root=layout.dinov3_root):
                resolved = _manifest_model(manifest, "sync")
                if resolved is None:
                    continue
                model = resolved[0]
                if model.path in known_paths or model.name.casefold() in known_names:
                    continue
                models.append(model)
                known_paths.add(model.path)
                known_names.add(model.name.casefold())

    unique: dict[str, DiscoveredModel] = {}
    for model in models:
        unique.setdefault(model.path, model)
    return sorted(
        unique.values(),
        key=lambda item: (_SOURCE_ORDER[item.source], item.name.casefold()),
    )


def _saved_path_matches(saved: str, current: str) -> bool:
    if saved == current:
        return True
    return os.name == "nt" and os.path.normcase(
        os.path.normpath(saved)
    ) == os.path.normcase(os.path.normpath(current))


def resolve_saved_model_path(
    saved: object,
    models: Sequence[DiscoveredModel],
) -> str | None:
    if not isinstance(saved, str) or not saved.strip():
        return None
    value = saved.strip()
    for model in models:
        if _saved_path_matches(value, model.path) or (
            model.checkpoint_path
            and _saved_path_matches(value, model.checkpoint_path)
        ):
            return model.path
    # Older lite builds put the built-in detector in the cloud-owned folder.
    # Preserve that saved selection when upgrading to the isolated bundle.
    normalized = value.replace('\\', '/').casefold()
    for model in models:
        if model.source == 'bundled' and normalized.endswith(
                '/model/detect/sync/' + Path(model.path).name.casefold()):
            return model.path
    named = [
        model
        for model in models
        if model.name == value or Path(model.path).name == value
    ]
    if not named:
        return None
    return next(
        (model.path for model in named if model.source == "user"),
        named[0].path,
    )
