"""Discovery and compatibility resolution for user and synchronized models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from .layout import ModelLayout

ModelKind = Literal["detect", "cls"]
ModelSource = Literal["user", "sync"]

_DETECT_EXTENSIONS = frozenset({".pt"})
_CLS_EXTENSIONS = frozenset({".pt", ".onnx", ".engine"})


@dataclass(frozen=True)
class DiscoveredModel:
    name: str
    path: str
    size_bytes: int | None
    source: ModelSource
    kind: ModelKind


def _scan(directory: Path, source: ModelSource, kind: ModelKind) -> list[DiscoveredModel]:
    extensions = _DETECT_EXTENSIONS if kind == "detect" else _CLS_EXTENSIONS
    if not directory.is_dir():
        return []
    items: list[DiscoveredModel] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
        items.append(
            DiscoveredModel(
                name=path.name,
                path=str(path.resolve()),
                size_bytes=size_bytes,
                source=source,
                kind=kind,
            )
        )
    return items


def discover_models(layout: ModelLayout, kind: ModelKind) -> list[DiscoveredModel]:
    if kind == "detect":
        directories = ((layout.detect_user, "user"), (layout.detect_sync, "sync"))
    elif kind == "cls":
        directories = ((layout.cls_user, "user"), (layout.cls_sync, "sync"))
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    models: list[DiscoveredModel] = []
    for directory, source in directories:
        models.extend(_scan(directory, source, kind))
    return models


def resolve_saved_model_path(
    saved: object,
    models: Sequence[DiscoveredModel],
) -> str | None:
    if not isinstance(saved, str) or not saved.strip():
        return None
    value = saved.strip()

    for model in models:
        if model.path == value:
            return model.path

    named = [model for model in models if model.name == value]
    if not named:
        return None
    for model in named:
        if model.source == "user":
            return model.path
    return named[0].path
