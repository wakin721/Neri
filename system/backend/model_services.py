"""Backend-facing model catalog helpers backed by the canonical Model layout."""
from __future__ import annotations

from pathlib import Path

from system.model_sync.catalog import discover_models
from system.model_sync.layout import get_model_layout

from .models import ModelInfo


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
    )


def list_available_models() -> list[ModelInfo]:
    layout = get_model_layout()
    return [_to_model_info(item) for item in discover_models(layout, "detect")]


def list_available_classification_models() -> list[ModelInfo]:
    layout = get_model_layout()
    return [_to_model_info(item) for item in discover_models(layout, "cls")]
