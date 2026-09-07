"""Compatibility facade for backend services with canonical model catalog helpers."""
from __future__ import annotations

from pathlib import Path

from system.model_sync.catalog import discover_models
from system.model_sync.layout import get_model_layout

from . import services_core as _core
from .models import ModelInfo

# Preserve the existing service surface, including private helpers used by
# regression tests, while keeping the large processing implementation unchanged.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


def model_directory() -> Path:
    """Return the canonical detection-model root exposed by the API."""

    return get_model_layout().root / "detect"


def classification_model_directory() -> Path:
    """Return the canonical classification-model root exposed by the API."""

    return get_model_layout().root / "cls"


def _catalog(kind: str) -> list[ModelInfo]:
    return [
        ModelInfo(
            name=item.name,
            path=item.path,
            size_bytes=item.size_bytes,
            source=item.source,
            kind=item.kind,
        )
        for item in discover_models(get_model_layout(), kind)
    ]


def list_available_models() -> list[ModelInfo]:
    """List user and synchronized detection models, keeping sources distinct."""

    return _catalog("detect")


def list_available_classification_models() -> list[ModelInfo]:
    """List user and synchronized classification models."""

    return _catalog("cls")
