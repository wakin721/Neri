"""Active API schemas for the Neri backend.

The legacy schema module is retained temporarily while the clean-break migration
is in progress. Active model discovery uses DINOv2 only.
"""
from typing import Literal

from .models_legacy import *  # noqa: F401,F403
from . import models_legacy as _legacy


class ModelInfo(_legacy.ModelInfo):
    """A model discovered under the canonical local model directory."""

    backend: Literal["yolo", "dinov2"] = "yolo"
