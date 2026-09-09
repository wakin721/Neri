"""Torch-free value types for DINOv3 base and learned prototype banks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .checkpoint import DINO_FEATURE_DIM


@dataclass(frozen=True)
class PrototypeRecord:
    species: str
    embedding: np.ndarray
    source: str
    registry_id: int | None = None
    registration_status: str | None = None

    def __post_init__(self) -> None:
        embedding = np.asarray(self.embedding, dtype=np.float32).copy()
        if embedding.shape != (DINO_FEATURE_DIM,) or not np.isfinite(embedding).all():
            raise ValueError("Expected a finite 768-dimensional prototype")
        embedding.setflags(write=False)
        object.__setattr__(self, "embedding", embedding)


@dataclass(frozen=True)
class PrototypeBank:
    formal: tuple[PrototypeRecord, ...]
    provisional: tuple[PrototypeRecord, ...] = ()
