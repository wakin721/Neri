"""Validated raw observations that can extend a frozen Memory classifier."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .checkpoint import DINO_FEATURE_DIM


@dataclass(frozen=True)
class MemoryExample:
    species: str
    embedding: np.ndarray
    camera_id: str
    source: str
    registry_id: int | None = None
    registration_status: str | None = None

    def __post_init__(self) -> None:
        vector = np.asarray(self.embedding, dtype=np.float32).copy()
        if vector.shape != (DINO_FEATURE_DIM,) or not np.isfinite(vector).all():
            raise ValueError("Memory evidence must be a finite 768-dimensional vector")
        if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4):
            raise ValueError("Memory evidence must be L2-normalized")
        if not self.species.strip() or not self.camera_id.strip():
            raise ValueError("Memory species and camera ID must be nonempty")
        vector.setflags(write=False)
        object.__setattr__(self, "embedding", vector)


@dataclass(frozen=True)
class MemoryBank:
    formal: tuple[MemoryExample, ...]
    provisional: tuple[MemoryExample, ...] = ()
