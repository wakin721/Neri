"""SimpleShot and deterministic prototype helpers for learned species."""
from __future__ import annotations

import hashlib
import random
from typing import Sequence, TypeVar

import numpy as np

T = TypeVar("T")
_FEATURE_DIM = 768


def normalize_embedding(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if (
        array.ndim != 1
        or array.shape[0] != _FEATURE_DIM
        or not np.isfinite(array).all()
    ):
        raise ValueError("Expected a finite 768-dimensional embedding")
    norm = float(np.linalg.norm(array))
    if norm <= 0:
        raise ValueError("Embedding must be non-zero")
    return array / norm


def build_prototype(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != _FEATURE_DIM or len(array) == 0:
        raise ValueError("Expected embeddings with shape (N, 768)")
    return normalize_embedding(
        np.stack([normalize_embedding(row) for row in array]).mean(axis=0)
    )


def cosine_similarity(embedding: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    vector = normalize_embedding(embedding)
    matrix = np.asarray(prototypes, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != _FEATURE_DIM:
        raise ValueError("Expected prototypes with shape (N, 768)")
    return np.stack([normalize_embedding(row) for row in matrix]) @ vector


def deterministic_event_sample(
    values: Sequence[T],
    *,
    count: int,
    seed_material: str,
) -> list[T]:
    items = list(values)
    if count <= 0 or not items:
        return []
    rnd = random.Random(
        int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    )
    indices = list(range(len(items)))
    rnd.shuffle(indices)
    return [items[index] for index in sorted(indices[: min(count, len(items))])]
