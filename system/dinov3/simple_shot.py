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


def _squared_distances(array: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    deltas = array[:, None, :] - centroids[None, :, :]
    return np.einsum("nkd,nkd->nk", deltas, deltas, optimize=True)


def _farthest_first_centroids(array: np.ndarray, count: int) -> np.ndarray:
    selected = [0]
    while len(selected) < count:
        centroids = array[np.asarray(selected, dtype=np.int64)]
        nearest_distance = _squared_distances(array, centroids).min(axis=1)
        nearest_distance[np.asarray(selected, dtype=np.int64)] = -1.0
        selected.append(int(np.argmax(nearest_distance)))
    return array[np.asarray(selected, dtype=np.int64)].copy()


def _centroids_are_separated(
    centroids: np.ndarray,
    *,
    max_centroid_cosine: float,
) -> bool:
    if len(centroids) < 2:
        return True
    norms = np.linalg.norm(centroids, axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        return False
    normalized = centroids / norms[:, None]
    similarities = normalized @ normalized.T
    upper = similarities[np.triu_indices(len(centroids), k=1)]
    return bool(np.all(upper <= max_centroid_cosine))


def deterministic_k_means(
    embeddings: np.ndarray,
    *,
    max_k: int,
    min_cluster_size: int = 4,
    max_centroid_cosine: float = 0.98,
    iterations: int = 20,
) -> np.ndarray:
    """Return the largest deterministically supported prototype set.

    The input is already in classifier feature space (for overlay prototypes
    this means ``event_embedding - feature_center``). Centroids are arithmetic
    means and are intentionally *not* L2-normalized because Multi-prototype
    class selection uses squared Euclidean distance. Cosine normalization is
    only used to decide whether candidate subclusters are sufficiently distinct.
    """

    array = np.asarray(embeddings, dtype=np.float32)
    if (
        array.ndim != 2
        or array.shape[1] != _FEATURE_DIM
        or len(array) == 0
        or not np.isfinite(array).all()
    ):
        raise ValueError("Expected finite embeddings with shape (N, 768)")
    if isinstance(max_k, bool) or max_k < 1:
        raise ValueError("max_k must be a positive integer")
    if isinstance(min_cluster_size, bool) or min_cluster_size < 1:
        raise ValueError("min_cluster_size must be a positive integer")
    if not -1.0 <= float(max_centroid_cosine) <= 1.0:
        raise ValueError("max_centroid_cosine must be between -1 and 1")

    supported_max = max(1, min(int(max_k), len(array) // min_cluster_size))
    for cluster_count in range(supported_max, 0, -1):
        if cluster_count == 1:
            centroid = array.mean(axis=0, keepdims=True).astype(np.float32)
            if float(np.linalg.norm(centroid[0])) <= 0:
                raise ValueError("Prototype centroid must be non-zero")
            return centroid

        centroids = _farthest_first_centroids(array, cluster_count)
        valid = True
        for _ in range(max(1, int(iterations))):
            distances = _squared_distances(array, centroids)
            labels = np.argmin(distances, axis=1)
            counts = np.bincount(labels, minlength=cluster_count)
            if int(counts.min()) < min_cluster_size:
                valid = False
                break
            next_centroids = np.stack(
                [array[labels == index].mean(axis=0) for index in range(cluster_count)]
            ).astype(np.float32)
            if np.allclose(next_centroids, centroids, atol=1e-7, rtol=0.0):
                centroids = next_centroids
                break
            centroids = next_centroids

        if not valid:
            continue

        distances = _squared_distances(array, centroids)
        labels = np.argmin(distances, axis=1)
        counts = np.bincount(labels, minlength=cluster_count)
        if int(counts.min()) < min_cluster_size:
            continue
        centroids = np.stack(
            [array[labels == index].mean(axis=0) for index in range(cluster_count)]
        ).astype(np.float32)
        if not _centroids_are_separated(
            centroids,
            max_centroid_cosine=float(max_centroid_cosine),
        ):
            continue
        return centroids

    # The loop always reaches k=1, but retain an explicit invariant failure.
    raise RuntimeError("Unable to build a deterministic prototype")


def deterministic_two_means(embeddings: np.ndarray, *, iterations: int = 12):
    """Legacy normalized two-way split retained until registry migration completes."""

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != _FEATURE_DIM or len(array) < 8:
        return None
    normalized = np.stack([normalize_embedding(row) for row in array])
    first = 0
    second = int(np.argmin(normalized @ normalized[first]))
    centroids = np.stack([normalized[first], normalized[second]])
    labels = np.zeros(len(normalized), dtype=np.int64)
    for _ in range(max(1, iterations)):
        labels = np.argmax(normalized @ centroids.T, axis=1)
        if min(np.bincount(labels, minlength=2)) < 4:
            return None
        next_centroids = np.stack(
            [build_prototype(normalized[labels == index]) for index in range(2)]
        )
        if np.allclose(next_centroids, centroids, atol=1e-6):
            centroids = next_centroids
            break
        centroids = next_centroids
    if float(centroids[0] @ centroids[1]) > 0.98:
        return None
    return centroids, labels
