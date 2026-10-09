"""Shared HDBSCAN clustering for discovery and learned species prototypes."""
from __future__ import annotations

from numbers import Integral
import numpy as np

DEFAULT_MIN_CLUSTER_SIZE = 4


def _positive_integer(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def hdbscan_labels(
    embeddings: np.ndarray, *, min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = None, cluster_selection_epsilon: float = 0.0,
) -> np.ndarray:
    """Return stable cluster IDs; -1 means insufficient density or evidence.

    Euclidean distance is applied in the caller's feature space. Unit CL2N
    vectors therefore retain the ordering of their cosine distances.
    """
    size = _positive_integer(min_cluster_size, "min_cluster_size", 2)
    samples = size if min_samples is None else _positive_integer(min_samples, "min_samples", 1)
    epsilon = float(cluster_selection_epsilon)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("cluster_selection_epsilon must be finite and nonnegative")
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 768 or not np.isfinite(array).all():
        raise ValueError("Expected finite embeddings with shape (N, 768)")
    if len(array) < max(size, samples, 2):
        return np.full(len(array), -1, dtype=np.int64)

    # Lazy import keeps optional DINOv2 modules loadable in minimal builds.
    # Missing dependencies must fail explicitly, never switch algorithms.
    from sklearn.cluster import HDBSCAN

    order = np.lexsort(array.T[::-1])
    fitted = HDBSCAN(
        min_cluster_size=size, min_samples=samples, metric="euclidean",
        cluster_selection_epsilon=epsilon, cluster_selection_method="eom",
        allow_single_cluster=True, n_jobs=1, copy=True,
    ).fit_predict(array[order])
    stable = np.full(len(array), -1, dtype=np.int64)
    next_id = 0
    mapping = {}
    for index, label in enumerate(fitted):
        if label < 0:
            continue
        if int(label) not in mapping:
            mapping[int(label)] = next_id
            next_id += 1
        stable[order[index]] = mapping[int(label)]
    return stable


def hdbscan_prototypes(
    embeddings: np.ndarray, *, max_prototypes: int,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> np.ndarray:
    """Summarize supported HDBSCAN groups as arithmetic prototype means.

    Retain the largest groups within the existing learning-stage budget;
    noise is excluded whenever supported groups exist. If labelled evidence
    has no supported group (including small evidence previews), retain its
    arithmetic mean as a single species summary, without inventing a cluster
    or running a second clustering algorithm.
    """
    limit = _positive_integer(max_prototypes, "max_prototypes", 1)
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 768 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Expected nonempty finite embeddings with shape (N, 768)")
    # Distance 0.2 between unit vectors corresponds to cosine similarity 0.98.
    labels = hdbscan_labels(
        array, min_cluster_size=min_cluster_size, cluster_selection_epsilon=0.2,
    )
    groups = [int(label) for label in np.unique(labels) if label >= 0]
    groups.sort(key=lambda label: (-int(np.sum(labels == label)), label))
    if groups:
        prototypes = np.stack([array[labels == label].mean(axis=0) for label in groups[:limit]])
    else:
        prototypes = array.mean(axis=0, keepdims=True)
    if np.any(np.linalg.norm(prototypes, axis=1) <= 1e-12):
        raise ValueError("Prototype centroid must be non-zero")
    return prototypes.astype(np.float32, copy=False)
