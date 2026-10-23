"""One density policy for rejected independent events in discovery and review."""
from __future__ import annotations

import numpy as np

MIN_REJECTION_CLUSTER_EVENTS = 5
REJECTION_MIN_SAMPLES = 3
REJECTION_DENSITY_POLICY = "raw-unit-hdbscan-eom-size5-samples3-no-single-v1"


def rejected_event_labels(features):
    """Cluster raw unit event means; -1 is noise, with no mean fallback."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite event feature rows")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Event features must have nonzero norm")
    if len(values) < MIN_REJECTION_CLUSTER_EVENTS:
        return np.full(len(values), -1, dtype=np.int64)
    try:
        from hdbscan import HDBSCAN
    except ImportError as exc:
        raise RuntimeError("Install requirements-within-seq.txt to use HDBSCAN") from exc
    values = values / norms
    # Deterministic input order also makes tied/duplicate event means stable.
    order = np.lexsort(values.T[::-1])
    fitted = HDBSCAN(
        min_cluster_size=MIN_REJECTION_CLUSTER_EVENTS,
        min_samples=REJECTION_MIN_SAMPLES, allow_single_cluster=False,
        metric="euclidean", cluster_selection_method="eom", core_dist_n_jobs=1,
    ).fit_predict(values[order])
    labels = np.full(len(values), -1, dtype=np.int64)
    labels[order] = fitted
    return labels
