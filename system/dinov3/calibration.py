"""Calibration helpers for DINOv3 Multi-prototype dual rejection."""
from __future__ import annotations

from typing import Any

import numpy as np


def _as_finite_vector(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def cl2n_transform(raw_features: np.ndarray, feature_center: np.ndarray) -> np.ndarray:
    """Apply the CL2N geometry used to fit the Multi-prototype head."""
    raw = np.asarray(raw_features, dtype=np.float32)
    center = np.asarray(feature_center, dtype=np.float32)
    if raw.ndim != 2 or center.shape != (raw.shape[1],):
        raise ValueError("Feature center must match the feature dimension")
    if not np.isfinite(raw).all() or not np.isfinite(center).all():
        raise ValueError("Features and feature center must be finite")
    centered = raw - center[None, :]
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("CL2N produced a zero or non-finite row")
    return centered / norms


def prototype_signals(features: np.ndarray, prototypes: np.ndarray) -> dict[str, np.ndarray]:
    """Return winning-prototype cosine and squared Euclidean distance."""
    x = np.asarray(features, dtype=np.float32)
    p = np.asarray(prototypes, dtype=np.float32)
    if x.ndim != 2 or p.ndim != 2 or x.shape[1] != p.shape[1] or len(p) == 0:
        raise ValueError("features/prototypes must be compatible non-empty matrices")
    if not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError("features/prototypes must be finite")
    x_norm = np.linalg.norm(x, axis=1)
    p_norm = np.linalg.norm(p, axis=1)
    if np.any(x_norm <= 1e-12) or np.any(p_norm <= 1e-12):
        raise ValueError("features/prototypes must have non-zero norm")

    dot = x @ p.T
    squared_distance = np.maximum(
        np.square(x).sum(axis=1, keepdims=True)
        + np.square(p).sum(axis=1)[None, :]
        - 2.0 * dot,
        0.0,
    )
    winner = np.argmin(squared_distance, axis=1)
    row = np.arange(len(x))
    cosine = dot / (x_norm[:, None] * p_norm[None, :])
    return {
        "winner_index": winner.astype(np.int64),
        "winner_cosine": cosine[row, winner].astype(np.float64),
        "winner_squared_distance": squared_distance[row, winner].astype(np.float64),
    }


def apply_multi_dual_gate(
    cosine_scores: np.ndarray,
    squared_distances: np.ndarray,
    *,
    cosine_threshold: float,
    squared_distance_threshold: float,
) -> np.ndarray:
    cosine = np.asarray(cosine_scores, dtype=np.float64)
    distance = np.asarray(squared_distances, dtype=np.float64)
    if cosine.shape != distance.shape:
        raise ValueError("cosine_scores and squared_distances must have the same shape")
    return (cosine >= float(cosine_threshold)) & (
        distance <= float(squared_distance_threshold)
    )


def species_macro_rejection(accepted: np.ndarray, labels: np.ndarray) -> float:
    accepted = np.asarray(accepted, dtype=bool)
    labels = np.asarray(labels).astype(str)
    if accepted.ndim != 1 or labels.ndim != 1 or len(accepted) != len(labels) or not len(labels):
        raise ValueError("accepted and labels must be aligned non-empty vectors")
    species = sorted(set(labels.tolist()))
    return float(
        np.mean(
            [np.mean(~accepted[labels == species_name]) for species_name in species]
        )
    )


def _threshold_candidates(values: np.ndarray, grid_size: int) -> np.ndarray:
    values = _as_finite_vector(values, name="threshold values")
    unique = np.unique(values)
    if len(unique) == 1:
        return unique
    candidates = np.concatenate(
        [
            [np.nextafter(unique[0], -np.inf)],
            (unique[:-1] + unique[1:]) / 2.0,
            [unique[-1]],
        ]
    )
    if len(candidates) <= grid_size:
        return candidates
    positions = np.unique(
        np.rint(np.linspace(0, len(candidates) - 1, grid_size)).astype(int)
    )
    return candidates[positions]


def calibrate_multi_dual(
    known_cosine: np.ndarray,
    known_squared_distance: np.ndarray,
    proxy_cosine: np.ndarray,
    proxy_squared_distance: np.ndarray,
    proxy_labels: np.ndarray,
    *,
    max_known_frr: float = 0.05,
    grid_size: int = 64,
) -> dict[str, Any]:
    """Jointly select cosine/L2 thresholds under a known-FRR constraint."""
    known_cosine = _as_finite_vector(known_cosine, name="known_cosine")
    known_distance = _as_finite_vector(
        known_squared_distance,
        name="known_squared_distance",
    )
    proxy_cosine = _as_finite_vector(proxy_cosine, name="proxy_cosine")
    proxy_distance = _as_finite_vector(
        proxy_squared_distance,
        name="proxy_squared_distance",
    )
    proxy_labels = np.asarray(proxy_labels).astype(str)
    if known_cosine.shape != known_distance.shape:
        raise ValueError("Known cosine/distance vectors must align")
    if proxy_cosine.shape != proxy_distance.shape or len(proxy_cosine) != len(proxy_labels):
        raise ValueError("Proxy cosine/distance/labels must align")
    if not 0 <= float(max_known_frr) < 1:
        raise ValueError("max_known_frr must be in [0, 1)")
    if int(grid_size) < 2:
        raise ValueError("grid_size must be >= 2")

    cosine_candidates = _threshold_candidates(
        np.concatenate([known_cosine, proxy_cosine]),
        int(grid_size),
    )
    distance_candidates = _threshold_candidates(
        np.concatenate([known_distance, proxy_distance]),
        int(grid_size),
    )

    best_key = None
    best = None
    for cosine_threshold in cosine_candidates:
        known_cosine_pass = known_cosine >= cosine_threshold
        for distance_threshold in distance_candidates:
            known_accept = known_cosine_pass & (known_distance <= distance_threshold)
            known_frr = float(np.mean(~known_accept))
            if known_frr > float(max_known_frr) + 1e-15:
                continue

            proxy_accept = (proxy_cosine >= cosine_threshold) & (
                proxy_distance <= distance_threshold
            )
            proxy_macro = species_macro_rejection(proxy_accept, proxy_labels)
            proxy_pooled = float(np.mean(~proxy_accept))
            objective = ((1.0 - known_frr) + proxy_macro) / 2.0
            key = (objective, -known_frr, proxy_macro, proxy_pooled)
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "mode": "multi_dual",
                    "cosine_threshold": float(cosine_threshold),
                    "squared_distance_threshold": float(distance_threshold),
                    "known_false_rejection_rate": known_frr,
                    "proxy_macro_rejection_rate": proxy_macro,
                    "proxy_pooled_rejection_rate": proxy_pooled,
                    "objective": float(objective),
                    "max_known_frr": float(max_known_frr),
                    "grid_size": int(grid_size),
                    "rule": (
                        "accept iff winner_cosine >= cosine_threshold AND "
                        "winner_squared_distance <= squared_distance_threshold"
                    ),
                }
    if best is None:
        raise RuntimeError("No feasible Multi-dual threshold pair found")
    return best
