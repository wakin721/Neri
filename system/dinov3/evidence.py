"""Shared weighted evidence validation and aggregation helpers."""
from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

import numpy as np


def _weighted_column_mean(rows_values: np.ndarray, row_weights: np.ndarray, total_weight: float) -> np.ndarray:
    """Compute exact order-independent weighted column mean using ``math.fsum``."""
    if len(rows_values) == 1:
        return np.asarray(rows_values[0], dtype=np.float64).copy()
    weighted = rows_values * row_weights[:, None]
    return np.asarray(
        [math.fsum(weighted[:, col].tolist()) / total_weight for col in range(weighted.shape[1])],
        dtype=np.float64,
    )


def validate_evidence_weights(evidence_weights: Any, length: int) -> np.ndarray:
    """Validate 1D positive finite row weights aligned to ``length`` rows."""
    raw = np.asarray(evidence_weights)
    if raw.dtype == bool or (
        raw.dtype == object and any(isinstance(item, (bool, np.bool_)) for item in raw.ravel())
    ):
        raise ValueError("evidence_weights must contain positive finite numeric values")
    try:
        weights = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("evidence_weights must contain positive finite numeric values") from error
    if weights.ndim != 1 or weights.shape != (length,):
        raise ValueError(f"evidence_weights must have shape ({length},)")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("evidence_weights must contain only positive finite values")
    return weights


def weighted_top75(
    values: np.ndarray,
    cameras: Sequence[Any],
    sequences: Sequence[Any],
    weights: Any,
) -> np.ndarray:
    """Compute weighted top-75% evidence-mass mean grouped by ``(camera, sequence)``."""
    vals = np.asarray(values, dtype=np.float64)
    w = validate_evidence_weights(weights, len(vals))
    if len(vals) == 0:
        return np.empty(0, dtype=np.float64)

    _, inverse = np.unique(
        np.column_stack((cameras, sequences)), axis=0, return_inverse=True
    )
    output = np.empty(len(vals), dtype=np.float64)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1, len(order)]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        indices = order[begin:end]
        g_vals = vals[indices]
        g_weights = w[indices]
        desc = np.argsort(-g_vals, kind="stable")
        s_vals = g_vals[desc]
        s_weights = g_weights[desc]
        target = 0.75 * math.fsum(s_weights.tolist())
        cum = np.cumsum(s_weights)
        cutoff = min(int(np.searchsorted(cum, target, side="left")), len(s_weights) - 1)
        prev_cum = float(cum[cutoff - 1]) if cutoff > 0 else 0.0
        rem = min(float(s_weights[cutoff]), max(0.0, target - prev_cum))
        num = math.fsum((s_vals[:cutoff] * s_weights[:cutoff]).tolist()) + rem * float(s_vals[cutoff])
        denom = math.fsum(s_weights[:cutoff].tolist()) + rem
        output[indices] = num / denom
    return output


def weighted_sequence_mean(
    values: np.ndarray,
    cameras: Sequence[Any],
    sequences: Sequence[Any],
    weights: Any,
) -> np.ndarray:
    """Compute weighted event mean grouped by ``(camera, sequence)``."""
    vals = np.asarray(values, dtype=np.float64)
    w = validate_evidence_weights(weights, len(vals))
    if len(vals) == 0:
        return np.empty(0, dtype=np.float64)

    _, inverse = np.unique(
        np.column_stack((cameras, sequences)), axis=0, return_inverse=True
    )
    output = np.empty(len(vals), dtype=np.float64)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1, len(order)]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        indices = order[begin:end]
        w_sum = math.fsum(w[indices].tolist())
        output[indices] = math.fsum((vals[indices] * w[indices]).tolist()) / w_sum
    return output


def weighted_fuse_group(
    values: np.ndarray,
    rows: Sequence[Sequence[int]],
    weights: np.ndarray,
    *,
    tau: float,
    consensus_boost: bool,
) -> Tuple[float, np.ndarray]:
    """Compute weighted sequence fusion weight and group score over collapsed content buckets."""
    content_scores = []
    content_weights = []
    for ii in rows:
        w_ii = weights[ii]
        w_sum = math.fsum(w_ii.tolist())
        content_scores.append(_weighted_column_mean(values[ii], w_ii, w_sum))
        content_weights.append(w_sum)
    scores_arr = np.stack(content_scores)
    weights_arr = np.asarray(content_weights, dtype=np.float64)
    total_weight = math.fsum(content_weights)
    # Bound effective sample count by unique content count to prevent repeats inflating counts.
    n_eff = min(total_weight, float(len(rows)))
    w_seq = n_eff / (n_eff + tau)
    if consensus_boost and len(rows) >= 2 and n_eff >= 2.0 - 1e-12:
        votes = scores_arr.argmax(axis=1)
        max_vote_weight = max(
            math.fsum(weights_arr[votes == cls_idx].tolist())
            for cls_idx in np.unique(votes)
        )
        if max_vote_weight * 3.0 >= total_weight * 2.0 - 1e-12 * total_weight:
            w_seq = max(w_seq, 0.80)
    group_score = _weighted_column_mean(scores_arr, weights_arr, total_weight)
    return w_seq, group_score
