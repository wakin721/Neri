"""Explicit population targets for a high-is-known rejection score.

Fit only on labeled auxiliary Unknown calibration data after each model update.
Species weighting targets macro recall; species_sequence also removes unequal
frame counts within each species. Neither target guarantees shifted-test recall.
"""
from __future__ import annotations

import numpy as np


def route_incremental_scores(context_scores, raw_scores, predictions, initial_classes):
    """Use the original feature path for predictions of newly inserted classes.

    Routing uses classifier predictions, never query truth. Route image scores
    before sequence aggregation. Calibrate the routed score afresh on C;
    thresholds from either input score are not transferable.
    """
    context_scores, raw_scores = [np.asarray(a, dtype=np.float64)
                                  for a in (context_scores, raw_scores)]
    predictions = np.asarray(predictions)
    if (context_scores.ndim != 1 or raw_scores.shape != context_scores.shape
            or predictions.shape != context_scores.shape):
        raise ValueError('Aligned one-dimensional scores and predictions required')
    if not np.isfinite(context_scores).all() or not np.isfinite(raw_scores).all():
        raise ValueError('Finite scores required')
    return np.where(np.isin(predictions, initial_classes), context_scores, raw_scores)


def unknown_weights(labels, *, sequences=None):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError('Nonempty one-dimensional species labels required')
    if sequences is not None:
        sequences = np.asarray(sequences)
        if sequences.shape != labels.shape:
            raise ValueError('Sequence IDs must align with species labels')
    classes = np.unique(labels)
    weights = np.empty(len(labels), dtype=np.float64)
    for cls in classes:
        selected = np.flatnonzero(labels == cls)
        if sequences is None:
            weights[selected] = 1 / (len(classes) * len(selected))
        else:
            _, inverse, counts = np.unique(sequences[selected], return_inverse=True,
                                           return_counts=True)
            weights[selected] = 1 / (len(classes) * len(counts) * counts[inverse])
    return weights / weights.sum()


def fit_unknown_threshold(scores, labels, *, target=.9, weighting='species', sequences=None):
    """Smallest observed cutoff meeting the specified empirical Unknown target.

    Reject using ``np.asarray(score, dtype=np.float64) < threshold``. All frames
    tied at the selected score are rejected, including at target=1. The caller
    must pass Unknown rows only; no evaluation labels or model fitting occur.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite nonempty one-dimensional scores required')
    if labels.shape != scores.shape:
        raise ValueError('Species labels must align with scores')
    if not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError('target must be in (0, 1]')
    if weighting == 'micro':
        weights = np.full(len(scores), 1 / len(scores))
    elif weighting == 'species':
        weights = unknown_weights(labels)
    elif weighting == 'species_sequence':
        if sequences is None:
            raise ValueError('species_sequence weighting requires sequence IDs')
        weights = unknown_weights(labels, sequences=sequences)
    else:
        raise ValueError('Unknown weighting')
    order = np.argsort(scores, kind='stable')
    cumulative = np.cumsum(weights[order])
    # Roundoff at an exact target must not select an unrelated next score.
    pos = min(int(np.searchsorted(cumulative, target - 1e-12)), len(scores) - 1)
    threshold = float(np.nextafter(scores[order[pos]], np.inf))
    rejected = scores < threshold
    return dict(threshold=threshold, target=float(target), weighting=weighting,
                unknown_images=len(scores), unknown_species=len(np.unique(labels)),
                achieved_weighted_recall=float(weights[rejected].sum()),
                achieved_micro_recall=float(rejected.mean()),
                achieved_species_macro_recall=float(np.mean([
                    rejected[labels == c].mean() for c in np.unique(labels)])))
