"""Frozen B/16 combined experiment recipe, without research-project imports."""
from __future__ import annotations

import numpy as np

from .balanced_unknown_calibration import fit_unknown_threshold
from .evidence import validate_evidence_weights, weighted_fuse_group, weighted_top75

PRODUCTION_RECIPE = "production_dual"
EXPERIMENT_RECIPE = "experiment_full"
PRODUCTION_DISPLAY_NAME = "Neri1.1"
PREVIEW_DISPLAY_NAME = "Neri1.2preview"
INFERENCE_RECIPES = (PRODUCTION_RECIPE, EXPERIMENT_RECIPE)


def head_type(recipe):
    if recipe not in INFERENCE_RECIPES:
        raise ValueError(f"Unsupported DINOv3 inference recipe: {recipe}")
    return "ridge_within_experiment" if recipe == EXPERIMENT_RECIPE else "ridge_within"


def row_zscore(values):
    values = np.asarray(values, np.float64)
    return (values-values.mean(1, keepdims=True))/np.maximum(values.std(1, keepdims=True), 1e-6)


def top_fraction(values, cameras, sequences, weights=None):
    """Preserve the experiment's input precision when averaging Top-75%."""
    if weights is not None:
        return weighted_top75(values, cameras, sequences, weights)
    _, inverse = np.unique(np.column_stack((cameras, sequences)), axis=0, return_inverse=True)
    output = np.empty(len(values), dtype=np.float64)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(inverse[order]))+1, len(order)]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        indices = order[begin:end]
        count = max(1, int(np.ceil(.75*len(indices))))
        output[indices] = np.sort(values[indices])[::-1][:count].mean()
    return output


def event_fuse(scores, sites, sequences, contents, weights=None):
    """Exact historical tau=.5 fusion, without production box/track splitting."""
    values = np.asarray(scores, np.float64)
    w_arr = validate_evidence_weights(weights, len(values)) if weights is not None else None
    if len(values) == 0:
        return np.empty((0, values.shape[1] if values.ndim == 2 else 0), dtype=np.float64)
    output = values.copy()
    groups = {}
    for i, (site, seq, content) in enumerate(zip(sites, sequences, contents)):
        groups.setdefault((str(site), str(seq)), {}).setdefault(str(content), []).append(i)
    for by_content in groups.values():
        rows = list(by_content.values())
        if w_arr is None:
            content_scores = np.stack([values[ii].mean(0) for ii in rows])
            count = len(rows)
            weight = count/(count+.5)
            if count >= 2 and np.bincount(content_scores.argmax(1)).max()*3 >= count*2:
                weight = max(weight, .8)
            mean = content_scores.mean(0)
        else:
            weight, mean = weighted_fuse_group(
                values, rows, w_arr, tau=0.5, consensus_boost=True
            )
        for ii in rows:
            output[ii] = (1-weight)*values[ii]+weight*mean
    return output


def pooled_evidence(within, norm, cameras, sequences, weights=None):
    from .model import sequence_mean
    top = np.sort(within, axis=1)[:, -2:]
    margin = 2*top[:, 1]-top[:, 0]
    if weights is None:
        return margin, top_fraction(margin, cameras, sequences), sequence_mean(norm, cameras, sequences)
    return (
        margin,
        top_fraction(margin, cameras, sequences, weights=weights),
        sequence_mean(norm, cameras, sequences, weights=weights),
    )


def fit_calibration(within, norm, labels, cameras, sequences, known, target):
    _, pooled, radial = pooled_evidence(within, norm, cameras, sequences)
    def stats(values):
        median = float(np.median(values[known]))
        scale = max(float(1.4826*np.median(abs(values[known]-median))), 1e-6)
        return dict(median=median, scale=scale)
    calibration = dict(margin=stats(pooled), radial=stats(radial), target=float(target))
    score = gate_score(pooled, radial, calibration)
    fit = fit_unknown_threshold(score[~known], labels[~known], target=target, weighting="species")
    calibration.update(threshold=float(fit["threshold"]), fit=fit)
    return calibration


def gate_score(pooled, radial, calibration):
    def standardized(values, key):
        info = calibration[key]
        return (values-info["median"])/info["scale"]
    return standardized(pooled, "margin")-.08*standardized(radial, "radial")


def predict(model, raw, cameras, sequences, sites, contents, adapted, evidence_weights=None):
    weights = (
        validate_evidence_weights(evidence_weights, len(raw))
        if evidence_weights is not None
        else None
    )
    within, ridge, norm, labels, support_cameras = model._prediction_evidence(raw, adapted)
    # Count cameras in the training supports, never in queries or calibration.
    counts = np.asarray([len(np.unique(support_cameras[labels == c])) for c in model.classes_])
    bias = .02*np.log(np.maximum(counts, 1)/3)
    blended = .55*row_zscore(within-bias[None, :])+.45*row_zscore(ridge)
    if weights is None:
        fused = event_fuse(blended, sites, sequences, contents)
        margin, pooled, radial = pooled_evidence(within, norm, cameras, sequences)
    else:
        fused = event_fuse(blended, sites, sequences, contents, weights=weights)
        margin, pooled, radial = pooled_evidence(within, norm, cameras, sequences, weights=weights)
    score = gate_score(pooled, radial, model._calibration)
    accepted = score >= model.threshold
    return dict(classes=model.classes, predictions=model.classes_[fused.argmax(1)], accepted=accepted,
                known_score=score, threshold=model.threshold, sequence_known_score=score,
                sequence_threshold=model.threshold, sequence_accepted=accepted,
                within_margins=margin, whitened_norms=norm, resolved_tracks=np.asarray(sequences),
                ridge_scores=ridge, within_scores=within, fused_scores=fused,
                classification_scores=blended, inference_recipe=EXPERIMENT_RECIPE, gate_mode="sequence")
