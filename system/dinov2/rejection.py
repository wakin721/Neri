"""Margin-adjusted dual rejection primitives for DINOv2."""
from __future__ import annotations

import math

from .checkpoint import DinoV2Rejection


def adjusted_distance_score(
    *,
    winner_squared_distance: float,
    class_margin: float,
    margin_weight: float,
    class_scale: float = 1.0,
) -> float:
    distance = float(winner_squared_distance)
    margin = float(class_margin)
    weight = float(margin_weight)
    scale = float(class_scale)
    if not all(math.isfinite(value) for value in (distance, margin, weight, scale)):
        raise ValueError("Margin rejection inputs must be finite")
    if distance < 0.0:
        raise ValueError("winner_squared_distance must be nonnegative")
    if weight < 0.0:
        raise ValueError("margin_weight must be nonnegative")
    if scale <= 0.0:
        raise ValueError("class_scale must be positive")
    return (-distance + weight * margin) / scale


def accepts_margin_rejection(
    cosine_score: float,
    adjusted_score: float,
    rejection: DinoV2Rejection,
) -> bool:
    cosine = float(cosine_score)
    score = float(adjusted_score)
    if not math.isfinite(cosine) or not math.isfinite(score):
        return False
    return (
        cosine >= rejection.cosine_threshold
        and score >= rejection.adjusted_distance_score_threshold
    )


def margin_route(
    cosine_score: float,
    adjusted_score: float,
    rejection: DinoV2Rejection,
) -> str:
    cosine = float(cosine_score)
    score = float(adjusted_score)
    cosine_pass = math.isfinite(cosine) and cosine >= rejection.cosine_threshold
    score_pass = (
        math.isfinite(score)
        and score >= rejection.adjusted_distance_score_threshold
    )
    if cosine_pass and score_pass:
        return "update_prototype"
    if not cosine_pass and not score_pass:
        return "new_mode_candidate"
    return "candidate"


def rejection_metadata(rejection: DinoV2Rejection) -> dict[str, object]:
    result: dict[str, object] = {
        "mode": rejection.mode,
        "cosine_threshold": rejection.cosine_threshold,
        "adjusted_distance_score_threshold": rejection.adjusted_distance_score_threshold,
        "margin_weight": rejection.margin_weight,
    }
    if rejection.class_distance_scales is not None:
        result["class_distance_scales"] = list(rejection.class_distance_scales)
    return result
