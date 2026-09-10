"""Open-set rejection configuration for DINOv3 Multi-prototype classifiers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

MULTI_DUAL_REJECTION_MODE = "multi_dual"


@dataclass(frozen=True)
class MultiDualRejectionConfig:
    cosine_threshold: float
    squared_distance_threshold: float

    def __post_init__(self) -> None:
        cosine = float(self.cosine_threshold)
        distance = float(self.squared_distance_threshold)
        if not math.isfinite(cosine) or cosine < -1.0 or cosine > 1.0:
            raise ValueError("cosine_threshold must be finite and within [-1, 1]")
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("squared_distance_threshold must be finite and non-negative")
        object.__setattr__(self, "cosine_threshold", cosine)
        object.__setattr__(self, "squared_distance_threshold", distance)

    def accepts(self, *, cosine_score: float, squared_distance: float) -> bool:
        cosine = float(cosine_score)
        distance = float(squared_distance)
        if not math.isfinite(cosine) or not math.isfinite(distance):
            return False
        return (
            cosine >= self.cosine_threshold
            and distance <= self.squared_distance_threshold
        )

    def as_dict(self) -> dict[str, float | str]:
        return {
            "mode": MULTI_DUAL_REJECTION_MODE,
            "cosine_threshold": self.cosine_threshold,
            "squared_distance_threshold": self.squared_distance_threshold,
        }


def parse_multi_dual_rejection(value: Any) -> MultiDualRejectionConfig | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("DINOv3 rejection config must be an object")
    mode = str(value.get("mode", "")).strip().lower()
    if mode != MULTI_DUAL_REJECTION_MODE:
        raise ValueError(
            f"Unsupported DINOv3 rejection mode: {mode or '<missing>'}"
        )
    if "cosine_threshold" not in value:
        raise ValueError("multi_dual rejection requires cosine_threshold")
    if "squared_distance_threshold" not in value:
        raise ValueError("multi_dual rejection requires squared_distance_threshold")
    return MultiDualRejectionConfig(
        cosine_threshold=value["cosine_threshold"],
        squared_distance_threshold=value["squared_distance_threshold"],
    )
