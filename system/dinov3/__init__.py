"""DINOv3 core package with joint Ridge-Within model."""

from .balanced_unknown_calibration import fit_unknown_threshold
from .hierarchical_ridge_head import HierarchicalRidgeHead
from .incremental_hierarchical_ridge_head import IncrementalHierarchicalRidgeHead
from .model import (
    DinoV3RidgeWithinModel,
    WithinReferenceUpdater,
    adaptive_sequence_fuse,
    resolve_target_tracks,
    score_cpu,
    sequence_mean,
    standardize,
    top75,
    unit,
)
from .species_feature_cache_ridge_head import SpeciesFeatureCacheRidgeHead

__all__ = [
    "DinoV3RidgeWithinModel",
    "WithinReferenceUpdater",
    "HierarchicalRidgeHead",
    "IncrementalHierarchicalRidgeHead",
    "SpeciesFeatureCacheRidgeHead",
    "fit_unknown_threshold",
    "adaptive_sequence_fuse",
    "resolve_target_tracks",
    "score_cpu",
    "top75",
    "sequence_mean",
    "standardize",
    "unit",
]
