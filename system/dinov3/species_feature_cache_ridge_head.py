"""Exact arbitrary old-species supplements retaining per-species feature cache.

Retains per-species raw feature caches and camera/sequence metadata, along with
global gram_sum, class_means, and class_counts. Avoids storing C full D-by-D
Gram matrices. Updates to existing or new classes recompute moments only for
the affected classes, adjusting the global Gram sum by subtracting the old
contribution and adding the new contribution.
"""

import math
from os import PathLike
from pathlib import Path

import numpy as np

from .hierarchical_ridge_head import (
    _metadata_vector,
    _portable_classes,
    _sorted_classes,
    hierarchical_weights,
)
from .incremental_hierarchical_ridge_head import IncrementalHierarchicalRidgeHead


class SpeciesFeatureCacheRidgeHead(IncrementalHierarchicalRidgeHead):
    """Multiclass ridge head with per-species feature cache for exact updates."""

    def __init__(self, ridge=0.01, weighting="hierarchical"):
        super().__init__(ridge=ridge, weighting=weighting)
        self.feature_cache_ = {}
        self.last_update_ = None

    def fit(self, features, labels, cameras, sequences):
        """Reset state and fit from scratch."""
        staged = type(self)(ridge=self.ridge, weighting=self.weighting)
        staged.insert(features, labels, cameras, sequences)
        self.__dict__.clear()
        self.__dict__.update(staged.__dict__)
        return self

    @staticmethod
    def _compute_class_moments(values, cameras, sequences, label, weighting):
        values_f64 = values.astype(np.float64, copy=False)
        if weighting == "hierarchical":
            weights = hierarchical_weights(
                np.full(len(values), label), cameras, sequences
            )
        else:
            weights = np.full(len(values), 1.0 / len(values), dtype=np.float64)
        mean = weights @ values_f64
        gram = values_f64.T @ (weights[:, None] * values_f64)
        return mean, gram

    def insert(self, features, labels, cameras, sequences):
        """Incorporate supplements for old classes and/or complete new classes."""
        try:
            raw_features = np.asarray(features)
            if np.issubdtype(raw_features.dtype, np.floating):
                features_arr = raw_features
            else:
                features_arr = raw_features.astype(np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("features must be a finite 2D numeric array") from error

        if (
            features_arr.ndim != 2
            or features_arr.shape[0] == 0
            or features_arr.shape[1] == 0
        ):
            raise ValueError("features must be a nonempty finite [N, D] matrix")
        if not np.isfinite(features_arr).all():
            raise ValueError("features must contain only finite values")

        length, dimension = features_arr.shape
        labels = _metadata_vector(labels, "labels", length)
        cameras = _portable_classes(_metadata_vector(cameras, "cameras", length))
        sequences = _portable_classes(_metadata_vector(sequences, "sequences", length))
        if cameras.dtype.kind == "O" or sequences.dtype.kind == "O":
            raise ValueError("camera and sequence metadata cannot be stored without pickle")
        incoming_classes = _portable_classes(_sorted_classes(labels))

        if self.classes_ is not None:
            if dimension != self.feature_dim_:
                raise ValueError("feature dimension must remain unchanged")
            if incoming_classes.dtype.kind != self.classes_.dtype.kind:
                raise ValueError(
                    "new labels must have the same dtype kind as existing labels"
                )
        else:
            if len(incoming_classes) < 2:
                raise ValueError(
                    "at least two classes are required for initialization"
                )

        staged_cache = dict(self.feature_cache_)
        old_feature_rows_read = 0
        delta_gram = np.zeros((dimension, dimension), dtype=np.float64)

        if self.classes_ is not None:
            staged_means = self.class_means_.copy()
            staged_counts = self.class_counts_.copy()
            existing_classes_list = list(self.classes_)
            existing_class_set = set(self.classes_)
        else:
            staged_means = np.empty((0, dimension), dtype=np.float64)
            staged_counts = np.empty(0, dtype=np.int64)
            existing_classes_list = []
            existing_class_set = set()

        new_class_labels = []
        new_class_means = []
        new_class_counts = []

        for label in incoming_classes:
            mask = labels == label
            inc_feat = features_arr[mask]
            inc_cam = cameras[mask]
            inc_seq = sequences[mask]

            if label in existing_class_set:
                old_feat, old_cam, old_seq = self.feature_cache_[label]
                old_feature_rows_read += len(old_feat)

                old_mean, old_q = self._compute_class_moments(
                    old_feat, old_cam, old_seq, label, self.weighting
                )

                new_feat = np.concatenate([old_feat, inc_feat], axis=0)
                new_cam = np.concatenate([old_cam, inc_cam], axis=0)
                new_seq = np.concatenate([old_seq, inc_seq], axis=0)

                new_mean, new_q = self._compute_class_moments(
                    new_feat, new_cam, new_seq, label, self.weighting
                )

                delta_gram += new_q - old_q
                idx = existing_classes_list.index(label)
                staged_means[idx] = new_mean
                staged_counts[idx] = len(new_feat)
                staged_cache[label] = (new_feat, new_cam, new_seq)
            else:
                new_mean, new_q = self._compute_class_moments(
                    inc_feat, inc_cam, inc_seq, label, self.weighting
                )
                delta_gram += new_q
                new_class_labels.append(label)
                new_class_means.append(new_mean)
                new_class_counts.append(len(inc_feat))
                staged_cache[label] = (inc_feat, inc_cam, inc_seq)

        if new_class_labels:
            if self.classes_ is not None:
                all_classes = np.concatenate([self.classes_, new_class_labels])
                all_means = np.concatenate([staged_means, new_class_means], axis=0)
                all_counts = np.concatenate([staged_counts, new_class_counts], axis=0)
            else:
                all_classes = np.asarray(new_class_labels)
                all_means = np.asarray(new_class_means, dtype=np.float64)
                all_counts = np.asarray(new_class_counts, dtype=np.int64)
            order = np.argsort(all_classes)
            all_classes = all_classes[order]
            all_means = all_means[order]
            all_counts = all_counts[order]
        else:
            all_classes = self.classes_
            all_means = staged_means
            all_counts = staged_counts

        if self.gram_sum_ is None:
            staged_gram_sum = delta_gram
        else:
            staged_gram_sum = self.gram_sum_ + delta_gram
        staged_gram_sum = (staged_gram_sum + staged_gram_sum.T) * 0.5

        if not np.isfinite(staged_gram_sum).all() or not np.isfinite(all_means).all():
            raise ValueError("sufficient statistics must be finite")

        coef, intercept = self._solve(all_means, staged_gram_sum, self.ridge)

        self.classes_ = all_classes
        self.class_means_ = all_means
        self.class_counts_ = all_counts
        self.gram_sum_ = staged_gram_sum
        self.feature_dim_ = dimension
        self.coef_ = coef
        self.intercept_ = intercept
        self.feature_cache_ = staged_cache
        self.sample_weights_ = None
        self.last_update_ = {
            "new_samples": int(length),
            "old_feature_rows_read": int(old_feature_rows_read),
            "affected_classes": incoming_classes.tolist(),
        }
        return self

    def partial_fit(self, features, labels, cameras, sequences):
        """Alias for insert supporting new and existing classes."""
        return self.insert(features, labels, cameras, sequences)

    def numeric_bytes(self):
        """Return memory footprint of all numeric and metadata ndarrays."""
        self._require_fitted()
        total = 0
        for arr in (
            self.class_means_,
            self.class_counts_,
            self.gram_sum_,
            self.coef_,
            self.intercept_,
            self.classes_,
        ):
            if isinstance(arr, np.ndarray):
                total += int(arr.nbytes)
        for feat, cam, seq in self.feature_cache_.values():
            for arr in (feat, cam, seq):
                if isinstance(arr, np.ndarray):
                    total += int(arr.nbytes)
        return total

    def save(self, path):
        """Save portable statistics and per-species feature cache without pickle."""
        self._require_fitted()
        arrays_to_save = {
            "format_version": np.asarray(1, dtype=np.int64),
            "classes": self.classes_,
            "class_means": self.class_means_,
            "class_counts": self.class_counts_,
            "gram_sum": self.gram_sum_,
            "ridge": np.asarray(self.ridge, dtype=np.float64),
            "weighting": np.asarray(self.weighting),
            "feature_dim": np.asarray(self.feature_dim_, dtype=np.int64),
        }
        for index, label in enumerate(self.classes_):
            feat, cam, seq = self.feature_cache_[label]
            arrays_to_save[f"cache_features_{index}"] = feat
            arrays_to_save[f"cache_cameras_{index}"] = _portable_classes(cam)
            arrays_to_save[f"cache_sequences_{index}"] = _portable_classes(seq)

        if hasattr(path, "write"):
            np.savez_compressed(path, **arrays_to_save)
        else:
            with Path(path).open("wb") as output:
                np.savez_compressed(output, **arrays_to_save)

    @classmethod
    def load(cls, path):
        """Load from archive without pickle and reconstruct exact parameters."""
        close_source = False
        if hasattr(path, "read"):
            source = path
        else:
            source = Path(path).open("rb")
            close_source = True
        try:
            with np.load(source, allow_pickle=False) as payload:
                required_base = {
                    "format_version",
                    "classes",
                    "class_means",
                    "class_counts",
                    "gram_sum",
                    "ridge",
                    "weighting",
                    "feature_dim",
                }
                files = set(payload.files)
                if not required_base.issubset(files):
                    raise ValueError("missing required ridge archive fields")

                format_version = payload["format_version"]
                ridge = payload["ridge"]
                weighting = payload["weighting"]
                feature_dim = payload["feature_dim"]

                for name, value, ndim in (
                    ("format_version", format_version, 0),
                    ("ridge", ridge, 0),
                    ("weighting", weighting, 0),
                    ("feature_dim", feature_dim, 0),
                ):
                    if value.ndim != ndim or value.size != 1:
                        raise ValueError(f"invalid {name} metadata")

                if format_version.item() != 1:
                    raise ValueError("unsupported archive version")

                ridge_value = float(ridge.item())
                weighting_value = str(weighting.item())
                raw_feature_dim = feature_dim.item()
                if raw_feature_dim != int(raw_feature_dim):
                    raise ValueError("feature_dim must be an integer")
                feature_dim_value = int(raw_feature_dim)

                if (
                    not math.isfinite(ridge_value)
                    or ridge_value <= 0.0
                    or weighting_value not in ("hierarchical", "class")
                    or feature_dim_value <= 0
                ):
                    raise ValueError("invalid ridge head metadata")

                classes = payload["classes"]
                means = payload["class_means"]
                counts = payload["class_counts"]
                gram = payload["gram_sum"]

                if (
                    classes.ndim != 1
                    or len(classes) < 2
                    or classes.dtype.kind == "O"
                    or not np.array_equal(classes, np.unique(classes))
                ):
                    raise ValueError("class labels must be unique, sorted and portable")
                _metadata_vector(classes, "classes", len(classes))

                num_classes = len(classes)
                if (
                    means.ndim != 2
                    or means.shape != (num_classes, feature_dim_value)
                    or gram.ndim != 2
                    or gram.shape != (feature_dim_value, feature_dim_value)
                    or counts.ndim != 1
                    or counts.shape != (num_classes,)
                    or counts.dtype.kind not in "iu"
                    or np.any(counts <= 0)
                ):
                    raise ValueError("invalid sufficient-statistic shapes or counts")

                if means.dtype.kind not in "fi" or gram.dtype.kind not in "fi":
                    raise ValueError("sufficient statistics must be real numeric arrays")

                means = means.astype(np.float64)
                gram = gram.astype(np.float64)

                if (
                    not np.isfinite(means).all()
                    or not np.isfinite(gram).all()
                    or not np.allclose(gram, gram.T, rtol=1e-12, atol=1e-12)
                ):
                    raise ValueError("invalid sufficient-statistic values")

                expected_files = set(required_base)
                for index in range(num_classes):
                    expected_files.add(f"cache_features_{index}")
                    expected_files.add(f"cache_cameras_{index}")
                    expected_files.add(f"cache_sequences_{index}")

                if files != expected_files:
                    raise ValueError("archive contains unexpected or missing cache fields")

                cache = {}
                for index, label in enumerate(classes):
                    feat = payload[f"cache_features_{index}"]
                    cam = payload[f"cache_cameras_{index}"]
                    seq = payload[f"cache_sequences_{index}"]
                    if cam.dtype.kind == "O" or seq.dtype.kind == "O":
                        raise ValueError("cached metadata cannot require pickle")

                    class_count = int(counts[index])
                    if (
                        feat.ndim != 2
                        or feat.shape != (class_count, feature_dim_value)
                        or feat.dtype.kind not in "fi"
                        or not np.isfinite(feat).all()
                    ):
                        raise ValueError(f"invalid cached features for class {label}")

                    if cam.ndim != 1 or len(cam) != class_count:
                        raise ValueError(f"invalid cached cameras for class {label}")
                    if seq.ndim != 1 or len(seq) != class_count:
                        raise ValueError(f"invalid cached sequences for class {label}")

                    _metadata_vector(cam, f"cache_cameras_{index}", class_count)
                    _metadata_vector(seq, f"cache_sequences_{index}", class_count)

                    cache[label] = (feat, cam, seq)
        finally:
            if close_source:
                source.close()

        head = cls(ridge=ridge_value, weighting=weighting_value)
        coef, intercept = head._solve(means, gram, ridge_value)
        head.classes_ = classes
        head.class_means_ = means
        head.class_counts_ = counts
        head.gram_sum_ = gram
        head.feature_dim_ = feature_dim_value
        head.coef_ = coef
        head.intercept_ = intercept
        head.feature_cache_ = cache
        head.sample_weights_ = None
        head.last_update_ = None
        return head
