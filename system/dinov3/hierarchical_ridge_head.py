"""Hierarchy-balanced multiclass ridge regression with an unpenalized intercept."""

from collections import Counter, defaultdict
import math
from numbers import Real
from os import PathLike
from pathlib import Path

import numpy as np


def _metadata_vector(values, name, length):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.shape[0] != length:
        raise ValueError(f"{name} must align with features")
    if array.size == 0:
        raise ValueError("At least one sample is required")
    for value in array:
        if isinstance(value, Real) and not math.isfinite(float(value)):
            raise ValueError(f"{name} must contain only finite values")
        try:
            hash(value)
        except TypeError as error:
            raise ValueError(f"{name} values must be hashable") from error
    return array


def _sorted_classes(labels):
    try:
        classes = np.unique(labels)
    except (TypeError, ValueError) as error:
        raise ValueError("labels must be mutually sortable") from error
    if classes.shape != (len(classes),):
        raise ValueError("labels must contain scalar values")
    return classes


def _class_balanced_weights(labels):
    labels = np.asarray(labels)
    try:
        classes, inverse, counts = np.unique(
            labels, return_inverse=True, return_counts=True
        )
    except (TypeError, ValueError) as error:
        raise ValueError("labels must be mutually sortable") from error
    weights = 1.0 / (len(classes) * counts[inverse])
    return weights.astype(np.float64, copy=False)


def hierarchical_weights(labels, cameras, sequences):
    """Return weights that equalize classes, cameras, sequences, and images.

    Each class receives total weight ``1 / number_of_classes``. Within a
    class, cameras share that weight equally. Within each class/camera pair,
    sequences share the camera weight equally, and images share their
    sequence weight equally.
    """

    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    length = len(labels)
    labels = _metadata_vector(labels, "labels", length)
    cameras = _metadata_vector(cameras, "cameras", length)
    sequences = _metadata_vector(sequences, "sequences", length)

    class_cameras = defaultdict(set)
    camera_sequences = defaultdict(set)
    sequence_counts = Counter()
    rows = []
    for label, camera, sequence in zip(labels, cameras, sequences):
        label = label.item() if isinstance(label, np.generic) else label
        camera = camera.item() if isinstance(camera, np.generic) else camera
        sequence = sequence.item() if isinstance(sequence, np.generic) else sequence
        rows.append((label, camera, sequence))
        class_cameras[label].add(camera)
        camera_sequences[(label, camera)].add(sequence)
        sequence_counts[(label, camera, sequence)] += 1

    class_mass = 1.0 / len(class_cameras)
    weights = np.empty(length, dtype=np.float64)
    for index, (label, camera, sequence) in enumerate(rows):
        camera_mass = class_mass / len(class_cameras[label])
        sequence_mass = camera_mass / len(camera_sequences[(label, camera)])
        weights[index] = sequence_mass / sequence_counts[(label, camera, sequence)]

    # Absorb the final floating-point rounding residual.
    weights /= weights.sum(dtype=np.float64)
    return weights


def _portable_classes(classes):
    """Return a NumPy-dtype class array that does not require pickle."""

    classes = np.asarray(classes)
    if classes.dtype.kind != "O":
        return classes
    portable = np.asarray(classes.tolist())
    if portable.dtype.kind == "O":
        raise ValueError("label dtype cannot be stored without pickle")
    return portable


class HierarchicalRidgeHead:
    """Weighted multiclass ridge classifier fitted by the normal equations."""

    def __init__(self, ridge=0.01, weighting="hierarchical"):
        if isinstance(ridge, (bool, np.bool_)) or not np.isscalar(ridge):
            raise ValueError("ridge must be a positive finite scalar")
        try:
            ridge = float(ridge)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("ridge must be a positive finite scalar") from error
        if not math.isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge must be a positive finite scalar")
        if weighting not in ("hierarchical", "class"):
            raise ValueError("weighting must be 'hierarchical' or 'class'")

        self.ridge = ridge
        self.weighting = str(weighting)
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.feature_dim_ = None
        self.sample_weights_ = None

    def fit(self, features, labels, cameras, sequences):
        try:
            features = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("features must be a finite 2D numeric array") from error
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features must have shape [N, D] with N, D > 0")
        if not np.isfinite(features).all():
            raise ValueError("features must contain only finite values")

        sample_count, feature_dim = features.shape
        labels = _metadata_vector(labels, "labels", sample_count)
        _metadata_vector(cameras, "cameras", sample_count)
        _metadata_vector(sequences, "sequences", sample_count)

        classes = _sorted_classes(labels)
        if len(classes) < 2:
            raise ValueError("at least two classes are required")
        classes = _portable_classes(classes)

        if self.weighting == "hierarchical":
            weights = hierarchical_weights(labels, cameras, sequences)
        else:
            weights = _class_balanced_weights(labels)
        if (
            weights.shape != (sample_count,)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0.0)
        ):
            raise ValueError("invalid sample weights")
        weights = weights / weights.sum(dtype=np.float64)

        targets = np.equal(labels[:, None], classes[None, :]).astype(np.float64)
        feature_mean = weights @ features
        target_mean = weights @ targets
        centered_features = features - feature_mean
        centered_targets = targets - target_mean
        weighted_centered_targets = weights[:, None] * centered_targets
        gram = centered_features.T @ (
            weights[:, None] * centered_features
        )
        cross_product = centered_features.T @ weighted_centered_targets
        gram.flat[:: feature_dim + 1] += self.ridge

        try:
            coef = np.linalg.solve(gram, cross_product)
        except np.linalg.LinAlgError as error:
            raise ValueError("ridge normal-equation system is singular") from error
        intercept = target_mean - feature_mean @ coef
        coef = coef.T
        if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
            raise ValueError("fit produced non-finite parameters")

        self.classes_ = classes.copy()
        self.coef_ = coef.astype(np.float64, copy=False)
        self.intercept_ = intercept.astype(np.float64, copy=False)
        self.feature_dim_ = feature_dim
        self.sample_weights_ = weights.astype(np.float64, copy=False)
        return self

    def _require_fitted(self):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("head has not been fitted")

    @property
    def classes(self):
        self._require_fitted()
        return self.classes_.copy()

    def scores(self, features):
        self._require_fitted()
        try:
            features = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("features must be a finite 2D numeric array") from error
        if features.ndim != 2 or features.shape[1] != self.feature_dim_:
            raise ValueError(
                f"features must have shape [N, {self.feature_dim_}]"
            )
        if not np.isfinite(features).all():
            raise ValueError("features must contain only finite values")
        scores = features @ self.coef_.T + self.intercept_
        if not np.isfinite(scores).all():
            raise ValueError("scores must contain only finite values")
        return scores

    def save(self, path):
        self._require_fitted()
        if self.classes_.dtype.kind == "O":
            raise ValueError("class labels cannot be saved without pickle")
        with Path(path).open("wb") as output:
            np.savez_compressed(
                output,
                classes=self.classes_,
                coef=self.coef_,
                intercept=self.intercept_,
                ridge=np.asarray(self.ridge, dtype=np.float64),
                weighting=np.asarray(self.weighting),
                feature_dim=np.asarray(self.feature_dim_, dtype=np.int64),
            )

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as source:
            with np.load(source, allow_pickle=False) as payload:
                expected = {
                    "classes",
                    "coef",
                    "intercept",
                    "ridge",
                    "weighting",
                    "feature_dim",
                }
                if set(payload.files) != expected:
                    raise ValueError("invalid ridge head archive fields")
                classes = payload["classes"]
                coef = payload["coef"]
                intercept = payload["intercept"]
                ridge = payload["ridge"]
                weighting = payload["weighting"]
                feature_dim = payload["feature_dim"]

                if classes.ndim != 1 or classes.size < 2 or classes.dtype.kind == "O":
                    raise ValueError("invalid class labels")
                try:
                    unique = np.unique(classes)
                except (TypeError, ValueError) as error:
                    raise ValueError("invalid class labels") from error
                if not np.array_equal(unique, classes):
                    raise ValueError("class labels must be unique and sorted")

                for name, value, ndim in (
                    ("ridge", ridge, 0),
                    ("weighting", weighting, 0),
                    ("feature_dim", feature_dim, 0),
                ):
                    if value.ndim != ndim or value.size != 1:
                        raise ValueError(f"invalid {name} metadata")
                ridge_value = float(ridge.item())
                weighting_value = str(weighting.item())
                feature_dim_value = feature_dim.item()
                if feature_dim_value != int(feature_dim_value):
                    raise ValueError("feature_dim must be an integer")
                feature_dim_value = int(feature_dim_value)
                if (
                    not math.isfinite(ridge_value)
                    or ridge_value <= 0.0
                    or weighting_value not in ("hierarchical", "class")
                    or feature_dim_value <= 0
                ):
                    raise ValueError("invalid ridge head metadata")

                if (
                    coef.ndim != 2
                    or coef.shape != (len(classes), feature_dim_value)
                    or intercept.ndim != 1
                    or intercept.shape != (len(classes),)
                    or not np.issubdtype(coef.dtype, np.number)
                    or not np.issubdtype(intercept.dtype, np.number)
                ):
                    raise ValueError("invalid ridge head parameter shapes")
                coef = coef.astype(np.float64)
                intercept = intercept.astype(np.float64)
                if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
                    raise ValueError("ridge head parameters must be finite")

        head = cls(ridge=ridge_value, weighting=weighting_value)
        head.classes_ = classes.copy()
        head.coef_ = coef.copy()
        head.intercept_ = intercept.copy()
        head.feature_dim_ = feature_dim_value
        return head
