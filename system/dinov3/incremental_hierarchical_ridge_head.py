"""Exact new-class updates for HierarchicalRidgeHead without old features.

Retain sum_c E_w[xx^T | c], each class's E_w[x | c], and sample counts.
Within-class expectations use the original camera/sequence/image balancing.
Every insertion supplies complete supports for NEW classes only. Updating an
existing class would require finer statistics and is deliberately rejected.
The fixed feature space (including whitening) must remain unchanged.
"""
from pathlib import Path

import numpy as np

from .hierarchical_ridge_head import (
    HierarchicalRidgeHead, _metadata_vector, _portable_classes,
    _sorted_classes, hierarchical_weights,
)


class IncrementalHierarchicalRidgeHead(HierarchicalRidgeHead):
    """Accumulate balanced moments, then solve a D-by-D ridge system.

    No optimizer or old-sample replay is used. This still solves the normal
    equations on each update; it does not promise a rank-one inverse update.
    ``sample_weights_`` is None because historical per-image state is not kept.
    """

    def __init__(self, ridge=0.01, weighting="hierarchical"):
        super().__init__(ridge=ridge, weighting=weighting)
        self.class_means_ = None
        self.class_counts_ = None
        self.gram_sum_ = None

    def fit(self, features, labels, cameras, sequences):
        staged = type(self)(self.ridge, self.weighting)
        staged.insert(features, labels, cameras, sequences)
        self.__dict__.update(staged.__dict__)
        return self

    @staticmethod
    def _solve(means, gram_sum, ridge):
        count, dimension = means.shape
        feature_mean = means.mean(axis=0)
        # Class mass is 1/C, including after adding new classes. Recompute the
        # intercept centering and keep the regularizer at its original scale.
        gram = gram_sum / count - np.outer(feature_mean, feature_mean)
        gram = (gram + gram.T) * 0.5
        gram.flat[::dimension + 1] += ridge
        cross = (means - feature_mean).T / count
        coef = np.linalg.solve(gram, cross).T
        intercept = np.full(count, 1.0 / count) - coef @ feature_mean
        if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
            raise ValueError("ridge solve produced non-finite parameters")
        return coef, intercept

    def insert(self, features, labels, cameras, sequences):
        """Read new-class supports once; commit only after a successful solve."""
        features = np.asarray(features, dtype=np.float64)
        if (features.ndim != 2 or not all(features.shape)
                or not np.isfinite(features).all()):
            raise ValueError("features must be a nonempty finite [N, D] matrix")
        length, dimension = features.shape
        labels = _metadata_vector(labels, "labels", length)
        cameras = _metadata_vector(cameras, "cameras", length)
        sequences = _metadata_vector(sequences, "sequences", length)
        incoming = _portable_classes(_sorted_classes(labels))
        if self.classes_ is not None:
            if dimension != self.feature_dim_:
                raise ValueError("feature dimension must remain unchanged")
            if incoming.dtype.kind != self.classes_.dtype.kind:
                raise ValueError("new labels must have the same dtype kind as existing labels")
            if np.isin(incoming, self.classes_).any():
                raise ValueError("Only new classes supported; supply each class's complete supports once")
            all_classes = np.concatenate((self.classes_, incoming))
        else:
            all_classes = incoming
        if len(all_classes) < 2:
            raise ValueError("at least two classes are required for initialization")

        gram_sum = (np.zeros((dimension, dimension), dtype=np.float64)
                    if self.gram_sum_ is None else self.gram_sum_.copy())
        means, counts = [], []
        for label in incoming:
            mask = labels == label
            values = features[mask]
            if self.weighting == "hierarchical":
                weights = hierarchical_weights(labels[mask], cameras[mask], sequences[mask])
            else:
                weights = np.full(len(values), 1.0 / len(values))
            means.append(weights @ values)
            counts.append(len(values))
            gram_sum += values.T @ (weights[:, None] * values)
        means = np.stack(means)
        counts = np.asarray(counts, dtype=np.int64)
        if self.classes_ is not None:
            means = np.concatenate((self.class_means_, means))
            counts = np.concatenate((self.class_counts_, counts))
        order = np.argsort(all_classes)
        all_classes, means, counts = all_classes[order], means[order], counts[order]
        if not np.isfinite(gram_sum).all() or not np.isfinite(means).all():
            raise ValueError("sufficient statistics must be finite")
        coef, intercept = self._solve(means, gram_sum, self.ridge)
        self.classes_, self.class_means_, self.class_counts_ = all_classes, means, counts
        self.gram_sum_, self.feature_dim_ = gram_sum, dimension
        self.coef_, self.intercept_ = coef, intercept
        self.sample_weights_ = None
        return self

    def partial_fit(self, features, labels, cameras, sequences):
        """Alias for new-class-only ``insert``; existing-class updates fail."""
        return self.insert(features, labels, cameras, sequences)

    def numeric_bytes(self):
        self._require_fitted()
        return sum(value.nbytes for value in (
            self.class_means_, self.class_counts_, self.gram_sum_,
            self.coef_, self.intercept_,
        ))

    def save(self, path):
        """Save portable statistics; parameters are reconstructed on load."""
        self._require_fitted()
        with Path(path).open("wb") as output:
            np.savez_compressed(
                output, format_version=np.asarray(1, dtype=np.int64),
                classes=self.classes_, class_means=self.class_means_,
                class_counts=self.class_counts_, gram_sum=self.gram_sum_,
                ridge=np.asarray(self.ridge), weighting=np.asarray(self.weighting),
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as payload:
            expected = {"format_version", "classes", "class_means", "class_counts",
                        "gram_sum", "ridge", "weighting"}
            if set(payload.files) != expected:
                raise ValueError("invalid incremental ridge archive fields")
            values = {key: payload[key] for key in expected}
        for key in ("format_version", "ridge", "weighting"):
            if values[key].ndim != 0:
                raise ValueError(f"invalid {key} metadata")
        if values["format_version"].item() != 1:
            raise ValueError("unsupported incremental ridge archive version")
        head = cls(ridge=values["ridge"].item(), weighting=values["weighting"].item())
        classes, means, counts, gram = [values[key] for key in
                                        ("classes", "class_means", "class_counts", "gram_sum")]
        if (classes.ndim != 1 or len(classes) < 2 or classes.dtype.kind == "O"
                or not np.array_equal(classes, np.unique(classes))):
            raise ValueError("class labels must be unique, sorted and portable")
        _metadata_vector(classes, "classes", len(classes))
        if (means.ndim != 2 or means.shape[0] != len(classes) or means.shape[1] == 0
                or gram.shape != (means.shape[1], means.shape[1])
                or counts.shape != (len(classes),) or counts.dtype.kind not in "iu"
                or np.any(counts <= 0)):
            raise ValueError("invalid sufficient-statistic shapes or counts")
        if means.dtype.kind not in "fi" or gram.dtype.kind not in "fi":
            raise ValueError("sufficient statistics must be real numeric arrays")
        means, gram = means.astype(np.float64), gram.astype(np.float64)
        if (not np.isfinite(means).all() or not np.isfinite(gram).all()
                or not np.allclose(gram, gram.T, rtol=1e-12, atol=1e-12)):
            raise ValueError("invalid sufficient-statistic values")
        coef, intercept = head._solve(means, gram, head.ridge)
        head.classes_, head.class_means_, head.class_counts_ = classes, means, counts
        head.gram_sum_, head.feature_dim_ = gram, means.shape[1]
        head.coef_, head.intercept_ = coef, intercept
        return head
