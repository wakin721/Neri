"""Batch implementation of the manuscript's Frozen Memory / Within-Seq route.

Consumes original unit encoder features, never the centered vectors stored in
legacy Memory checkpoints. This is a method reimplementation, not a verified
export of the manuscript's experimental model.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import math

import numpy as np


def _unit(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("Features have a zero or nonfinite norm after transformation")
    return values / norms


def _features(values, dimension: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != dimension or not len(array):
        raise ValueError(f"Expected nonempty features of shape (N, {dimension})")
    if not np.isfinite(array).all() or not np.allclose(
        np.linalg.norm(array, axis=1), 1.0, atol=1e-4, rtol=0.0
    ):
        raise ValueError("Expected finite, original L2-normalized encoder features")
    return _unit(array)


def _ids(values, count: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (count,) or array.dtype.kind not in "US":
        raise ValueError(f"{name} must contain one string per feature row")
    if any(not value.strip() for value in array):
        raise ValueError(f"{name} must not contain blank IDs")
    return array.astype(str)


def _groups(*columns) -> list[np.ndarray]:
    groups = defaultdict(list)
    for index, key in enumerate(zip(*columns)):
        groups[key].append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def hierarchical_weights(labels, cameras, sequences) -> np.ndarray:
    """Equal class, camera-within-class, and sequence-within-camera mass."""
    weights = np.zeros(len(labels), dtype=np.float64)
    classes = np.unique(labels)
    for species in classes:
        species_cameras = np.unique(cameras[labels == species])
        for camera in species_cameras:
            mask = (labels == species) & (cameras == camera)
            camera_sequences = np.unique(sequences[mask])
            for sequence in camera_sequences:
                selected = mask & (sequences == sequence)
                weights[selected] = 1.0 / (
                    len(classes) * len(species_cameras)
                    * len(camera_sequences) * int(selected.sum())
                )
    return weights


def select_neighbors(similarity: np.ndarray) -> np.ndarray:
    """Largest relative gap among distinct sequences; ties select smaller k."""
    count = similarity.shape[1]
    if count < 2:
        return np.ones(len(similarity), dtype=np.int64)
    limit = min(17, count)
    nearest = np.sort(
        np.partition(1.0 - similarity, limit - 1, axis=1)[:, :limit], axis=1
    )
    nearest = np.maximum(nearest, 1e-6)
    jumps = (nearest[:, 1:] - nearest[:, :-1]) / nearest[:, :-1]
    return np.argmax(jumps, axis=1) + 1


def calibration_threshold(scores, target: float = 0.90) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Auxiliary-unknown calibration scores must be finite and nonempty")
    if not math.isfinite(target) or not 0 < target <= 1:
        raise ValueError("Unknown recall target must be in (0, 1]")
    rank = math.ceil(len(values) * target) - 1
    return float(np.nextafter(values[rank], np.inf))


class WithinSeqModel:
    """Fixed initial transforms and a separately scored Memory classifier.

    The classifier uses k=3 and class-balanced centering, consistent with Neri's
    Memory scoring family. The manuscript does not provide its original export;
    exact experimental equivalence requires comparing to that implementation.
    """

    def __init__(self, features, labels, cameras, sequences, *, dimension: int = 768):
        self.dimension = dimension
        self.raw = _features(features, dimension)
        n = len(self.raw)
        self.labels = _ids(labels, n, "bank labels")
        self.cameras = _ids(cameras, n, "bank camera IDs")
        self.sequences = _ids(sequences, n, "bank sequence IDs")
        self.classes = np.unique(self.labels)
        if len(self.classes) < 2:
            raise ValueError("At least two bank classes are required for the class margin")

        class_means = np.stack([self.raw[self.labels == c].mean(axis=0) for c in self.classes])
        self.classification_center = class_means.mean(axis=0)
        self.memory = _unit(self.raw - self.classification_center)
        self.memory_centroids = _unit(np.stack([
            self.memory[self.labels == c].mean(axis=0) for c in self.classes
        ]))

        weights = hierarchical_weights(self.labels, self.cameras, self.sequences)
        self.within_center = weights @ self.raw
        residuals = self.raw.copy()
        for c in self.classes:
            mask = self.labels == c
            center = np.average(self.raw[mask], axis=0, weights=weights[mask])
            residuals[mask] -= center
        covariance = (residuals * weights[:, None]).T @ residuals
        scale = float(np.trace(covariance) / dimension)
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("Initial bank has insufficient within-class variation for whitening")
        covariance = 0.9 * covariance + 0.1 * scale * np.eye(dimension)
        self.whitening = np.linalg.cholesky(np.linalg.solve(covariance, np.eye(dimension)))

        groups = _groups(self.labels, self.cameras, self.sequences)
        self.sequence_raw = _unit(np.stack([self.raw[g].mean(axis=0) for g in groups]))
        self.sequence_labels = np.asarray([self.labels[g[0]] for g in groups])
        self.sequence_cameras = np.asarray([self.cameras[g[0]] for g in groups])
        self.sequence_ids = np.asarray([self.sequences[g[0]] for g in groups])
        self.sequence_memory = self._transform(self.sequence_raw)
        self.within_centroids = _unit(np.stack([
            self.sequence_memory[self.sequence_labels == c].mean(axis=0) for c in self.classes
        ]))
        self._distinct_sequences = _groups(self.sequence_cameras, self.sequence_ids)
        # Precompute class/camera partitions, avoiding masks on every query batch.
        self._memory_groups = self._camera_groups(self.labels, self.cameras)
        self._within_groups = self._camera_groups(self.sequence_labels, self.sequence_cameras)

    def _camera_groups(self, labels, cameras):
        return [
            [np.flatnonzero((labels == c) & (cameras == camera))
             for camera in np.unique(cameras[labels == c])]
            for c in self.classes
        ]

    def _transform(self, raw):
        return _unit((raw - self.within_center) @ self.whitening)

    @staticmethod
    def _scores(query, memory, centroids, groups, k):
        similarity = query @ memory.T
        scores = np.empty((len(query), len(groups)), dtype=np.float64)
        for c, camera_groups in enumerate(groups):
            pooled = np.column_stack([similarity[:, indices].max(axis=1) for indices in camera_groups])
            ordered = np.sort(pooled, axis=1)[:, ::-1]
            selected_k = np.minimum(k, ordered.shape[1])
            local = np.cumsum(ordered, axis=1)[np.arange(len(query)), selected_k - 1] / selected_k
            scores[:, c] = 0.5 * local + 0.5 * (query @ centroids[c])
        return scores

    def with_examples(self, examples):
        """Extend sequence memory without refitting either frozen transform."""
        if not examples:
            return self
        result = copy.copy(self)
        raw = np.stack([example.embedding for example in examples])
        result.sequence_raw = np.concatenate((self.sequence_raw, raw))
        result.sequence_labels = np.r_[self.sequence_labels, [e.species for e in examples]]
        result.sequence_cameras = np.r_[self.sequence_cameras, [e.camera_id for e in examples]]
        result.sequence_ids = np.r_[self.sequence_ids, [f"dynamic:{i}" for i in range(len(examples))]]
        result.sequence_memory = np.concatenate((self.sequence_memory, self._transform(raw)))
        result.classes = np.unique(result.sequence_labels)
        result.raw = np.concatenate((self.raw, raw))
        result.labels = np.r_[self.labels, [e.species for e in examples]]
        result.cameras = np.r_[self.cameras, [e.camera_id for e in examples]]
        result.memory = np.concatenate((self.memory, _unit(raw - self.classification_center)))
        result.memory_centroids = _unit(np.stack([
            result.memory[result.labels == c].mean(axis=0) for c in result.classes
        ]))
        result._memory_groups = result._camera_groups(result.labels, result.cameras)
        result.within_centroids = _unit(np.stack([
            result.sequence_memory[result.sequence_labels == c].mean(axis=0) for c in result.classes
        ]))
        result._distinct_sequences = _groups(result.sequence_cameras, result.sequence_ids)
        result._within_groups = result._camera_groups(result.sequence_labels, result.sequence_cameras)
        return result

    def score_batch(self, features, cameras, sequences, *, batch_size: int = 128, classify: bool = True):
        raw = _features(features, self.dimension)
        cameras = _ids(cameras, len(raw), "query camera IDs")
        sequences = _ids(sequences, len(raw), "query sequence IDs")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        predictions, image_scores, selected = [], [], []
        for start in range(0, len(raw), batch_size):
            chunk = raw[start:start + batch_size]
            if classify:
                classification = self._scores(
                    _unit(chunk - self.classification_center), self.memory,
                    self.memory_centroids, self._memory_groups, np.full(len(chunk), 3),
                )
                predictions.extend(self.classes[np.argmax(classification, axis=1)].tolist())
            query = self._transform(chunk)
            similarity = query @ self.sequence_memory.T
            distinct = np.column_stack([similarity[:, group].max(axis=1) for group in self._distinct_sequences])
            k = select_neighbors(distinct)
            scores = self._scores(query, self.sequence_memory, self.within_centroids, self._within_groups, k)
            top = np.sort(scores, axis=1)[:, -2:]
            # Word equation: m_W(x) = 2*s_(1)(x) - s_(2)(x).
            image_scores.extend((2.0 * top[:, 1] - top[:, 0]).tolist())
            selected.extend(k.tolist())
        image_scores = np.asarray(image_scores)
        sequence_scores = image_scores.copy()
        for group in _groups(cameras, sequences):
            sequence_scores[group] = image_scores[group].mean()
        return {
            "known_species": predictions,
            "image_knownness": image_scores,
            "sequence_knownness": sequence_scores,
            "selected_k": selected,
        }


def cluster_rejected_sequences(features, cameras, sequences, rejected):
    """HDBSCAN on raw sequence means; None means accepted, -1 means noise."""
    rejected = np.asarray(rejected, dtype=bool)
    labels = [None] * len(features)
    groups = _groups(cameras, sequences)
    candidates = []
    for group in groups:
        if rejected[group].any():
            if not rejected[group].all():
                raise ValueError("A sequence must have one shared rejection decision")
            candidates.append(group)
    if not candidates:
        return labels, 0
    means = _unit(np.stack([features[g].mean(axis=0) for g in candidates]))
    if len(means) < 5:
        partition = np.full(len(means), -1, dtype=np.int64)
    else:
        try:
            from hdbscan import HDBSCAN
        except ImportError as exc:
            raise RuntimeError("Install requirements-within-seq.txt to use HDBSCAN") from exc
        partition = HDBSCAN(
            min_cluster_size=5, min_samples=3, allow_single_cluster=False,
            metric="euclidean", cluster_selection_method="eom", core_dist_n_jobs=1,
        ).fit_predict(means)
    for group, label in zip(candidates, partition):
        for index in group:
            labels[int(index)] = int(label)
    return labels, len(candidates)


def discover_within_seq(
    bank_features, bank_labels, bank_camera_ids, bank_sequence_ids,
    calibration_features, calibration_camera_ids, calibration_sequence_ids,
    query_features, query_camera_ids, query_sequence_ids, *, target: float = 0.90,
):
    """Fit only on initial Known; calibrate on Aux unknown; score an unlabeled batch."""
    model = WithinSeqModel(bank_features, bank_labels, bank_camera_ids, bank_sequence_ids)
    calibration = _features(calibration_features, model.dimension)
    query = _features(query_features, model.dimension)
    ccam = _ids(calibration_camera_ids, len(calibration), "calibration camera IDs")
    cseq = _ids(calibration_sequence_ids, len(calibration), "calibration sequence IDs")
    qcam = _ids(query_camera_ids, len(query), "query camera IDs")
    qseq = _ids(query_sequence_ids, len(query), "query sequence IDs")
    pools = [set(zip(model.cameras, model.sequences)), set(zip(ccam, cseq)), set(zip(qcam, qseq))]
    if any(pools[i] & pools[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Bank, auxiliary calibration and query sequences must be disjoint")
    cal = model.score_batch(calibration, ccam, cseq)
    threshold = calibration_threshold(cal["sequence_knownness"], target)
    result = model.score_batch(query, qcam, qseq)
    scores = result.pop("sequence_knownness")
    rejected = scores < threshold
    clusters, sequence_count = cluster_rejected_sequences(query, qcam, qseq, rejected)
    return {
        "method": "frozen_memory_within_seq_hdbscan",
        "implementation": "manuscript_reimplementation_v1",
        "classification_neighbors": 3,
        "known_species": result["known_species"],
        "species": ["Unknown" if reject else species for species, reject in zip(result["known_species"], rejected)],
        "selected_k": result["selected_k"],
        "image_knownness": result["image_knownness"].tolist(),
        "sequence_knownness": scores.tolist(),
        "threshold": threshold,
        "target_aux_unknown_recall": target,
        "empirical_aux_unknown_recall": float(np.mean(cal["sequence_knownness"] < threshold)),
        "rejected": rejected.tolist(),
        "novel_cluster": clusters,
        "candidate_sequence_count": sequence_count,
        "noise_image_count": sum(label == -1 for label in clusters),
        "review_budget_images": math.ceil(int(rejected.sum()) * 0.03),
    }
