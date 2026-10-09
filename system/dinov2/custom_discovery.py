"""Calibrated Memory and density gate followed by event-balanced HDBSCAN.

Inputs are frozen, unit DINOv2 event embeddings. Calibration rows must be
independent known examples; query rows form the mixed batch being discovered.
No query labels are read or used to fit the gate or clusters.
"""
from __future__ import annotations

import numpy as np

from .clustering import DEFAULT_MIN_CLUSTER_SIZE, hdbscan_labels
from .classifier import DinoV2Classifier
from .memory_classifier import MemoryDinoV2Classifier, _normalize_rows


TARGET_FRR = 0.05
DENSITY_NEIGHBORS = 5


def _threshold(scores: np.ndarray) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Known calibration scores must be nonempty and finite")
    rank = int(np.floor((len(values) + 1) * TARGET_FRR))
    return float(values[rank - 1]) if rank else float(-np.finfo(np.float64).max)


def _local_similarity(
    calibration: np.ndarray, query: np.ndarray, calibration_sequence: np.ndarray,
    query_sequence: np.ndarray, calibration_camera: np.ndarray | None,
    query_camera: np.ndarray | None, *, batch_size: int = 128,
) -> np.ndarray:
    """Fifth eligible DU neighbor; exclude the same sequence and camera."""
    all_features = np.concatenate((calibration, query))
    all_sequence = np.concatenate((calibration_sequence, query_sequence))
    all_camera = None if query_camera is None else np.concatenate((calibration_camera, query_camera))
    output = np.empty(len(all_features), dtype=np.float32)
    for start in range(0, len(all_features), batch_size):
        stop = min(start + batch_size, len(all_features))
        similarity = all_features[start:stop] @ query.T
        similarity[all_sequence[start:stop, None] == query_sequence[None, :]] = -np.inf
        if all_camera is not None:
            similarity[all_camera[start:stop, None] == query_camera[None, :]] = -np.inf
        if similarity.shape[1] < DENSITY_NEIGHBORS:
            raise ValueError("At least five eligible query neighbors are required")
        fifth = np.partition(similarity, -DENSITY_NEIGHBORS, axis=1)[:, -DENSITY_NEIGHBORS]
        if not np.isfinite(fifth).all():
            raise ValueError("Some rows have fewer than five neighbors outside their sequence/camera")
        output[start:stop] = fifth
    return output


def _event_labels(features, sequence_ids, *, min_cluster_size, min_samples):
    """One mean unit vector per sequence; map event labels back to images."""
    sequences, inverse = np.unique(sequence_ids, return_inverse=True)
    events = np.stack([features[inverse == index].mean(axis=0) for index in range(len(sequences))])
    events = _normalize_rows(events)
    labels = hdbscan_labels(events, min_cluster_size=min_cluster_size, min_samples=min_samples)
    return labels[inverse]


def discover(
    checkpoint, calibration_features, query_features, calibration_sequence_ids,
    query_sequence_ids, *, min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = None, calibration_camera_ids=None,
    query_camera_ids=None,
) -> dict[str, object]:
    """Assign accepted rows to known species and rejected rows to novel IDs.

    Camera exclusion is used only when both camera arrays are supplied. The
    checkpoint's 4% deployment threshold is not reused as a 5% discovery gate.
    """
    if checkpoint.head_type not in {"memory", "memory_no_centroid", "memory_within_seq"}:
        raise ValueError("Custom discovery requires a Memory checkpoint")
    calibration = DinoV2Classifier._validate_features(calibration_features)
    query = DinoV2Classifier._validate_features(query_features)
    if not len(calibration) or not len(query):
        raise ValueError("Calibration and query batches must be nonempty")
    cseq, qseq = np.asarray(calibration_sequence_ids).astype(str), np.asarray(query_sequence_ids).astype(str)
    if cseq.shape != (len(calibration),) or qseq.shape != (len(query),):
        raise ValueError("Sequence IDs must align with feature rows")
    if any(not value.strip() for value in np.concatenate((cseq, qseq))):
        raise ValueError("Sequence IDs must be nonempty")
    if (calibration_camera_ids is None) != (query_camera_ids is None):
        raise ValueError("Both camera arrays are required for cross-camera density")
    ccam = qcam = None
    if calibration_camera_ids is not None:
        ccam, qcam = np.asarray(calibration_camera_ids).astype(str), np.asarray(query_camera_ids).astype(str)
        if ccam.shape != cseq.shape or qcam.shape != qseq.shape:
            raise ValueError("Camera IDs must align with feature rows")
        if any(not value.strip() for value in np.concatenate((ccam, qcam))):
            raise ValueError("Camera IDs must be nonempty")
    # Validate HDBSCAN settings even when the gate rejects no rows.
    hdbscan_labels(np.empty((0, 768), dtype=np.float32),
                   min_cluster_size=min_cluster_size, min_samples=min_samples)
    center = checkpoint.feature_center.numpy()
    cal_centered = _normalize_rows(calibration - center[None, :])
    query_centered = _normalize_rows(query - center[None, :])
    classifier = MemoryDinoV2Classifier(checkpoint)
    predictions = []
    knownness = []
    for batch in (calibration, query):
        batch_predictions = [prediction for start in range(0, len(batch), 128)
                             for prediction in classifier.classify_features(batch[start:start + 128])]
        predictions.append(np.asarray([prediction.best_known_species for prediction in batch_predictions]))
        knownness.append(np.asarray([prediction.known_score for prediction in batch_predictions]))
    centered = np.concatenate((cal_centered, query_centered))
    known_similarity = np.empty(len(centered), dtype=np.float32)
    for start in range(0, len(centered), 128):
        known_similarity[start:start + 128] = (centered[start:start + 128] @ checkpoint.features.T).max(axis=1)
    local_similarity = _local_similarity(cal_centered, query_centered, cseq, qseq, ccam, qcam)
    ratio = np.log(np.maximum(1.0 - known_similarity, 1e-6) / np.maximum(1.0 - local_similarity, 1e-6))
    ncal = len(calibration)
    standardized = []
    for values in (ratio, np.concatenate(knownness)):
        median = np.median(values[:ncal])
        scale = max(float(1.4826 * np.median(np.abs(values[:ncal] - median))), 1e-6)
        standardized.append((values - median) / scale)
    known_score = -(standardized[0] - standardized[1])
    threshold = _threshold(known_score[:ncal])
    rejected = known_score[ncal:] < threshold
    labels = np.full(len(query), -1, dtype=np.int64)
    count = int(rejected.sum())
    if count:
        event_labels = _event_labels(
            query_centered[rejected], qseq[rejected],
            min_cluster_size=min_cluster_size, min_samples=min_samples,
        )
        # -1 is reserved for accepted known rows; -2 identifies novel noise.
        labels[rejected] = np.where(event_labels >= 0, event_labels, -2)
    return {
        "known_species": predictions[1].tolist(),
        "novel_cluster": labels.tolist(),
        "threshold": threshold,
        "target_known_calibration_frr": TARGET_FRR,
        "empirical_calibration_frr": float(np.mean(known_score[:ncal] < threshold)),
        "rejected_count": count,
        "occupied_clusters": int(len(np.unique(labels[labels >= 0]))),
        "noise_count": int(np.sum(labels == -2)),
        "rejected_mask": rejected.tolist(),
        "algorithm": "hdbscan",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_cluster_size if min_samples is None else min_samples,
        "cross_camera_density": qcam is not None,
        "density_neighbors": DENSITY_NEIGHBORS,
    }
