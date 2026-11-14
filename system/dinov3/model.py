"""DINOv3 joint Ridge-Within classification and unknown rejection model.

Faithfully implements:
- Dual-path hierarchical Ridge head (via SpeciesFeatureCacheRidgeHead)
- Within-class metric geometry reference updater and score_cpu calculation
- top75_radial08 margin MAD and radial MAD gate score aggregation
- Species-balanced Macro90 unknown rejection threshold calibration
- Adaptive sequence classification fusion with optional repeat-content collapsing
- Atomic serialization/deserialization to .neri-model and .zip archives without pickle
- Atomic partial_fit with calibration invalidation and snapshot rollback on error
"""

from __future__ import annotations

from copy import deepcopy
import io
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid
import zipfile

import numpy as np

from .balanced_unknown_calibration import fit_unknown_threshold
from .hierarchical_ridge_head import _metadata_vector, _portable_classes, _sorted_classes, hierarchical_weights
from .species_feature_cache_ridge_head import SpeciesFeatureCacheRidgeHead


def unit(x: np.ndarray) -> np.ndarray:
    """Use the experiment's float32 unit projection, including event means."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


class WithinReferenceUpdater:
    """Accumulate event counts and unit-sum vectors under frozen geometry."""

    def __init__(self, origin: np.ndarray, transform: np.ndarray):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.transform = np.asarray(transform, dtype=np.float64)
        if self.origin.ndim != 1:
            raise ValueError("origin must be a 1D vector")
        if self.transform.ndim != 2 or self.transform.shape[0] != self.origin.shape[0]:
            raise ValueError("transform must be a 2D matrix matching origin dimension")
        self.groups: Dict[Tuple[str, str, str], Tuple[int, np.ndarray]] = {}

    def append(
        self,
        features: np.ndarray,
        labels: Sequence[Any],
        cameras: Sequence[Any],
        sequences: Sequence[Any],
    ) -> None:
        """Incorporate new raw features into event sum-of-units."""
        features_arr = unit(features)
        length = len(features_arr)
        labels_arr = _metadata_vector(labels, "labels", length)
        cameras_arr = _metadata_vector(cameras, "cameras", length)
        sequences_arr = _metadata_vector(sequences, "sequences", length)

        for x, label, camera, sequence in zip(
            features_arr, labels_arr, cameras_arr, sequences_arr
        ):
            key = (str(label), str(camera), str(sequence))
            if key in self.groups:
                count, total = self.groups[key]
                self.groups[key] = (count + 1, total + x.astype(np.float64))
            else:
                self.groups[key] = (1, x.astype(np.float64).copy())

    def bank(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute whitened event centroids, event labels, and event cameras."""
        keys = sorted(self.groups)
        if not keys:
            dim = self.origin.shape[0]
            return (
                np.empty((0, dim), dtype=np.float64),
                np.empty(0, dtype=str),
                np.empty(0, dtype=str),
            )
        events = unit(np.stack([self.groups[k][1] / self.groups[k][0] for k in keys]))
        whitened = unit((events - self.origin) @ self.transform)
        event_labels = np.asarray([k[0] for k in keys], dtype=str)
        event_cameras = np.asarray([k[1] for k in keys], dtype=str)
        return whitened, event_labels, event_cameras


def score_cpu(
    query: np.ndarray,
    bank: np.ndarray,
    labels: np.ndarray,
    cameras: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    """Score query features against reference event bank with local camera pooling."""
    query = np.asarray(query, dtype=np.float32)
    if query.ndim == 1:
        query = query[None, :]
    if len(query) == 0:
        return np.empty((0, len(classes)), dtype=np.float64)

    labels = np.asarray(labels, dtype=str)
    cameras = np.asarray(cameras, dtype=str)
    columns = []

    for label in classes:
        mask = labels == str(label)
        if not np.any(mask):
            columns.append(np.zeros(len(query), dtype=np.float64))
            continue
        b = bank[mask]
        cam = cameras[mask]
        similarity = query @ b.T
        unique_cams = np.unique(cam)
        pooled = np.column_stack([similarity[:, cam == c].max(axis=1) for c in unique_cams])
        k = min(3, pooled.shape[1])
        local = np.sort(pooled, axis=1)[:, -k:].mean(axis=1)
        center = unit(b.mean(axis=0, keepdims=True))[0]
        global_sim = query @ center
        columns.append(0.5 * local + 0.5 * global_sim)

    return np.column_stack(columns)


def top75(
    values: np.ndarray,
    cameras: np.ndarray,
    sequences: Sequence[Any],
) -> np.ndarray:
    """Compute top-75% mean margin grouped by (camera, sequence)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)

    _, inv = np.unique(
        np.column_stack((cameras, sequences)), axis=0, return_inverse=True
    )
    output = np.empty(len(values), dtype=np.float64)
    order = np.argsort(inv, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1, len(order)]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        ii = order[begin:end]
        count = max(1, int(np.ceil(0.75 * len(ii))))
        output[ii] = np.sort(values[ii])[-count:].mean()
    return output


def sequence_mean(
    values: np.ndarray,
    cameras: np.ndarray,
    sequences: Sequence[Any],
) -> np.ndarray:
    """Compute event mean values grouped by (camera, sequence)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)

    _, inverse = np.unique(
        np.column_stack((cameras, sequences)), axis=0, return_inverse=True
    )
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return (sums / counts)[inverse]


def standardize(
    values: np.ndarray,
    known_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Standardize using median and MAD scale derived strictly from known samples."""
    values = np.asarray(values, dtype=np.float64)
    known_mask = np.asarray(known_mask, dtype=bool)
    if not np.any(known_mask):
        raise ValueError("At least one known sample is required to standardize")

    ref = values[known_mask]
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med)))
    scale = max(float(1.4826 * mad), 1e-12)
    standardized = (values - med) / scale
    return standardized, {"median": med, "scale": scale}


def _parse_frame_box(content_str: str) -> Tuple[str, Optional[str]]:
    """Split content identifier into (frame_key, box_key) when formatted as '{frame}:{box_idx}'."""
    if ":" in content_str:
        prefix, suffix = content_str.rsplit(":", 1)
        if prefix and suffix.isdigit() and prefix.lower() not in {
            "cal",
            "single",
            "item",
            "crop",
            "row",
            "test",
        }:
            return prefix, suffix
    return content_str, None


def resolve_target_tracks(
    features: np.ndarray,
    cameras: Sequence[Any],
    sequences: Sequence[Any],
    contents: Sequence[Any],
    box_accepted: Optional[np.ndarray] = None,
    sim_threshold: float = 0.55,
) -> np.ndarray:
    """Resolve per-target trajectories within each (camera, sequence) group.

    - Boxes failing per-box independent acceptance (`box_accepted == False`) are
      isolated into singleton tracks so they never contaminate or borrow from
      valid target trajectories.
    - Distinct boxes from the same photo (`{frame_key}:{box_index}`) are strictly
      prevented from sharing a track and are linked across frames greedily via
      one-to-one unit feature cosine similarity.
    """
    feat_unit = unit(np.asarray(features, dtype=np.float32))
    length = len(feat_unit)
    if length == 0:
        return np.empty(0, dtype=str)

    cameras_arr = np.asarray(cameras, dtype=str)
    sequences_arr = np.asarray(sequences, dtype=str)
    contents_arr = np.asarray(contents, dtype=str)
    acc_arr = (
        np.ones(length, dtype=bool)
        if box_accepted is None
        else np.asarray(box_accepted, dtype=bool)
    )

    resolved = np.empty(length, dtype=object)
    groups: Dict[Tuple[str, str], List[int]] = {}
    for row_idx in range(length):
        cam = str(cameras_arr[row_idx])
        seq = str(sequences_arr[row_idx])
        if not acc_arr[row_idx]:
            resolved[row_idx] = f"{seq}#rej_{row_idx}"
        else:
            groups.setdefault((cam, seq), []).append(row_idx)

    for (_, seq), rows in groups.items():
        if len(rows) <= 1:
            for r in rows:
                resolved[r] = seq
            continue

        frames: List[List[List[int]]] = []
        frame_idx_map: Dict[str, int] = {}
        slot_maps: List[Dict[str, int]] = []
        for pos, r in enumerate(rows):
            c_str = str(contents_arr[r])
            frame_key, box_key = _parse_frame_box(c_str)
            slot_key = f"box:{box_key}" if box_key is not None else f"content:{c_str}:{pos}"
            if frame_key not in frame_idx_map:
                frame_idx_map[frame_key] = len(frames)
                frames.append([])
                slot_maps.append({})
            f_idx = frame_idx_map[frame_key]
            s_map = slot_maps[f_idx]
            if slot_key not in s_map:
                s_map[slot_key] = len(frames[f_idx])
                frames[f_idx].append([])
            frames[f_idx][s_map[slot_key]].append(r)

        has_multi_box_frame = any(len(f_slots) > 1 for f_slots in frames)
        if not has_multi_box_frame:
            for r in rows:
                resolved[r] = seq
            continue

        tracks: List[List[int]] = []
        track_vecs: List[np.ndarray] = []
        for f_slots in frames:
            slot_vecs = [
                unit(feat_unit[s_rows].mean(axis=0, keepdims=True))[0]
                for s_rows in f_slots
            ]
            if not tracks:
                for s_rows, s_vec in zip(f_slots, slot_vecs):
                    tracks.append(list(s_rows))
                    track_vecs.append(s_vec)
                continue

            candidates: List[Tuple[float, int, int]] = []
            for t_idx, t_vec in enumerate(track_vecs):
                for s_idx, s_vec in enumerate(slot_vecs):
                    sim = float(np.dot(t_vec, s_vec))
                    if sim >= sim_threshold:
                        candidates.append((sim, t_idx, s_idx))
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            used_t: set[int] = set()
            used_s: set[int] = set()
            for _, t_idx, s_idx in candidates:
                if t_idx in used_t or s_idx in used_s:
                    continue
                used_t.add(t_idx)
                used_s.add(s_idx)
                tracks[t_idx].extend(f_slots[s_idx])
                track_vecs[t_idx] = unit(
                    feat_unit[tracks[t_idx]].mean(axis=0, keepdims=True)
                )[0]
            for s_idx, s_rows in enumerate(f_slots):
                if s_idx not in used_s:
                    tracks.append(list(s_rows))
                    track_vecs.append(slot_vecs[s_idx])

        if len(tracks) == 1:
            for r in tracks[0]:
                resolved[r] = seq
        else:
            for t_idx, t_rows in enumerate(tracks):
                t_name = f"{seq}#track_{t_idx:02d}"
                for r in t_rows:
                    resolved[r] = t_name

    return resolved.astype(str)


def adaptive_sequence_fuse(
    scores: np.ndarray,
    sites: np.ndarray,
    sequences: np.ndarray,
    contents: np.ndarray,
    tau: float = 1.0,
    consensus_boost: bool = True,
) -> np.ndarray:
    """Posterior classification sequence fusion with repeat-content collapsing."""
    values = np.asarray(scores, dtype=np.float64)
    if len(values) == 0:
        return np.empty((0, values.shape[1] if values.ndim == 2 else 0), dtype=np.float64)
    output = values.copy()
    parsed_contents = [_parse_frame_box(str(contents[row])) for row in range(len(values))]
    frame_boxes: Dict[Tuple[str, str, str], set] = {}
    for row, (site, seq) in enumerate(zip(sites, sequences)):
        frame_key, box_key = parsed_contents[row]
        if box_key is not None:
            frame_boxes.setdefault((str(site), str(seq), frame_key), set()).add(box_key)

    groups: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
    for row, (site, seq) in enumerate(zip(sites, sequences)):
        c_str = str(contents[row])
        frame_key, box_key = parsed_contents[row]
        site_s, seq_s = str(site), str(seq)
        sub_seq = (
            f"{seq_s}#box_{box_key}"
            if box_key is not None and len(frame_boxes.get((site_s, seq_s, frame_key), ())) > 1
            else seq_s
        )
        groups.setdefault((site_s, sub_seq), {}).setdefault(
            c_str, []
        ).append(row)

    for by_content in groups.values():
        rows = list(by_content.values())
        content_scores = np.stack([values[ii].mean(0) for ii in rows])
        n_content = len(rows)
        w_seq = n_content / (n_content + tau)
        if consensus_boost and n_content >= 2:
            votes = content_scores.argmax(axis=1)
            mode_count = int(np.max(np.bincount(votes)))
            if mode_count * 3 >= n_content * 2:
                w_seq = max(w_seq, 0.80)
        group_score = content_scores.mean(0)
        for ii in rows:
            output[ii] = (1.0 - w_seq) * values[ii] + w_seq * group_score
    return output


def _fit_hierarchical_within_geometry(
    raw_features: np.ndarray,
    labels: np.ndarray,
    cameras: np.ndarray,
    sequences: np.ndarray,
    shrinkage: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit hierarchical within-class origin and precision transform from initial bank."""
    features = np.asarray(raw_features, dtype=np.float64)
    dim = features.shape[1]
    weights = hierarchical_weights(labels, cameras, sequences)
    classes, inverse = np.unique(labels, return_inverse=True)
    centers = np.stack([np.average(features[inverse == i], axis=0, weights=weights[inverse == i])
                        for i in range(len(classes))])
    origin = np.average(features, axis=0, weights=weights)
    residual = features - centers[inverse]
    cov = (residual.T * weights) @ residual
    scale = max(float(np.trace(cov) / dim), 1e-12)
    cov = (1.0 - shrinkage) * cov + shrinkage * scale * np.eye(dim)
    precision = np.linalg.inv(cov)
    transform = np.linalg.cholesky(precision)
    return origin, transform


class DinoV3RidgeWithinModel:
    """Joint DINOv3 Ridge classifier and Within metric unknown rejection model."""

    def __init__(
        self,
        ridge: float = 0.01,
        weighting: str = "hierarchical",
        tau: float = 1.0,
        consensus_boost: bool = True,
        target_recall: float = 0.90,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge penalty must be a positive finite float")
        if weighting not in ("hierarchical", "class"):
            raise ValueError("weighting must be 'hierarchical' or 'class'")

        self.ridge = float(ridge)
        self.weighting = str(weighting)
        self.tau = float(tau)
        self.consensus_boost = bool(consensus_boost)
        self.target_recall = float(target_recall)
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be positive and finite")
        if not math.isfinite(self.target_recall) or not 0 < self.target_recall <= 1:
            raise ValueError("target_recall must be in (0, 1]")
        self.metadata: Dict[str, Any] = dict(metadata or {})

        self.ridge_head: Optional[SpeciesFeatureCacheRidgeHead] = None
        self.within_updater: Optional[WithinReferenceUpdater] = None
        self.origin_: Optional[np.ndarray] = None
        self.transform_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_dim_: Optional[int] = None

        self._calibration: Optional[Dict[str, Any]] = None
        self._threshold: Optional[float] = None
        self._box_threshold: Optional[float] = None
        self.is_calibrated: bool = False

    @property
    def dimension(self) -> Optional[int]:
        """Feature embedding dimension."""
        return self.feature_dim_

    @property
    def provenance(self) -> Dict[str, Any]:
        """Provenance metadata dictionary (alias to metadata)."""
        return self.metadata

    @provenance.setter
    def provenance(self, value: Dict[str, Any]) -> None:
        self.metadata = dict(value or {})

    @property
    def calibration(self) -> Optional[Dict[str, Any]]:
        """Active calibration details, or None if uncalibrated."""
        return deepcopy(self._calibration)

    @property
    def threshold(self) -> Optional[float]:
        """Rejection threshold cutoff (high is known)."""
        return self._threshold

    @property
    def box_threshold(self) -> Optional[float]:
        """Per-box independent rejection threshold cutoff (high is known)."""
        return self._box_threshold

    @property
    def classes(self) -> Optional[np.ndarray]:
        """Array of known species class labels."""
        return self.classes_.copy() if self.classes_ is not None else None

    def _require_fitted(self) -> None:
        if self.ridge_head is None or self.within_updater is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    def whiten(self, x: np.ndarray) -> np.ndarray:
        """Whiten features using frozen origin and transform."""
        self._require_fitted()
        x_f64 = np.asarray(x, dtype=np.float64)
        if x_f64.ndim != 2 or x_f64.shape[1] != self.feature_dim_ or not np.isfinite(x_f64).all():
            raise ValueError("features must be a finite matrix matching model dimension")
        return unit((x_f64 - self.origin_) @ self.transform_)

    def fit(
        self,
        raw_features: np.ndarray,
        labels: Sequence[Any],
        cameras: Sequence[Any],
        sequences: Sequence[Any],
        origin: Optional[np.ndarray] = None,
        transform: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DinoV3RidgeWithinModel":
        """Fit joint model from scratch, resetting any prior state and calibration."""
        try:
            raw = np.asarray(raw_features)
            if np.issubdtype(raw.dtype, np.floating):
                features = raw
            else:
                features = raw.astype(np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("raw_features must be a finite 2D numeric matrix") from error

        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("raw_features must be a nonempty finite [N, D] matrix")
        if not np.isfinite(features).all():
            raise ValueError("raw_features must contain only finite values")

        length, dim = features.shape
        labels_arr = _metadata_vector(labels, "labels", length)
        cameras_arr = _portable_classes(_metadata_vector(cameras, "cameras", length))
        sequences_arr = _portable_classes(_metadata_vector(sequences, "sequences", length))
        sorted_classes = _portable_classes(_sorted_classes(labels_arr))
        if len(sorted_classes) < 2:
            raise ValueError("At least two distinct classes are required to fit")

        if (origin is None) != (transform is None):
            raise ValueError("Supply both origin and transform")
        if origin is not None and transform is not None:
            origin_arr = np.asarray(origin, dtype=np.float64)
            transform_arr = np.asarray(transform, dtype=np.float64)
            if origin_arr.shape != (dim,) or transform_arr.shape != (dim, dim):
                raise ValueError("origin and transform shapes must match feature dimension")
            if not np.isfinite(origin_arr).all() or not np.isfinite(transform_arr).all():
                raise ValueError("origin and transform must contain finite values")
        else:
            origin_arr, transform_arr = _fit_hierarchical_within_geometry(
                features, labels_arr, cameras_arr, sequences_arr
            )

        updater = WithinReferenceUpdater(origin_arr, transform_arr)
        updater.append(features, labels_arr, cameras_arr, sequences_arr)
        whitened = unit((np.asarray(features, dtype=np.float64) - origin_arr) @ transform_arr)
        head = SpeciesFeatureCacheRidgeHead(ridge=self.ridge, weighting=self.weighting)
        head.fit(whitened, labels_arr, cameras_arr, sequences_arr)
        new_metadata = dict(self.metadata if metadata is None else metadata)
        json.dumps(new_metadata, allow_nan=False)
        self.origin_ = origin_arr.copy()
        self.transform_ = transform_arr.copy()
        self.feature_dim_ = dim
        self.within_updater = updater
        self.ridge_head = head
        self.classes_ = head.classes_
        self.metadata = new_metadata

        self._calibration = None
        self._threshold = None
        self.is_calibrated = False
        return self

    def partial_fit(
        self,
        raw_features: np.ndarray,
        labels: Sequence[Any],
        cameras: Sequence[Any],
        sequences: Sequence[Any],
    ) -> "DinoV3RidgeWithinModel":
        """Incorporate supplements for old and/or new classes with frozen geometry.

        Guarantees update atomicity: any error reverts the model state cleanly.
        Invalidates rejection calibration, requiring a subsequent call to calibrate().
        """
        self._require_fitted()
        try:
            raw = np.asarray(raw_features)
            if np.issubdtype(raw.dtype, np.floating):
                features = raw
            else:
                features = raw.astype(np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("raw_features must be a finite 2D numeric matrix") from error

        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("raw_features must be a nonempty finite [N, D] matrix")
        if not np.isfinite(features).all():
            raise ValueError("raw_features must contain only finite values")
        if features.shape[1] != self.feature_dim_:
            raise ValueError("feature dimension must remain unchanged")

        length = features.shape[0]
        labels_arr = _metadata_vector(labels, "labels", length)
        cameras_arr = _portable_classes(_metadata_vector(cameras, "cameras", length))
        sequences_arr = _portable_classes(_metadata_vector(sequences, "sequences", length))

        # Snapshot state for transaction atomicity
        prev_ridge_head = deepcopy(self.ridge_head)
        prev_within_updater = deepcopy(self.within_updater)
        prev_classes = self.classes_.copy() if self.classes_ is not None else None
        prev_calibrated = self.is_calibrated
        prev_calibration = deepcopy(self._calibration)
        prev_threshold = self._threshold
        prev_box_threshold = self._box_threshold

        try:
            whitened = self.whiten(features)
            self.ridge_head.partial_fit(whitened, labels_arr, cameras_arr, sequences_arr)
            self.within_updater.append(features, labels_arr, cameras_arr, sequences_arr)
            self.classes_ = self.ridge_head.classes_

            # Invalidate calibration on successful update
            self.is_calibrated = False
            self._calibration = None
            self._threshold = None
            self._box_threshold = None
        except Exception:
            # Rollback to exact snapshot on any failure
            self.ridge_head = prev_ridge_head
            self.within_updater = prev_within_updater
            self.classes_ = prev_classes
            self.is_calibrated = prev_calibrated
            self._calibration = prev_calibration
            self._threshold = prev_threshold
            self._box_threshold = prev_box_threshold
            raise

        return self

    def scores(
        self,
        raw_features: np.ndarray,
        adapted_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute Ridge classification scores (N, C)."""
        self._require_fitted()
        query_raw = adapted_features if adapted_features is not None else raw_features
        query_arr = np.asarray(query_raw, dtype=np.float64)
        if query_arr.ndim == 1:
            query_arr = query_arr[None, :]
        if query_arr.ndim != 2 or query_arr.shape[1] != self.feature_dim_:
            raise ValueError("feature dimension must match model dimension")
        whitened = self.whiten(query_arr)
        return self.ridge_head.scores(whitened)

    def _compute_within_scores(
        self,
        raw_features: np.ndarray,
        adapted_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute Within metric scores (N, C) using score_cpu against the event bank."""
        self._require_fitted()
        query_raw = adapted_features if adapted_features is not None else raw_features
        query_arr = np.asarray(query_raw, dtype=np.float64)
        if query_arr.ndim == 1:
            query_arr = query_arr[None, :]
        if query_arr.shape[1] != self.feature_dim_:
            raise ValueError("feature dimension must match model dimension")
        whitened = self.whiten(query_arr)
        b, el, ec = self.within_updater.bank()
        return score_cpu(whitened, b, el, ec, self.classes_)

    def calibrate(
        self,
        raw_features: np.ndarray,
        labels: Sequence[Any],
        cameras: Sequence[Any],
        sequences: Sequence[Any],
        sites: Optional[Sequence[Any]] = None,
        contents: Optional[Sequence[Any]] = None,
        adapted_features: Optional[np.ndarray] = None,
        unknown_mask: Optional[Sequence[bool]] = None,
        target_recall: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calibrate rejection gate using known standardization and Unknown Macro90.

        Parameters:
            raw_features: (N, D) query features
            labels: (N,) species labels
            cameras: (N,) camera/location IDs used for gate grouping
            sequences: (N,) event/sequence IDs used for top75 aggregation
            sites: (N,) optional location/camera component IDs
            contents: (N,) optional content/image hash IDs
            adapted_features: (N, D) optional context-adapted feature vectors
            unknown_mask: (N,) bool array indicating auxiliary unknown rows
            target_recall: target macro recall for unknown rejection (default self.target_recall)
        """
        self._require_fitted()
        raw = np.asarray(raw_features, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != self.feature_dim_ or not np.isfinite(raw).all():
            raise ValueError("raw_features must match model dimension")
        if adapted_features is not None and (np.shape(adapted_features) != raw.shape or not np.isfinite(adapted_features).all()):
            raise ValueError("adapted_features must be finite and align with raw_features")
        length = raw.shape[0]
        if length == 0:
            raise ValueError("Calibration requires at least one sample")

        labels_arr = _metadata_vector(labels, "labels", length)
        cameras_arr = _portable_classes(_metadata_vector(cameras, "cameras", length))
        sequences_arr = _portable_classes(_metadata_vector(sequences, "sequences", length))

        if unknown_mask is None:
            u_mask = ~np.isin(labels_arr, self.classes_)
        else:
            u_mask = np.asarray(unknown_mask, dtype=bool)
            if u_mask.shape != (length,):
                raise ValueError("unknown_mask must align with features")

        k_mask = ~u_mask
        if not np.any(k_mask):
            raise ValueError("Calibration requires at least one known sample to standardize")
        if not np.any(u_mask):
            raise ValueError("Calibration requires at least one unknown sample to fit Macro90 cutoff")

        target = float(target_recall if target_recall is not None else self.target_recall)
        if not math.isfinite(target) or not (0 < target <= 1):
            raise ValueError("target_recall must be in (0, 1]")
        if not np.array_equal(u_mask, ~np.isin(labels_arr, self.classes_)):
            raise ValueError("unknown_mask must match the current known/unknown class roles")

        within_mat = self._compute_within_scores(raw, adapted_features=adapted_features)
        raw_top = np.sort(within_mat, axis=1)[:, -2:]
        margin = 2 * raw_top[:, 1] - raw_top[:, 0]

        query_feat = adapted_features if adapted_features is not None else raw
        z = (np.asarray(query_feat, dtype=np.float64) - self.origin_) @ self.transform_
        norm = np.linalg.norm(z, axis=1)

        # Stage 1: Per-box independent gate standardization and calibration
        z_box_margin, box_m_meta = standardize(margin, k_mask)
        z_box_radial, box_r_meta = standardize(norm, k_mask)
        box_known_score = z_box_margin - 0.08 * z_box_radial

        unknown_labels = labels_arr[u_mask]
        box_target = min(0.80, (0.80 / 0.90) * target)
        box_fit_res = fit_unknown_threshold(
            box_known_score[u_mask], unknown_labels, target=box_target, weighting="species"
        )
        box_threshold = float(box_fit_res["threshold"])
        box_accepted = box_known_score >= box_threshold

        # Stage 2: Same-target trajectory resolution and sequence gate calibration
        contents_arr = (
            _portable_classes(_metadata_vector(contents, "contents", length))
            if contents is not None
            else np.asarray([f"item_{i}" for i in range(length)], dtype=str)
        )
        resolved_tracks = resolve_target_tracks(
            query_feat, cameras_arr, sequences_arr, contents_arr, box_accepted=box_accepted
        )
        pooled_margin = top75(margin, cameras_arr, resolved_tracks)
        radial = sequence_mean(norm, cameras_arr, resolved_tracks)

        z_margin, m_meta = standardize(pooled_margin, k_mask)
        z_radial, r_meta = standardize(radial, k_mask)
        combined_score = z_margin - 0.08 * z_radial

        sentinel = float(np.min(combined_score)) - 1.0
        effective_u_scores = np.where(box_accepted[u_mask], combined_score[u_mask], sentinel)
        fit_res = fit_unknown_threshold(
            effective_u_scores, unknown_labels, target=target, weighting="species"
        )
        if float(fit_res["threshold"]) <= sentinel or not np.any(box_accepted[u_mask]):
            fit_res = fit_unknown_threshold(
                combined_score[u_mask], unknown_labels, target=target, weighting="species"
            )

        self._threshold = float(fit_res["threshold"])
        self._box_threshold = box_threshold
        self._calibration = {
            "threshold": self._threshold,
            "target": target,
            "margin": m_meta,
            "radial": r_meta,
            "fit": fit_res,
            "box_threshold": self._box_threshold,
            "box_target": box_target,
            "box_margin": box_m_meta,
            "box_radial": box_r_meta,
            "box_fit": box_fit_res,
        }
        self.is_calibrated = True
        return deepcopy(self._calibration)

    def recalibrate(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for calibrate()."""
        return self.calibrate(*args, **kwargs)

    def predict(
        self,
        raw_features: np.ndarray,
        cameras: Sequence[Any],
        sequences: Sequence[Any],
        sites: Optional[Sequence[Any]] = None,
        contents: Optional[Sequence[Any]] = None,
        adapted_features: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Perform joint prediction, adaptive sequence fusion, and unknown gating.

        Returns:
            dict containing:
                'classes': np.ndarray of known class names
                'predictions': np.ndarray of string class predictions
                'accepted': np.ndarray of booleans (known_score >= threshold)
                'known_score': np.ndarray of float64 gate scores
                'threshold': float cutoff
                'ridge_scores': np.ndarray (N, C) raw ridge scores
                'within_scores': np.ndarray (N, C) raw within scores
                'fused_scores': np.ndarray (N, C) sequence-fused classification scores
        """
        self._require_fitted()
        if not self.is_calibrated or self._calibration is None:
            raise RuntimeError(
                "Model rejection gate is not calibrated. Call calibrate() before predict()."
            )

        raw = np.asarray(raw_features, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != self.feature_dim_ or not np.isfinite(raw).all():
            raise ValueError("raw_features must match model dimension")
        if adapted_features is not None and (np.shape(adapted_features) != raw.shape or not np.isfinite(adapted_features).all()):
            raise ValueError("adapted_features must be finite and align with raw_features")
        length = raw.shape[0]
        if length == 0:
            box_thresh = float(
                self._box_threshold if self._box_threshold is not None else self._threshold
            )
            return {
                "classes": self.classes_.copy(),
                "predictions": np.empty(0, dtype=self.classes_.dtype),
                "accepted": np.empty(0, dtype=bool),
                "known_score": np.empty(0, dtype=np.float64),
                "threshold": float(self._threshold),
                "box_known_score": np.empty(0, dtype=np.float64),
                "box_threshold": box_thresh,
                "box_accepted": np.empty(0, dtype=bool),
                "sequence_known_score": np.empty(0, dtype=np.float64),
                "sequence_threshold": float(self._threshold),
                "sequence_accepted": np.empty(0, dtype=bool),
                "within_margins": np.empty(0, dtype=np.float64),
                "whitened_norms": np.empty(0, dtype=np.float64),
                "resolved_tracks": np.empty(0, dtype=str),
                "ridge_scores": np.empty((0, len(self.classes_)), dtype=np.float64),
                "within_scores": np.empty((0, len(self.classes_)), dtype=np.float64),
                "fused_scores": np.empty((0, len(self.classes_)), dtype=np.float64),
            }

        cameras_arr = _portable_classes(_metadata_vector(cameras, "cameras", length))
        sequences_arr = _portable_classes(_metadata_vector(sequences, "sequences", length))
        sites_arr = (
            _portable_classes(_metadata_vector(sites, "sites", length))
            if sites is not None
            else cameras_arr
        )
        contents_arr = (
            _portable_classes(_metadata_vector(contents, "contents", length))
            if contents is not None
            else np.asarray([f"item_{i}" for i in range(length)], dtype=str)
        )

        within_mat = self._compute_within_scores(raw, adapted_features=adapted_features)
        raw_top = np.sort(within_mat, axis=1)[:, -2:]
        margin = 2 * raw_top[:, 1] - raw_top[:, 0]

        query_feat = adapted_features if adapted_features is not None else raw
        z = (np.asarray(query_feat, dtype=np.float64) - self.origin_) @ self.transform_
        norm = np.linalg.norm(z, axis=1)

        box_m_meta = self._calibration.get("box_margin", self._calibration["margin"])
        box_r_meta = self._calibration.get("box_radial", self._calibration["radial"])
        box_threshold = float(
            self._calibration.get(
                "box_threshold",
                self._box_threshold if self._box_threshold is not None else self._threshold,
            )
        )
        z_box_margin = (margin - box_m_meta["median"]) / box_m_meta["scale"]
        z_box_radial = (norm - box_r_meta["median"]) / box_r_meta["scale"]
        box_known_score = z_box_margin - 0.08 * z_box_radial
        box_accepted = box_known_score >= box_threshold

        resolved_tracks = resolve_target_tracks(
            query_feat, cameras_arr, sequences_arr, contents_arr, box_accepted=box_accepted
        )

        ridge_mat = self.scores(raw, adapted_features=adapted_features)
        fused_scores = adaptive_sequence_fuse(
            ridge_mat,
            sites_arr,
            resolved_tracks,
            contents_arr,
            tau=self.tau,
            consensus_boost=self.consensus_boost,
        )
        pred_indices = fused_scores.argmax(axis=1)
        predictions = self.classes_[pred_indices]

        pooled_margin = top75(margin, cameras_arr, resolved_tracks)
        radial = sequence_mean(norm, cameras_arr, resolved_tracks)

        m_meta = self._calibration["margin"]
        r_meta = self._calibration["radial"]
        z_margin = (pooled_margin - m_meta["median"]) / m_meta["scale"]
        z_radial = (radial - r_meta["median"]) / r_meta["scale"]
        sequence_known_score = z_margin - 0.08 * z_radial
        sequence_threshold = float(self._threshold)
        sequence_accepted = sequence_known_score >= sequence_threshold

        accepted = box_accepted & sequence_accepted
        box_delta = box_known_score - box_threshold
        seq_delta = sequence_known_score - sequence_threshold
        effective_delta = np.where(
            accepted,
            0.5 * box_delta + 0.5 * seq_delta,
            np.minimum(box_delta, seq_delta),
        )
        known_score = sequence_threshold + effective_delta

        return {
            "classes": self.classes_.copy(),
            "predictions": predictions,
            "accepted": accepted,
            "known_score": known_score,
            "threshold": sequence_threshold,
            "box_known_score": box_known_score,
            "box_threshold": box_threshold,
            "box_accepted": box_accepted,
            "sequence_known_score": sequence_known_score,
            "sequence_threshold": sequence_threshold,
            "sequence_accepted": sequence_accepted,
            "within_margins": margin,
            "whitened_norms": norm,
            "resolved_tracks": resolved_tracks,
            "ridge_scores": ridge_mat,
            "within_scores": within_mat,
            "fused_scores": fused_scores,
        }

    def save(self, path: Union[str, Path, os.PathLike]) -> None:
        """Atomically serialize model parameters, feature caches, and calibration.

        Produces a clean zip archive (.neri-model or .zip) containing no pickle files.
        """
        self._require_fitted()
        target_path = Path(path).resolve()
        parent = target_path.parent
        parent.mkdir(parents=True, exist_ok=True)

        temp_name = f".{target_path.name}.tmp_{uuid.uuid4().hex}"
        temp_path = parent / temp_name

        try:
            with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                manifest_data = {
                    "head_type": "ridge_within",
                    "schema_version": 1,
                    "backend": "dinov3",
                    "dimension": int(self.feature_dim_),
                    "is_calibrated": bool(self.is_calibrated),
                    "threshold": float(self._threshold) if self._threshold is not None else None,
                    "classes": [str(c) for c in self.classes_],
                    "ridge": float(self.ridge),
                    "weighting": str(self.weighting),
                    "tau": float(self.tau),
                    "consensus_boost": bool(self.consensus_boost),
                    "target_recall": float(self.target_recall),
                    "metadata": self.metadata,
                }
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                )

                # Geometry
                geo_bytes = io.BytesIO()
                np.savez_compressed(
                    geo_bytes,
                    origin=self.origin_,
                    transform=self.transform_,
                )
                archive.writestr("geometry.npz", geo_bytes.getvalue())

                # Ridge head (pure npz without pickle)
                ridge_bytes = io.BytesIO()
                self.ridge_head.save(ridge_bytes)
                archive.writestr("ridge_head.npz", ridge_bytes.getvalue())

                # Within updater groups
                groups_keys = list(self.within_updater.groups.keys())
                u_labels = np.asarray([k[0] for k in groups_keys], dtype=str)
                u_cameras = np.asarray([k[1] for k in groups_keys], dtype=str)
                u_sequences = np.asarray([k[2] for k in groups_keys], dtype=str)
                u_counts = np.asarray(
                    [self.within_updater.groups[k][0] for k in groups_keys], dtype=np.int64
                )
                u_totals = (
                    np.stack([self.within_updater.groups[k][1] for k in groups_keys], axis=0)
                    if groups_keys
                    else np.empty((0, self.feature_dim_), dtype=np.float64)
                )

                within_bytes = io.BytesIO()
                np.savez_compressed(
                    within_bytes,
                    labels=u_labels,
                    cameras=u_cameras,
                    sequences=u_sequences,
                    counts=u_counts,
                    totals=u_totals,
                )
                archive.writestr("within_updater.npz", within_bytes.getvalue())

                # Calibration state if active
                if self.is_calibrated and self._calibration is not None:
                    cal_data = {
                        "threshold": float(self._calibration["threshold"]),
                        "target": float(self._calibration["target"]),
                        "margin": {
                            "median": float(self._calibration["margin"]["median"]),
                            "scale": float(self._calibration["margin"]["scale"]),
                        },
                        "radial": {
                            "median": float(self._calibration["radial"]["median"]),
                            "scale": float(self._calibration["radial"]["scale"]),
                        },
                        "fit": {
                            k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                            for k, v in self._calibration["fit"].items()
                        },
                    }
                    if "box_threshold" in self._calibration:
                        cal_data["box_threshold"] = float(self._calibration["box_threshold"])
                    if "box_target" in self._calibration:
                        cal_data["box_target"] = float(self._calibration["box_target"])
                    if "box_margin" in self._calibration:
                        cal_data["box_margin"] = {
                            "median": float(self._calibration["box_margin"]["median"]),
                            "scale": float(self._calibration["box_margin"]["scale"]),
                        }
                    if "box_radial" in self._calibration:
                        cal_data["box_radial"] = {
                            "median": float(self._calibration["box_radial"]["median"]),
                            "scale": float(self._calibration["box_radial"]["scale"]),
                        }
                    if "box_fit" in self._calibration and isinstance(self._calibration["box_fit"], dict):
                        cal_data["box_fit"] = {
                            k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                            for k, v in self._calibration["box_fit"].items()
                        }
                    archive.writestr(
                        "calibration.json",
                        json.dumps(cal_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    )

            # Atomic replace
            os.replace(temp_path, target_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    @classmethod
    def load(cls, path: Union[str, Path, os.PathLike]) -> "DinoV3RidgeWithinModel":
        """Load joint model from archive without pickle and reconstruct exact parameters."""
        source_path = Path(path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {source_path}")

        with zipfile.ZipFile(source_path, mode="r") as archive:
            namelist = set(archive.namelist())
            if "manifest.json" not in namelist or "geometry.npz" not in namelist:
                raise ValueError("Archive is missing required model files")

            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("Model manifest must be an object")
            if manifest.get("head_type") != "ridge_within":
                raise ValueError(f"Unsupported head_type: {manifest.get('head_type')}")
            if manifest.get("schema_version") != 1:
                raise ValueError(f"Unsupported schema_version: {manifest.get('schema_version')}")
            if manifest.get("backend") != "dinov3":
                raise ValueError(f"Unsupported backend: {manifest.get('backend')}")
            expected = {"manifest.json", "geometry.npz", "ridge_head.npz", "within_updater.npz"}
            if manifest.get("is_calibrated") is True:
                expected.add("calibration.json")
            if (type(manifest.get("is_calibrated")) is not bool or namelist != expected
                    or len(archive.namelist()) != len(namelist)):
                raise ValueError("Invalid archive fields or calibrated state")
            if type(manifest.get("dimension")) is not int or manifest["dimension"] <= 0:
                raise ValueError("Invalid feature dimension")

            # Instantiate model
            model = cls(
                ridge=manifest["ridge"],
                weighting=manifest["weighting"],
                tau=manifest.get("tau", 1.0),
                consensus_boost=manifest.get("consensus_boost", True),
                target_recall=manifest.get("target_recall", 0.90),
                metadata=manifest.get("metadata", {}),
            )
            model.feature_dim_ = int(manifest["dimension"])

            # Geometry
            with np.load(io.BytesIO(archive.read("geometry.npz")), allow_pickle=False) as geo:
                model.origin_ = geo["origin"].astype(np.float64)
                model.transform_ = geo["transform"].astype(np.float64)
            dimension = model.feature_dim_
            if (model.origin_.shape != (dimension,) or model.transform_.shape != (dimension, dimension)
                    or not np.isfinite(model.origin_).all() or not np.isfinite(model.transform_).all()):
                raise ValueError("Invalid geometry shape or finite values")

            # Ridge head
            if "ridge_head.npz" not in namelist:
                raise ValueError("Archive is missing ridge_head.npz")
            model.ridge_head = SpeciesFeatureCacheRidgeHead.load(
                io.BytesIO(archive.read("ridge_head.npz"))
            )
            model.classes_ = model.ridge_head.classes_
            if (model.ridge_head.feature_dim_ != dimension
                    or manifest.get("classes") != list(map(str, model.classes_))
                    or model.ridge_head.ridge != model.ridge
                    or model.ridge_head.weighting != model.weighting):
                raise ValueError("Ridge state does not match model metadata")

            # Within updater
            if "within_updater.npz" not in namelist:
                raise ValueError("Archive is missing within_updater.npz")
            with np.load(io.BytesIO(archive.read("within_updater.npz")), allow_pickle=False) as u_data:
                updater = WithinReferenceUpdater(model.origin_, model.transform_)
                u_labels = u_data["labels"]
                u_cameras = u_data["cameras"]
                u_sequences = u_data["sequences"]
                u_counts = u_data["counts"]
                u_totals = u_data["totals"]
                count = len(u_labels)
                if (u_labels.ndim != 1 or u_cameras.shape != (count,) or u_sequences.shape != (count,)
                        or u_counts.shape != (count,) or u_counts.dtype.kind not in "iu"
                        or np.any(u_counts <= 0) or u_totals.shape != (count, dimension)
                        or not np.isfinite(u_totals).all()):
                    raise ValueError("Invalid Within counts, totals or metadata shapes")
                for i in range(len(u_labels)):
                    key = (str(u_labels[i]), str(u_cameras[i]), str(u_sequences[i]))
                    if key in updater.groups:
                        raise ValueError("Duplicate Within event key")
                    updater.groups[key] = (int(u_counts[i]), u_totals[i].astype(np.float64))
                if set(map(str, model.classes_)) != set(map(str, u_labels)):
                    raise ValueError("Within classes do not match Ridge classes")
                for label, samples in zip(model.classes_, model.ridge_head.class_counts_):
                    if sum(n for (c, _, _), (n, total) in updater.groups.items() if c == str(label)) != samples:
                        raise ValueError("Within counts do not match Ridge support counts")
                model.within_updater = updater

            # Calibration
            if manifest.get("is_calibrated") and "calibration.json" in namelist:
                cal_data = json.loads(archive.read("calibration.json").decode("utf-8"))
                if not isinstance(cal_data, dict):
                    raise ValueError("Calibration must be an object")
                model._threshold = float(cal_data["threshold"])
                model._box_threshold = float(cal_data.get("box_threshold", model._threshold))
                model._calibration = cal_data
                model.is_calibrated = True
                if (not math.isfinite(model._threshold) or manifest.get("threshold") != model._threshold
                        or not math.isfinite(model._box_threshold)
                        or not math.isfinite(cal_data["target"]) or not 0 < cal_data["target"] <= 1):
                    raise ValueError("Invalid calibration threshold or target")
                if "box_target" in cal_data and (
                    not math.isfinite(cal_data["box_target"]) or not 0 < cal_data["box_target"] <= 1
                ):
                    raise ValueError("Invalid calibration box_target")
                for name in ("margin", "radial"):
                    stats = cal_data[name]
                    if (not math.isfinite(stats["median"]) or not math.isfinite(stats["scale"])
                            or stats["scale"] <= 0):
                        raise ValueError("Invalid calibration normalization scale or median")
                for name in ("box_margin", "box_radial"):
                    if name in cal_data:
                        stats = cal_data[name]
                        if (not math.isfinite(stats["median"]) or not math.isfinite(stats["scale"])
                                or stats["scale"] <= 0):
                            raise ValueError("Invalid box calibration normalization scale or median")
            else:
                if manifest.get("threshold") is not None:
                    raise ValueError("Uncalibrated model cannot declare a threshold")
                model._threshold = None
                model._box_threshold = None
                model._calibration = None
                model.is_calibrated = False

        return model
