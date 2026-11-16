"""Pure NumPy/Python causal camera-history context adapter for DINOv3.

Faithfully implements the frozen causal history protocol without PyTorch nn dependencies:
- HistoryConfig: days=7, max_history=64, short_term=0, seed=20261105
- FastCausalHistoryIndex: strictly past, same-camera, deduplicated object history
- adapt_with_history: temperature=0.1, alpha=0.1 matching adapt_numpy
- Event grouping: camera + 30-minute events; missing timestamp creates independent sequence
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from heapq import heapify, heappop, heappush
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class HistoryConfig:
    days: float = 7.0
    max_history: int = 64
    short_term: int = 0
    seed: int = 20261105

    def __post_init__(self):
        if not math.isfinite(self.days) or self.days <= 0:
            raise ValueError("days must be positive and finite")
        if self.max_history < 0 or self.short_term < 0:
            raise ValueError("history capacities must be nonnegative")


@dataclass(frozen=True)
class HistoryBatch:
    """Global feature-row indices; -1 padding represents an invalid entry."""

    indices: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        return self.indices >= 0

    def numpy(self) -> np.ndarray:
        return self.indices

    def __getitem__(self, item: Any) -> Any:
        return self.indices[item]

    def __len__(self) -> int:
        return len(self.indices)


def _first(record: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and (not isinstance(value, str) or value.strip()):
            return value
    return None


def parse_capture_time(value: Any) -> datetime | None:
    """Parse capture timestamp from string, datetime, or numeric epoch."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, (int, float)):
        if math.isnan(value) or not math.isfinite(value):
            return None
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "-1", ""}:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def camera_id_for_path(path: str | Path, root: str | Path | None = None) -> str:
    """Derive stable camera ID from file directory relative to input root."""
    source = Path(path).expanduser()
    camera_directory = source.parent.resolve()
    if root is not None:
        try:
            resolved_root = Path(root).expanduser().resolve()
            relative = source.resolve().relative_to(resolved_root)
            camera_directory = (
                (resolved_root / relative.parts[0]).resolve()
                if len(relative.parts) > 1
                else resolved_root
            )
        except (OSError, ValueError):
            pass
    normalized = os.path.normcase(str(camera_directory)).replace("\\", "/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class _HistorySequence:
    """Time-sorted rows with a range-minimum tree for frozen random ranks."""

    def __init__(self, rows, times, ranks, identities):
        self.rows = rows
        self.times = [times[i] for i in rows]
        self.ranks = ranks
        self.identities = identities
        self.size = 1 << (len(rows) - 1).bit_length()
        self.tree = [-1] * (2 * self.size)
        self.tree[self.size:self.size + len(rows)] = rows
        for node in range(self.size - 1, 0, -1):
            children = [i for i in self.tree[2 * node:2 * node + 2] if i >= 0]
            self.tree[node] = min(children, key=lambda i: (ranks[i], i)) if children else -1

    def best(self, start, stop, seen):
        """Return the lowest-ranked eligible row without scanning the window."""
        left, right = start + self.size, stop + self.size
        nodes = []
        while left < right:
            if left & 1:
                nodes.append(left)
                left += 1
            if right & 1:
                right -= 1
                nodes.append(right)
            left //= 2
            right //= 2
        if not nodes:
            return None
        row = min((self.tree[n] for n in nodes), key=lambda i: (self.ranks[i], i))
        if seen.isdisjoint(self.identities[row]):
            return row
        # Only descend into additional tree nodes when content aliases exclude
        # a range's winner. The heap preserves the exact (rank, row) ordering.
        pending = [(self.ranks[self.tree[n]], self.tree[n], n) for n in nodes]
        heapify(pending)
        while pending:
            _, row, node = heappop(pending)
            if seen.isdisjoint(self.identities[row]):
                return row
            if node < self.size:
                for child in (2 * node, 2 * node + 1):
                    candidate = self.tree[child]
                    if candidate >= 0:
                        heappush(pending, (self.ranks[candidate], candidate, child))
        return None


class FastCausalHistoryIndex:
    """Pure NumPy/Python causal history index.

    Deduplicated object history restricted strictly to the past within config.days.
    Missing capture times yield empty history (no past context and excluded as candidates).
    """

    identity_keys = ("content_hash", "content_id", "crop_path", "file", "path", "image_id")

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        config: HistoryConfig | None = None,
        candidate_indices: Sequence[int] | None = None,
    ):
        self.config = config or HistoryConfig()
        self.cameras: list[str] = []
        for r in records:
            cam = _first(r, ("camera_component", "camera_id", "location"))
            self.cameras.append(str(cam if cam is not None else "single"))

        self.sequences: list[str] = []
        for i, r in enumerate(records):
            seq = _first(r, ("seq_id", "sequence", "sequence_id"))
            self.sequences.append(str(seq if seq is not None else f"seq_{i}"))

        self.times: list[float | None] = []
        for r in records:
            raw_time = _first(r, ("datetime", "date_taken", "captured_at", "timestamp"))
            dt = parse_capture_time(raw_time)
            self.times.append(dt.timestamp() if dt is not None else None)

        self.identities: list[frozenset[tuple[str, str]]] = []
        for i, record in enumerate(records):
            idents = []
            for key in self.identity_keys:
                val = record.get(key)
                if val is not None and str(val).strip():
                    idents.append((key, str(val).strip()))
            if not idents:
                idents.append(("row_index", str(i)))
            self.identities.append(frozenset(idents))

        self._by_camera: dict[str, list[int]] = {}
        target_indices = (
            range(len(records)) if candidate_indices is None else candidate_indices
        )
        for index in target_indices:
            idx = int(index)
            if not 0 <= idx < len(records):
                raise IndexError(idx)
            # Missing timestamps are NOT eligible as historical candidates
            if self.times[idx] is None:
                continue
            self._by_camera.setdefault(self.cameras[idx], []).append(idx)

        self._camera_times: dict[str, list[float]] = {}
        for camera, indices in self._by_camera.items():
            indices[:] = sorted(set(indices), key=lambda i: (self.times[i], i))
            self._camera_times[camera] = [self.times[i] for i in indices]

        self._ranks: list[bytes] = [
            self._rank(self.cameras[i], self.sequences[i], sorted(self.identities[i]))
            for i in range(len(records))
        ]

        self._sequence_ranks: dict[tuple[str, str], bytes] = {}
        self._camera_sequences: dict[str, dict[str, _HistorySequence]] = {}
        for camera, indices in self._by_camera.items():
            rows_by_sequence: dict[str, list[int]] = {}
            for index in indices:
                key = (camera, self.sequences[index])
                rows_by_sequence.setdefault(key[1], []).append(index)
                if key not in self._sequence_ranks:
                    self._sequence_ranks[key] = self._rank(*key)
            self._camera_sequences[camera] = {
                seq: _HistorySequence(rows, self.times, self._ranks, self.identities)
                for seq, rows in rows_by_sequence.items()
            }

    def _rank(self, *parts: Any) -> bytes:
        return hashlib.sha256(repr((self.config.seed, parts)).encode("utf-8")).digest()

    def _select(self, query: int) -> list[int]:
        if not 0 <= query < len(self.times):
            raise IndexError(query)
        config = self.config
        if config.max_history == 0 or self.times[query] is None:
            return []
        camera, query_sequence = self.cameras[query], self.sequences[query]
        query_time = self.times[query]
        earliest = query_time - config.days * 86400
        sequences = self._camera_sequences.get(camera, {})
        seen = set(self.identities[query])
        selected: list[int] = []

        short_limit = min(config.short_term, config.max_history)
        short_sequence = sequences.get(query_sequence)
        if short_limit and short_sequence is not None:
            start = bisect_left(short_sequence.times, earliest)
            stop = bisect_left(short_sequence.times, query_time)
            for position in range(stop - 1, start - 1, -1):
                index = short_sequence.rows[position]
                if seen.isdisjoint(self.identities[index]):
                    selected.append(index)
                    seen.update(self.identities[index])
                    if len(selected) == short_limit:
                        break
        if len(selected) == config.max_history:
            return selected

        ordered = []
        for seq, sequence in sequences.items():
            if seq == query_sequence:
                continue
            start = bisect_left(sequence.times, earliest)
            stop = bisect_left(sequence.times, query_time)
            if start == stop:
                continue
            # Preserve chronological insertion order even in the unlikely case
            # of equal sequence ranks. Only aliases of the query affect it.
            first = start
            while first < stop and not self.identities[query].isdisjoint(
                self.identities[sequence.rows[first]]
            ):
                first += 1
            if first < stop:
                row = sequence.rows[first]
                ordered.append((self._sequence_ranks[camera, seq], self.times[row], row, seq, start, stop))
        heapify(ordered)
        while ordered and len(selected) < config.max_history:
            _, _, _, seq, start, stop = heappop(ordered)
            index = sequences[seq].best(start, stop, seen)
            if index is None:
                continue
            selected.append(index)
            seen.update(self.identities[index])

        return selected

    def build(self, query_indices: Sequence[int] | None = None) -> HistoryBatch:
        queries = range(len(self.times)) if query_indices is None else query_indices
        rows = [self._select(int(i)) for i in queries]
        indices = np.full((len(rows), self.config.max_history), -1, dtype=np.int64)
        for row, values in enumerate(rows):
            if values:
                indices[row, : len(values)] = values
        return HistoryBatch(indices)


def unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project vectors to the unit sphere using float32/float64 precision."""
    x = np.asarray(x)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def adapt_with_history(
    raw: np.ndarray,
    history: np.ndarray | HistoryBatch,
    query_indices: np.ndarray | Sequence[int] | None = None,
    *,
    temperature: float = 0.1,
    alpha: float = 0.1,
    chunk_size: int = 128,
    progress_fn=None,
) -> np.ndarray:
    """Adapt query features against historical representations via Q/K attention.

    Computes:
        logits = einsum('bd,bmd->bm', unit(q), unit(h)) / temperature
        weights = softmax(logits)
        pooled = einsum('bm,bmd->bd', weights, h)
        output = unit((1 - alpha) * q + alpha * pooled)
    """
    raw = np.asarray(raw, dtype=np.float32)
    history_arr = np.asarray(history.indices if isinstance(history, HistoryBatch) else history, dtype=np.int64)

    if query_indices is None:
        if len(history_arr) != len(raw):
            raise ValueError(
                f"history length ({len(history_arr)}) must match raw length ({len(raw)}) when query_indices is None"
            )
        queries = np.arange(len(raw), dtype=np.int64)
        output = raw.copy()
    else:
        queries = np.asarray(query_indices, dtype=np.int64)
        if len(history_arr) != len(queries):
            raise ValueError(
                f"history length ({len(history_arr)}) must match query_indices length ({len(queries)})"
            )
        output = raw[queries].copy()

    if len(queries) == 0:
        return output

    normalized = unit(raw)
    for begin in range(0, len(queries), chunk_size):
        end = min(begin + chunk_size, len(queries))
        hh = history_arr[begin:end]
        valid = hh >= 0
        active = valid.any(axis=1)
        if not active.any():
            if progress_fn is not None:
                progress_fn(end, len(queries))
            continue

        q = raw[queries[begin:end]][active]
        h_indices = hh[active].clip(min=0)
        h = raw[h_indices]
        mask = valid[active]
        h[~mask] = 0.0

        logits = np.einsum(
            "bd,bmd->bm", normalized[queries[begin:end]][active], normalized[h_indices]
        ) / temperature
        logits[~mask] = -np.inf
        max_logits = np.max(logits, axis=1, keepdims=True)
        max_logits = np.where(np.isfinite(max_logits), max_logits, 0.0)
        weights = np.exp(logits - max_logits)
        weights[~mask] = 0.0
        sums = weights.sum(axis=1, keepdims=True)
        weights = np.where(sums > 0, weights / np.maximum(sums, 1e-12), 0.0)

        pooled = np.einsum("bm,bmd->bd", weights, h)
        output[begin + np.flatnonzero(active)] = unit((1.0 - alpha) * q + alpha * pooled)
        if progress_fn is not None:
            progress_fn(end, len(queries))

    return output.astype(np.float32)


# Direct alias matching experiment protocol scripts
adapt_numpy = adapt_with_history


def _bbox_metrics(
    box_a: Any,
    box_b: Any,
) -> tuple[float, float, float, float] | None:
    """Compute (iou, center_norm_dist, size_ratio, spatial_sim) between two xyxy boxes."""
    if box_a is None or box_b is None:
        return None
    try:
        if len(box_a) < 4 or len(box_b) < 4:
            return None
        x1_a, y1_a, x2_a, y2_a = (float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3]))
        x1_b, y1_b, x2_b, y2_b = (float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3]))
    except (TypeError, ValueError, IndexError):
        return None

    w_a = max(0.0, x2_a - x1_a)
    h_a = max(0.0, y2_a - y1_a)
    w_b = max(0.0, x2_b - x1_b)
    h_b = max(0.0, y2_b - y1_b)
    area_a = w_a * h_a
    area_b = w_b * h_b

    inter_w = max(0.0, min(x2_a, x2_b) - max(x1_a, x1_b))
    inter_h = max(0.0, min(y2_a, y2_b) - max(y1_a, y1_b))
    inter = inter_w * inter_h
    union = area_a + area_b - inter
    iou = inter / union if union > 1e-9 else 0.0

    cx_a = 0.5 * (x1_a + x2_a)
    cy_a = 0.5 * (y1_a + y2_a)
    cx_b = 0.5 * (x1_b + x2_b)
    cy_b = 0.5 * (y1_b + y2_b)
    diag = max(0.5 * (math.hypot(w_a, h_a) + math.hypot(w_b, h_b)), 1.0)
    center_norm_dist = math.hypot(cx_a - cx_b, cy_a - cy_b) / diag

    max_area = max(area_a, area_b)
    size_ratio = min(area_a, area_b) / max_area if max_area > 1e-9 else 1.0
    spatial_sim = 0.50 * iou + 0.35 * math.exp(-0.75 * center_norm_dist) + 0.15 * size_ratio
    return iou, center_norm_dist, size_ratio, spatial_sim


def _extract_unit_embedding(record: Mapping[str, Any]) -> np.ndarray | None:
    emb = _first(record, ("raw_embedding", "embedding", "feature"))
    if emb is None:
        return None
    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return None
    return arr / norm


def _frame_and_slot_keys(record: Mapping[str, Any], fallback_idx: int) -> tuple[str, str]:
    frame = _first(record, ("path", "image_id", "source_sha256"))
    box_idx = _first(record, ("box_index", "crop_index"))
    content = _first(record, ("content_id",))
    if frame is not None and str(frame) != "":
        frame_key = str(frame)
    elif content is not None and ":" in str(content):
        prefix, suffix = str(content).rsplit(":", 1)
        frame_key = prefix if suffix.isdigit() else str(content)
    elif content is not None and str(content) != "":
        frame_key = str(content)
    else:
        frame_key = f"frame_{fallback_idx}"

    if box_idx is not None and str(box_idx) != "":
        slot_key = f"box:{int(box_idx)}"
    elif content is not None and str(content) != "":
        slot_key = f"content:{content}"
    else:
        slot_key = f"row:{fallback_idx}"
    return frame_key, slot_key


def _assign_cluster_tracks(
    cluster: list[dict[str, Any]],
    event_key: str,
) -> None:
    """Partition a 30-minute camera event into per-target trajectories (track_id).

    Enforces hard same-photo exclusion (distinct boxes from the same frame never
    share a track) and links boxes across frames via greedy one-to-one spatial
    and visual similarity matching.
    """
    if not cluster:
        return

    frames: list[list[list[int]]] = []
    frame_index_by_key: dict[str, int] = {}
    slot_maps_per_frame: list[dict[str, int]] = []

    for idx, item in enumerate(cluster):
        frame_key, slot_key = _frame_and_slot_keys(item, idx)
        if frame_key not in frame_index_by_key:
            frame_index_by_key[frame_key] = len(frames)
            frames.append([])
            slot_maps_per_frame.append({})
        f_idx = frame_index_by_key[frame_key]
        slot_map = slot_maps_per_frame[f_idx]
        if slot_key not in slot_map:
            slot_map[slot_key] = len(frames[f_idx])
            frames[f_idx].append([])
        s_idx = slot_map[slot_key]
        frames[f_idx][s_idx].append(idx)

    has_multi_box_frame = any(len(f_slots) > 1 for f_slots in frames)
    embeddings = [_extract_unit_embedding(item) for item in cluster]
    bboxes = [_first(item, ("bbox",)) for item in cluster]
    box_indices = [_first(item, ("box_index", "crop_index")) for item in cluster]
    tracks: list[list[int]] = []
    track_last_rep: list[int] = []

    for f_slots in frames:
        if not tracks:
            for slot_rows in f_slots:
                tracks.append(list(slot_rows))
                track_last_rep.append(slot_rows[-1])
            continue

        candidates: list[tuple[float, int, int]] = []
        for t_idx, last_row in enumerate(track_last_rep):
            emb_a = embeddings[last_row]
            box_a = bboxes[last_row]
            box_idx_a = box_indices[last_row]

            for s_idx, slot_rows in enumerate(f_slots):
                current_row = slot_rows[0]
                emb_b = embeddings[current_row]
                box_b = bboxes[current_row]
                box_idx_b = box_indices[current_row]

                emb_sim: float | None = None
                if emb_a is not None and emb_b is not None and emb_a.shape == emb_b.shape:
                    emb_sim = float(np.dot(emb_a, emb_b))
                spatial = _bbox_metrics(box_a, box_b)

                if emb_sim is not None and spatial is not None:
                    _, center_norm_dist, _, spatial_sim = spatial
                    if emb_sim < 0.50 or (center_norm_dist > 4.0 and emb_sim < 0.70):
                        continue
                    score = 0.70 * emb_sim + 0.30 * spatial_sim
                elif emb_sim is not None:
                    min_emb = 0.55 if has_multi_box_frame else 0.35
                    if emb_sim < min_emb:
                        continue
                    score = emb_sim
                elif spatial is not None:
                    _, center_norm_dist, size_ratio, spatial_sim = spatial
                    if has_multi_box_frame and (center_norm_dist > 2.5 or size_ratio < 0.25):
                        continue
                    score = spatial_sim
                else:
                    if not has_multi_box_frame and len(tracks) == 1 and len(f_slots) == 1:
                        score = 1.0
                    elif box_idx_a is not None and box_idx_b is not None and int(box_idx_a) == int(box_idx_b):
                        score = 0.5
                    else:
                        continue

                candidates.append((score, t_idx, s_idx))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        used_tracks: set[int] = set()
        used_slots: set[int] = set()
        for _, t_idx, s_idx in candidates:
            if t_idx in used_tracks or s_idx in used_slots:
                continue
            used_tracks.add(t_idx)
            used_slots.add(s_idx)
            slot_rows = f_slots[s_idx]
            tracks[t_idx].extend(slot_rows)
            track_last_rep[t_idx] = slot_rows[-1]

        for s_idx, slot_rows in enumerate(f_slots):
            if s_idx not in used_slots:
                tracks.append(list(slot_rows))
                track_last_rep.append(slot_rows[-1])

    multi_track = len(tracks) > 1
    for t_idx, row_indices in enumerate(tracks):
        track_key = f"{event_key}:track_{t_idx:02d}" if multi_track else event_key
        for r_idx in row_indices:
            cluster[r_idx]["event_id"] = event_key
            cluster[r_idx]["track_id"] = track_key
            cluster[r_idx]["sequence_id"] = track_key


def group_job_events(
    records: Sequence[Mapping[str, Any]],
    gap_seconds: int = 1800,
) -> list[dict[str, Any]]:
    """Group observations into 30-minute camera events and per-target trajectories.

    Missing capture times form single-item independent sequences with no temporal history.
    """
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        cam = str(_first(r, ("camera_id", "camera_component", "location")) or "single")
        item = dict(r)
        item["datetime"] = parse_capture_time(_first(r, ("datetime", "date_taken", "captured_at")))
        by_camera.setdefault(cam, []).append(item)

    output: list[dict[str, Any]] = []
    for camera_id, items in sorted(by_camera.items()):
        missing = [x for x in items if x["datetime"] is None]
        timed = [x for x in items if x["datetime"] is not None]

        missing_sorted = sorted(
            missing,
            key=lambda x: (
                str(_first(x, ("path", "image_id", "content_id")) or ""),
                int(_first(x, ("box_index", "crop_index")) or 0),
            ),
        )
        missing_frame_counts: dict[str, int] = {}
        for item in missing_sorted:
            f_key = str(_first(item, ("path", "image_id", "content_id")) or "")
            missing_frame_counts[f_key] = missing_frame_counts.get(f_key, 0) + 1
        missing_frame_seen: dict[str, int] = {}
        for item in missing_sorted:
            f_key = str(_first(item, ("path", "image_id", "content_id")) or "")
            sub_idx = missing_frame_seen.get(f_key, 0)
            missing_frame_seen[f_key] = sub_idx + 1
            event_key = f"missing:{f_key}"
            box_idx = _first(item, ("box_index", "crop_index"))
            if missing_frame_counts[f_key] > 1:
                suffix = int(box_idx) if box_idx is not None else sub_idx
                track_key = f"{event_key}:track_{suffix:02d}"
            else:
                track_key = event_key
            item_copy = dict(item)
            item_copy["camera_id"] = camera_id
            item_copy["event_id"] = event_key
            item_copy["track_id"] = track_key
            item_copy["sequence_id"] = track_key
            item_copy["timestamp_missing"] = True
            item_copy["datetime"] = None
            output.append(item_copy)

        timed.sort(
            key=lambda x: (
                x["datetime"],
                str(_first(x, ("path", "image_id", "content_id")) or ""),
                int(_first(x, ("box_index", "crop_index")) or 0),
            )
        )
        current_cluster: list[dict[str, Any]] = []
        cluster_start: datetime | None = None

        for item in timed:
            dt = item["datetime"]
            if current_cluster:
                prev_dt = current_cluster[-1]["datetime"]
                if (dt - prev_dt).total_seconds() >= gap_seconds:
                    seq_key = f"event:{camera_id}:{cluster_start.isoformat()}"
                    _assign_cluster_tracks(current_cluster, seq_key)
                    for it in current_cluster:
                        it["camera_id"] = camera_id
                        it["timestamp_missing"] = False
                        output.append(it)
                    current_cluster = []
                    cluster_start = None

            if not current_cluster:
                cluster_start = dt
            current_cluster.append(dict(item))

        if current_cluster:
            seq_key = f"event:{camera_id}:{cluster_start.isoformat()}"
            _assign_cluster_tracks(current_cluster, seq_key)
            for it in current_cluster:
                it["camera_id"] = camera_id
                it["timestamp_missing"] = False
                output.append(it)

    return output
