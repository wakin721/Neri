"""DINOv3 sequence aggregation and crop collection processor.

Collects raw embeddings and detection boxes across batch-level processing,
then performs complete-sequence temporal causal context adaptation and joint
re-classification at job completion.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .context import (
    FastCausalHistoryIndex,
    HistoryConfig,
    adapt_with_history,
    camera_id_for_path,
    group_job_events,
    parse_capture_time,
)
from .observation_support import make_dinov3_observation_id

logger = logging.getLogger(__name__)

SEQUENCE_STATUS_COMPLETE = "complete"
SEQUENCE_STATUS_PARTIAL = "partial"
SEQUENCE_STATUS_PENDING = "pending"
SEQUENCE_STATUS_FAILED = "failed"
_SEQUENCE_SAVE_CHUNK_SIZE = 512


class DinoV3SequenceFinalizationError(RuntimeError):
    """Raised when a full DINOv3 sequence summary cannot be completed safely."""


class _SequenceCancelled(Exception):
    pass


@dataclass
class DinoV3CropRecord:
    image_path: str
    box_index: int
    bbox: tuple[float, float, float, float]
    detection_confidence: float
    raw_embedding: np.ndarray
    initial_prediction: Any = None
    initial_candidate: dict[str, Any] = field(default_factory=dict)


def _image_file_digest(path: str | Path) -> str:
    """Compute SHA-256 hex digest of image file for content identity."""
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except (AttributeError, OSError):
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return hashlib.sha256(os.path.normcase(os.path.abspath(str(path))).encode("utf-8")).hexdigest()


class DinoV3CropCollector:
    """Deduplicating collector for crop embeddings during batch detection."""

    def __init__(self):
        # Index by photo as well as box to keep checkpointing/restore linear
        # in job size, while deduplicating OOM retries.
        self._records_by_path: dict[str, dict[int, DinoV3CropRecord]] = {}

    def record_crop(self, record: DinoV3CropRecord) -> None:
        path = os.path.normcase(os.path.abspath(record.image_path))
        self._records_by_path.setdefault(path, {})[record.box_index] = record

    def discard_paths(self, paths: Iterable[str | Path]) -> None:
        """Discard any recorded crops for the given paths before batch retry."""
        targets = {os.path.normcase(os.path.abspath(str(p))) for p in paths}
        for path in targets:
            self._records_by_path.pop(path, None)

    def clear_path(self, path: str | Path) -> None:
        self.discard_paths([path])

    def record_crops(self, records: Sequence[DinoV3CropRecord]) -> None:
        for r in records:
            self.record_crop(r)

    def get_crops_for_path(self, path: str | Path) -> list[DinoV3CropRecord]:
        norm_path = os.path.normcase(os.path.abspath(str(path)))
        matching = self._records_by_path.get(norm_path, {}).values()
        return sorted(matching, key=lambda c: c.box_index)

    def get_all_crops(self) -> list[DinoV3CropRecord]:
        return [r for crops in self._records_by_path.values() for r in crops.values()]

    def clear(self) -> None:
        self._records_by_path.clear()


def _annotate_sequence_state(
    item: Any,
    *,
    status: str,
    reason: str | None = None,
    processed_count: int | None = None,
    total_count: int | None = None,
    apply_data_fn: Callable[[Any, dict[str, Any]], Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    if getattr(item, "error", None) and status in {SEQUENCE_STATUS_PENDING, SEQUENCE_STATUS_PARTIAL}:
        status = SEQUENCE_STATUS_FAILED
    complete = status == SEQUENCE_STATUS_COMPLETE
    pending = status in {SEQUENCE_STATUS_PENDING, SEQUENCE_STATUS_PARTIAL}
    data = dict(getattr(item, "detection_data", {}) or {})
    if status == SEQUENCE_STATUS_FAILED:
        data["error"] = getattr(item, "error", None)
    data["sequence_completed"] = complete
    data["sequence_pending"] = pending
    data["sequence_status"] = status
    if reason:
        data["sequence_reason"] = reason
    else:
        data.pop("sequence_reason", None)
    if processed_count is not None:
        data["sequence_processed_count"] = int(processed_count)
    if total_count is not None:
        data["sequence_total_count"] = int(total_count)

    source_boxes = data.get("检测框")
    if not isinstance(source_boxes, list):
        source_boxes = list(getattr(item, "detection_boxes", []) or [])
    boxes: list[dict[str, Any]] = []
    for raw_box in source_boxes:
        if not isinstance(raw_box, dict):
            continue
        box = dict(raw_box)
        if not complete:
            box.pop("rejection_cluster", None)
        box["sequence_pending"] = pending
        box["sequence_status"] = status
        raw_candidates = box.get("候选项")
        if isinstance(raw_candidates, list):
            candidates = []
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, dict):
                    continue
                candidate = dict(raw_candidate)
                candidate["sequence_pending"] = pending
                candidate["sequence_status"] = status
                candidates.append(candidate)
            box["候选项"] = candidates
        boxes.append(box)
    data["检测框"] = boxes

    if apply_data_fn is not None:
        return apply_data_fn(item, data), data
    return item.model_copy(
        update={
            "detection_boxes": boxes,
            "detection_data": data,
        }
    ), data


def mark_dinov3_sequence_incomplete(
    results: list[Any],
    *,
    reason: str,
    status: str = SEQUENCE_STATUS_PENDING,
    processed_count: int | None = None,
    total_count: int | None = None,
    input_path: Path | None = None,
    save_batch_fn: Callable[[list[tuple[Path, dict[str, Any]]], Path], None] | None = None,
    apply_data_fn: Callable[[Any, dict[str, Any]], Any] | None = None,
) -> list[Any]:
    """Mark results as intentionally incomplete without inventing a summary."""
    if status not in {SEQUENCE_STATUS_PARTIAL, SEQUENCE_STATUS_PENDING}:
        raise ValueError(f"Unsupported incomplete sequence status: {status}")
    updated_results: list[Any] = []
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for item in results:
        updated_item, data = _annotate_sequence_state(
            item,
            status=status,
            reason=reason,
            processed_count=processed_count,
            total_count=total_count,
            apply_data_fn=apply_data_fn,
        )
        updated_results.append(updated_item)
        path = getattr(updated_item, "path", None)
        if path:
            payloads.append((Path(path), data))
    if save_batch_fn is not None and payloads:
        _save_in_chunks(payloads, input_path, save_batch_fn)
    return updated_results


def _save_in_chunks(
    payloads: Sequence[tuple[Path, dict[str, Any]]],
    input_path: Path | None,
    save_batch_fn: Callable[[list[tuple[Path, dict[str, Any]]], Path], None],
    progress_fn=None,
) -> None:
    if not payloads:
        return
    root = input_path or (payloads[0][0].parent if payloads else Path.cwd())
    for start in range(0, len(payloads), _SEQUENCE_SAVE_CHUNK_SIZE):
        save_batch_fn(list(payloads[start:start + _SEQUENCE_SAVE_CHUNK_SIZE]), root)
        if progress_fn is not None:
            progress_fn(min(start + _SEQUENCE_SAVE_CHUNK_SIZE, len(payloads)), len(payloads))


def ensure_dinov3_sequences_completed(results: Sequence[Any]) -> None:
    """Fail closed when any result still represents an incomplete sequence."""
    incomplete: list[str] = []
    for item in results:
        data = dict(getattr(item, "detection_data", {}) or {})
        if (
            getattr(item, "error", None)
            and data.get("sequence_status") == SEQUENCE_STATUS_FAILED
            and data.get("sequence_pending") is False
        ):
            continue
        if getattr(item, "error", None) or data.get("sequence_completed") is not True:
            incomplete.append(str(getattr(item, "path", getattr(item, "filename", "<unknown>"))))
    if incomplete:
        preview = ", ".join(incomplete[:3])
        suffix = f" (+{len(incomplete) - 3})" if len(incomplete) > 3 else ""
        raise DinoV3SequenceFinalizationError(
            f"DINOv3 sequence summary incomplete for {len(incomplete)} results: {preview}{suffix}"
        )


def _expected_detection_box_indices(item: Any) -> tuple[int, ...] | None:
    data = dict(getattr(item, "detection_data", {}) or {})
    raw_indices = data.get("detection_box_indices")
    if isinstance(raw_indices, list):
        try:
            parsed_indices = [int(index) for index in raw_indices]
        except (TypeError, ValueError):
            raise DinoV3SequenceFinalizationError(
                f"DINOv3 detection box indices are invalid for {getattr(item, 'path', '')}"
            )
        if any(index < 0 for index in parsed_indices) or len(set(parsed_indices)) != len(parsed_indices):
            raise DinoV3SequenceFinalizationError(
                f"DINOv3 detection box indices are inconsistent for {getattr(item, 'path', '')}"
            )
        indices = tuple(sorted(parsed_indices))
        raw_count = data.get("detection_box_count")
        if raw_count is not None and (
            type(raw_count) is not int
            or raw_count < 0
            or raw_count != len(indices)
        ):
            raise DinoV3SequenceFinalizationError(
                f"DINOv3 detection box count/index mismatch for {getattr(item, 'path', '')}"
            )
        return indices
    raw_count = data.get("detection_box_count")
    if type(raw_count) is int and raw_count >= 0:
        return tuple(range(raw_count))
    return None


def _has_completed_empty_detection(item: Any) -> bool:
    data = dict(getattr(item, "detection_data", {}) or {})
    boxes = data.get("检测框")
    return (
        data.get("detection_completed") is True
        and isinstance(boxes, list)
        and not boxes
        and _expected_detection_box_indices(item) == ()
    )


def _mark_completed_without_crops(
    results: list[Any],
    *,
    complete_sequence: bool,
    incomplete_reason: str,
    processed_count: int,
    total_photos: int | None,
    input_path: Path,
    save_batch_fn: Callable[[list[tuple[Path, dict[str, Any]]], Path], None] | None,
    apply_data_fn: Callable[[Any, dict[str, Any]], Any] | None,
    mutate_job: Callable[..., None] | None,
    progress_fn=None,
) -> list[Any]:
    status = SEQUENCE_STATUS_COMPLETE if complete_sequence else SEQUENCE_STATUS_PARTIAL
    updated_results: list[Any] = []
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for item in results:
        updated_item, data = _annotate_sequence_state(
            item,
            status=status,
            reason=None if complete_sequence else incomplete_reason,
            processed_count=None if complete_sequence else processed_count,
            total_count=None if complete_sequence else total_photos,
            apply_data_fn=apply_data_fn,
        )
        updated_results.append(updated_item)
        if getattr(updated_item, "path", None):
            payloads.append((Path(updated_item.path), data))
    if save_batch_fn is not None:
        _save_in_chunks(payloads, input_path, save_batch_fn, progress_fn=progress_fn)
    if mutate_job is not None:
        mutate_job(
            results=updated_results,
            message=(
                f"Neri1.1序列聚合完成: {len(updated_results)} 张空照片已确认"
                if complete_sequence
                else f"Neri1.1序列部分聚合: {len(updated_results)} 张空照片，任务仍需继续"
            ),
        )
    if complete_sequence:
        ensure_dinov3_sequences_completed(updated_results)
    return updated_results


def finalize_photo_job(
    detector: Any,
    results: list[Any],
    input_path: Path,
    *,
    is_cancelled: Callable[[], bool] | None = None,
    mutate_job: Callable[..., None] | None = None,
    save_batch_fn: Callable[[list[tuple[Path, dict[str, Any]]], Path], None] | None = None,
    apply_data_fn: Callable[[Any, dict[str, Any]], Any] | None = None,
    job_id: str | None = None,
    complete_sequence: bool = True,
    incomplete_reason: str = "incomplete_job",
    total_photos: int | None = None,
    progress_fn: Callable[[str, int, int], None] | None = None,
) -> list[Any]:
    """Hook called when all photo batches have finished inference.

    Groups all crops in the job into camera + 30-min events, computes frozen
    causal history context via FastCausalHistoryIndex, adapts features via Q/K
    attention, and runs a single full-sequence classifier.classify_features pass.
    Rewrites detection_data preserving rejected Unknown boxes and candidate metadata.
    """
    if complete_sequence and is_cancelled is not None and is_cancelled():
        logger.info("DINOv3 job cancelled before finalization; skipping final sequence aggregation")
        return results

    processed_count = len(results)
    partial_status = SEQUENCE_STATUS_PARTIAL
    classifier = getattr(detector, "dinov3_classifier", None)
    if classifier is None:
        classifier = getattr(detector, "dino_classifier", None)
    if classifier is None or getattr(classifier, "backend", None) != "dinov3":
        if complete_sequence:
            raise DinoV3SequenceFinalizationError("DINOv3 classifier is unavailable")
        return mark_dinov3_sequence_incomplete(
            results,
            reason=incomplete_reason,
            processed_count=processed_count,
            total_count=total_photos,
            input_path=input_path,
            save_batch_fn=save_batch_fn,
            apply_data_fn=apply_data_fn,
        )

    collector: DinoV3CropCollector | None = getattr(detector, "_dinov3_collector", None)
    if collector is None or not hasattr(collector, "get_crops_for_path"):
        if complete_sequence:
            raise DinoV3SequenceFinalizationError("DINOv3 crop collector is unavailable")
        return mark_dinov3_sequence_incomplete(
            results,
            reason=incomplete_reason,
            processed_count=processed_count,
            total_count=total_photos,
            input_path=input_path,
            save_batch_fn=save_batch_fn,
            apply_data_fn=apply_data_fn,
        )

    def report(stage, done=0, total=0):
        if complete_sequence and is_cancelled is not None and is_cancelled():
            raise _SequenceCancelled()
        if progress_fn is not None:
            progress_fn(stage, done, total)
        if complete_sequence and is_cancelled is not None and is_cancelled():
            raise _SequenceCancelled()

    try:
        report("校验照片特征", 0, len(results))
        failed_items = [item for item in results if getattr(item, "error", None)]
        if failed_items:
            # A bad source file has no embedding to aggregate. Keep its error
            # visible while completing the full available sequence of valid photos.
            valid_items = [item for item in results if not getattr(item, "error", None)]
            completed_valid = (
                finalize_photo_job(
                    detector, valid_items, input_path,
                    is_cancelled=is_cancelled, save_batch_fn=save_batch_fn,
                    apply_data_fn=apply_data_fn, job_id=job_id,
                    complete_sequence=complete_sequence,
                    incomplete_reason=incomplete_reason, total_photos=total_photos,
                    progress_fn=progress_fn,
                )
                if valid_items else []
            )
            if complete_sequence and is_cancelled is not None and is_cancelled():
                completed_by_path = {str(item.path): item for item in completed_valid}
                return [completed_by_path.get(str(item.path), item) for item in results]
            failed_results = []
            failed_payloads = []
            for item in failed_items:
                updated, data = _annotate_sequence_state(
                    item, status=SEQUENCE_STATUS_FAILED, reason="file_error",
                    apply_data_fn=apply_data_fn,
                )
                failed_results.append(updated)
                failed_payloads.append((Path(item.path), data))
            if complete_sequence and is_cancelled is not None and is_cancelled():
                by_path = {str(item.path): item for item in [*completed_valid, *failed_results]}
                return [by_path[str(item.path)] for item in results]
            if save_batch_fn is not None:
                _save_in_chunks(failed_payloads, input_path, save_batch_fn)
            by_path = {str(item.path): item for item in [*completed_valid, *failed_results]}
            updated_results = [by_path[str(item.path)] for item in results]
            if complete_sequence and not (is_cancelled and is_cancelled()):
                ensure_dinov3_sequences_completed(updated_results)
            if mutate_job is not None:
                mutate_job(results=updated_results)
            return updated_results

        crop_items: list[tuple[Any, DinoV3CropRecord]] = []
        crops_by_path: dict[str, list[DinoV3CropRecord]] = {}
        for item in results:
            item_path = getattr(item, "path", None)
            if not item_path:
                continue
            crops = collector.get_crops_for_path(item_path)
            path_key = os.path.normcase(os.path.abspath(str(item_path)))
            crops_by_path[path_key] = crops
            for crop in crops:
                crop_items.append((item, crop))

        for photo_index, item in enumerate(results):
            item_path = getattr(item, "path", None)
            if not item_path:
                continue
            path_key = os.path.normcase(os.path.abspath(str(item_path)))
            crops = crops_by_path.get(path_key, [])
            data = dict(getattr(item, "detection_data", {}) or {})
            if data.get("detection_completed") is False:
                raise DinoV3SequenceFinalizationError(
                    f"DINOv3 detection output is incomplete for {item_path}"
                )
            expected_indices = _expected_detection_box_indices(item)
            actual_indices = tuple(sorted(crop.box_index for crop in crops))
            if expected_indices is not None and actual_indices != expected_indices:
                raise DinoV3SequenceFinalizationError(
                    "DINOv3 crop collection is incomplete for "
                    f"{item_path}: expected box indices {list(expected_indices)}, "
                    f"collected {list(actual_indices)}"
                )
            if not crops and not _has_completed_empty_detection(item):
                raise DinoV3SequenceFinalizationError(
                    "DINOv3 sequence summary requires completed detection output for empty photo: "
                    f"{item_path}"
                )
            report("校验照片特征", photo_index + 1, len(results))

        if not crop_items:
            logger.info(
                "DINOv3 finalization found no crops across %d successfully completed empty photos",
                len(results),
            )
            report("写入序列结果", 0, len(results))
            updated = _mark_completed_without_crops(
                results,
                complete_sequence=complete_sequence,
                incomplete_reason=incomplete_reason,
                processed_count=processed_count,
                total_photos=total_photos,
                input_path=input_path,
                save_batch_fn=save_batch_fn,
                apply_data_fn=apply_data_fn,
                mutate_job=mutate_job,
                progress_fn=lambda done, total: report("写入序列结果", done, total),
            )
            report("写入序列结果", len(results), len(results))
            return updated

        logger.info("DINOv3 finalization starting: %d total crops across %d photos", len(crop_items), len(results))

        # Precompute file SHA-256 digests for unique images for true content deduplication
        image_digests: dict[str, str] = {}
        digest_total = len({str(item.path) for item, _ in crop_items})
        report("计算照片指纹", 0, digest_total)
        for item, _ in crop_items:
            p = str(item.path)
            if p not in image_digests:
                image_digests[p] = _image_file_digest(p)
                report("计算照片指纹", len(image_digests), digest_total)

        # Prepare metadata for event grouping
        event_input_records: list[dict[str, Any]] = []
        for crop_idx, (item, crop) in enumerate(crop_items):
            path_str = str(item.path)
            cam_id = camera_id_for_path(path_str, root=input_path)
            date_taken = getattr(item, "date_taken", None)
            content_hash = f"{image_digests[path_str]}:{crop.box_index}"
            event_input_records.append(
                {
                    "crop_idx": crop_idx,
                    "box_index": int(crop.box_index),
                    "bbox": list(crop.bbox),
                    "raw_embedding": crop.raw_embedding,
                    "path": path_str,
                    "camera_id": cam_id,
                    "date_taken": date_taken,
                    "content_hash": content_hash,
                    "content_id": content_hash,
                }
            )

        report("划分事件与目标轨迹")
        grouped_events = group_job_events(event_input_records, gap_seconds=1800)
        by_crop_idx = {g["crop_idx"]: g for g in grouped_events}

        history_records: list[dict[str, Any]] = []
        track_sequence_ids: list[str] = []
        for crop_idx in range(len(crop_items)):
            g = by_crop_idx[crop_idx]
            event_id = str(g.get("event_id", g["sequence_id"]))
            track_id = str(g.get("track_id", g["sequence_id"]))
            track_sequence_ids.append(track_id)
            history_records.append(
                {
                    "camera_id": g["camera_id"],
                    "seq_id": event_id,
                    "datetime": g.get("datetime"),
                    "content_hash": g["content_hash"],
                    "content_id": g["content_hash"],
                }
            )

        if complete_sequence and is_cancelled is not None and is_cancelled():
            return results

        # Build causal history index
        config = HistoryConfig(days=7.0, max_history=64, short_term=0, seed=20261105)
        report("构建因果历史", 0, len(history_records))
        history_index = FastCausalHistoryIndex(history_records, config=config)
        history = np.full((len(history_records), config.max_history), -1, dtype=np.int64)
        for start in range(0, len(history_records), 256):
            end = min(start + 256, len(history_records))
            history[start:end] = history_index.build(range(start, end)).indices
            report("构建因果历史", end, len(history_records))

        # Stack raw embeddings preserving source precision
        raw_features = np.stack([crop.raw_embedding for _, crop in crop_items], axis=0)
        report("整合历史特征", 0, len(raw_features))
        adapted_features = adapt_with_history(
            raw_features, history, temperature=0.1, alpha=0.1,
            progress_fn=lambda done, total: report("整合历史特征", done, total),
        )

        camera_ids = [r["camera_id"] for r in history_records]
        sequence_ids = track_sequence_ids
        if getattr(getattr(classifier, "model", None), "inference_recipe", None) == "experiment_full":
            sequence_ids = [r["seq_id"] for r in history_records]
        content_ids = [r["content_hash"] for r in history_records]

        if complete_sequence and is_cancelled is not None and is_cancelled():
            return results

        try:
            from threadpoolctl import threadpool_limits
            t_ctx = threadpool_limits(limits=4)
        except Exception:
            t_ctx = nullcontext()

        report("全序列分类")
        with t_ctx:
            predictions = classifier.classify_features(
                raw_features,
                camera_ids=camera_ids,
                sequence_ids=sequence_ids,
                content_ids=content_ids,
                site_ids=camera_ids,
                adapted_features=adapted_features,
            )

        # Check cancellation immediately after classification before writing DB
        if complete_sequence and is_cancelled is not None and is_cancelled():
            logger.info("DINOv3 job cancelled after classification; skipping DB writes")
            return results

        if len(predictions) != len(crop_items):
            raise RuntimeError(
                f"DINOv3 classifier returned {len(predictions)} predictions for {len(crop_items)} crops"
            )

        # One fit across the entire task, after final sequence decisions. Use
        # raw crop features and individual tracks even for experiment_full.
        rejection_clusters = {}
        if complete_sequence:
            report("聚类拒识目标")
            from .rejection_clustering import (
                ClusteringCancelled, cluster_rejections, failed_rejections,
            )
            rejected = [i for i, pred in enumerate(predictions) if not bool(pred.accepted)]
            rejected_ids = [make_dinov3_observation_id(crop_items[i][0].path,
                            crop_items[i][1].box_index, crop_items[i][1].bbox) for i in rejected]
            fingerprint = getattr(classifier, "model_fingerprint", None)
            if not isinstance(fingerprint, str):
                fingerprint = "unversioned"
            try:
                # Directory proxies for clustering must not depend on which
                # selected input root happened to be listed first. Keep the
                # original classifier context/gates unchanged; regroup target
                # tracks using full source directories for this browsing step.
                cluster_groups = group_job_events([
                    {**record, "camera_id": camera_id_for_path(record["path"])}
                    for record in event_input_records
                ]) if rejected else []
                cluster_by_crop = {g["crop_idx"]: g for g in cluster_groups}
                rejection_clusters = cluster_rejections(
                    raw_features[rejected], rejected_ids,
                    [cluster_by_crop[i]["camera_id"] for i in rejected],
                    [cluster_by_crop[i]["track_id"] for i in rejected],
                    [f"{image_digests[str(crop_items[i][0].path)]}:"
                     f"{tuple(float(v) for v in crop_items[i][1].bbox)}" for i in rejected],
                    model_fingerprint=fingerprint, scope_id=job_id or "",
                    is_cancelled=is_cancelled,
                )
            except ClusteringCancelled:
                return results
            except Exception:
                logger.exception("HDBSCAN rejection clustering failed; preserving sequence decisions")
                rejection_clusters = failed_rejections(
                    rejected_ids, model_fingerprint=fingerprint, scope_id=job_id or "")
            if is_cancelled is not None and is_cancelled():
                return results

        # Map predictions back by (path_key, box_index)
        final_preds_by_path: dict[str, dict[int, tuple[DinoV3CropRecord, Any, dict[str, Any]]]] = {}
        for crop_idx, ((item, crop), pred) in enumerate(zip(crop_items, predictions)):
            key = os.path.normcase(os.path.abspath(str(item.path)))
            final_preds_by_path.setdefault(key, {})[crop.box_index] = (
                crop,
                pred,
                by_crop_idx[crop_idx],
            )

        updated_results: list[Any] = []
        detections_to_save: list[tuple[Path, dict[str, Any]]] = []

        report("生成序列结果", 0, len(results))
        for item in results:
            key = os.path.normcase(os.path.abspath(str(item.path)))
            item_preds = final_preds_by_path.get(key)
            if not item_preds:
                expected_indices = _expected_detection_box_indices(item)
                if expected_indices != () or not _has_completed_empty_detection(item):
                    raise DinoV3SequenceFinalizationError(
                        f"DINOv3 crop collection is incomplete or detection is unconfirmed for {item.path}"
                    )
                updated_item, existing_data = _annotate_sequence_state(
                    item,
                    status=SEQUENCE_STATUS_COMPLETE if complete_sequence else partial_status,
                    reason=None if complete_sequence else incomplete_reason,
                    processed_count=None if complete_sequence else processed_count,
                    total_count=None if complete_sequence else total_photos,
                    apply_data_fn=apply_data_fn,
                )
                updated_results.append(updated_item)
                detections_to_save.append((Path(item.path), existing_data))
                report("生成序列结果", len(updated_results), len(results))
                continue

            new_boxes: list[dict[str, Any]] = []
            species_counts: dict[str, int] = {}
            confidences: list[float] = []

            for box_index in sorted(item_preds.keys()):
                crop, pred, group_meta = item_preds[box_index]
                accepted = bool(pred.accepted)
                best_known = str(pred.best_known_species)
                box_name = best_known if accepted else "Unknown"
                box_conf = float(pred.confidence)
                box_known_score = float(
                    pred.box_known_score
                    if getattr(pred, "box_known_score", None) is not None
                    else pred.known_score
                )
                box_threshold = float(
                    pred.box_threshold
                    if getattr(pred, "box_threshold", None) is not None
                    else pred.threshold
                )
                box_accepted = bool(
                    pred.box_accepted
                    if getattr(pred, "box_accepted", None) is not None
                    else accepted
                )
                box_confidence = float(
                    pred.box_confidence
                    if getattr(pred, "box_confidence", None) is not None
                    else box_conf
                )
                gate_mode = getattr(pred, "gate_mode", "box_and_sequence")
                inference_recipe = getattr(pred, "inference_recipe", "production_dual")
                if gate_mode == "sequence":
                    box_known_score = box_threshold = box_accepted = box_confidence = None
                sequence_known_score = float(
                    pred.sequence_known_score
                    if getattr(pred, "sequence_known_score", None) is not None
                    else pred.known_score
                )
                sequence_threshold = float(
                    pred.sequence_threshold
                    if getattr(pred, "sequence_threshold", None) is not None
                    else pred.threshold
                )
                sequence_accepted = bool(
                    pred.sequence_accepted
                    if getattr(pred, "sequence_accepted", None) is not None
                    else accepted
                )
                sequence_confidence = float(
                    pred.sequence_confidence
                    if getattr(pred, "sequence_confidence", None) is not None
                    else box_conf
                )
                raw_ridge_score = float(
                    pred.raw_ridge_score
                    if getattr(pred, "raw_ridge_score", None) is not None
                    else pred.ridge_score
                )
                within_margin = (
                    float(pred.within_margin)
                    if getattr(pred, "within_margin", None) is not None
                    else None
                )
                whitened_norm = (
                    float(pred.whitened_norm)
                    if getattr(pred, "whitened_norm", None) is not None
                    else None
                )
                track_id = str(
                    getattr(pred, "track_id", None)
                    or group_meta.get("track_id")
                    or group_meta.get("sequence_id")
                    or ""
                )
                event_id = str(
                    group_meta.get("event_id")
                    or group_meta.get("sequence_id")
                    or ""
                )
                obs_id = make_dinov3_observation_id(item.path, box_index, crop.bbox)

                candidate = pred.as_candidate(detection_confidence=crop.detection_confidence)
                candidate["observation_id"] = obs_id
                candidate["sequence_pending"] = not complete_sequence
                candidate["sequence_status"] = (
                    SEQUENCE_STATUS_COMPLETE if complete_sequence else partial_status
                )
                if not accepted:
                    candidate["name"] = "Unknown"
                candidate["best_known_species"] = best_known
                candidate["predicted_species"] = best_known
                candidate["accepted"] = accepted
                candidate["gate_mode"] = gate_mode
                candidate["inference_recipe"] = inference_recipe
                candidate["known_score"] = float(pred.known_score)
                candidate["threshold"] = float(pred.threshold)
                candidate["score_threshold"] = float(pred.threshold)
                candidate["box_known_score"] = box_known_score
                candidate["box_threshold"] = box_threshold
                candidate["box_accepted"] = box_accepted
                candidate["box_confidence"] = box_confidence
                candidate["sequence_known_score"] = sequence_known_score
                candidate["sequence_threshold"] = sequence_threshold
                candidate["sequence_accepted"] = sequence_accepted
                candidate["sequence_confidence"] = sequence_confidence
                candidate["ridge_score"] = float(pred.ridge_score)
                candidate["raw_ridge_score"] = raw_ridge_score
                candidate["within_score"] = float(pred.within_score)
                if within_margin is not None:
                    candidate["within_margin"] = within_margin
                if whitened_norm is not None:
                    candidate["whitened_norm"] = whitened_norm
                if track_id:
                    candidate["track_id"] = track_id
                if event_id:
                    candidate["event_id"] = event_id

                # Format candidate list for inspection
                cand_list = [candidate]
                if hasattr(pred, "candidates") and pred.candidates:
                    for c in pred.candidates:
                        if isinstance(c, dict):
                            cand_list.append(dict(c))

                box_dict = {
                    "物种": box_name,
                    "置信度": box_conf,
                    "边界框": list(crop.bbox),
                    "候选项": cand_list,
                    "observation_id": obs_id,
                    "accepted": accepted,
                    "gate_mode": gate_mode,
                    "inference_recipe": inference_recipe,
                    "confidence_kind": getattr(pred, "confidence_kind", "sigmoid_gate_margin"),
                    "frame_classification_score": getattr(pred, "frame_classification_score", None),
                    "frame_classification_margin": getattr(pred, "frame_classification_margin", None),
                    "frame_classification_confidence": (
                        box_conf if getattr(pred, "confidence_kind", "") == "sigmoid_frame_class_margin" else None
                    ),
                    "known_score": float(pred.known_score),
                    "threshold": float(pred.threshold),
                    "box_known_score": box_known_score,
                    "box_threshold": box_threshold,
                    "box_accepted": box_accepted,
                    "box_confidence": box_confidence,
                    "sequence_known_score": sequence_known_score,
                    "sequence_threshold": sequence_threshold,
                    "sequence_accepted": sequence_accepted,
                    "sequence_confidence": sequence_confidence,
                    "ridge_score": float(pred.ridge_score),
                    "raw_ridge_score": raw_ridge_score,
                    "within_score": float(pred.within_score),
                    "raw_det_conf": float(crop.detection_confidence),
                    "best_known_species": best_known,
                    "sequence_pending": not complete_sequence,
                    "sequence_status": (
                        SEQUENCE_STATUS_COMPLETE if complete_sequence else partial_status
                    ),
                }
                if within_margin is not None:
                    box_dict["within_margin"] = within_margin
                if whitened_norm is not None:
                    box_dict["whitened_norm"] = whitened_norm
                if track_id:
                    box_dict["track_id"] = track_id
                if event_id:
                    box_dict["event_id"] = event_id
                if not accepted and obs_id in rejection_clusters:
                    box_dict["rejection_cluster"] = rejection_clusters[obs_id]
                new_boxes.append(box_dict)
                species_counts[box_name] = species_counts.get(box_name, 0) + 1
                confidences.append(box_conf)

            min_conf_str = f"{min(confidences):.3f}" if confidences else None
            species_str = ",".join(species_counts.keys()) if species_counts else "无"
            count_str = ",".join(str(species_counts[k]) for k in species_counts) if species_counts else "0"

            existing_data = dict(getattr(item, "detection_data", {}) or {})
            existing_data.update(
                {
                    "物种名称": species_str,
                    "物种数量": count_str,
                    "最低置信度": min_conf_str,
                    "检测框": new_boxes,
                    "sequence_completed": complete_sequence,
                    "sequence_pending": not complete_sequence,
                    "sequence_status": (
                        SEQUENCE_STATUS_COMPLETE if complete_sequence else partial_status
                    ),
                    "confidence_priority": existing_data.get("confidence_priority", "classification"),
                }
            )
            if complete_sequence:
                existing_data.pop("sequence_reason", None)
                existing_data.pop("sequence_processed_count", None)
                existing_data.pop("sequence_total_count", None)
            else:
                existing_data["sequence_reason"] = incomplete_reason
                existing_data["sequence_processed_count"] = processed_count
                if total_photos is not None:
                    existing_data["sequence_total_count"] = int(total_photos)

            if apply_data_fn is not None:
                updated_item = apply_data_fn(item, existing_data)
            else:
                updated_item = item.model_copy(
                    update={
                        "species": list(species_counts.keys()),
                        "confidence": min(confidences) if confidences else None,
                        "detection_boxes": new_boxes,
                        "detection_data": existing_data,
                    }
                )

            updated_results.append(updated_item)
            detections_to_save.append((Path(item.path), existing_data))
            report("生成序列结果", len(updated_results), len(results))

        if save_batch_fn is not None and detections_to_save:
            report("写入序列结果", 0, len(detections_to_save))
            _save_in_chunks(
                detections_to_save, input_path, save_batch_fn,
                progress_fn=lambda done, total: report("写入序列结果", done, total),
            )

        if mutate_job is not None:
            mutate_job(
                results=updated_results,
                message=(
                    f"Neri1.1序列聚合完成: {len(updated_results)} 张照片已确认"
                    if complete_sequence
                    else f"Neri1.1序列部分聚合: {len(updated_results)} 张照片，任务仍需继续"
                ),
            )

        logger.info(
            "DINOv3 %s finalization completed: %d items updated",
            "full" if complete_sequence else "partial",
            len(updated_results),
        )
        if complete_sequence:
            ensure_dinov3_sequences_completed(updated_results)
        return updated_results
    except _SequenceCancelled:
        return results
    finally:
        collector.clear()
