"""Photo sequences and independent video trajectories with optional temporal context."""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import math
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from system.config import SUPPORTED_IMAGE_EXTENSIONS
from .processor import DinoV3CropCollector, DinoV3SequenceFinalizationError
from .video import video_frame_key

MEDIA_SEQUENCE_VERSION = 4


def video_result_signature(signature):
    return hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def completed_video_is_current(item, signature):
    data = item.detection_data or {}
    if (item.error or data.get("sequence_completed") is not True
            or data.get("capture_time", {}).get("timestamp_reliable") is True
            or data.get("video_inference_signature") != signature):
        return False
    try:
        stamp = Path(item.path).stat()
        return data.get("video_source_stamp") == [stamp.st_size, stamp.st_mtime_ns]
    except OSError:
        return False


def capture_time_camera_clock(value):
    """User-confirmed shared photo/video clock; no timezone adjustment."""
    from system.backend.media_time import camera_clock_time

    return camera_clock_time(value)


def trusted_video_time(item):
    metadata = (item.detection_data or {}).get("capture_time", {})
    if metadata.get("timestamp_reliable") is not True:
        return None
    return capture_time_camera_clock(metadata.get("date_taken") or item.date_taken)


def finalize_mixed_media_job(
    detector, results, input_path, *, video_sequences, save_batch_fn,
    apply_data_fn, build_video_data_fn, is_cancelled=None, progress_fn=None,
    complete_sequence=True, incomplete_reason="incomplete_job", job_id=None,
    burst_settings=None,
):
    """Reuse the calibrated photo pipeline, mapping virtual frames back to files.

    Virtual frame keys are in-memory identities, never image paths to be decoded
    or written. Their raw features and true source metadata come from the cache.
    """
    from . import processor
    surviving = []
    for item in results:
        try:
            Path(item.path).stat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        surviving.append(item)
    results = surviving
    photos = [item for item in results if Path(item.path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    entries = {str(item.path): video_sequences[str(item.path)] for item in results
               if str(item.path) in video_sequences and not item.error}
    if not entries:
        finalized = processor.finalize_photo_job(
            detector, photos, input_path, save_batch_fn=save_batch_fn, apply_data_fn=apply_data_fn,
            is_cancelled=is_cancelled, progress_fn=progress_fn, complete_sequence=complete_sequence,
            incomplete_reason=incomplete_reason, total_photos=len(photos), job_id=job_id,
        ) if photos else []
        by_path = {item.path: item for item in finalized}
        return [by_path.get(item.path, item) for item in results]
    if is_cancelled and is_cancelled():
        return results
    from .video_reference import resolve_video_references
    references = resolve_video_references(results, **(burst_settings or {}))

    def validate_sources():
        for source_path, (_, _, info) in entries.items():
            stamp = info.get("source_stamp")
            if stamp is not None:
                try:
                    current = Path(source_path).stat()
                    unchanged = stamp == [current.st_size, current.st_mtime_ns]
                except OSError:
                    unchanged = False
                if not unchanged:
                    raise DinoV3SequenceFinalizationError("视频源文件在联合序列汇总前已变化，请重新处理")

    validate_sources()

    combined = DinoV3CropCollector()
    proxies = []
    metadata = {}
    frame_keys = {}
    original_collector = detector._dinov3_collector
    active_photos = []
    for item in photos:
        digest = original_collector.image_digest(item.path)
        try:
            stat = Path(item.path).stat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        active_photos.append(item)
        captured = capture_time_camera_clock(item.date_taken)
        proxies.append(item.model_copy(update={"date_taken": captured.isoformat() if captured else None}))
        combined.record_crops(original_collector.get_crops_for_path(item.path))
        if digest:
            combined.record_image_digest(item.path, digest, (stat.st_size, stat.st_mtime_ns))
    removed_photos = {item.path for item in photos} - {item.path for item in active_photos}
    photos = active_photos
    results = [item for item in results if item.path not in removed_photos]
    for source_path, (item, frames, info) in entries.items():
        reference = references[source_path]
        captured = capture_time_camera_clock(reference["reference_time"])
        fps = float(info.get("fps") or 25)
        if not math.isfinite(fps) or fps <= 0:
            raise DinoV3SequenceFinalizationError("Invalid sampled video FPS")
        ordered = sorted(frames, key=lambda frame: frame[0])
        indices = [frame[1] for frame in ordered]
        if (not ordered or indices != info.get("frame_points")
                or len(set(indices)) != len(indices)
                or [frame[0] for frame in ordered] != list(range(len(ordered)))):
            raise DinoV3SequenceFinalizationError("视频采样帧不完整或顺序无效")
        frame_keys[source_path] = []
        offset = 0
        for order, index, data, crops in ordered:
            key = video_frame_key(source_path, index)
            frame_keys[source_path].append((order, index, key))
            indices = {crop.box_index: offset + rank for rank, crop in enumerate(crops)}
            metadata[key] = dict(source_path=source_path, frame_index=index,
                timestamp=index / fps, observation_indices=indices,
                reference_source=reference["source"],
                content_digest=data.get("_frame_digest") or f"{source_path}:frame:{index}")
            offset += len(crops)
            frame_time = (captured + timedelta(seconds=index / fps)
                          if captured is not None and reference["source"] == "video" else captured)
            proxies.append(item.model_copy(update={
                "path": key, "date_taken": frame_time.isoformat() if frame_time is not None else None,
                "detection_data": data, "detection_boxes": [], "species": [], "error": None,
            }))
            combined.record_crops([replace(crop, image_path=key) for crop in crops])
    adapter = SimpleNamespace(dinov3_classifier=detector.dinov3_classifier, _dinov3_collector=combined)
    try:
        finalized = processor.finalize_photo_job(
            adapter, proxies, input_path, media_metadata=metadata,
            is_cancelled=is_cancelled, progress_fn=progress_fn, save_batch_fn=None,
            apply_data_fn=apply_data_fn, complete_sequence=complete_sequence,
            incomplete_reason=incomplete_reason, total_photos=len(proxies), job_id=job_id,
        )
        if is_cancelled and is_cancelled():
            return results
        by_key = {item.path: item for item in finalized}
        by_source = {}
        for item in photos:
            by_source[item.path] = apply_data_fn(item, by_key[item.path].detection_data)
        for source_path, (item, _, info) in entries.items():
            frames = [(order, index, by_key[key].detection_data)
                      for order, index, key in frame_keys[source_path]]
            data = build_video_data_fn(Path(source_path), info, frames)
            data.update(capture_time=item.detection_data.get("capture_time"),
                        video_reference=references[source_path], classification_scope="video_track",
                        sequence_completed=complete_sequence, sequence_pending=not complete_sequence,
                        sequence_status="complete" if complete_sequence else "partial")
            if item.detection_data.get("burst_group_id") is not None:
                data["burst_group_id"] = item.detection_data["burst_group_id"]
            if not complete_sequence:
                data["sequence_reason"] = incomplete_reason
            by_source[source_path] = apply_data_fn(item, data)
        for item in by_source.values():
            item.detection_data.update(media_sequence_version=MEDIA_SEQUENCE_VERSION,
                                       camera_clock_policy="recorded_wall_time", evidence_balance="within_sequence")
        updated = [by_source.get(item.path, item) for item in results]
        if complete_sequence:
            processor.ensure_dinov3_sequences_completed(list(by_source.values()))
        if is_cancelled and is_cancelled():
            return results
        validate_sources()
        if save_batch_fn:
            processor._save_in_chunks([(Path(item.path), item.detection_data) for item in by_source.values()],
                                     input_path, save_batch_fn)
        return updated
    finally:
        combined.clear()
        original_collector.clear()
