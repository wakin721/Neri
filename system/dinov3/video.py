"""Complete sampled-video sequences without temporary images or re-encoding."""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import os

import numpy as np

from .context import camera_id_for_path, group_job_events
from .observation_support import make_dinov3_observation_id


def video_frame_key(path, frame_index):
    """Internal collector key only; no file is created at this path."""
    return f"{path}.neri-frame-{frame_index:012d}"


def group_video_trajectories(records, path):
    """Associate targets using relative clip time, keeping real history time separate."""
    if not records:
        return []
    identity = hashlib.sha256(os.path.normcase(os.path.abspath(str(path))).encode("utf-8")).hexdigest()
    event_id = f"video:{identity}"
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    relative = [{**record, "camera_id": event_id,
                 "date_taken": epoch + timedelta(seconds=record["frame_seconds"])} for record in records]
    span = max(record["frame_seconds"] for record in records) - min(record["frame_seconds"] for record in records)
    grouped = group_job_events(relative, gap_seconds=max(1800, int(span) + 1))
    original = {record["crop_idx"]: record for record in records}
    tracks = {}
    output = []
    from .context import parse_capture_time
    for group in grouped:
        track = tracks.setdefault(group["track_id"], f"{event_id}:track_{len(tracks):02d}")
        record = original[group["crop_idx"]]
        output.append({**group, "camera_id": record["camera_id"],
                       "datetime": parse_capture_time(record.get("date_taken")),
                       "date_taken": record.get("date_taken"),
                       "event_id": event_id, "track_id": track, "sequence_id": track})
    return output


def finalize_video_frames(classifier, path, frames, *, fps, input_path, check_cancelled):
    """Return finalized (order, frame index, data) records for one whole clip.

    Frames carry crop embeddings captured immediately after each successful
    inference batch, before a retry or subsequent call can replace its state.
    Only sampled features/metadata survive decoding, never full-sized pixels.
    """
    check_cancelled()
    ordered = sorted(frames, key=lambda frame: frame[0])
    crops = []
    records = []
    camera = camera_id_for_path(path, root=input_path)
    for order, frame_index, data, frame_crops in ordered:
        check_cancelled()
        expected = data.get("detection_box_indices")
        if data.get("error") or data.get("detection_completed") is not True:
            raise RuntimeError(data.get("error") or "视频帧未返回完整检测结果")
        if expected is None or sorted(expected) != sorted(crop.box_index for crop in frame_crops):
            raise RuntimeError(f"视频帧特征不完整: frame={frame_index}")
        for crop in frame_crops:
            records.append(dict(
                crop_idx=len(crops), camera_id=camera,
                path=video_frame_key(path, frame_index), box_index=crop.box_index,
                bbox=crop.bbox, raw_embedding=crop.raw_embedding,
                detector_category=crop.detector_category,
                date_taken=None, frame_seconds=frame_index / fps,
            ))
            crops.append(crop)

    predictions = []
    groups = {}
    if crops:
        # Force a single clip event even for long clips, while retaining the
        # existing same-frame exclusion and visual/spatial target association.
        grouped = group_video_trajectories(records, path)
        groups = {record["crop_idx"]: record for record in grouped}
        features = np.stack([crop.raw_embedding for crop in crops])
        sequences = [groups[i]["track_id"] for i in range(len(crops))]
        content_ids = [f"{record['path']}:{record['box_index']}" for record in records]
        check_cancelled()
        try:
            from threadpoolctl import threadpool_limits
            limits = threadpool_limits(limits=4)
        except ImportError:
            limits = nullcontext()
        with limits:
            # All samples belong to this event; there are no prior events to
            # adapt with. Do not use another clip merely because it decoded first.
            predictions = classifier.classify_features(
                features, camera_ids=[camera] * len(crops), sequence_ids=sequences,
                content_ids=content_ids, site_ids=[camera] * len(crops),
                adapted_features=features,
            )
        check_cancelled()
        if len(predictions) != len(crops):
            raise RuntimeError("DINOv3视频分类结果数量与目标数量不一致")

    output = []
    offset = 0
    for order, frame_index, data, frame_crops in ordered:
        check_cancelled()
        boxes = []
        counts = Counter()
        confidences = []
        for crop in frame_crops:
            prediction = predictions[offset]
            group = groups[offset]
            # Offset is global within this video's persisted box array.
            observation_id = make_dinov3_observation_id(path, offset, crop.bbox)
            offset += 1
            if crop.detector_category in {"person", "vehicle"}:
                from .detection_categories import detector_category_box
                box = detector_category_box(crop.detector_category, crop.detection_confidence, crop.bbox,
                    observation_id=observation_id, track_id=group["track_id"], event_id=group["event_id"])
                boxes.append(box)
                counts[box["物种"]] += 1
                confidences.append(crop.detection_confidence)
                continue
            name = prediction.best_known_species if prediction.accepted else "Unknown"
            candidate = prediction.as_candidate(detection_confidence=crop.detection_confidence)
            candidate.update(name=name, observation_id=observation_id,
                             sequence_pending=False, sequence_status="complete",
                             track_id=group["track_id"], event_id=group["event_id"])
            boxes.append({
                **candidate, "物种": name, "置信度": prediction.confidence,
                "边界框": list(crop.bbox), "候选项": [candidate, *prediction.candidates],
            })
            counts[name] += 1
            confidences.append(prediction.confidence)
        output.append((order, frame_index, {
            **data, "检测框": boxes,
            "物种名称": ",".join(counts) if counts else "空",
            "物种数量": ",".join(map(str, counts.values())) if counts else "空",
            "最低置信度": min(confidences) if confidences else None,
            "sequence_completed": True, "sequence_pending": False, "sequence_status": "complete",
        }))
    return output
