"""Replay a real model on disposable camera media for V2 policy scenarios.

Only the output directory is written. Model weights, learning state and original
media are read-only. This is a functional smoke test, not a calibration benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any

SCENARIOS = ("own", "burst", "none")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for real-model smoke verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--runtime", choices=("litert", "cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="own")
    return parser


def _write_exif_photo(image, destination: Path, exif_timestamp: str) -> None:
    exif = image.getexif()
    exif[36867] = exif_timestamp
    image.save(destination, exif=exif)


def _write_avi_clip(bgr, destination: Path, *, embedded_icrd: bytes | None) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("Unable to create the disposable smoke video")
    try:
        for index in range(100):
            writer.write(np.roll(bgr, index % 4, axis=1))
    finally:
        writer.release()
    if embedded_icrd is None:
        return
    chunk = b"ICRD" + struct.pack("<I", len(embedded_icrd)) + embedded_icrd + b"\0" * (len(embedded_icrd) % 2)
    info = b"INFO" + chunk
    trailer = b"LIST" + struct.pack("<I", len(info)) + info
    with destination.open("r+b") as stream:
        stream.seek(4)
        size = struct.unpack("<I", stream.read(4))[0]
        stream.seek(4)
        stream.write(struct.pack("<I", size + len(trailer)))
        stream.seek(0, 2)
        stream.write(trailer)


def prepare_disposable_camera(camera_dir: Path, source_image: Path, scenario: str) -> tuple[list[Path], Path]:
    """Create disposable camera media matching ``scenario`` (own|burst|none)."""
    import cv2
    import numpy as np
    from PIL import Image

    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported verification scenario: {scenario!r}")
    camera_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image) as original:
        image = original.convert("RGB").resize((640, 480))
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    if scenario == "burst":
        photo_specs = [
            ("IMG_0001.jpg", "2026:11:16 08:00:00"),
            ("IMG_0002.jpg", "2026:11:16 08:00:01"),
            ("IMG_0003.jpg", "2026:11:16 08:00:02"),
        ]
        video_name = "IMG_0004.avi"
        embedded_icrd = None
    elif scenario == "own":
        photo_specs = [("photo.jpg", "2026:11:16 08:00:00")]
        video_name = "clip.avi"
        embedded_icrd = b"2026-11-16T08:00:10\0"
    else:
        photo_specs = [("photo.jpg", "2026:11:16 08:00:00")]
        video_name = "clip.avi"
        embedded_icrd = None

    expected_names = {name for name, _ in photo_specs} | {video_name}
    for known in ("photo.jpg", "clip.avi", "IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg", "IMG_0004.avi"):
        candidate = camera_dir / known
        if candidate.exists() and known not in expected_names:
            candidate.unlink()

    photos: list[Path] = []
    for name, exif_ts in photo_specs:
        photo_path = camera_dir / name
        _write_exif_photo(image, photo_path, exif_ts)
        photos.append(photo_path)
    video_path = camera_dir / video_name
    _write_avi_clip(bgr, video_path, embedded_icrd=embedded_icrd)
    return photos, video_path


def validate_job_results(job: Any, calls: list[dict[str, Any]], scenario: str) -> None:
    """Assert V2 video track isolation, sequence IDs, reference time, and provenance."""
    from system.backend.models import JobState
    from system.config import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
    from system.dinov3.observation_support import parse_dinov3_observation_id
    from system.dinov3.mixed_media import MEDIA_SEQUENCE_VERSION

    if job.state != JobState.COMPLETED or job.failed_count:
        raise RuntimeError(f"Mixed job did not complete: {job.error}; {[i.error for i in job.results]}")

    image_exts = {ext.lower() for ext in SUPPORTED_IMAGE_EXTENSIONS}
    video_exts = {ext.lower() for ext in SUPPORTED_VIDEO_EXTENSIONS}
    photo_results = [item for item in job.results if Path(item.path).suffix.lower() in image_exts]
    video_results = [item for item in job.results if Path(item.path).suffix.lower() in video_exts]

    expected_photo_count = 3 if scenario == "burst" else 1
    assert len(photo_results) == expected_photo_count
    assert len(video_results) == 1
    assert len(job.results) == expected_photo_count + 1
    assert len(calls) == 1

    call = calls[0]
    assert not call.get("has_evidence_weights") and "evidence_weights" not in call
    all_seq_ids = list(call["sequence_ids"])
    assert call["crops"] == len(all_seq_ids)

    photo_box_count = sum(len(item.detection_boxes) for item in photo_results)
    video_box_count = sum(len(item.detection_boxes) for item in video_results)
    assert photo_box_count > 0 and video_box_count > 0
    assert len(all_seq_ids) == photo_box_count + video_box_count

    photo_seq_ids = all_seq_ids[:photo_box_count]
    video_seq_ids = all_seq_ids[photo_box_count:]
    assert set(photo_seq_ids).isdisjoint(video_seq_ids)
    assert all(isinstance(seq_id, str) and seq_id.startswith("video:") for seq_id in video_seq_ids)

    for item in job.results:
        assert item.detection_data.get("media_sequence_version") == MEDIA_SEQUENCE_VERSION
        assert item.detection_data.get("sequence_completed") is True
        assert len(item.detection_boxes) > 0
        boxes = item.detection_data.get("检测框") or []
        assert len(boxes) == len(item.detection_boxes)
        for index, box in enumerate(boxes):
            source, mapped_index, _ = parse_dinov3_observation_id(box["observation_id"])
            assert source == Path(item.path).resolve() and mapped_index == index

    video_item = video_results[0]
    vdata = video_item.detection_data
    assert vdata.get("classification_scope") == "video_track"
    sampled_frames = vdata.get("sampled_frames") or []
    assert len(sampled_frames) > 0

    by_frame: dict[int, list[str]] = {}
    persisted_video_tracks: list[str] = []
    for box in vdata["检测框"]:
        frame_index = box.get("frame_index")
        timestamp = box.get("timestamp")
        track_id = box.get("track_id")
        assert frame_index in sampled_frames
        assert isinstance(timestamp, (int, float)) and timestamp >= 0
        assert isinstance(track_id, str) and track_id.startswith("video:")
        by_frame.setdefault(frame_index, []).append(track_id)
        persisted_video_tracks.append(track_id)

    for frame_tracks in by_frame.values():
        assert len(set(frame_tracks)) == len(frame_tracks)
    assert persisted_video_tracks == video_seq_ids

    capture_time = vdata.get("capture_time") or {}
    ref = vdata.get("video_reference")
    assert isinstance(ref, dict)

    if scenario == "own":
        assert video_item.date_taken is not None
        assert capture_time.get("timestamp_reliable") is True
        assert capture_time.get("date_taken") is not None
        assert ref.get("source") == "video"
        assert ref.get("precision") == "exact"
        assert ref.get("reference_time") == "2026-11-16T00:00:10+00:00"
        assert ref.get("burst_group_id") is None
        assert ref.get("group_members") == []
        assert ref.get("group_end_time") is None
    elif scenario == "burst":
        assert video_item.date_taken is None
        assert capture_time.get("timestamp_reliable") is False
        assert capture_time.get("date_taken") is None
        assert ref.get("source") == "burst_group"
        assert ref.get("precision") == "group_anchor"
        assert ref.get("reference_time") == "2026-11-16T00:00:00+00:00"
        assert ref.get("group_end_time") == "2026-11-16T00:00:02+00:00"
        assert isinstance(ref.get("burst_group_id"), str) and bool(ref["burst_group_id"])
        expected_members = {str(Path(item.path).resolve()) for item in [*photo_results, video_item]}
        actual_members = {str(Path(member).resolve()) for member in (ref.get("group_members") or [])}
        assert len(ref.get("group_members") or []) == 4 and actual_members == expected_members
    else:
        assert video_item.date_taken is None
        assert capture_time.get("timestamp_reliable") is False
        assert capture_time.get("date_taken") is None
        assert ref.get("source") == "none"
        assert ref.get("precision") == "unknown"
        assert ref.get("reference_time") is None
        assert ref.get("burst_group_id") is None
        assert ref.get("group_members") == []
        assert ref.get("group_end_time") is None


def main():
    parser = build_parser()
    args = parser.parse_args()
    root, output = args.application_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    yolo_config = output / "yolo-config"
    yolo_config.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config)
    sys.path.insert(0, str(root))
    from system.backend import services
    from system.backend.models import CreateJobRequest, JobState, JobSummary, ProcessingOptions
    from system.dino.image_processor import ImageProcessor
    from system.dinov3.runtime import load_dinov3_model

    device = "cuda" if args.runtime == "gpu" else "cpu"
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU smoke requested but CUDA is unavailable")
    camera = output / "camera"
    prepare_disposable_camera(camera, args.source_image, args.scenario)

    print(f"Loading real models: {args.runtime} (scenario={args.scenario})", flush=True)
    runtime = load_dinov3_model(args.manifest.resolve(), device=device, use_shared_learning=False)
    detector = ImageProcessor(str(args.detector.resolve()))
    if detector.model is None:
        raise RuntimeError("Detector failed to load")
    detector.load_dino_classifier(runtime.classifier)
    if args.runtime != "litert":
        detector._legacy._determine_device = lambda half: (device, False)
    calls = []
    classify = runtime.classifier.classify_features
    def measured(features, **kwargs):
        if "evidence_weights" in kwargs:
            raise AssertionError("Classifier must not receive evidence_weights under V2")
        calls.append(dict(crops=len(features), sequence_ids=list(kwargs["sequence_ids"])))
        return classify(features, **kwargs)
    runtime.classifier.classify_features = measured
    services._legacy._load_detector = lambda *a: detector
    services.job_state_path = services._legacy.job_state_path = lambda: output / "job-state.json"
    manager = services.ProcessingJobManager()
    try:
        request = CreateJobRequest(input_dir=str(camera), output_dir=str(output / "export"),
            options=ProcessingOptions(enable_detection=True, model_path=str(args.detector.resolve()),
                classification_model_path=str(args.manifest.resolve()), batch_size=2,
                imgsz=640, vid_stride=3, video_mode="fast", use_fp16=False, use_augment=False,
                confidence=0.1))
        now = services.utc_now()
        manager._jobs["mixed-smoke"] = JobSummary(id="mixed-smoke", state=JobState.QUEUED,
            input_dir=str(camera), created_at=now, updated_at=now)
        manager._job_requests["mixed-smoke"] = request
        started = time.perf_counter()
        manager._run_job("mixed-smoke", request, [])
        seconds = time.perf_counter() - started
        job = manager.get_job("mixed-smoke")
        validate_job_results(job, calls, args.scenario)
        report = dict(passed=True, scenario=args.scenario, runtime=args.runtime,
            application_root=str(root), seconds=seconds, classification_calls=calls,
            files=[dict(path=i.path, date_taken=i.date_taken, boxes=len(i.detection_boxes),
            species=i.species, data=i.detection_data) for i in job.results],
            note="Real models on a disposable still-derived mixed pair; not an accuracy/calibration benchmark")
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("passed", "scenario", "runtime", "seconds", "classification_calls")}), flush=True)
    finally:
        manager._executor.shutdown(wait=True)
        runtime.close()


if __name__ == "__main__":
    main()
