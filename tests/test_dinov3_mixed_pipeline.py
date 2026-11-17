from pathlib import Path
import struct

import cv2
import numpy as np
import pytest
from PIL import Image

from system.backend import services
from system.backend.models import CreateJobRequest, JobState, JobSummary, ProcessingOptions
from tests.test_dinov3_video_fast import Classifier, make_detector
from tests.test_dinov3_video_job import _setup_dinov3_job_env


def write_pair(root, include_time=True):
    photo = root / "photo.jpg"
    image = Image.new("RGB", (96, 64), (80, 50, 20))
    exif = image.getexif()
    exif[36867] = "2026:11:16 08:00:00"
    image.save(photo, exif=exif)
    video = root / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (96, 64))
    assert writer.isOpened()
    for i in range(100):
        writer.write(np.full((64, 96, 3), [20, 50, i * 2], np.uint8))
    writer.release()
    if not include_time:
        return photo, video
    # Camera-style embedded AVI capture date, readable without ffprobe/PyTorch.
    value = b"2026-11-16T08:00:10\0"
    chunk = b"ICRD" + struct.pack("<I", len(value)) + value + b"\0" * (len(value) % 2)
    info = b"INFO" + chunk
    with video.open("r+b") as stream:
        stream.seek(4)
        previous = struct.unpack("<I", stream.read(4))[0]
        trailer = b"LIST" + struct.pack("<I", len(info)) + info
        stream.seek(4)
        stream.write(struct.pack("<I", previous + len(trailer)))
        stream.seek(0, 2)
        stream.write(trailer)
    return photo, video


@pytest.mark.parametrize("scenario", ["video", "burst_group", "none"])
@pytest.mark.parametrize("previous_failure", [False, True])
def test_real_avi_metadata_and_photo_exif_use_track_time_priority(monkeypatch, tmp_path, scenario, previous_failure):
    photo, video = write_pair(tmp_path, include_time=scenario == "video")
    photos = [photo]
    if scenario == "burst_group":
        renamed = photo.with_name("IMG_0001.jpg")
        photo.rename(renamed)
        photos = [renamed]
        for index in (2, 3):
            destination = photo.with_name(f"IMG_{index:04d}.jpg")
            with Image.open(renamed) as image:
                exif = image.getexif()
                exif[36867] = f"2026:11:16 08:00:0{index - 1}"
                image.save(destination, exif=exif)
            photos.append(destination)
        renamed_video = video.with_name("IMG_0004.avi")
        video.rename(renamed_video)
        video = renamed_video
    build_metadata = services._legacy._build_metadata_item
    legacy, _, model, manifest, _ = _setup_dinov3_job_env(monkeypatch, tmp_path, [*photos, video])
    if previous_failure:
        monkeypatch.setattr(legacy, "load_media_indexes", lambda *a, **kw: ({str(video): {"error": "previous failure"}}, {}))
    monkeypatch.setattr(legacy, "_build_metadata_item", build_metadata)
    classifier = Classifier()
    detector = make_detector(classifier)
    detector._legacy.release_batch_images = lambda: None
    detector._legacy.preload_batch_data = lambda paths: detector.preload_frame_data([cv2.imread(p) for p in paths])
    monkeypatch.setattr(legacy, "_load_detector", lambda *args: detector)
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(
        enable_detection=True, model_path=str(model), classification_model_path=str(manifest),
        batch_size=2, imgsz=640, vid_stride=3, use_augment=False,
    ))
    manager = services.ProcessingJobManager()
    try:
        now = services.utc_now()
        manager._jobs["job"] = JobSummary(id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
                                        created_at=now, updated_at=now)
        manager._job_requests["job"] = request
        manager._run_job("job", request, [])
        job = manager.get_job("job")
        assert job.state == JobState.COMPLETED, job.error
        assert job.failed_count == 0 and all(item.error is None for item in job.results)
        assert len(classifier.calls) == 1 and sum(classifier.encoder.batch_sizes) == len(photos) + 3
        assert len(set(classifier.calls[0][1]["sequence_ids"])) == 2
        assert "evidence_weights" not in classifier.calls[0][1]
        assert [len(item.detection_boxes) for item in job.results] == [1] * len(photos) + [3]
        result = job.results[-1]
        assert result.date_taken == ("2026-11-16T08:00:10" if scenario == "video" else None)
        assert result.detection_data["video_reference"]["source"] == scenario
        assert [box["frame_index"] for box in result.detection_data["检测框"]] == [25, 50, 75]
    finally:
        manager._executor.shutdown()
