from pathlib import Path
from dataclasses import replace

import pytest

from system.backend import services
from system.backend.models import CreateJobRequest, JobState, JobSummary, ProcessingOptions
from system.dinov3.media_cache import DinoV3VideoCache
from system.dinov3.video import video_frame_key
from tests.test_dinov3_mixed_sequences import fixture, record
from tests.test_dinov3_video_job import _setup_dinov3_job_env


def setup(monkeypatch, tmp_path, *, cancel_first=False, start="2026-11-16T00:00:10+00:00"):
    detector, items, videos = fixture(tmp_path, start=start)
    first = Path(items[1].path)
    second = tmp_path / "second.avi"
    second.write_bytes(b"second video")
    second_item = items[1].model_copy(update={"filename": second.name, "path": str(second)})
    second_frames = [(order, index, data, [replace(crop, image_path=video_frame_key(second, index)) for crop in crops])
                     for order, index, data, crops in videos[str(first)][1]]
    videos[str(second)] = (second_item, second_frames, videos[str(first)][2])
    files = [Path(items[0].path), first, second]
    legacy, state_path, model, manifest, exported = _setup_dinov3_job_env(monkeypatch, tmp_path, files)
    monkeypatch.setattr(legacy, "_load_detector", lambda *a: detector)
    encoded_photos, encoded_videos = [], []
    def photos(det, paths, photo_items, *args, **kwargs):
        encoded_photos.extend(paths)
        for path in paths:
            det._dinov3_collector.record_crop(record(path))
        return [items[0]]
    monkeypatch.setattr(legacy, "_detect_image_batch", photos)
    control = {"cancel": cancel_first, "manager": None}
    def fast(det, paths, incoming, req, root, *, on_item_ready, on_video_features=None, **kwargs):
        output = []
        for index, path in enumerate(paths):
            encoded_videos.append(path)
            assert on_video_features is not None
            item, frames, info = videos[str(path)]
            pending = on_video_features(index, item, frames, info)
            assert not pending.detection_data["sequence_completed"]
            output.append(pending)
            on_item_ready(index, pending)
            if control["cancel"]:
                control["manager"]._cancelled_jobs.add("job")
                if control.get("return_on_cancel"):
                    return output
                raise RuntimeError("用户取消任务")
        return output
    monkeypatch.setattr(legacy, "_detect_video_fast_batch", fast)
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(
        enable_detection=True, model_path=str(model), classification_model_path=str(manifest),
        batch_size=2, video_mode="fast"))
    manager = services.ProcessingJobManager()
    now = services.utc_now()
    manager._jobs["job"] = JobSummary(id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
                                    created_at=now, updated_at=now)
    manager._job_requests["job"] = request
    control["manager"] = manager
    return manager, request, detector, encoded_photos, encoded_videos, control, state_path, files


def test_job_defers_photos_until_all_video_features_join_the_same_event(monkeypatch, tmp_path):
    manager, request, detector, photos, videos, control, state, files = setup(monkeypatch, tmp_path)
    try:
        manager._run_job("job", request, [])
        job = manager.get_job("job")
        assert job.state == JobState.COMPLETED, job.error
        assert photos == files[:1] and videos == files[1:]
        assert [item.path for item in job.results] == list(map(str, files))
        assert all(item.detection_data["sequence_completed"] for item in job.results)
        assert len(detector.dinov3_classifier.calls) == 1
        raw, context = detector.dinov3_classifier.calls[0]
        assert len(raw) == 7 and len(set(context["sequence_ids"])) == 3
        assert "evidence_weights" not in context
        assert not DinoV3VideoCache.path_for(state, "job").exists()
    finally:
        manager._executor.shutdown()


@pytest.mark.parametrize("start", [None, "2026-11-16T00:00:10+00:00"])
def test_cancel_and_resume_reuses_video_features_then_refinalizes_mixed_event(monkeypatch, tmp_path, start):
    manager, request, detector, photos, videos, control, state, files = setup(monkeypatch, tmp_path, cancel_first=True, start=start)
    restarted = None
    try:
        manager._run_job("job", request, [])
        cancelled = manager.get_job("job")
        assert cancelled.state == JobState.CANCELLED
        assert detector.dinov3_classifier.calls == []
        assert DinoV3VideoCache.path_for(state, "job").exists()
        assert all(not item.detection_data.get("sequence_completed") for item in cancelled.results)
        control["cancel"] = False
        restarted = services.ProcessingJobManager()
        control["manager"] = restarted
        restarted.resume_job("job")
        restarted._executor.shutdown(wait=True)
        finished = restarted.get_job("job")
        assert finished.state == JobState.COMPLETED, finished.error
        assert photos == files[:1]
        assert videos == files[1:]
        assert len(detector.dinov3_classifier.calls) == 1
        assert detector.dinov3_classifier.calls[0][0].shape == (7, 768)
        assert all(item.detection_data["sequence_completed"] for item in finished.results)
        assert not DinoV3VideoCache.path_for(state, "job").exists()
    finally:
        manager._executor.shutdown()
        if restarted:
            restarted._executor.shutdown()


@pytest.mark.parametrize("storage_failure", [False, True])
def test_cancel_at_normal_video_return_always_marks_partial_even_if_db_write_fails(monkeypatch, tmp_path, storage_failure):
    manager, request, detector, photos, videos, control, state, files = setup(monkeypatch, tmp_path, cancel_first=True)
    control["return_on_cancel"] = True
    def save(pairs, root):
        if storage_failure and any(data.get("sequence_status") == "partial" for _, data in pairs):
            raise OSError("disconnected input drive")
    monkeypatch.setattr(services._legacy, "_save_detection_data_batch", save)
    try:
        manager._run_job("job", request, [])
        job = manager.get_job("job")
        assert job.state == JobState.CANCELLED
        assert all(item.detection_data["sequence_status"] == "partial" for item in job.results)
        assert all(item.detection_data["sequence_reason"] == "cancelled" for item in job.results)
    finally:
        manager._executor.shutdown()


def test_cancel_after_joint_db_save_clears_complete_status_before_export(monkeypatch, tmp_path):
    manager, request, detector, photos, videos, control, state, files = setup(monkeypatch, tmp_path)
    saved_complete = False
    def save(pairs, root):
        nonlocal saved_complete
        if any(data.get("sequence_completed") for _, data in pairs):
            saved_complete = True
    checked = services._legacy._checked_media_file
    def check(path):
        if saved_complete:
            manager._cancelled_jobs.add("job")
        return checked(path)
    monkeypatch.setattr(services._legacy, "_save_detection_data_batch", save)
    monkeypatch.setattr(services._legacy, "_checked_media_file", check)
    try:
        manager._run_job("job", request, [])
        job = manager.get_job("job")
        assert job.state == JobState.CANCELLED
        assert all(item.detection_data["sequence_status"] == "partial" for item in job.results)
        assert all(not item.detection_data["sequence_completed"] for item in job.results)
    finally:
        manager._executor.shutdown()
