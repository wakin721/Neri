from pathlib import Path
import json

import pytest

from system.backend import services
from system.backend.models import (
    CreateJobRequest,
    DetectionItem,
    JobState,
    JobSummary,
    ProcessingOptions,
)


def _request(input_dir: Path) -> CreateJobRequest:
    return CreateJobRequest(
        input_dir=str(input_dir),
        options=ProcessingOptions(
            enable_detection=True,
            model_path="fake.pt",
            batch_size=1,
            thread_count=1,
            video_mode="skip",
        ),
    )


def _job(job_id: str, input_dir: Path) -> JobSummary:
    now = services.utc_now()
    return JobSummary(
        id=job_id,
        state=JobState.QUEUED,
        input_dir=str(input_dir),
        created_at=now,
        updated_at=now,
    )


def _patch_state_path(monkeypatch, state_path: Path) -> None:
    resolver = lambda: state_path
    monkeypatch.setattr(services, "job_state_path", resolver)
    monkeypatch.setattr(services._legacy, "job_state_path", resolver)


def test_resume_reuses_first_scan_after_restart_and_skips_deleted_pending_files(monkeypatch, tmp_path):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    root = tmp_path / "media"
    root.mkdir()
    first, second, deleted = [root / name for name in ("1.jpg", "2.jpg", "3.jpg")]
    for path in (first, second, deleted):
        path.write_bytes(b"fixture")
    request = _request(root)
    request.options.enable_detection = False
    manager = services.ProcessingJobManager()
    manager._jobs["cached"] = _job("cached", root)
    manager._job_requests["cached"] = request
    legacy = services._legacy
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *args: None)
    metadata = legacy._build_fast_metadata_item

    def cancel_after_first(path):
        item = metadata(path)
        manager._cancelled_jobs.add("cached")
        return item

    monkeypatch.setattr(legacy, "_build_fast_metadata_item", cancel_after_first)
    restarted = None
    try:
        manager._run_job("cached", request, [])
        paused = manager.get_job("cached")
        assert paused.state == JobState.CANCELLED and paused.processed == 1
        (root / "new.jpg").write_bytes(b"new file belongs to a new task")
        deleted.unlink()
        monkeypatch.setattr(legacy, "_build_fast_metadata_item", metadata)
        monkeypatch.setattr(legacy, "_resolve_job_inputs",
                            lambda *args, **kwargs: pytest.fail("resume must not scan source directories"))
        restarted = services.ProcessingJobManager()
        messages = []
        mutate = restarted._mutate_job
        def capture_message(*args, **kwargs):
            messages.append(kwargs.get("message", ""))
            return mutate(*args, **kwargs)
        monkeypatch.setattr(restarted, "_mutate_job", capture_message)
        restarted.resume_job("cached")
        restarted._executor.shutdown(wait=True)
        finished = restarted.get_job("cached")
        assert finished.state == JobState.COMPLETED
        assert finished.processed == finished.total == 2
        assert [item.path for item in finished.results] == [str(first), str(second)]
        assert "正在预扫描输入文件" not in messages
        restarted.delete_job("cached")
        import sqlite3
        with sqlite3.connect(restarted._state_store.path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM job_inputs").fetchone()[0] == 0
    finally:
        manager._executor.shutdown()
        if restarted is not None:
            restarted._executor.shutdown()


def test_cached_inputs_are_scoped_to_task_and_video_filter(monkeypatch, tmp_path):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    (tmp_path / "photo.jpg").write_bytes(b"photo")
    (tmp_path / "video.mp4").write_bytes(b"video")
    manager = services.ProcessingJobManager()
    request = _request(tmp_path)
    manager._jobs["one"] = _job("one", tmp_path)
    manager._job_requests["one"] = request
    try:
        _, paths = manager._resolve_inputs("one", request)
        assert [path.name for path in paths] == ["photo.jpg"]
        (tmp_path / "new.jpg").write_bytes(b"new")
        assert manager._resolve_inputs("one", request)[1] == paths
        changed = request.model_copy(deep=True)
        changed.options.video_mode = "fast"
        assert {path.name for path in manager._resolve_inputs("one", changed)[1]} == {
            "photo.jpg", "new.jpg", "video.mp4",
        }
        manager._jobs["two"] = _job("two", tmp_path)
        manager._job_requests["two"] = request
        assert {path.name for path in manager._resolve_inputs("two", request)[1]} == {"photo.jpg", "new.jpg"}
    finally:
        manager._executor.shutdown()


def test_failed_or_missing_snapshot_falls_back_to_one_complete_scan(monkeypatch, tmp_path):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    manager = services.ProcessingJobManager()
    request = _request(tmp_path)
    manager._jobs["old"] = _job("old", tmp_path)
    manager._job_requests["old"] = request
    calls = []
    def scan(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise services.MediaStorageError(tmp_path, "扫描", OSError("disconnected"))
        return tmp_path, [tmp_path / "photo.jpg"]
    monkeypatch.setattr(services._legacy, "_resolve_job_inputs", scan)
    try:
        with pytest.raises(services.MediaStorageError):
            manager._resolve_inputs("old", request)
        manager._resolve_inputs("old", request)
        manager._resolve_inputs("old", request)
        assert len(calls) == 2
        import sqlite3
        with sqlite3.connect(manager._state_store.path) as conn:
            conn.execute("UPDATE job_inputs SET paths='broken json'")
        manager._resolve_inputs("old", request)
        manager._resolve_inputs("old", request)
        assert len(calls) == 3
    finally:
        manager._executor.shutdown()


def test_cached_inputs_do_not_treat_offline_root_as_deleted_files(monkeypatch, tmp_path):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    root = tmp_path / "media"
    root.mkdir()
    (root / "photo.jpg").write_bytes(b"fixture")
    manager = services.ProcessingJobManager()
    request = _request(root)
    manager._jobs["offline"] = _job("offline", root)
    manager._job_requests["offline"] = request
    try:
        manager._resolve_inputs("offline", request)
        root.rename(tmp_path / "disconnected")
        with pytest.raises(ValueError, match="输入路径不存在"):
            manager._resolve_inputs("offline", request)
    finally:
        manager._executor.shutdown()


def test_completed_image_batch_is_checkpointed_while_job_is_running(
    monkeypatch,
    tmp_path,
):
    state_path = tmp_path / "job_state.json"
    _patch_state_path(monkeypatch, state_path)

    image_path = tmp_path / "camera-01" / "0001.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"image")

    request = _request(tmp_path)
    job_id = "restart-resume"
    manager = services.ProcessingJobManager(max_workers=1)
    manager._jobs[job_id] = _job(job_id, tmp_path)
    manager._job_requests[job_id] = request

    item = DetectionItem(
        filename=image_path.name,
        path=str(image_path),
        file_type="image",
    )

    class FakeDetector:
        pass

    legacy = services._legacy
    monkeypatch.setattr(legacy, "_validate_dinov2_job_options", lambda *args: None)
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        legacy,
        "_resolve_job_inputs",
        lambda request, cancelled=None: (tmp_path, [image_path]),
    )
    monkeypatch.setattr(legacy, "_preview_detection_db_roots", lambda *args, **kwargs: [])
    monkeypatch.setattr(legacy, "_load_detection_index", lambda *args, **kwargs: {})
    monkeypatch.setattr(legacy, "_load_detector", lambda *args, **kwargs: FakeDetector())
    monkeypatch.setattr(legacy, "_build_metadata_item", lambda path: item)
    monkeypatch.setattr(
        legacy,
        "_detect_image_batch",
        lambda detector, batch, batch_items, request, input_path, preloaded_data=None, inference_batcher=None: [item],
    )
    monkeypatch.setattr(legacy, "_export_results", lambda *args, **kwargs: None)

    snapshots: list[tuple[JobState, int]] = []
    original_save = manager._save_state_unlocked

    def capture_save() -> None:
        current = manager._jobs[job_id]
        snapshots.append((current.state, current.processed))
        original_save()

    monkeypatch.setattr(manager, "_save_state_unlocked", capture_save)

    try:
        manager._run_job(job_id, request, [])
    finally:
        manager._executor.shutdown(wait=False, cancel_futures=True)

    assert (JobState.RUNNING, 1) in snapshots


def test_failed_state_checkpoint_does_not_corrupt_previous_resume_state(
    monkeypatch,
    tmp_path,
):
    state_path = tmp_path / "job_state.json"
    _patch_state_path(monkeypatch, state_path)

    manager = services.ProcessingJobManager(max_workers=1)
    job_id = "atomic-state"
    manager._jobs[job_id] = _job(job_id, tmp_path)
    manager._job_requests[job_id] = _request(tmp_path)

    manager._save_state_unlocked()
    manager._jobs[job_id] = manager._jobs[job_id].model_copy(update={"processed": 1})
    original_dumps = json.dumps

    def fail_dumps(payload, *args, **kwargs):
        if isinstance(payload, dict) and payload.get("filename") == "fail.jpg":
            raise OSError("simulated interrupted checkpoint")
        return original_dumps(payload, *args, **kwargs)

    manager._jobs[job_id].results.append(DetectionItem(filename="fail.jpg", path="fail.jpg", file_type="image"))
    monkeypatch.setattr(json, "dumps", fail_dumps)

    try:
        manager._save_state_unlocked()
    finally:
        manager._executor.shutdown(wait=False, cancel_futures=True)

    saved = manager._state_store.load()["jobs"][job_id]
    assert saved["processed"] == 0
    assert saved["results"] == []


def test_legacy_history_migrates_once_and_incremental_progress_survives_restart(monkeypatch, tmp_path):
    state_path = tmp_path / "job_state.json"
    _patch_state_path(monkeypatch, state_path)
    job = _job("legacy", tmp_path)
    job.results = [DetectionItem(filename="old.jpg", path="old.jpg", file_type="image")]
    job.processed = 1
    state_path.write_text(json.dumps({"jobs": {job.id: job.model_dump()},
                                     "requests": {job.id: _request(tmp_path).model_dump()}}), encoding="utf-8")
    original_json = state_path.read_bytes()
    manager = services.ProcessingJobManager()
    try:
        next_item = DetectionItem(filename="new.jpg", path="new.jpg", file_type="image")
        manager._mutate_job(job.id, state=JobState.RUNNING, processed=2,
                            results=[*manager._jobs[job.id].results, next_item])
        restarted = services.ProcessingJobManager()
        try:
            saved = restarted.get_job(job.id)
            assert saved.state == JobState.CANCELLED
            assert saved.processed == 2
            assert [item.filename for item in saved.results] == ["old.jpg", "new.jpg"]
            restarted.clear_jobs()
            assert restarted._state_store.load() == {"jobs": {}, "requests": {}}
        finally:
            restarted._executor.shutdown()
    finally:
        manager._executor.shutdown()
    assert state_path.read_bytes() == original_json


def test_progress_serializes_only_new_results(monkeypatch, tmp_path):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    manager = services.ProcessingJobManager()
    manager._jobs["history"] = _job("history", tmp_path)
    manager._jobs["history"].results = [DetectionItem(filename="old.jpg", path="old.jpg", file_type="image")]
    manager._jobs["active"] = _job("active", tmp_path)
    manager._job_requests["active"] = _request(tmp_path)
    manager._save_state_unlocked()
    old = manager._jobs["history"].results[0]
    original_dump = DetectionItem.model_dump
    calls = []

    def record_dump(item, *args, **kwargs):
        assert item is not old, "history must not be serialized for an active batch"
        calls.append(item.filename)
        return original_dump(item, *args, **kwargs)

    monkeypatch.setattr(DetectionItem, "model_dump", record_dump)
    try:
        first = DetectionItem(filename="first.jpg", path="first.jpg", file_type="image")
        second = DetectionItem(filename="second.jpg", path="second.jpg", file_type="image")
        manager._mutate_job("active", processed=1, results=[first])
        manager._mutate_job("active", processed=2, results=[first, second])
        assert calls == ["first.jpg", "second.jpg"]
        assert len(manager._state_store.load()["jobs"]["active"]["results"]) == 2
        # Replacing an earlier result also persists corrections, without duplicate positions.
        corrected = first.model_copy(update={"species": ["corrected"]})
        manager._mutate_job("active", results=[corrected, second])
        saved_results = manager._state_store.load()["jobs"]["active"]["results"]
        assert len(saved_results) == 2
        assert saved_results[0]["species"] == ["corrected"]
    finally:
        manager._executor.shutdown()


def test_checkpoint_rebuilds_after_software_cache_is_removed(monkeypatch, tmp_path):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    manager = services.ProcessingJobManager()
    try:
        manager._jobs["history"] = _job("history", tmp_path)
        manager._jobs["history"].results = [DetectionItem(filename="a.jpg", path="a.jpg", file_type="image")]
        manager._save_state_unlocked()
        manager._state_store.path.unlink()
        manager._save_state_unlocked()
        assert len(manager._state_store.load()["jobs"]["history"]["results"]) == 1
    finally:
        manager._executor.shutdown()


@pytest.mark.parametrize("stage", ["_validate_dinov2_job_options", "_start_batch_log_session"])
def test_startup_failure_releases_job_and_checkpoints_error(monkeypatch, tmp_path, stage):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    manager = services.ProcessingJobManager()
    request = _request(tmp_path)
    manager._jobs["startup"] = _job("startup", tmp_path)
    manager._job_requests["startup"] = request

    def fail(*args, **kwargs):
        raise ValueError("startup fixture failure")

    monkeypatch.setattr(services._legacy, stage, fail)
    try:
        manager._run_job("startup", request, [])
        job = manager.get_job("startup")
        assert job.state == JobState.FAILED
        assert not job.active
        assert job.error == "startup fixture failure"
        assert manager._state_store.load()["jobs"]["startup"]["state"] == "failed"
    finally:
        manager._executor.shutdown()


@pytest.mark.parametrize("cancelled", [False, True])
def test_fast_video_interruption_keeps_unpublished_completed_results(monkeypatch, tmp_path, cancelled):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    legacy = services._legacy
    paths = [tmp_path / f"video-{index}.mp4" for index in range(3)]
    for path in paths:
        path.write_bytes(b"video fixture")
    request = _request(tmp_path)
    request.options.video_mode = "fast"
    manager = services.ProcessingJobManager()
    manager._jobs["videos"] = _job("videos", tmp_path)
    manager._job_requests["videos"] = request
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *args: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *args, **kwargs: (tmp_path, paths))
    monkeypatch.setattr(legacy, "load_media_indexes", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(legacy, "_load_detector", lambda *args: object())
    monkeypatch.setattr(legacy, "_build_metadata_item", lambda path: DetectionItem(
        filename=path.name, path=str(path), file_type="video"))
    monkeypatch.setattr(legacy.time, "monotonic", lambda: 1000.0)

    def interrupt_batch(detector, video_paths, items, *args, on_item_ready, **kwargs):
        on_item_ready(0, items[0])
        on_item_ready(1, items[1])
        # The second completion is held back by the 200 ms UI throttle.
        assert manager.get_job("videos").processed == 1
        if cancelled:
            manager._cancelled_jobs.add("videos")
            raise RuntimeError("用户取消任务")
        raise services.MediaStorageError(paths[2], "保存检测结果", OSError("disk disconnected"))

    monkeypatch.setattr(legacy, "_detect_video_fast_batch", interrupt_batch)
    try:
        manager._run_job("videos", request, [])
        job = manager.get_job("videos")
        assert job.state == (JobState.CANCELLED if cancelled else JobState.FAILED)
        assert not job.active
        assert job.processed == 2
        assert [item.path for item in job.results] == [str(path) for path in paths[:2]]
        saved = manager._state_store.load()["jobs"]["videos"]
        assert saved["processed"] == 2
        assert len(saved["results"]) == 2
    finally:
        manager._executor.shutdown()
