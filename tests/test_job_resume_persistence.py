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


def test_restart_and_summary_poll_never_read_historical_results(monkeypatch, tmp_path):
    import sqlite3
    from system.backend.job_state_store import JobStateStore

    state_path = tmp_path / "job_state.json"
    _patch_state_path(monkeypatch, state_path)
    history = _job("history", tmp_path)
    history.state = JobState.COMPLETED
    history.results = [DetectionItem(filename="old.jpg", path="old.jpg", file_type="image")]
    store = JobStateStore(state_path)
    store.save({history.id: history}, {})
    # Even an unreadable detail must not affect startup or summary polling.
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE results SET payload='invalid json'")
    manager = services.ProcessingJobManager()
    try:
        assert manager._unloaded_job_ids == {history.id}
        assert manager.list_jobs(include_results=False)[0].results == []
        with pytest.raises(json.JSONDecodeError):
            manager.get_job(history.id)
        assert manager._unloaded_job_ids == {history.id}
        manager._save_state_unlocked()
        with sqlite3.connect(store.path) as conn:
            assert conn.execute("SELECT payload FROM results").fetchone()[0] == 'invalid json'
    finally:
        manager._executor.shutdown()


def test_unloaded_results_survive_startup_cancellation_and_other_checkpoints(monkeypatch, tmp_path):
    from system.backend.job_state_store import JobStateStore

    state_path = tmp_path / "job_state.json"
    _patch_state_path(monkeypatch, state_path)
    history = _job("history", tmp_path)
    history.state = JobState.RUNNING
    history.processed = 1
    history.results = [DetectionItem(filename="old.jpg", path="old.jpg", file_type="image")]
    store = JobStateStore(state_path)
    store.save({history.id: history}, {history.id: _request(tmp_path)})
    manager = services.ProcessingJobManager()
    try:
        assert manager.list_jobs(include_results=False)[0].state == JobState.CANCELLED
        manager._jobs["new"] = _job("new", tmp_path)
        manager._mutate_job("new", processed=1, results=[
            DetectionItem(filename="new.jpg", path="new.jpg", file_type="image")
        ])
        assert len(store.load()["jobs"][history.id]["results"]) == 1
        assert history.id in manager._unloaded_job_ids
        calls = []
        load_results = manager._state_store.load_results
        def record(job_id):
            calls.append(job_id)
            return load_results(job_id)
        monkeypatch.setattr(manager._state_store, "load_results", record)
        assert manager.get_job(history.id).results[0].filename == "old.jpg"
        manager.get_job(history.id)
        assert calls == [history.id]
        manager._mutate_job(history.id, results=[])
        assert store.load()["jobs"][history.id]["results"] == []
        # Deleting a still-unloaded task must not hydrate unrelated records.
        restarted = services.ProcessingJobManager()
        try:
            monkeypatch.setattr(restarted._state_store, "load_results",
                                lambda _: pytest.fail("delete must not load results"))
            restarted.delete_job("new")
            assert set(store.load()["jobs"]) == {history.id}
            restarted.clear_jobs()
            assert store.load() == {"jobs": {}, "requests": {}}
        finally:
            restarted._executor.shutdown()
    finally:
        manager._executor.shutdown()


def test_failed_file_retry_hydrates_only_its_source_task(monkeypatch, tmp_path):
    from system.backend.job_state_store import JobStateStore

    state_path = tmp_path / "job_state.json"
    _patch_state_path(monkeypatch, state_path)
    history = _job("failed", tmp_path)
    history.state = JobState.COMPLETED
    history.failed_count = 1
    history.results = [
        DetectionItem(filename="ok.jpg", path="ok.jpg", file_type="image"),
        DetectionItem(filename="bad.jpg", path="bad.jpg", file_type="image", error="failed"),
    ]
    JobStateStore(state_path).save({history.id: history}, {history.id: _request(tmp_path)})
    manager = services.ProcessingJobManager()
    try:
        requests = []
        monkeypatch.setattr(manager, "create_job", lambda request: requests.append(request))
        manager.retry_failed_job(history.id)
        assert requests[0].input_paths == ["bad.jpg"]
        assert len(manager.get_job(history.id).results) == 2
    finally:
        manager._executor.shutdown()


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
    monkeypatch.setattr(legacy, "_validate_dino_job_options", lambda *args: None)
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
    original_dump = DetectionItem.model_dump_json

    def fail_dump(item, *args, **kwargs):
        if item.filename == "fail.jpg":
            raise OSError("simulated interrupted checkpoint")
        return original_dump(item, *args, **kwargs)

    manager._jobs[job_id].results.append(DetectionItem(filename="fail.jpg", path="fail.jpg", file_type="image"))
    monkeypatch.setattr(DetectionItem, "model_dump_json", fail_dump)

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
    original_dump = DetectionItem.model_dump_json
    calls = []

    def record_dump(item, *args, **kwargs):
        assert item is not old, "history must not be serialized for an active batch"
        calls.append(item.filename)
        return original_dump(item, *args, **kwargs)

    monkeypatch.setattr(DetectionItem, "model_dump_json", record_dump)
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


def test_progress_appends_without_reading_history_or_changing_snapshots(monkeypatch, tmp_path):
    error_reads = []
    result_reads = []
    result_scans = []

    class TrackedItem(DetectionItem):
        def __getattribute__(self, name):
            if name == "error":
                error_reads.append(self.filename)
            return super().__getattribute__(name)

    class TrackedResults(list):
        def __iter__(self):
            result_scans.append(len(self))
            return super().__iter__()

        def __getitem__(self, index):
            result_reads.append(index)
            return super().__getitem__(index)

    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    manager = services.ProcessingJobManager()
    job_id = "active"
    first = TrackedItem(filename="first.jpg", path="first.jpg", file_type="image", error="bad image")
    manager._jobs[job_id] = _job(job_id, tmp_path).model_copy(update={
        "processed": 1, "failed_count": 1, "results": TrackedResults([first]),
    })

    try:
        manager._save_state_unlocked()
        snapshot = manager.get_job(job_id)
        result_scans.clear()
        result_reads.clear()
        error_reads.clear()
        next_items = [
            TrackedItem(filename=f"{index}.jpg", path=f"{index}.jpg", file_type="image",
                        error="bad image" if index % 2 == 0 else None)
            for index in range(12)
        ]
        for start in range(0, len(next_items), 3):
            end = start + 3
            manager._mutate_job(job_id, processed=1 + end, appended_results=next_items[start:end])

        assert error_reads == [item.filename for item in next_items]
        assert result_reads == list(range(1, 1 + len(next_items)))
        assert result_scans == []
        assert [item.filename for item in snapshot.results] == [first.filename]
        assert (snapshot.processed, snapshot.failed_count) == (1, 1)
        saved = manager._state_store.load()["jobs"][job_id]
        assert [item["filename"] for item in saved["results"]] == [first.filename, *error_reads]
        assert (saved["processed"], saved["failed_count"]) == (13, 7)
    finally:
        manager._executor.shutdown()


@pytest.mark.parametrize("cancel_after", [None, 3])
def test_parallel_video_progress_appends_or_inserts_in_input_order(monkeypatch, tmp_path, cancel_after):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    paths = [tmp_path / f"{index}.mp4" for index in range(4)]
    for path in paths:
        path.touch()

    def detect(_detector, path, item, *_args, **_kwargs):
        if path == paths[0]:
            raise ValueError("video inference failed")
        return item

    completion_order = [2, 0, 3, 1]

    def ordered_futures(futures):
        by_index = {index: future for future, index in futures.items()}
        return (by_index[index] for index in completion_order)

    legacy = services._legacy
    monkeypatch.setattr(legacy, "as_completed", ordered_futures)
    monkeypatch.setattr(legacy, "_tracked_video_worker_count", lambda _: 2)
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda _: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *_args, **_kwargs: (tmp_path, paths))
    monkeypatch.setattr(legacy, "load_media_indexes", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(legacy, "_load_detector", lambda *_: object())
    monkeypatch.setattr(legacy, "_build_metadata_item", lambda path: DetectionItem(
        filename=path.name, path=str(path), file_type="video",
    ))
    monkeypatch.setattr(legacy, "_detect_video", detect)
    monkeypatch.setattr(legacy, "_clear_torch_runtime_cache", lambda: None)
    monkeypatch.setattr(legacy, "_export_results", lambda *_: None)

    manager = services.ProcessingJobManager()
    original_mutate = manager._mutate_job
    snapshots = []
    replacements = []

    def record_progress(job_id, **changes):
        original_mutate(job_id, **changes)
        if changes.get("message", "").startswith("正在并发处理视频"):
            snapshots.append(manager.get_job(job_id))
            replacements.append("results" in changes)
            if len(snapshots) == cancel_after:
                manager.cancel_job(job_id)

    monkeypatch.setattr(manager, "_mutate_job", record_progress)
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(
        enable_detection=True, model_path="fake.pt", batch_size=1, video_mode="all",
    ))
    try:
        queued = manager.create_job(request)
        manager._executor.shutdown()
        final = manager.get_job(queued.id)
        assert final.state == (JobState.CANCELLED if cancel_after else JobState.COMPLETED), final.error
        assert replacements == [False, True, False, True][:cancel_after]
        for count, snapshot in enumerate(snapshots, 1):
            completed = sorted(completion_order[:count])
            assert [item.filename for item in snapshot.results] == [paths[index].name for index in completed]
            assert snapshot.processed == count
            assert snapshot.failed_count == int(0 in completed)

        restarted = services.ProcessingJobManager()
        try:
            restored = restarted.get_job(queued.id)
            assert restored.state == final.state
            assert restored.results == snapshots[-1].results
            assert restored.processed == len(snapshots)
            assert restored.failed_count == 1
        finally:
            restarted._executor.shutdown()
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


@pytest.mark.parametrize("stage", ["_validate_dino_job_options", "_start_batch_log_session"])
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
