from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from system.backend import services
from system.backend.models import CreateJobRequest, DetectionItem, JobState, JobSummary, ProcessingOptions
from system.dinov2.example_writer import ExampleWriter
from system.dinov2 import registry_examples


@pytest.mark.parametrize("bound", ["items", "bytes"])
def test_writer_overlaps_work_and_bounds_pending_crops(bound, tmp_path):
    writing, release, submitting = threading.Event(), threading.Event(), threading.Event()
    saved = []

    def write(path, crop):
        writing.set()
        assert release.wait(3)
        saved.append((path.name, int(crop[0, 0, 0]), threading.current_thread().name))
        return True

    frame = np.full((32, 32, 3), 30, np.uint8)
    writer = ExampleWriter(write, max_items=2 if bound == "items" else 100,
                           max_bytes=frame.nbytes * 2 if bound == "bytes" else 1024**2)
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            writer.submit(tmp_path / "1.jpg", frame)
            assert writing.wait(3)
            frame.fill(60)
            writer.submit(tmp_path / "2.jpg", frame)
            writer.submit(tmp_path / "1.jpg", frame)  # Coalesce the pending path.

            def third():
                submitting.set()
                writer.submit(tmp_path / "3.jpg", frame)

            pending = executor.submit(third)
            assert submitting.wait(3)
            assert not pending.done()
            release.set()
            pending.result(timeout=3)
            writer.flush()
        finally:
            release.set()
            writer.close()
    assert [(name, color) for name, color, _ in saved] == [("1.jpg", 30), ("2.jpg", 60), ("3.jpg", 60)]
    assert all(name == "neri-example-writer" for _, _, name in saved)
    assert not any(thread.name == "neri-example-writer" for thread in threading.enumerate())


def test_writer_drains_and_reports_failed_writes(tmp_path):
    def fail(path, crop):
        raise OSError("disk disconnected")

    writer = ExampleWriter(fail)
    writer.submit(tmp_path / "failed.jpg", np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(OSError, match="disk disconnected") as error:
        writer.flush()
    assert error.value.filename == str(tmp_path / "failed.jpg")
    with pytest.raises(OSError, match="disk disconnected"):
        writer.close()
    assert not any(thread.name == "neri-example-writer" for thread in threading.enumerate())


def test_background_examples_own_pixels_after_frame_cache_closes(tmp_path, monkeypatch):
    source = tmp_path / "video.avi"
    source.touch()
    store = tmp_path / "feedback.sqlite3"
    frame = np.full((32, 32, 3), [20, 90, 180], np.uint8)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: pytest.fail("must reuse decoded frame"))
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("must reuse decoded frame"))
    with registry_examples.background_example_writes():
        with registry_examples.example_frame_cache(video_frames=[(source, 25, 1.0, frame)]):
            assert registry_examples.persist_feedback_observation_example(
                store, "observation", source_path=source, bbox=(0, 0, 32, 32), frame_index=25,
            )
        frame.fill(0)
    path = registry_examples.feedback_observation_example_path(store, "observation")
    decoded = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    assert np.max(np.abs(decoded[160, 160].astype(int) - [20, 90, 180])) < 4
    assert registry_examples._EXAMPLE_WRITER.get() is None


@pytest.mark.parametrize("ending", ["success", "cancel", "write_failure"])
def test_job_waits_for_background_writes_before_completion(ending, tmp_path, monkeypatch):
    legacy = services._legacy
    state_path = tmp_path / "state.json"
    for module in (services, legacy):
        monkeypatch.setattr(module, "job_state_path", lambda: state_path)
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"fixture")
    writing, release = threading.Event(), threading.Event()

    def write(path, crop):
        writing.set()
        assert release.wait(5)
        if ending == "write_failure":
            raise OSError("disk disconnected")
        return True

    monkeypatch.setattr(registry_examples, "_write_crop", write)
    monkeypatch.setattr(legacy, "_validate_dinov2_job_options", lambda *args: None)
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *args, **kwargs: (tmp_path, [photo]))
    monkeypatch.setattr(legacy, "load_media_indexes", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(legacy, "_build_metadata_item", lambda path: DetectionItem(
        filename=path.name, path=str(path), file_type="jpg"))
    monkeypatch.setattr(legacy, "_serialize_detector_output", lambda *args: {})
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda *args: None)
    monkeypatch.setattr(legacy, "_export_results", lambda *args: None)
    monkeypatch.setattr(legacy, "_load_detector", lambda *args: SimpleNamespace(
        detect_batch_species=lambda paths, **kwargs: [{} for _ in paths]))

    def persist(*args, **kwargs):
        registry_examples._EXAMPLE_WRITER.get().submit(
            tmp_path / "example.jpg", np.zeros((320, 320, 3), np.uint8),
            source_path=photo,
        )

    monkeypatch.setattr(legacy, "_persist_dinov2_observations", persist)
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(
        enable_detection=True, model_path="fake.pt", batch_size=1, video_mode="skip",
    ))
    manager = services.ProcessingJobManager(max_workers=1)
    now = services.utc_now()
    manager._jobs["job"] = JobSummary(id="job", state=JobState.QUEUED,
                                      input_dir=str(tmp_path), created_at=now, updated_at=now)
    manager._job_requests["job"] = request
    try:
        future = manager._executor.submit(manager._run_job, "job", request, [])
        assert writing.wait(5)
        assert not future.done()
        assert manager.get_job("job").state == JobState.RUNNING
        if ending == "cancel":
            manager.cancel_job("job")
        release.set()
        future.result(timeout=5)
        result = manager.get_job("job")
        expected = {"success": JobState.COMPLETED, "cancel": JobState.CANCELLED, "write_failure": JobState.FAILED}
        assert result.state == expected[ending]
        if ending == "write_failure":
            assert "disk disconnected" in result.error
            assert result.processed == 0 and not result.results
    finally:
        release.set()
        manager._executor.shutdown(wait=True, cancel_futures=True)
    assert not any(thread.name == "neri-example-writer" for thread in threading.enumerate())
