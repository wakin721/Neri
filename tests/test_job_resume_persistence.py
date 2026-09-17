from pathlib import Path

from system.backend import services_legacy as services
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


def test_completed_image_batch_is_checkpointed_while_job_is_running(
    monkeypatch,
    tmp_path,
):
    state_path = tmp_path / "job_state.json"
    monkeypatch.setattr(services, "job_state_path", lambda: state_path)

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

    monkeypatch.setattr(services, "_validate_dinov2_job_options", lambda *args: None)
    monkeypatch.setattr(services, "_start_batch_log_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        services,
        "_resolve_job_inputs",
        lambda request, cancelled=None: (tmp_path, [image_path]),
    )
    monkeypatch.setattr(services, "_preview_detection_db_roots", lambda *args, **kwargs: [])
    monkeypatch.setattr(services, "_load_detection_index", lambda *args, **kwargs: {})
    monkeypatch.setattr(services, "_load_detector", lambda *args, **kwargs: FakeDetector())
    monkeypatch.setattr(services, "_build_metadata_item", lambda path: item)
    monkeypatch.setattr(
        services,
        "_detect_image_batch",
        lambda detector, batch, batch_items, request, input_path, preloaded_data=None: [item],
    )
    monkeypatch.setattr(services, "_export_results", lambda *args, **kwargs: None)

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
    monkeypatch.setattr(services, "job_state_path", lambda: state_path)

    manager = services.ProcessingJobManager(max_workers=1)
    job_id = "atomic-state"
    manager._jobs[job_id] = _job(job_id, tmp_path)
    manager._job_requests[job_id] = _request(tmp_path)

    previous = '{"previous":"valid"}'
    state_path.write_text(previous, encoding="utf-8")
    original_write_text = Path.write_text

    def fail_after_partial_write(self, text, *args, **kwargs):
        if self.parent == state_path.parent and self.name.startswith(state_path.name):
            original_write_text(self, "{", encoding="utf-8")
            raise OSError("simulated interrupted checkpoint")
        return original_write_text(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_after_partial_write)

    try:
        manager._save_state_unlocked()
    finally:
        manager._executor.shutdown(wait=False, cancel_futures=True)

    assert state_path.read_text(encoding="utf-8") == previous
