import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from system.backend import main_core_legacy as backend
from system.backend.models_legacy import JobState, JobSummary


def _job_summary(**changes):
    values = {
        "id": "retry-job",
        "state": JobState.QUEUED,
        "input_dir": "failed.jpg",
        "total": 1,
        "processed": 0,
        "failed_count": 0,
        "message": "任务已加入队列",
        "created_at": "2026-11-13T00:00:00Z",
        "updated_at": "2026-11-13T00:00:00Z",
    }
    values.update(changes)
    return JobSummary(**values)


def test_job_summary_tracks_failed_count():
    job = _job_summary(
        state=JobState.COMPLETED,
        total=3,
        processed=3,
        failed_count=1,
        message="处理完成，1 个文件失败",
    )

    assert job.failed_count == 1
    assert _job_summary().failed_count == 0
    with pytest.raises(ValidationError):
        _job_summary(failed_count=-1)


def test_retry_failed_endpoint_returns_new_job(monkeypatch):
    expected = _job_summary()

    class Manager:
        def __init__(self):
            self.job_ids = []

        def retry_failed_job(self, job_id):
            self.job_ids.append(job_id)
            return expected

    manager = Manager()
    monkeypatch.setattr(backend, "job_manager", manager)
    route = next(
        route
        for route in backend.app.routes
        if getattr(route, "path", None) == "/api/jobs/{job_id}/retry-failed"
    )
    response = backend.retry_failed_job("source-job")

    assert route.status_code == 202
    assert response.id == "retry-job"
    assert response.failed_count == 0
    assert manager.job_ids == ["source-job"]


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (backend.JobNotFoundError("missing"), 404),
        (ValueError("没有失败文件"), 400),
    ],
)
def test_retry_failed_endpoint_maps_manager_errors(
    monkeypatch,
    error,
    status_code,
):
    class Manager:
        def retry_failed_job(self, _job_id):
            raise error

    monkeypatch.setattr(backend, "job_manager", Manager())
    with pytest.raises(HTTPException) as caught:
        backend.retry_failed_job("source-job")
    assert caught.value.status_code == status_code
