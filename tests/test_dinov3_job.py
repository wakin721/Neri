from pathlib import Path
from types import SimpleNamespace

from system.backend import services
from system.backend.models import CreateJobRequest, DetectionItem, JobState, JobSummary, ProcessingOptions


def test_dino_serializer_distinguishes_empty_detection_from_missing_output():
    legacy = services._legacy
    detector = SimpleNamespace(dinov3_classifier=object(), translation_dict={})

    empty = legacy._serialize_detector_output(
        detector,
        {"detect_results": [SimpleNamespace(boxes=[], names={})]},
    )
    assert empty["detection_completed"] is True
    assert empty["detection_box_count"] == 0
    assert empty["detection_box_indices"] == []
    assert empty["检测框"] == []

    missing = legacy._serialize_detector_output(detector, {"detect_results": []})
    assert missing["detection_completed"] is False
    assert missing["detection_box_count"] == 0
    assert missing["detection_box_indices"] == []


def test_resumed_joint_job_reencodes_complete_sequences_before_finalizing(monkeypatch, tmp_path):
    legacy = services._legacy
    state_path = tmp_path / "state.json"
    for module in (services, legacy):
        monkeypatch.setattr(module, "job_state_path", lambda: state_path)
    photos = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    for path in photos:
        path.write_bytes(b"fixture")
    monkeypatch.setattr(legacy, "_validate_dino_job_options", lambda *a: None)
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *a, **kw: (tmp_path, photos))
    monkeypatch.setattr(legacy, "_resolve_classification_model_path", lambda p: Path(p))
    monkeypatch.setattr(legacy, "_dino_manifest_payload", lambda p: dict(backend="dinov3"))
    monkeypatch.setattr(legacy, "load_media_indexes", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(legacy, "_build_metadata_item", lambda p: DetectionItem(filename=p.name, path=str(p), file_type="jpg"))
    monkeypatch.setattr(legacy, "_serialize_detector_output", lambda *a: {})
    monkeypatch.setattr(legacy, "_persist_dino_observations", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_selected_inference_class_ids", lambda *a: None)
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_export_results", lambda *a, **kw: None)
    detected = []
    class Detector:
        def detect_batch_species(self, paths, **kwargs):
            detected.extend(paths)
            return [{} for _ in paths]
    monkeypatch.setattr(legacy, "_load_detector", lambda *a: Detector())
    import system.dinov3.processor as processor
    finalized = []
    def finalize(detector, results, root, **kwargs):
        assert detected == list(map(str, photos))
        assert len(results) == 2
        assert kwargs["is_cancelled"]() is False
        finalized.append(True)
        return [
            item.model_copy(
                update={
                    "species": ["joined"],
                    "detection_data": {
                        **(item.detection_data or {}),
                        "sequence_completed": True,
                        "sequence_pending": False,
                        "sequence_status": "complete",
                    },
                }
            )
            for item in results
        ]
    monkeypatch.setattr(processor, "finalize_photo_job", finalize)
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(
        enable_detection=True, model_path="detector.pt", classification_model_path="joint.neri.json",
        batch_size=1, video_mode="skip"))
    manager = services.ProcessingJobManager()
    try:
        now = services.utc_now()
        manager._jobs["job"] = JobSummary(id="job", state=JobState.QUEUED, input_dir=str(tmp_path), created_at=now, updated_at=now)
        manager._job_requests["job"] = request
        old = DetectionItem(filename=photos[0].name, path=str(photos[0]), file_type="jpg", species=["old"])
        manager._run_job("job", request, [old])
        finished = manager.get_job("job")
        assert finished.state == JobState.COMPLETED, finished.error
        assert finalized == [True]
        assert [item.species for item in finished.results] == [["joined"], ["joined"]]
        assert finished.processed == 2
    finally:
        manager._executor.shutdown()


def test_cancelled_joint_job_checkpoints_all_processed_photos_as_partial(
    monkeypatch,
    tmp_path,
):
    legacy = services._legacy
    state_path = tmp_path / "state.json"
    for module in (services, legacy):
        monkeypatch.setattr(module, "job_state_path", lambda: state_path)
    photos = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    for path in photos:
        path.write_bytes(b"fixture")
    monkeypatch.setattr(legacy, "_validate_dino_job_options", lambda *a: None)
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *a, **kw: (tmp_path, photos))
    monkeypatch.setattr(legacy, "_resolve_classification_model_path", lambda p: Path(p))
    monkeypatch.setattr(legacy, "_dino_manifest_payload", lambda p: dict(backend="dinov3"))
    monkeypatch.setattr(legacy, "load_media_indexes", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(
        legacy,
        "_build_metadata_item",
        lambda p: DetectionItem(filename=p.name, path=str(p), file_type="jpg"),
    )
    monkeypatch.setattr(legacy, "_serialize_detector_output", lambda *a: {})
    monkeypatch.setattr(legacy, "_persist_dino_observations", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_selected_inference_class_ids", lambda *a: None)
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_export_results", lambda *a, **kw: None)

    class Detector:
        pass

    manager = services.ProcessingJobManager()
    monkeypatch.setattr(legacy, "_load_detector", lambda *a: Detector())
    detect_calls = 0

    def detect_batch(detector, batch, batch_items, *args, **kwargs):
        nonlocal detect_calls
        detect_calls += 1
        if detect_calls == 2:
            manager._cancelled_jobs.add("job")
        return batch_items

    monkeypatch.setattr(legacy, "_detect_image_batch", detect_batch)

    import system.dinov3.processor as processor
    checkpoints = []

    def finalize(detector, results, root, **kwargs):
        assert kwargs["complete_sequence"] is False
        assert kwargs["incomplete_reason"] == "cancelled"
        checkpoints.append([item.path for item in results])
        return [
            item.model_copy(
                update={
                    "detection_data": {
                        **(item.detection_data or {}),
                        "sequence_completed": False,
                        "sequence_pending": True,
                        "sequence_status": "partial",
                    }
                }
            )
            for item in results
        ]

    monkeypatch.setattr(processor, "finalize_photo_job", finalize)
    request = CreateJobRequest(
        input_dir=str(tmp_path),
        options=ProcessingOptions(
            enable_detection=True,
            model_path="detector.pt",
            classification_model_path="joint.neri.json",
            batch_size=1,
            video_mode="skip",
        ),
    )
    try:
        now = services.utc_now()
        manager._jobs["job"] = JobSummary(
            id="job",
            state=JobState.QUEUED,
            input_dir=str(tmp_path),
            created_at=now,
            updated_at=now,
        )
        manager._job_requests["job"] = request
        manager._run_job("job", request, [])
        finished = manager.get_job("job")
        assert finished.state == JobState.CANCELLED
        assert finished.processed == 2
        assert [item.path for item in finished.results] == list(map(str, photos))
        assert checkpoints == [list(map(str, photos))]
        assert all(
            item.detection_data["sequence_pending"] is True
            and item.detection_data["sequence_completed"] is False
            and item.detection_data["sequence_status"] == "partial"
            for item in finished.results
        )
    finally:
        manager._executor.shutdown()
