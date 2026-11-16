from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from system.backend import services
from system.backend.models import (
    CreateJobRequest,
    DetectionItem,
    JobState,
    JobSummary,
    ProcessingOptions,
)
from system.dinov3.job_cache import DinoV3JobCache
from system.dinov3.processor import DinoV3CropCollector, DinoV3CropRecord


def _photo_item(path: Path) -> DetectionItem:
    return DetectionItem(
        filename=path.name,
        path=str(path),
        file_type=path.suffix.lstrip("."),
        detection_data={
            "detection_completed": True,
            "detection_box_count": 1,
            "detection_box_indices": [0],
            "检测框": [{"边界框": [1, 2, 20, 30]}],
            "sequence_pending": True,
        },
    )


def _completed_video_item(path: Path, species: str = "fox") -> DetectionItem:
    return DetectionItem(
        filename=path.name,
        path=str(path),
        file_type=path.suffix.lstrip("."),
        species=[species],
        detection_boxes=[{"物种": species, "sequence_status": "complete"}],
        detection_data={
            "video_source": str(path),
            "video_mode": "fast",
            "物种名称": species,
            "物种数量": "1",
            "检测框": [{"物种": species, "sequence_status": "complete"}],
            "sequence_completed": True,
            "sequence_pending": False,
            "sequence_status": "complete",
        },
    )


def _crop_record(path: Path) -> DinoV3CropRecord:
    return DinoV3CropRecord(
        str(path),
        0,
        (1, 2, 20, 30),
        0.9,
        np.array([0.5, 0.5, 0.0], dtype=np.float32),
    )


def _setup_dinov3_job_env(monkeypatch, tmp_path, files):
    legacy = services._legacy
    state_path = tmp_path / "state.json"
    for module in (services, legacy):
        monkeypatch.setattr(module, "job_state_path", lambda: state_path)
    model_path = tmp_path / "detector.pt"
    model_path.write_bytes(b"detector")
    manifest_path = tmp_path / "joint.neri.json"
    manifest_path.write_text('{"backend":"dinov3"}', encoding="utf-8")

    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *a, **kw: (tmp_path, list(files)))
    monkeypatch.setattr(legacy, "_resolve_model_path", lambda p: model_path)
    monkeypatch.setattr(legacy, "_resolve_classification_model_path", lambda p: manifest_path)
    monkeypatch.setattr(
        legacy,
        "_dino_manifest_payload",
        lambda p: {"backend": "dinov3", "display_name": "Neri1.2preview"},
    )
    monkeypatch.setattr(
        services,
        "_dino_manifest_payload",
        lambda p: {"backend": "dinov3", "display_name": "Neri1.2preview"},
    )
    monkeypatch.setattr(legacy, "load_media_indexes", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(
        legacy,
        "_build_metadata_item",
        lambda p: DetectionItem(filename=p.name, path=str(p), file_type=p.suffix.lstrip(".")),
    )
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda *a, **kw: None)
    exported = []
    monkeypatch.setattr(legacy, "_export_results", lambda out, items: exported.extend(items))
    return legacy, state_path, model_path, manifest_path, exported


def test_dinov3_job_allows_fast_video_and_still_guards_all_mode(monkeypatch, tmp_path):
    video = tmp_path / "clip.avi"
    video.write_bytes(b"video")
    legacy, _, model_path, manifest_path, _ = _setup_dinov3_job_env(
        monkeypatch, tmp_path, [video],
    )

    services._validate_dino_job_options(str(model_path), str(manifest_path), "fast")
    services._validate_dino_job_options(str(model_path), str(manifest_path), "skip")
    with pytest.raises(ValueError, match="快速识别.*跳过视频"):
        services._validate_dino_job_options(str(model_path), str(manifest_path), "all")

    # Even when _validate_dino_job_options is bypassed, the runner still guards 'all' with videos.
    monkeypatch.setattr(legacy, "_validate_dino_job_options", lambda *a: None)
    request = CreateJobRequest(
        input_dir=str(tmp_path),
        options=ProcessingOptions(
            enable_detection=True,
            model_path=str(model_path),
            classification_model_path=str(manifest_path),
            video_mode="all",
        ),
    )
    manager = services.ProcessingJobManager()
    try:
        now = services.utc_now()
        manager._jobs["job"] = JobSummary(
            id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
            created_at=now, updated_at=now,
        )
        manager._job_requests["job"] = request
        manager._run_job("job", request, [])
        finished = manager.get_job("job")
        assert finished.state == JobState.FAILED
        assert "快速识别" in (finished.error or "")
    finally:
        manager._executor.shutdown()


def test_mixed_dinov3_fast_video_job_finalizes_photos_only_and_preserves_order(
    monkeypatch, tmp_path,
):
    p1 = tmp_path / "01.jpg"
    v1 = tmp_path / "02.avi"
    p2 = tmp_path / "03.jpg"
    v2 = tmp_path / "04.mp4"
    files = [p1, v1, p2, v2]
    for path in files:
        path.write_bytes(path.name.encode())

    legacy, state_path, model_path, manifest_path, exported = _setup_dinov3_job_env(
        monkeypatch, tmp_path, files,
    )
    detector = SimpleNamespace(
        dinov3_classifier=SimpleNamespace(backend="dinov3", display_name="Neri1.2preview"),
        _dinov3_collector=DinoV3CropCollector(),
    )
    monkeypatch.setattr(legacy, "_load_detector", lambda *a: detector)

    def fake_detect_images(det, batch, batch_items, *args, **kwargs):
        out = []
        for path in batch:
            det._dinov3_collector.record_crop(_crop_record(path))
            out.append(_photo_item(path))
        return out

    monkeypatch.setattr(legacy, "_detect_image_batch", fake_detect_images)

    import system.dinov3.processor as processor
    finalized_photo_paths = []

    def fake_finalize_photos(det, results, root, **kwargs):
        finalized_photo_paths.append([item.path for item in results])
        return [
            item.model_copy(
                update={
                    "species": ["deer"],
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

    monkeypatch.setattr(processor, "finalize_photo_job", fake_finalize_photos)

    fast_video_paths = []

    def fake_fast_videos(det, paths, items, req, root, *, on_item_ready=None, **kwargs):
        fast_video_paths.append([str(p) for p in paths])
        out = []
        for index, path in enumerate(paths):
            item = _completed_video_item(path, "fox")
            out.append(item)
            if on_item_ready is not None:
                on_item_ready(index, item)
        return out

    monkeypatch.setattr(legacy, "_detect_video_fast_batch", fake_fast_videos)

    request = CreateJobRequest(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        options=ProcessingOptions(
            enable_detection=True,
            model_path=str(model_path),
            classification_model_path=str(manifest_path),
            batch_size=2,
            video_mode="fast",
        ),
    )
    manager = services.ProcessingJobManager()
    try:
        now = services.utc_now()
        manager._jobs["job"] = JobSummary(
            id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
            created_at=now, updated_at=now,
        )
        manager._job_requests["job"] = request
        manager._run_job("job", request, [])
        finished = manager.get_job("job")
        assert finished.state == JobState.COMPLETED, finished.error
        assert finished.processed == 4
        assert finalized_photo_paths == [[str(p1), str(p2)]]
        assert fast_video_paths == [[str(v1), str(v2)]]
        assert [item.path for item in finished.results] == [str(p) for p in files]
        assert [item.species for item in finished.results] == [["deer"], ["fox"], ["deer"], ["fox"]]
        assert all(item.detection_data.get("sequence_completed") is True for item in finished.results)
        assert [item.path for item in exported] == [str(p) for p in files]
        assert not DinoV3JobCache.path_for(state_path, "job").exists()
    finally:
        manager._executor.shutdown()


def test_video_only_dinov3_fast_job_skips_photo_finalization(monkeypatch, tmp_path):
    v1 = tmp_path / "01.avi"
    v2 = tmp_path / "02.mp4"
    files = [v1, v2]
    for path in files:
        path.write_bytes(path.name.encode())

    legacy, _, model_path, manifest_path, _ = _setup_dinov3_job_env(
        monkeypatch, tmp_path, files,
    )
    detector = SimpleNamespace(
        dinov3_classifier=SimpleNamespace(backend="dinov3", display_name="Neri1.2preview"),
        _dinov3_collector=DinoV3CropCollector(),
    )
    monkeypatch.setattr(legacy, "_load_detector", lambda *a: detector)
    import system.dinov3.processor as processor
    monkeypatch.setattr(
        processor,
        "finalize_photo_job",
        lambda *a, **kw: pytest.fail("finalize_photo_job must not run for video-only jobs"),
    )
    monkeypatch.setattr(
        legacy,
        "_detect_video_fast_batch",
        lambda det, paths, items, req, root, *, on_item_ready=None, **kw: [
            (on_item_ready(i, item) if on_item_ready else None) or item
            for i, item in enumerate(_completed_video_item(p, "hare") for p in paths)
        ],
    )
    request = CreateJobRequest(
        input_dir=str(tmp_path),
        options=ProcessingOptions(
            enable_detection=True,
            model_path=str(model_path),
            classification_model_path=str(manifest_path),
            video_mode="fast",
        ),
    )
    manager = services.ProcessingJobManager()
    try:
        now = services.utc_now()
        manager._jobs["job"] = JobSummary(
            id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
            created_at=now, updated_at=now,
        )
        manager._job_requests["job"] = request
        manager._run_job("job", request, [])
        finished = manager.get_job("job")
        assert finished.state == JobState.COMPLETED, finished.error
        assert [item.path for item in finished.results] == [str(v1), str(v2)]
    finally:
        manager._executor.shutdown()


def test_resume_after_video_cancel_restores_photo_cache_and_preserves_completed_videos(
    monkeypatch, tmp_path,
):
    p1 = tmp_path / "01.jpg"
    v1 = tmp_path / "02.avi"
    p2 = tmp_path / "03.jpg"
    v2 = tmp_path / "04.avi"
    files = [p1, v1, p2, v2]
    for path in files:
        path.write_bytes(path.name.encode())

    legacy, state_path, model_path, manifest_path, _ = _setup_dinov3_job_env(
        monkeypatch, tmp_path, files,
    )
    detected_photos = []
    detected_videos = []
    finalized_inputs = []

    def load_detector(*a):
        return SimpleNamespace(
            dinov3_classifier=SimpleNamespace(backend="dinov3", display_name="Neri1.2preview"),
            _dinov3_collector=DinoV3CropCollector(),
        )

    monkeypatch.setattr(legacy, "_load_detector", load_detector)

    def fake_detect_images(det, batch, batch_items, *args, **kwargs):
        out = []
        for path in batch:
            detected_photos.append(path)
            det._dinov3_collector.record_crop(_crop_record(path))
            out.append(_photo_item(path))
        return out

    monkeypatch.setattr(legacy, "_detect_image_batch", fake_detect_images)

    import system.dinov3.processor as processor

    def fake_finalize_photos(det, results, root, **kwargs):
        finalized_inputs.append([item.path for item in results])
        for item in results:
            assert len(det._dinov3_collector.get_crops_for_path(item.path)) == 1
        return [
            item.model_copy(
                update={
                    "species": ["deer"],
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

    monkeypatch.setattr(processor, "finalize_photo_job", fake_finalize_photos)

    cancel_after_first_video = True
    active_manager = None

    def fake_fast_videos(det, paths, items, req, root, *, on_item_ready=None, cancelled=None, **kw):
        detected_videos.append([str(p) for p in paths])
        out = []
        for index, path in enumerate(paths):
            if cancelled and cancelled():
                raise RuntimeError("用户取消任务")
            item = _completed_video_item(path, "fox")
            out.append(item)
            if on_item_ready is not None:
                on_item_ready(index, item)
            if cancel_after_first_video and index == 0:
                active_manager._cancelled_jobs.add("job")
                raise RuntimeError("用户取消任务")
        return out

    monkeypatch.setattr(legacy, "_detect_video_fast_batch", fake_fast_videos)

    request = CreateJobRequest(
        input_dir=str(tmp_path),
        options=ProcessingOptions(
            enable_detection=True,
            model_path=str(model_path),
            classification_model_path=str(manifest_path),
            batch_size=2,
            video_mode="fast",
        ),
    )
    manager = services.ProcessingJobManager()
    restarted = None
    try:
        active_manager = manager
        now = services.utc_now()
        manager._jobs["job"] = JobSummary(
            id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
            created_at=now, updated_at=now,
        )
        manager._job_requests["job"] = request
        manager._run_job("job", request, [])

        cancelled_job = manager.get_job("job")
        assert cancelled_job.state == JobState.CANCELLED
        assert cancelled_job.processed == 3
        assert {item.path for item in cancelled_job.results} == {str(p1), str(p2), str(v1)}
        assert DinoV3JobCache.path_for(state_path, "job").exists()

        cancel_after_first_video = False
        restarted = services.ProcessingJobManager()
        active_manager = restarted
        restarted.resume_job("job")
        restarted._executor.shutdown(wait=True)

        finished = restarted.get_job("job")
        assert finished.state == JobState.COMPLETED, finished.error
        assert detected_photos == [p1, p2]
        assert detected_videos == [[str(v1), str(v2)], [str(v2)]]
        assert finalized_inputs == [[str(p1), str(p2)], [str(p1), str(p2)]]
        assert [item.path for item in finished.results] == [str(p) for p in files]
        assert [item.species for item in finished.results] == [["deer"], ["fox"], ["deer"], ["fox"]]
        assert all(item.detection_data.get("sequence_completed") is True for item in finished.results)
        assert not DinoV3JobCache.path_for(state_path, "job").exists()
    finally:
        manager._executor.shutdown()
        if restarted is not None:
            restarted._executor.shutdown()


def test_resume_retries_failed_or_incomplete_videos_while_preserving_completed_ones(
    monkeypatch, tmp_path,
):
    v_done = tmp_path / "01.avi"
    v_err = tmp_path / "02.avi"
    v_incomplete = tmp_path / "03.avi"
    files = [v_done, v_err, v_incomplete]
    for path in files:
        path.write_bytes(path.name.encode())

    legacy, _, model_path, manifest_path, _ = _setup_dinov3_job_env(
        monkeypatch, tmp_path, files,
    )
    detector = SimpleNamespace(
        dinov3_classifier=SimpleNamespace(backend="dinov3", display_name="Neri1.2preview"),
        _dinov3_collector=DinoV3CropCollector(),
    )
    monkeypatch.setattr(legacy, "_load_detector", lambda *a: detector)
    rerun_paths = []

    def fake_fast_videos(det, paths, items, req, root, *, on_item_ready=None, **kw):
        rerun_paths.append([str(p) for p in paths])
        out = []
        for index, path in enumerate(paths):
            item = _completed_video_item(path, "badger")
            out.append(item)
            if on_item_ready is not None:
                on_item_ready(index, item)
        return out

    monkeypatch.setattr(legacy, "_detect_video_fast_batch", fake_fast_videos)

    initial_results = [
        _completed_video_item(v_done, "fox"),
        DetectionItem(filename=v_err.name, path=str(v_err), file_type="avi", error="视频快速检测失败: boom"),
        DetectionItem(
            filename=v_incomplete.name,
            path=str(v_incomplete),
            file_type="avi",
            detection_data={"sequence_completed": False, "sequence_pending": True},
        ),
    ]
    request = CreateJobRequest(
        input_dir=str(tmp_path),
        options=ProcessingOptions(
            enable_detection=True,
            model_path=str(model_path),
            classification_model_path=str(manifest_path),
            video_mode="fast",
        ),
    )
    manager = services.ProcessingJobManager()
    try:
        now = services.utc_now()
        manager._jobs["job"] = JobSummary(
            id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
            created_at=now, updated_at=now,
        )
        manager._job_requests["job"] = request
        manager._run_job("job", request, initial_results)
        finished = manager.get_job("job")
        assert finished.state == JobState.COMPLETED, finished.error
        assert rerun_paths == [[str(v_err), str(v_incomplete)]]
        assert [item.path for item in finished.results] == [str(p) for p in files]
        assert [item.species for item in finished.results] == [["fox"], ["badger"], ["badger"]]
    finally:
        manager._executor.shutdown()


def test_photo_cancellation_checkpoint_does_not_corrupt_completed_videos(
    monkeypatch, tmp_path,
):
    p1 = tmp_path / "01.jpg"
    v1 = tmp_path / "02.avi"
    p2 = tmp_path / "03.jpg"
    files = [p1, v1, p2]
    for path in files:
        path.write_bytes(path.name.encode())

    legacy, _, model_path, manifest_path, _ = _setup_dinov3_job_env(
        monkeypatch, tmp_path, files,
    )
    detector = SimpleNamespace(
        dinov3_classifier=SimpleNamespace(backend="dinov3", display_name="Neri1.2preview"),
        _dinov3_collector=DinoV3CropCollector(),
    )
    monkeypatch.setattr(legacy, "_load_detector", lambda *a: detector)

    manager = services.ProcessingJobManager()

    def fake_detect_images(det, batch, batch_items, *args, **kwargs):
        manager._cancelled_jobs.add("job")
        return [_photo_item(path) for path in batch]

    monkeypatch.setattr(legacy, "_detect_image_batch", fake_detect_images)

    import system.dinov3.processor as processor
    checkpoint_inputs = []

    def failing_finalize(det, results, root, **kwargs):
        checkpoint_inputs.append([item.path for item in results])
        raise RuntimeError("trigger fallback mark_dinov3_sequence_incomplete")

    monkeypatch.setattr(processor, "finalize_photo_job", failing_finalize)

    request = CreateJobRequest(
        input_dir=str(tmp_path),
        options=ProcessingOptions(
            enable_detection=True,
            model_path=str(model_path),
            classification_model_path=str(manifest_path),
            batch_size=1,
            video_mode="fast",
        ),
    )
    try:
        now = services.utc_now()
        manager._jobs["job"] = JobSummary(
            id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
            created_at=now, updated_at=now,
        )
        manager._job_requests["job"] = request
        completed_v1 = _completed_video_item(v1, "fox")
        manager._run_job("job", request, [completed_v1])
        cancelled = manager.get_job("job")
        assert cancelled.state == JobState.CANCELLED
        assert checkpoint_inputs == [[str(p1)]]
        by_path = {item.path: item for item in cancelled.results}
        assert by_path[str(p1)].detection_data["sequence_status"] == "partial"
        assert by_path[str(p1)].detection_data["sequence_completed"] is False
        assert by_path[str(v1)].detection_data["sequence_completed"] is True
        assert by_path[str(v1)].detection_data["sequence_pending"] is False
        assert by_path[str(v1)].detection_data["sequence_status"] == "complete"
        assert by_path[str(v1)].species == ["fox"]
    finally:
        manager._executor.shutdown()


def test_failed_video_in_mixed_dinov3_job_is_reported_without_failing_healthy_items(
    monkeypatch, tmp_path,
):
    p1 = tmp_path / "01.jpg"
    v_ok = tmp_path / "02.avi"
    v_bad = tmp_path / "03.avi"
    files = [p1, v_ok, v_bad]
    for path in files:
        path.write_bytes(path.name.encode())

    legacy, _, model_path, manifest_path, exported = _setup_dinov3_job_env(
        monkeypatch, tmp_path, files,
    )
    detector = SimpleNamespace(
        dinov3_classifier=SimpleNamespace(backend="dinov3", display_name="Neri1.2preview"),
        _dinov3_collector=DinoV3CropCollector(),
    )
    monkeypatch.setattr(legacy, "_load_detector", lambda *a: detector)
    monkeypatch.setattr(
        legacy,
        "_detect_image_batch",
        lambda det, batch, batch_items, *a, **kw: [
            det._dinov3_collector.record_crop(_crop_record(p)) or _photo_item(p)
            for p in batch
        ],
    )
    import system.dinov3.processor as processor
    monkeypatch.setattr(
        processor,
        "finalize_photo_job",
        lambda det, results, root, **kw: [
            item.model_copy(
                update={
                    "species": ["deer"],
                    "detection_data": {
                        **(item.detection_data or {}),
                        "sequence_completed": True,
                        "sequence_pending": False,
                        "sequence_status": "complete",
                    },
                }
            )
            for item in results
        ],
    )

    def fake_fast_videos(det, paths, items, req, root, *, on_item_ready=None, **kw):
        out = []
        for index, path in enumerate(paths):
            if path == v_bad:
                item = items[index].model_copy(update={"error": "视频快速检测失败: 无法从视频中提取有效帧"})
            else:
                item = _completed_video_item(path, "fox")
            out.append(item)
            if on_item_ready is not None:
                on_item_ready(index, item)
        return out

    monkeypatch.setattr(legacy, "_detect_video_fast_batch", fake_fast_videos)

    request = CreateJobRequest(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        options=ProcessingOptions(
            enable_detection=True,
            model_path=str(model_path),
            classification_model_path=str(manifest_path),
            video_mode="fast",
        ),
    )
    manager = services.ProcessingJobManager()
    try:
        now = services.utc_now()
        manager._jobs["job"] = JobSummary(
            id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
            created_at=now, updated_at=now,
        )
        manager._job_requests["job"] = request
        manager._run_job("job", request, [])
        finished = manager.get_job("job")
        assert finished.state == JobState.COMPLETED, finished.error
        assert finished.failed_count == 1
        assert [item.path for item in exported] == [str(p1), str(v_ok)]
        by_path = {item.path: item for item in finished.results}
        assert by_path[str(v_bad)].error == "视频快速检测失败: 无法从视频中提取有效帧"
    finally:
        manager._executor.shutdown()


def test_dinov3_job_fails_closed_if_successful_video_lacks_sequence_completed(
    monkeypatch, tmp_path,
):
    video = tmp_path / "01.avi"
    video.write_bytes(b"video")
    legacy, _, model_path, manifest_path, _ = _setup_dinov3_job_env(
        monkeypatch, tmp_path, [video],
    )
    detector = SimpleNamespace(
        dinov3_classifier=SimpleNamespace(backend="dinov3", display_name="Neri1.2preview"),
        _dinov3_collector=DinoV3CropCollector(),
    )
    monkeypatch.setattr(legacy, "_load_detector", lambda *a: detector)
    monkeypatch.setattr(
        legacy,
        "_detect_video_fast_batch",
        lambda det, paths, items, req, root, *, on_item_ready=None, **kw: [
            DetectionItem(
                filename=paths[0].name,
                path=str(paths[0]),
                file_type="avi",
                detection_data={"video_mode": "fast"},
            )
        ],
    )
    request = CreateJobRequest(
        input_dir=str(tmp_path),
        options=ProcessingOptions(
            enable_detection=True,
            model_path=str(model_path),
            classification_model_path=str(manifest_path),
            video_mode="fast",
        ),
    )
    manager = services.ProcessingJobManager()
    try:
        now = services.utc_now()
        manager._jobs["job"] = JobSummary(
            id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
            created_at=now, updated_at=now,
        )
        manager._job_requests["job"] = request
        manager._run_job("job", request, [])
        finished = manager.get_job("job")
        assert finished.state == JobState.FAILED
        assert "incomplete" in (finished.error or "").lower()
    finally:
        manager._executor.shutdown()
