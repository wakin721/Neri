from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from system.backend import inference_batching, services, video_frames
from system.backend.inference_batching import InferenceBatcher
from system.backend.models import CreateJobRequest, DetectionItem, ProcessingOptions


def test_auto_is_the_default_and_survives_cpu_request_normalization(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference_batching, "XPU_ENABLED", False)
    monkeypatch.setattr(services._legacy, "detect_gpu_available", lambda: False)
    options = ProcessingOptions(enable_detection=True, model_path="model.pt", use_fp16=True)
    request = CreateJobRequest(input_dir="media", options=options)
    effective = services._legacy._effective_processing_request(request)
    assert effective.options.batch_size == 0
    assert not effective.options.use_fp16
    assert InferenceBatcher(effective.options).batch_size == 1
    restored = CreateJobRequest.model_validate(effective.model_dump(mode="json"))
    assert restored.options.batch_size == 0


@pytest.mark.parametrize("value", [-1, 65])
def test_invalid_batch_sizes_are_rejected(value):
    with pytest.raises(ValidationError):
        ProcessingOptions(batch_size=value)


def test_gpu_auto_adapts_to_memory_resolution_and_precision(monkeypatch):
    monkeypatch.setattr(inference_batching, "_gpu_memory_available", lambda: 16 * 1024**3)
    options = ProcessingOptions(imgsz=1920, use_fp16=False, use_augment=False)
    high_res = InferenceBatcher(options).batch_size
    half = InferenceBatcher(options.model_copy(update={"use_fp16": True})).batch_size
    low_res = InferenceBatcher(options.model_copy(update={"imgsz": 640})).batch_size
    augmented = InferenceBatcher(options.model_copy(update={"use_augment": True})).batch_size
    assert 1 < high_res < half <= 32
    assert high_res < low_res <= 32
    assert 1 <= augmented <= high_res
    monkeypatch.setattr(inference_batching, "_gpu_memory_available", lambda: 4 * 1024**3)
    assert InferenceBatcher(options).batch_size < high_res
    assert InferenceBatcher(ProcessingOptions(batch_size=16)).batch_size == 16


@pytest.mark.parametrize("memory", [None, 0, 512 * 1024**2])
def test_unknown_or_low_memory_uses_one_item(memory, monkeypatch):
    monkeypatch.setattr(inference_batching, "_gpu_memory_available", lambda: memory)
    assert InferenceBatcher(ProcessingOptions()).batch_size == 1


def test_auto_retries_without_losing_paths_or_preloaded_indices(monkeypatch):
    monkeypatch.setattr(inference_batching, "_automatic_batch_size", lambda _: 4)
    batcher = InferenceBatcher(ProcessingOptions())
    calls, cleanups, slices = [], [], []

    def detect(paths, *, preloaded_data):
        calls.append(len(paths))
        if len(paths) > 2:
            raise RuntimeError("CUDA out of memory")
        slices.append((list(paths), preloaded_data))
        return [{"path": path} for path in paths]

    detector = SimpleNamespace(
        detect_batch_species=detect,
        cleanup_runtime_cache=lambda **kwargs: cleanups.append(kwargs),
    )
    paths = ["a.jpg", "missing.jpg", "c.jpg", "d.jpg"]
    loaded = ([0, 2, 3], ["a-bgr", "c-bgr", "d-bgr"], ["a-rgb", "c-rgb", "d-rgb"])
    batches = list(batcher.detect(detector, paths, preloaded_data=loaded))
    assert calls == [4, 2, 2]
    assert [row["path"] for batch in batches for row in batch.detections] == paths
    assert slices == [
        (paths[:2], ([0], ["a-bgr"], ["a-rgb"])),
        (paths[2:], ([0, 1], ["c-bgr", "d-bgr"], ["c-rgb", "d-rgb"])),
    ]
    assert cleanups == [{"clear_cuda_cache": True}]
    list(batcher.detect(detector, paths))
    assert calls == [4, 2, 2, 2, 2]


@pytest.mark.parametrize("batch_size,count,message", [
    (0, 2, "invalid model output"),
    (0, 1, "CUDA out of memory"),
    (8, 4, "CUDA out of memory"),
])
def test_terminal_errors_propagate_without_retries(batch_size, count, message, monkeypatch):
    monkeypatch.setattr(inference_batching, "_automatic_batch_size", lambda _: 8)
    calls = []

    def fail(paths, **kwargs):
        calls.append(paths)
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match=message):
        list(InferenceBatcher(ProcessingOptions(batch_size=batch_size)).detect(
            SimpleNamespace(detect_batch_species=fail), ["image.jpg"] * count,
        ))
    assert len(calls) == 1


def test_legacy_detector_reports_oom_instead_of_returning_empty_results(monkeypatch, tmp_path):
    from system import image_processor_legacy

    def fail(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    processor = image_processor_legacy.ImageProcessor.__new__(image_processor_legacy.ImageProcessor)
    processor.model = fail
    processor.dinov2_classifier = None
    processor._determine_device = lambda enabled: ("cpu", False)
    processor._sync_device = lambda device: None
    monkeypatch.setattr(image_processor_legacy, "_runtime_logs_dir", lambda *args: str(tmp_path))
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        processor.detect_batch_species(["a.jpg", "b.jpg"], preloaded_data=(
            [0, 1], [frame, frame], [frame, frame],
        ))


@pytest.mark.parametrize("video", [False, True])
def test_backend_auto_split_keeps_results_and_observations_aligned(video, tmp_path, monkeypatch):
    legacy = services._legacy
    monkeypatch.setattr(inference_batching, "_automatic_batch_size", lambda _: 4)
    calls, saved, observations, persisted = [], [], [], []
    frames = [np.full((8, 8, 3), index, np.uint8) for index in range(4)]

    def preload(values):
        return list(range(len(values))), values, values

    def drain():
        result = list(observations)
        observations.clear()
        return result

    def detect(paths, *, preloaded_data, **kwargs):
        calls.append(len(paths))
        observations.clear()
        if len(paths) > 2:
            # Partial failed observations must be discarded before the retry.
            observations.append((0, -1))
            raise RuntimeError("CUDA out of memory")
        values = [int(frame[0, 0, 0]) for frame in preloaded_data[1]]
        observations.extend(enumerate(values))
        return [{"物种名称": f"species-{value}", "物种数量": "1"} for value in values]

    def persist(detector, paths, items, root, *, preloaded_data, frame_indices=None, **kwargs):
        for index, value in detector.drain_dinov2_observations():
            assert int(preloaded_data[1][index][0, 0, 0]) == value
            assert Path(items[index].path) == paths[index]
            persisted.append((paths[index].name, frame_indices[index] if frame_indices else None, value))

    detector = SimpleNamespace(
        model=object(), cls_model=None, preload_frame_data=preload,
        detect_batch_species=detect, drain_dinov2_observations=drain,
        cleanup_runtime_cache=lambda **kwargs: None,
    )
    monkeypatch.setattr(legacy, "_serialize_detector_output", lambda _, result: result)
    monkeypatch.setattr(legacy, "_persist_dinov2_observations", persist)
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda values, _: saved.extend(values))
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(vid_stride=4))
    if video:
        path = tmp_path / "video.avi"

        def sample(path, count, emit, **kwargs):
            for index, frame in enumerate(frames):
                emit(frame, index * 25, index, 25)
            return dict(frame_points=[0, 25, 50, 75], fps=25, width=8, height=8, total_frames=100)

        monkeypatch.setattr(video_frames, "sample_video_frames", sample)
        item = DetectionItem(filename=path.name, path=str(path), file_type="avi")
        results = legacy._detect_video_fast_batch(detector, [path], [item], request, tmp_path)
        assert saved[0][1]["sampled_frames"] == [0, 25, 50, 75]
        assert persisted == [(path.name, index * 25, index) for index in range(4)]
        assert results[0].species == [f"species-{index}" for index in range(4)]
    else:
        paths = [tmp_path / f"{index}.jpg" for index in range(4)]
        for path in paths:
            path.touch()
        items = [DetectionItem(filename=path.name, path=str(path), file_type="jpg") for path in paths]
        results = legacy._detect_image_batch(
            detector, paths, items, request, tmp_path, preloaded_data=preload(frames),
        )
        assert [path for path, _ in saved] == paths
        assert [item.species for item in results] == [[f"species-{index}"] for index in range(4)]
        assert persisted == [(path.name, None, index) for index, path in enumerate(paths)]
    assert calls == [4, 2, 2]
    assert all(item.error is None for item in results)
    assert request.options.batch_size == 0
