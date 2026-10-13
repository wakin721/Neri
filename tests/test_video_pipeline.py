from contextlib import closing
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import cv2
import numpy as np
import pytest

from system.backend import services, video_frames
from system.backend.models import CreateJobRequest, ProcessingOptions


def preload(frames):
    return list(range(len(frames))), frames, [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (96, 64))
    assert writer.isOpened()
    try:
        for index in range(100):
            writer.write(np.full((64, 96, 3), [index * 2, 40, 180], dtype=np.uint8))
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("stride,expected", [(None, [25, 50, 75]), (20, [0, 20, 40, 60, 80])])
def test_decoded_frames_keep_indices_colors_and_skip_jpeg_round_trip(video, monkeypatch, stride, expected):
    def forbidden(*args, **kwargs):
        pytest.fail("Video decoding must not write or read temporary JPEGs")

    monkeypatch.setattr(cv2, "imwrite", forbidden)
    monkeypatch.setattr(cv2, "imread", forbidden)
    frames = []
    info = video_frames.sample_video_frames(
        video, 3, lambda frame, index, order, fps: frames.append((frame, index, order, fps)),
        frame_stride=stride,
    )
    assert info["frame_points"] == expected
    assert [index for _, index, _, _ in frames] == expected
    capture = cv2.VideoCapture(str(video))
    try:
        for frame, index, order, fps in frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, expected_frame = capture.read()
            assert ok and np.array_equal(frame, expected_frame)
            assert order == expected.index(index) and fps == 25
    finally:
        capture.release()


def test_single_video_uses_multiple_frames_per_inference_and_preserves_results(video, monkeypatch):
    legacy = services._legacy
    batches = []
    saved = []

    def detect(paths, *, preloaded_data, **kwargs):
        frames = preloaded_data[1]
        batches.append(len(frames))
        return [{"物种名称": "fox", "物种数量": "1", "最低置信度": 0.8,
                 "检测框": [{"物种": "fox", "置信度": 0.8, "边界框": [1, 2, 20, 30]}]} for _ in frames]

    detector = SimpleNamespace(
        model=object(), cls_model=None, preload_frame_data=preload,
        detect_batch_species=detect,
    )
    monkeypatch.setattr(legacy, "_serialize_detector_output", lambda _detector, detection: detection)
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda values, root: saved.extend(values))
    monkeypatch.setattr(cv2, "imwrite", lambda *args: pytest.fail("intermediate JPEG"))
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("intermediate JPEG"))
    request = CreateJobRequest(input_dir=str(video.parent), options=ProcessingOptions(batch_size=16))
    results = legacy._detect_video_fast_batch(
        detector, [video], [legacy._build_fast_metadata_item(video)], request, video.parent,
    )
    assert results[0].error is None
    assert batches == [3]
    assert len(saved) == 1 and saved[0][0] == video
    data = saved[0][1]
    assert data["sampled_frames"] == [25, 50, 75]
    assert [box["frame_index"] for box in data["检测框"]] == [25, 50, 75]
    assert [box["timestamp"] for box in data["检测框"]] == [1.0, 2.0, 3.0]
    assert data["物种名称"] == "fox" and data["物种数量"] == "1"


def test_interleaved_videos_keep_their_own_frame_results(tmp_path, monkeypatch):
    legacy = services._legacy
    first_emitted = threading.Event()
    second_emitted = threading.Event()
    paths = [tmp_path / "first.avi", tmp_path / "second.avi"]

    def sample(path, count, emit, **kwargs):
        if path == paths[0]:
            emit(np.full((8, 8, 3), 10, np.uint8), 10, 0, 10)
            first_emitted.set()
            assert second_emitted.wait(3)
            emit(np.full((8, 8, 3), 20, np.uint8), 20, 1, 10)
        else:
            assert first_emitted.wait(3)
            emit(np.full((8, 8, 3), 50, np.uint8), 10, 0, 10)
            second_emitted.set()
            emit(np.full((8, 8, 3), 60, np.uint8), 20, 1, 10)
        return dict(frame_points=[10, 20], fps=10, width=8, height=8, total_frames=30)

    def detect(paths, *, preloaded_data, **kwargs):
        return [{"物种名称": f"species-{frame[0, 0, 0]}", "物种数量": "1"}
                for frame in preloaded_data[1]]

    monkeypatch.setattr(video_frames, "sample_video_frames", sample)
    monkeypatch.setattr(legacy, "_fast_video_sample_worker_count", lambda count: 2)
    monkeypatch.setattr(legacy, "_serialize_detector_output", lambda detector, detection: detection)
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda *args: None)
    detector = SimpleNamespace(model=object(), cls_model=None, preload_frame_data=preload,
                               detect_batch_species=detect)
    results = legacy._detect_video_fast_batch(
        detector, paths, [legacy._build_fast_metadata_item(path) for path in paths],
        CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(batch_size=2)), tmp_path,
    )
    assert all(result.error is None for result in results)
    assert results[0].species == ["species-10", "species-20"]
    assert results[1].species == ["species-50", "species-60"]
    assert [result.path for result in results] == [str(path) for path in paths]


def test_classification_only_full_video_streams_frames_with_stride(video, monkeypatch):
    legacy = services._legacy
    batches = []

    def detect(paths, **kwargs):
        batches.append(len(paths))
        return [{} for _ in paths]

    detector = SimpleNamespace(model=None, cls_model=object(), preload_frame_data=preload,
                               detect_batch_species=detect)
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda *args: None)
    result = legacy._detect_video(
        detector, video, legacy._build_fast_metadata_item(video),
        CreateJobRequest(input_dir=str(video.parent), options=ProcessingOptions(
            batch_size=3, video_mode="all", vid_stride=20,
        )), video.parent,
    )
    assert result.error is None
    assert batches == [3, 2]
    assert result.detection_data["sampled_frames"] == [0, 20, 40, 60, 80]


def test_frame_preprocessing_uses_original_pixels_and_rgb_order():
    from system.image_processor_legacy import ImageProcessor

    processor = ImageProcessor.__new__(ImageProcessor)
    bgr = np.full((8, 8, 3), [12, 90, 200], np.uint8)
    indices, processed, rgb = processor.preload_frame_data([bgr])
    assert indices == [0]
    assert np.array_equal(processed[0], bgr)
    assert np.array_equal(rgb[0], bgr[:, :, ::-1])


def test_decoder_falls_back_when_opencv_rejects_thread_option(video, monkeypatch):
    original = cv2.VideoCapture
    unsupported = SimpleNamespace(isOpened=lambda: False, release=lambda: released.append(True))
    released = []
    calls = []

    def open_video(*args):
        calls.append(args)
        return unsupported if len(args) > 1 else original(*args)

    monkeypatch.setattr(cv2, "CAP_PROP_N_THREADS", 70, raising=False)
    monkeypatch.setattr(cv2, "VideoCapture", open_video)
    info = video_frames.sample_video_frames(video, 1, lambda *args: None, decoder_threads=2)
    assert info["frame_points"] == [50]
    assert len(calls) == 2 and len(calls[1]) == 1 and released == [True]


def test_pipeline_prepares_next_batch_while_consumer_is_busy(monkeypatch):
    next_prepared = threading.Event()
    calls = []

    def sample(path, count, emit, *, check_cancelled, **kwargs):
        for index in range(8):
            check_cancelled()
            emit(np.full((8, 8, 3), index, np.uint8), index, index, 25)
        return {"frame_points": list(range(8))}

    def prepare(frames):
        calls.append(len(frames))
        if len(calls) >= 2:
            next_prepared.set()
        return preload(frames)

    monkeypatch.setattr(video_frames, "sample_video_frames", sample)
    with closing(video_frames.prepared_video_batches(
        [Path("one.avi")], sample_count=8, batch_size=2, workers=1,
        decoder_threads=1, preload=prepare,
    )) as stream:
        first = next(stream)
        assert isinstance(first, video_frames.PreparedVideoBatch)
        assert next_prepared.wait(3), "CPU preprocessing should advance while inference holds the first batch"
        assert len(first.frames) == 2


def test_long_video_has_bounded_live_frames_and_close_releases_blocked_workers(monkeypatch):
    references = []
    sampling_finished = threading.Event()
    second_prepared = threading.Event()
    prepare_calls = 0

    def sample(path, count, emit, *, check_cancelled, **kwargs):
        try:
            for index in range(10_000):
                check_cancelled()
                frame = np.zeros((32, 32, 3), np.uint8)
                references.append(weakref.ref(frame))
                emit(frame, index, index, 25)
        finally:
            sampling_finished.set()

    def prepare(frames):
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls >= 2:
            second_prepared.set()
        return preload(frames)

    monkeypatch.setattr(video_frames, "sample_video_frames", sample)
    stream = video_frames.prepared_video_batches(
        [Path("long.avi")], sample_count=3, frame_stride=1, batch_size=2,
        workers=1, decoder_threads=1, preload=prepare,
        buffer_bytes=32 * 32 * 3 * 2,
    )
    with closing(stream):
        assert isinstance(next(stream), video_frames.PreparedVideoBatch)
        assert second_prepared.wait(3)
        assert sum(reference() is not None for reference in references) <= 16
        assert len(references) < 100
    assert sampling_finished.wait(1)
    assert not any(thread.name.startswith("neri-video-") for thread in threading.enumerate())


def test_large_frames_split_batches_to_host_memory_budget(monkeypatch):
    def sample(path, count, emit, **kwargs):
        for index in range(7):
            emit(np.zeros((32, 32, 3), np.uint8), index, index, 25)
        return {"frame_points": list(range(7))}

    monkeypatch.setattr(video_frames, "sample_video_frames", sample)
    with closing(video_frames.prepared_video_batches(
        [Path("large.avi")], sample_count=7, batch_size=16, workers=1,
        decoder_threads=1, preload=preload, batch_bytes=32 * 32 * 3 * 2 * 2,
    )) as stream:
        sizes = [len(value.frames) for value in stream if isinstance(value, video_frames.PreparedVideoBatch)]
    assert sizes == [2, 2, 2, 1]


def test_cancelled_consumer_stops_decoders_with_full_buffers(monkeypatch):
    cancelled = threading.Event()
    finished = threading.Event()

    def sample(path, count, emit, *, check_cancelled, **kwargs):
        try:
            for index in range(1000):
                check_cancelled()
                emit(np.zeros((32, 32, 3), np.uint8), index, index, 25)
        finally:
            finished.set()

    monkeypatch.setattr(video_frames, "sample_video_frames", sample)
    with closing(video_frames.prepared_video_batches(
        [Path("long.avi")], sample_count=3, batch_size=1, workers=1,
        decoder_threads=1, preload=preload, cancelled=cancelled.is_set,
    )) as stream:
        next(stream)
        cancelled.set()
        with pytest.raises(video_frames.VideoPipelineCancelled):
            next(stream)
    assert finished.wait(1)


def test_preprocessing_failure_reaches_consumer_and_stops_decoder(monkeypatch):
    finished = threading.Event()

    def sample(path, count, emit, *, check_cancelled, **kwargs):
        try:
            for index in range(1000):
                check_cancelled()
                emit(np.zeros((8, 8, 3), np.uint8), index, index, 25)
        finally:
            finished.set()

    def broken_preload(frames):
        raise ValueError("broken preprocessing")

    monkeypatch.setattr(video_frames, "sample_video_frames", sample)
    with closing(video_frames.prepared_video_batches(
        [Path("long.avi")], sample_count=3, batch_size=1, workers=1,
        decoder_threads=1, preload=broken_preload,
    )) as stream:
        with pytest.raises(ValueError, match="broken preprocessing"):
            list(stream)
    assert finished.wait(1)


def test_one_broken_video_does_not_hide_other_results(video, tmp_path, monkeypatch):
    legacy = services._legacy
    broken = tmp_path / "broken.avi"
    broken.write_bytes(b"not a video")
    detector = SimpleNamespace(
        model=object(), cls_model=None, preload_frame_data=preload,
        detect_batch_species=lambda paths, **kwargs: [{} for _ in paths],
    )
    saved = []
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda values, root: saved.extend(values))
    paths = [broken, video]
    results = legacy._detect_video_fast_batch(
        detector, paths, [legacy._build_fast_metadata_item(path) for path in paths],
        CreateJobRequest(input_dir=str(tmp_path)), tmp_path,
    )
    assert results[0].error and results[1].error is None
    assert [path for path, _ in saved] == [video]


@pytest.mark.parametrize("cores,count,expected", [(1, 20, 1), (4, 20, 2), (16, 20, 8), (16, 2, 2)])
def test_automatic_decoding_concurrency(cores, count, expected, monkeypatch):
    monkeypatch.setattr(services._legacy.os, "cpu_count", lambda: cores)
    assert services._legacy._fast_video_sample_worker_count(count) == expected


def test_legacy_thread_choice_is_accepted_but_not_persisted_or_used(monkeypatch):
    monkeypatch.setattr(services._legacy, "detect_gpu_available", lambda: False)
    request = CreateJobRequest(input_dir="media", options=ProcessingOptions(
        thread_count=8, batch_size=16, use_fp16=True, enable_detection=True, model_path="model.pt",
    ))
    effective = services._legacy._effective_processing_request(request)
    assert effective.options.batch_size == 1 and not effective.options.use_fp16
    assert "thread_count" not in effective.model_dump()["options"]
    assert services._legacy._tracked_video_worker_count(10) == 1


def test_video_examples_reuse_exact_frames_without_reopening_video(video, monkeypatch):
    from system.backend.dinov2_persistence_fast import persist_runtime_observations
    from system.dinov2.registry_examples import feedback_observation_example_path

    embedding = np.zeros(768, np.float32)
    embedding[0] = 1
    observations = [SimpleNamespace(
        result_index=index, box_index=box, observation_id=f"{index}-{box}", embedding=embedding,
        accepted=True, species="fox", source="checkpoint", registry_id=None,
        known_score=0.9, threshold=0.5, bbox=(0, 0, 32, 32), best_known_species="fox",
    ) for index in range(3) for box in range(4)]
    feedback = SimpleNamespace(path=video.parent / "feedback.sqlite3", persist_observations=lambda rows: None)
    detector = SimpleNamespace(dinov2_feedback=feedback, dinov2_registry=None,
                               drain_dinov2_observations=lambda: observations)
    colors = [[20, 90, 170], [180, 30, 70], [70, 160, 30]]
    frames = [np.full((32, 32, 3), color, np.uint8) for color in colors]
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: pytest.fail("reopened video"))
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("reread frame"))
    persist_runtime_observations(
        detector, [video] * 3, [SimpleNamespace(date_taken=None)] * 3, video.parent,
        source_paths=[video] * 3, frame_indices=[25, 50, 75], timestamp_seconds=[1, 2, 3],
        preloaded_data=preload(frames),
    )
    for index, color in enumerate(colors):
        for box in range(4):
            path = feedback_observation_example_path(feedback.path, f"{index}-{box}")
            decoded = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            assert np.max(np.abs(decoded[160, 160].astype(int) - color)) < 4


def test_video_example_fallback_cache_distinguishes_frames_and_videos(video, tmp_path):
    from system.dinov2.registry_examples import _read_frame, example_frame_cache

    other = tmp_path / "other.avi"
    other.write_bytes(video.read_bytes())
    with patch.object(cv2, "VideoCapture", wraps=cv2.VideoCapture) as capture:
        with example_frame_cache():
            first = _read_frame(video, frame_index=25)
            assert _read_frame(video, frame_index=25) is first
            second = _read_frame(video, frame_index=50)
            assert _read_frame(other, frame_index=25) is not first
            assert not np.array_equal(first, second)
        with example_frame_cache():
            assert _read_frame(video, frame_index=25) is not first
    assert capture.call_count == 4
