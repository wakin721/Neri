from contextlib import closing
from pathlib import Path
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from system.backend import services, video_frames
from system.backend.models import CreateJobRequest, DetectionItem, JobState, JobSummary, ProcessingOptions


@pytest.mark.parametrize("seek_cost,expected_seeks", [(0.005, 3), (0.5, 1)])
def test_sampling_measures_seek_cost_without_changing_frames(seek_cost, expected_seeks, monkeypatch):
    clock, seeks, grabs = [0.0], [], []

    class Capture:
        index = 0
        released = False

        def isOpened(self):
            return True

        def get(self, key):
            return {cv2.CAP_PROP_FRAME_COUNT: 900, cv2.CAP_PROP_FPS: 30,
                    cv2.CAP_PROP_FRAME_WIDTH: 8, cv2.CAP_PROP_FRAME_HEIGHT: 8,
                    cv2.CAP_PROP_POS_FRAMES: self.index}[key]

        def set(self, key, value):
            seeks.append(value)
            clock[0] += seek_cost
            self.index = value
            return True

        def grab(self):
            grabs.append(self.index)
            clock[0] += 0.001
            self.index += 1
            return True

        def retrieve(self):
            return True, np.full((8, 8, 3), (self.index - 1) % 256, np.uint8)

        def read(self):
            self.grab()
            return self.retrieve()

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(video_frames, "_open_video", lambda *args: capture)
    monkeypatch.setattr(video_frames.time, "perf_counter", lambda: clock[0])
    frames = []
    info = video_frames.sample_video_frames(
        Path("sample.mp4"), 3,
        lambda frame, index, order, fps: frames.append((index, int(frame[0, 0, 0]))),
    )
    assert info["frame_points"] == [225, 450, 675]
    assert frames == [(index, index % 256) for index in [225, 450, 675]]
    assert len(seeks) == expected_seeks
    assert capture.released
    assert len(grabs) < (50 if expected_seeks == 3 else 500)


@pytest.mark.parametrize("seek_behavior", ["rejected", "keyframe"])
def test_inexact_seeks_fall_back_to_counting_frames(seek_behavior, monkeypatch):
    captures = []

    class Capture:
        index = 0
        released = False

        def isOpened(self):
            return True

        def get(self, key):
            return {cv2.CAP_PROP_FRAME_COUNT: 400, cv2.CAP_PROP_FPS: 25,
                    cv2.CAP_PROP_FRAME_WIDTH: 8, cv2.CAP_PROP_FRAME_HEIGHT: 8,
                    cv2.CAP_PROP_POS_FRAMES: self.index}[key]

        def set(self, key, value):
            self.index = max(0, value - 10)
            return seek_behavior != "rejected"

        def grab(self):
            self.index += 1
            return True

        def retrieve(self):
            return True, np.full((8, 8, 3), (self.index - 1) % 256, np.uint8)

        def read(self):
            self.grab()
            return self.retrieve()

        def release(self):
            self.released = True

    def open_video(*args):
        capture = Capture()
        captures.append(capture)
        return capture

    monkeypatch.setattr(video_frames, "_open_video", open_video)
    frames = []
    video_frames.sample_video_frames(Path("sample.mp4"), 3, lambda frame, index, *args: frames.append((index, int(frame[0, 0, 0]))))
    assert frames == [(100, 100), (200, 200), (300, 44)]
    assert len(captures) == 2 and all(capture.released for capture in captures)


@pytest.mark.parametrize("reported_count", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("stride,expected", [(None, [25, 50, 75]), (20, [0, 20, 40, 60, 80])])
def test_missing_frame_count_does_not_reduce_video_to_first_frame(reported_count, stride, expected, monkeypatch):
    captures = []

    class Capture:
        def __init__(self):
            self.index = 0
            self.released = False

        def isOpened(self):
            return True

        def get(self, key):
            return {cv2.CAP_PROP_FRAME_COUNT: reported_count, cv2.CAP_PROP_FPS: 25,
                    cv2.CAP_PROP_FRAME_WIDTH: 8, cv2.CAP_PROP_FRAME_HEIGHT: 8,
                    cv2.CAP_PROP_POS_FRAMES: self.index}[key]

        def set(self, key, value):
            self.index = value
            return True

        def grab(self):
            if self.index >= 100:
                return False
            self.index += 1
            return True

        def retrieve(self):
            return True, np.full((8, 8, 3), self.index - 1, np.uint8)

        def read(self):
            return self.retrieve() if self.grab() else (False, None)

        def release(self):
            self.released = True

    def open_video(*args):
        capture = Capture()
        captures.append(capture)
        return capture

    monkeypatch.setattr(video_frames, "_open_video", open_video)
    frames = []
    info = video_frames.sample_video_frames(
        Path("no-frame-count.mp4"), 3,
        lambda frame, index, order, fps: frames.append((index, int(frame[0, 0, 0]))),
        frame_stride=stride,
    )
    assert info["frame_points"] == expected
    assert info["total_frames"] == 100
    assert frames == [(index, index) for index in expected]
    assert len(captures) == (2 if stride is None else 1)
    assert all(capture.released for capture in captures)


def test_missing_frame_count_scan_is_cancellable_and_releases_decoder(monkeypatch):
    grabbed, released = [], []
    capture = SimpleNamespace(
        isOpened=lambda: True,
        get=lambda key: 0,
        grab=lambda: grabbed.append(True) or True,
        release=lambda: released.append(True),
    )
    monkeypatch.setattr(video_frames, "_open_video", lambda *args: capture)

    def check_cancelled():
        if len(grabbed) == 2:
            raise video_frames.VideoPipelineCancelled("cancelled during count")

    with pytest.raises(video_frames.VideoPipelineCancelled, match="during count"):
        video_frames.sample_video_frames(
            Path("no-frame-count.mp4"), 3, lambda *args: pytest.fail("emitted before count finished"),
            check_cancelled=check_cancelled,
        )
    assert len(grabbed) == 2
    assert released == [True]


def preload(frames):
    return list(range(len(frames))), frames, frames


def test_video_prefetch_adopts_larger_batches_without_losing_frames(monkeypatch):
    allow_rest = threading.Event()
    size = [1]

    def sample(path, count, emit, **kwargs):
        for index in range(9):
            if index == 1:
                assert allow_rest.wait(3)
            emit(np.full((8, 8, 3), index, np.uint8), index, index, 25)
        return {"frame_points": list(range(9))}

    monkeypatch.setattr(video_frames, "sample_video_frames", sample)
    with closing(video_frames.prepared_video_batches(
        [Path("video.avi")], sample_count=9, batch_size=lambda: size[0],
        workers=1, decoder_threads=1, preload=preload,
    )) as stream:
        first = next(stream)
        size[0] = 4
        allow_rest.set()
        batches = [first] + [value for value in stream if isinstance(value, video_frames.PreparedVideoBatch)]
    assert [len(batch.frames) for batch in batches] == [1, 4, 4]
    assert [frame.frame_index for batch in batches for frame in batch.frames] == list(range(9))


def test_next_video_starts_during_current_inference_with_bounded_scheduling(monkeypatch):
    started, second_started = [], threading.Event()

    def sample(path, count, emit, **kwargs):
        started.append(path)
        if len(started) == 2:
            second_started.set()
        emit(np.zeros((8, 8, 3), np.uint8), 0, 0, 25)
        return {"frame_points": [0]}

    monkeypatch.setattr(video_frames, "sample_video_frames", sample)
    with closing(video_frames.prepared_video_batches(
        [Path(f"{index}.avi") for index in range(1000)], sample_count=1,
        batch_size=1, workers=1, decoder_threads=1, preload=preload,
    )) as stream:
        assert isinstance(next(stream), video_frames.PreparedVideoBatch)
        assert second_started.wait(3)
        assert len(started) < 10
    assert not any(thread.name.startswith("neri-video-") for thread in threading.enumerate())


@pytest.mark.parametrize("ending", ["success", "cancel", "storage_failure"])
def test_fast_videos_checkpoint_before_slow_video_finishes(ending, tmp_path, monkeypatch):
    legacy = services._legacy
    first_saved, cancelled = threading.Event(), threading.Event()
    paths = [tmp_path / "fast.avi", tmp_path / "slow.avi"]
    saved, callbacks = [], []

    def sample(path, count, emit, *, check_cancelled, **kwargs):
        if path == paths[1]:
            assert first_saved.wait(3), "partial batch must run before the slow video finishes"
            check_cancelled()
        emit(np.zeros((8, 8, 3), np.uint8), 25, 0, 25)
        return dict(frame_points=[25], fps=25, width=8, height=8, total_frames=100)

    def save(values, root):
        if ending == "storage_failure" and any(path == paths[1] for path, _ in values):
            raise legacy.MediaStorageError(paths[1], "保存检测结果", OSError("disk removed"))
        saved.extend(path for path, _ in values)

    def on_ready(index, item):
        assert paths[index] in saved
        callbacks.append(index)
        if index == 0:
            if ending == "cancel":
                cancelled.set()
            first_saved.set()

    monkeypatch.setattr(video_frames, "sample_video_frames", sample)
    monkeypatch.setattr(legacy, "_fast_video_sample_worker_count", lambda _: 2)
    monkeypatch.setattr(legacy, "_save_detection_data_batch", save)
    detector = SimpleNamespace(model=object(), cls_model=None, preload_frame_data=preload,
                               detect_batch_species=lambda paths, **kwargs: [{} for _ in paths])
    items = [DetectionItem(filename=path.name, path=str(path), file_type="avi") for path in paths]
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(batch_size=8))
    def run():
        return legacy._detect_video_fast_batch(detector, paths, items, request, tmp_path,
                                               cancelled=cancelled.is_set, on_item_ready=on_ready)
    if ending == "success":
        results = run()
        assert all(item.error is None for item in results)
        assert callbacks == [0, 1]
    else:
        with pytest.raises(RuntimeError, match="取消|disk removed"):
            run()
        assert callbacks == [0]
    assert not any(thread.name.startswith("neri-video-") for thread in threading.enumerate())


def test_job_streams_all_videos_with_lazy_metadata_and_ordered_final_results(tmp_path, monkeypatch):
    legacy = services._legacy
    paths = [tmp_path / f"{index}.avi" for index in range(3)]
    for path in paths:
        path.touch()
    first_progress = threading.Event()
    metadata_reads, progress = [], []
    for module in (services, legacy):
        monkeypatch.setattr(module, "job_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(legacy, "_validate_dinov2_job_options", lambda *args: None)
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *args, **kwargs: (tmp_path, paths))
    monkeypatch.setattr(legacy, "load_media_indexes", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda *args: None)
    monkeypatch.setattr(legacy, "_export_results", lambda *args: None)
    monkeypatch.setattr(legacy, "_fast_video_sample_worker_count", lambda _: 2)

    def build(path):
        metadata_reads.append(path)
        return DetectionItem(filename=path.name, path=str(path), file_type="avi")

    def sample(path, count, emit, **kwargs):
        if path == paths[1]:
            assert first_progress.wait(3)
        emit(np.zeros((8, 8, 3), np.uint8), 25, 0, 25)
        return dict(frame_points=[25], fps=25, width=8, height=8, total_frames=100)

    monkeypatch.setattr(legacy, "_build_metadata_item", build)
    monkeypatch.setattr(video_frames, "sample_video_frames", sample)
    monkeypatch.setattr(legacy, "_load_detector", lambda *args: SimpleNamespace(
        model=object(), cls_model=None, preload_frame_data=preload,
        detect_batch_species=lambda paths, **kwargs: [{} for _ in paths],
    ))
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(
        enable_detection=True, model_path="fake.pt", batch_size=1, video_mode="fast",
    ))
    manager = services.ProcessingJobManager(max_workers=1)
    now = services.utc_now()
    manager._jobs["job"] = JobSummary(id="job", state=JobState.QUEUED,
                                      input_dir=str(tmp_path), created_at=now, updated_at=now)
    manager._job_requests["job"] = request
    mutate = manager._mutate_job

    def capture_progress(job_id, **changes):
        mutate(job_id, **changes)
        if changes.get("processed") == 1:
            assert metadata_reads == [paths[0]]
            progress.append(1)
            first_progress.set()

    monkeypatch.setattr(manager, "_mutate_job", capture_progress)
    try:
        manager._run_job("job", request, [])
        result = manager.get_job("job")
        assert result.state == JobState.COMPLETED and result.processed == 3
        assert [item.path for item in result.results] == [str(path) for path in paths]
        assert all(item.error is None for item in result.results)
        assert progress and sorted(metadata_reads) == paths
    finally:
        first_progress.set()
        manager._executor.shutdown(wait=True, cancel_futures=True)
