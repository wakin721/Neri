"""Bounded, cancellable video decoding and CPU preprocessing ahead of inference."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
import math
import statistics
from typing import Any, Callable, Iterator

import cv2


RAW_FRAME_BUFFER_BYTES = 128 * 1024 * 1024
PREPARED_BATCH_BYTES = 256 * 1024 * 1024


class VideoPipelineCancelled(RuntimeError):
    pass


@dataclass
class VideoFrame:
    video_index: int
    frame_order: int
    frame_index: int
    fps: float
    bgr: Any


@dataclass
class VideoSampleResult:
    video_index: int
    info: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class PreparedVideoBatch:
    frames: list[VideoFrame]
    preloaded_data: tuple
    preload_seconds: float


class _FrameBuffer:
    def __init__(self, max_items: int, max_bytes: int, check_cancelled: Callable):
        self._items = deque()
        self._bytes = 0
        self._condition = threading.Condition()
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._check_cancelled = check_cancelled

    def put(self, item, size: int = 0) -> None:
        with self._condition:
            # Always admit one oversized frame, otherwise a high-resolution
            # video could deadlock even with an empty buffer.
            while self._items and (
                len(self._items) >= self._max_items
                or self._bytes + size > self._max_bytes
            ):
                self._check_cancelled()
                self._condition.wait(0.05)
            self._check_cancelled()
            self._items.append((item, size))
            self._bytes += size
            self._condition.notify_all()

    def get(self, timeout: float | None = None):
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._condition:
            while not self._items:
                self._check_cancelled()
                remaining = deadline - time.monotonic() if deadline is not None else 0.05
                if remaining <= 0:
                    raise Empty
                self._condition.wait(min(0.05, remaining))
            self._check_cancelled()
            item, size = self._items.popleft()
            self._bytes -= size
            self._condition.notify_all()
            return item


def _open_video(path: Path, decoder_threads: int):
    thread_option = getattr(cv2, "CAP_PROP_N_THREADS", None)
    if thread_option is not None:
        try:
            capture = cv2.VideoCapture(
                str(path), cv2.CAP_FFMPEG, [thread_option, decoder_threads]
            )
            if capture.isOpened():
                return capture
            capture.release()
        except cv2.error:
            pass
    # Older OpenCV builds and non-FFmpeg codecs may not support that option.
    return cv2.VideoCapture(str(path))


def sample_video_frames(
    path: Path,
    sample_count: int,
    emit: Callable,
    *,
    frame_stride: int | None = None,
    check_cancelled: Callable = lambda: None,
    decoder_threads: int = 1,
) -> dict[str, Any]:
    """Emit decoded BGR frames without creating intermediate image files."""
    check_cancelled()
    capture = _open_video(path, decoder_threads)
    points_read: list[int] = []
    try:
        if not capture.isOpened():
            raise RuntimeError("无法打开视频文件")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        def accept(frame, index):
            check_cancelled()
            emit(frame, index, len(points_read), fps)
            points_read.append(index)

        if frame_stride is not None:
            stride = max(1, int(frame_stride))
            index = 0
            while True:
                check_cancelled()
                if not capture.grab():
                    break
                if index % stride == 0:
                    ok, frame = capture.retrieve()
                    if ok:
                        accept(frame, index)
                index += 1
        else:
            sample_count = max(1, int(sample_count))
            points = sorted({
                min(total_frames - 1, max(0, round(total_frames * (i + 1) / (sample_count + 1))))
                for i in range(sample_count)
            }) if total_frames > 0 else [0]
            # Measure this video's decoder instead of assuming a fixed GOP.
            # A small sequential probe estimates per-frame cost; the first
            # sparse seek measures keyframe/preroll overhead. Dense samples
            # continue sequentially and sparse short-GOP samples use seeks.
            next_index = 0
            grab_times: list[float] = []
            seek_seconds = None
            seek_supported = True

            def grab_until(point):
                nonlocal next_index
                while next_index <= point:
                    check_cancelled()
                    started = time.perf_counter()
                    if not capture.grab():
                        return False, None
                    if next_index >= 3:
                        grab_times.append(time.perf_counter() - started)
                        if len(grab_times) > 32:
                            del grab_times[0]
                    next_index += 1
                return capture.retrieve()

            if len(points) > 1 and points[0] >= 64:
                grab_until(11)
            for point in points:
                check_cancelled()
                gap = point - next_index
                per_frame = statistics.median(grab_times) if grab_times else None
                use_seek = seek_supported and gap >= 32 and (
                    seek_seconds is None or per_frame is None
                    # Keep sequential decoding when the expected costs are
                    # close; seek/preroll timings vary with GOP position.
                    or gap * per_frame > seek_seconds * 1.5
                )
                if use_seek:
                    started = time.perf_counter()
                    positioned = capture.set(cv2.CAP_PROP_POS_FRAMES, point)
                    ok, frame = capture.read() if positioned else (False, None)
                    actual_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
                    exact = (
                        ok and math.isfinite(actual_position)
                        and abs(actual_position - (point + 1)) < 0.5
                    )
                    if exact:
                        seek_seconds = time.perf_counter() - started
                        next_index = point + 1
                    else:
                        # Some backends ignore seeks or land on keyframes.
                        # Restart and count frames so sample labels stay exact.
                        seek_supported = False
                        capture.release()
                        capture = _open_video(path, decoder_threads)
                        next_index = 0
                        ok, frame = grab_until(point)
                else:
                    ok, frame = grab_until(point)
                if ok:
                    accept(frame, point)
        if not points_read:
            raise RuntimeError("无法从视频中提取有效帧")
        return dict(frame_points=points_read, width=width, height=height,
                    fps=fps, total_frames=total_frames)
    finally:
        capture.release()


def prepared_video_batches(
    paths: list[Path],
    *,
    sample_count: int,
    batch_size: int | Callable[[], int],
    workers: int,
    decoder_threads: int,
    preload: Callable,
    frame_stride: int | None = None,
    cancelled: Callable | None = None,
    buffer_bytes: int = RAW_FRAME_BUFFER_BYTES,
    batch_bytes: int = PREPARED_BATCH_BYTES,
) -> Iterator[PreparedVideoBatch | VideoSampleResult]:
    """Keep decoding and preprocessing running while the caller runs the GPU.

    Raw frames are bounded by count and bytes. Only one prepared batch can wait
    in the output queue, plus the batch being prepared and the one in inference.
    A caller that exits early must close this iterator to join all workers.
    """
    stop = threading.Event()

    def check_cancelled():
        if stop.is_set() or (cancelled is not None and cancelled()):
            raise VideoPipelineCancelled("用户取消任务")

    current_batch_size = batch_size if callable(batch_size) else lambda: batch_size
    raw = _FrameBuffer(max(2, current_batch_size() * 2), buffer_bytes, check_cancelled)
    ready: Queue = Queue(maxsize=1)

    def publish(value):
        while True:
            check_cancelled()
            try:
                ready.put(value, timeout=0.05)
                return
            except Full:
                continue

    def sample(index: int):
        def emit(bgr, frame_index, frame_order, fps):
            raw.put(VideoFrame(index, frame_order, frame_index, fps, bgr), bgr.nbytes)

        try:
            info = sample_video_frames(
                paths[index], sample_count, emit, frame_stride=frame_stride,
                check_cancelled=check_cancelled, decoder_threads=decoder_threads,
            )
            raw.put(VideoSampleResult(index, info=info))
        except VideoPipelineCancelled:
            return
        except Exception as exc:
            try:
                raw.put(VideoSampleResult(index, error=str(exc)))
            except VideoPipelineCancelled:
                pass

    def produce():
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="neri-video-decode") as executor:
                # Keep only one task per decoder pending, including when a job
                # contains hundreds of thousands of files.
                next_video = min(workers, len(paths))
                for index in range(next_video):
                    executor.submit(sample, index)
                frames: list[VideoFrame] = []
                size = 0
                completed = 0

                def flush():
                    nonlocal frames, size
                    if not frames:
                        return
                    check_cancelled()
                    started = time.perf_counter()
                    loaded = preload([frame.bgr for frame in frames])
                    if loaded is None or len(loaded[0]) != len(frames):
                        raise RuntimeError("视频帧预处理失败")
                    publish(PreparedVideoBatch(frames, loaded, time.perf_counter() - started))
                    frames = []
                    size = 0

                try:
                    while completed < len(paths):
                        try:
                            value = raw.get(timeout=0.05 if frames else None)
                        except Empty:
                            # Do not leave the GPU idle waiting for a slow file
                            # to fill the last few slots of a partial batch.
                            flush()
                            continue
                        if isinstance(value, VideoSampleResult):
                            completed += 1
                            if next_video < len(paths):
                                executor.submit(sample, next_video)
                                next_video += 1
                            publish(value)
                            continue
                        # BGR plus RGB preprocessing copies determine the
                        # actual host memory needed by an inference batch.
                        frame_bytes = value.bgr.nbytes * 2
                        if frames and size + frame_bytes > batch_bytes:
                            flush()
                        frames.append(value)
                        size += frame_bytes
                        if len(frames) >= current_batch_size() or size >= batch_bytes:
                            flush()
                    flush()
                except BaseException:
                    # Wake blocked decoders before the executor joins them.
                    stop.set()
                    raise
            publish(None)
        except BaseException as exc:
            # Preprocessing failures must reach the consumer even after stop
            # has been set to release the decoders.
            while not consumer_done.is_set():
                try:
                    ready.put(exc, timeout=0.05)
                    break
                except Full:
                    continue

    consumer_done = threading.Event()
    producer = threading.Thread(target=produce, name="neri-video-preload", daemon=True)
    producer.start()
    try:
        while True:
            if cancelled is not None and cancelled():
                raise VideoPipelineCancelled("用户取消任务")
            try:
                value = ready.get(timeout=0.05)
            except Empty:
                continue
            if value is None:
                break
            if isinstance(value, BaseException):
                raise value
            yield value
    finally:
        consumer_done.set()
        stop.set()
        producer.join()
