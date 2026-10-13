"""Choose inference batches and retry automatic batches with less GPU memory."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
import math
import time
from typing import Any, Callable, Iterator

from system.config import XPU_ENABLED
from system.adaptive_batch import AdaptiveBatchSize, accelerator_memory, is_accelerator_oom
from .models import ProcessingOptions

logger = logging.getLogger(__name__)


def prefetched_path_batches(paths, batch_size: Callable[[], int], preload=None):
    """Bounded photo lookahead that picks up learned sizes as the job advances."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neri-image-preload") if preload else None
    pending = deque()
    offset = 0
    try:
        while pending or offset < len(paths):
            while len(pending) < 3 and offset < len(paths):
                end = min(len(paths), offset + max(1, batch_size()))
                batch = paths[offset:end]
                future = executor.submit(preload, [str(path) for path in batch]) if executor else None
                pending.append((batch, future))
                offset = end
            yield pending.popleft()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def _gpu_memory_available() -> int | None:
    """Return current free accelerator memory; CPU/unknown devices use one item."""
    try:
        import torch

        device = torch.cuda if torch.cuda.is_available() else None
        if device is None and XPU_ENABLED:
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                device = xpu
        if device is not None:
            free, _ = device.mem_get_info()
            return max(0, int(free))
    except Exception:
        logger.debug("Cannot query accelerator memory", exc_info=True)
    return None


def _automatic_batch_size(options: ProcessingOptions) -> int:
    free = _gpu_memory_available()
    if free is None:
        return 1
    # Keep room for models loaded on first inference, classification crops and
    # other applications. This is a conservative starting estimate, not a
    # model-specific memory guarantee; inference can reduce it after an OOM.
    budget = max(0, free - 1024**3) * 0.6
    input_size = math.ceil(options.imgsz / 32) * 32
    bytes_per_image = 256 * 1024**2 * (input_size / 640) ** 2
    if options.use_fp16:
        bytes_per_image *= 0.5
    if options.use_augment:
        bytes_per_image *= 1.5
    limit = max(1, min(32, int(budget / bytes_per_image)))
    return 1 << (limit.bit_length() - 1)


def _slice_preloaded(data, start: int, end: int, total: int):
    if data is None or (start == 0 and end == total):
        return data
    indices, images, rgb_images = data
    positions = [position for position, index in enumerate(indices) if start <= index < end]
    return (
        [indices[position] - start for position in positions],
        [images[position] for position in positions],
        [rgb_images[position] for position in positions],
    )


@dataclass
class InferenceBatch:
    start: int
    end: int
    detections: list[dict[str, Any]]
    preloaded_data: Any


class InferenceBatcher:
    """Remember a safe batch size for one job without changing saved settings."""

    def __init__(self, options: ProcessingOptions):
        self.automatic = options.batch_size == 0
        size = (
            _automatic_batch_size(options) if self.automatic else options.batch_size
        )
        self._tuner = AdaptiveBatchSize(size, maximum=32, enabled=self.automatic, label="Detection")
        if self.automatic:
            logger.info("Automatic inference batch selected: size=%d", self.batch_size)

    @property
    def batch_size(self) -> int:
        return self._tuner.size

    def detect(
        self,
        detector,
        paths: list[str],
        *,
        preloaded_data=None,
        check_cancelled: Callable[[], None] | None = None,
        **kwargs,
    ) -> Iterator[InferenceBatch]:
        """Consume/persist each yielded batch before advancing (DINO state is per call)."""
        start = 0
        while start < len(paths):
            if check_cancelled is not None:
                check_cancelled()
            end = min(len(paths), start + self.batch_size)
            loaded = _slice_preloaded(preloaded_data, start, end, len(paths))
            memory_before = accelerator_memory() if self.automatic else None
            started = time.perf_counter()
            try:
                detections = detector.detect_batch_species(
                    paths[start:end], preloaded_data=loaded, **kwargs,
                )
            except Exception as exc:
                if not self.automatic or end - start <= 1 or not is_accelerator_oom(exc):
                    raise
                self._tuner.reduce_after_oom(end - start)
                logger.warning(
                    "GPU memory exhausted; retrying automatic batch with size=%d",
                    self.batch_size,
                )
            else:
                self._tuner.observe(
                    end - start, time.perf_counter() - started, memory_before,
                    accelerator_memory() if self.automatic else None,
                )
                yield InferenceBatch(start, end, detections, loaded)
                start = end
                continue

            # Release the failed call's traceback before emptying the cache so
            # tensors referenced by that stack can actually be reclaimed.
            drain = getattr(detector, "drain_dinov2_observations", None)
            if callable(drain):
                drain()
            cleanup = getattr(detector, "cleanup_runtime_cache", None)
            if callable(cleanup):
                cleanup(clear_cuda_cache=True)
