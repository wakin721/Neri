"""Choose inference batches and retry automatic batches with less GPU memory."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import gc
import logging
import math
import time
from typing import Any, Callable, Iterator

from system.config import XPU_ENABLED
from system.adaptive_batch import AdaptiveBatchSize, accelerator_memory, is_accelerator_oom
from system.host_memory import PhotoMemoryGuard, HostMemoryPressure, is_host_memory_error
from .models import ProcessingOptions

logger = logging.getLogger(__name__)


def prefetched_path_batches(paths, batch_size: Callable[[], int], preload=None, *,
                            memory_guard=None, check_cancelled=None):
    """One byte-budgeted lookahead, with admission before any full-image decode.

    The caller must release its previous Future/data before advancing. Unlike
    the former three-batch queue, only the current batch and one lookahead can
    own full-resolution image buffers. GPU tuning remains an upper bound.
    """
    if preload is None:
        offset = 0
        while offset < len(paths):
            if check_cancelled:
                check_cancelled()
            end = min(len(paths), offset + max(1, batch_size()))
            yield paths[offset:end], None
            offset = end
        return
    guard = memory_guard or PhotoMemoryGuard()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neri-image-preload")
    offset = 0
    pending = None
    retry_ceiling = None

    def submit(start, count, cost):
        batch = paths[start:start+count]
        def guarded_load():
            guard.ensure_decode(batch)
            return preload([str(path) for path in batch])
        return start, batch, executor.submit(guarded_load), cost

    try:
        while pending is not None or offset < len(paths):
            if check_cancelled:
                check_cancelled()
            if pending is None:
                maximum = min(batch_size(), retry_ceiling) if retry_ceiling else batch_size()
                count, cost = guard.choose(paths[offset:], maximum, leave_lookahead=True)
                pending = submit(offset, count, cost)
                offset += count
            start, batch, future, cost = pending
            pending = None
            retry = False
            try:
                while True:
                    if check_cancelled:
                        check_cancelled()
                    try:
                        future.result(timeout=.2)
                        break
                    except TimeoutError:
                        if future.done():
                            raise
                guard.observe_decode(batch, future.result())
            except Exception as exc:
                if not is_host_memory_error(exc):
                    raise
                if len(batch) <= 1:
                    raise HostMemoryPressure(
                        "单张照片解码也超出安全内存预算，请释放内存后继续任务。") from exc
                retry_ceiling = max(1, len(batch)//2)
                offset = start
                logger.warning("Host memory pressure before inference; retrying photo batch <= %d",
                               retry_ceiling)
                retry = True
            if retry:
                future = None
                gc.collect()
                continue
            cost = sum(guard.estimate(path) for path in batch)
            # Current allocations may have reduced headroom since admission.
            # Do not decode additional photos when the OS is under pressure.
            if cost // 2 > guard.budget():
                if len(batch) > 1:
                    retry_ceiling = max(1, len(batch)//2)
                    offset = start
                    future = None
                    gc.collect()
                    continue
                raise HostMemoryPressure("图片已解码，但系统内存余量不足以安全开始推理；请释放内存后继续任务。")
            if offset < len(paths):
                maximum = min(batch_size(), retry_ceiling) if retry_ceiling else batch_size()
                count, next_cost = guard.choose(paths[offset:], maximum, reserved=cost, optional=True)
                if count:
                    pending = submit(offset, count, next_cost)
                    offset += count
            yield batch, future
            future = None
            batch = None
    finally:
        if pending is not None:
            pending[2].cancel()
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
        self.host_guard = PhotoMemoryGuard()
        self.automatic = options.batch_size == 0
        size = (
            _automatic_batch_size(options) if self.automatic else options.batch_size
        )
        self._tuner = AdaptiveBatchSize(size, maximum=32, enabled=self.automatic, label="Detection")
        if self.automatic:
            logger.info("Automatic inference GPU batch ceiling: size=%d; photo RAM/commit guard enabled",
                        self.batch_size)

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
                accelerator_oom = is_accelerator_oom(exc)
                host_oom = is_host_memory_error(exc)
                if not self.automatic or end - start <= 1 or not (accelerator_oom or host_oom):
                    if host_oom:
                        raise HostMemoryPressure("推理所需的主内存不足，请释放内存后继续任务。") from exc
                    raise
                self._tuner.reduce_after_oom(end - start)
                logger.warning(
                    "%s memory exhausted; retrying automatic batch with size=%d",
                    "GPU" if accelerator_oom else "Host", self.batch_size,
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
            cleanup = getattr(detector, "cleanup_runtime_cache", None)
            if callable(cleanup):
                cleanup(clear_cuda_cache=True)
