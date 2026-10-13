"""Job-local batch tuning shared by detection and crop embedding inference."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import statistics

from system.config import XPU_ENABLED

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcceleratorMemory:
    free: int
    total: int
    allocated: int
    peak: int


def accelerator_memory(device: str | None = None) -> AcceleratorMemory | None:
    try:
        import torch

        if device == "cpu":
            return None
        accelerator = torch.cuda if torch.cuda.is_available() else None
        if accelerator is None and XPU_ENABLED:
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                accelerator = xpu
        if accelerator is None:
            return None
        free, total = accelerator.mem_get_info()
        return AcceleratorMemory(
            int(free), int(total), int(accelerator.memory_allocated()),
            int(accelerator.max_memory_allocated()),
        )
    except Exception:
        return None


def is_accelerator_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        type(exc).__name__ == "OutOfMemoryError"
        or ("out of memory" in message and any(
            name in message for name in ("cuda", "xpu", "gpu")
        ))
    )


class AdaptiveBatchSize:
    """Try larger batches only with memory headroom and measured throughput.

    Warmup and incomplete batches do not drive tuning. A slower trial or OOM
    limits future growth for this job, avoiding repeated costly probes. Peak
    counters are read, never reset: nested encoders and concurrent jobs must not
    corrupt each other's measurements. Historical peaks give a conservative
    upper bound on working memory.
    """

    def __init__(self, size: int, *, maximum: int, enabled: bool, label: str):
        self.size = max(1, size)
        self.maximum = max(self.size, maximum)
        self.enabled = enabled
        self.label = label
        self._warmed = False
        self._samples: list[float] = []
        self._trial: tuple[int, float] | None = None

    def reduce_after_oom(self, failed_count: int) -> None:
        self.size = max(1, failed_count // 2)
        self.maximum = self.size
        self._samples.clear()
        self._trial = None
        logger.warning("%s OOM: reducing batch to %d", self.label, self.size)

    def observe(
        self, count: int, seconds: float,
        before: AcceleratorMemory | None, after: AcceleratorMemory | None,
    ) -> None:
        if not self.enabled or count != self.size or seconds <= 0:
            return
        if not self._warmed:
            self._warmed = True
            return
        if before is None or after is None:
            return
        self._samples.append(seconds / count)
        if len(self._samples) < 3:
            return
        cost = statistics.median(self._samples)
        self._samples.clear()
        if self._trial is not None:
            previous_size, previous_cost = self._trial
            self._trial = None
            if cost > previous_cost * 1.05:
                self.size = self.maximum = previous_size
                logger.info("%s: larger batch was slower; retaining %d", self.label, self.size)
                return
        candidate = min(self.maximum, self.size * 2)
        if candidate <= self.size:
            return
        reserve = max(512 * 1024**2, after.total * 0.1)
        headroom = max(0, after.free - reserve)
        working = max(1024**2, after.peak - before.allocated, after.allocated - before.allocated)
        extra = working / count * (candidate - count) * 1.5
        if extra > headroom:
            return
        self._trial = (self.size, cost)
        self.size = candidate
        logger.info("%s: testing batch %d with %.1f MiB headroom", self.label, self.size, headroom / 1024**2)
