"""Bound full-resolution photo allocations before native decoders are called."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import ctypes
import logging
import os

logger = logging.getLogger(__name__)
MIB = 1024**2
GIB = 1024**3


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in (
            "total", "available", "commit_limit", "commit_available",
            "virtual_total", "virtual_available", "extended",
        )
    ]


@dataclass(frozen=True)
class HostMemory:
    total: int
    available: int
    commit_available: int | None = None


def host_memory() -> HostMemory | None:
    """Windows commit headroom is independent of available physical RAM.

    MEMORYSTATUSEX.ullAvailPageFile reports what this process can commit,
    not the free space on a disk or the remaining size of a pagefile.
    """
    try:
        if os.name == "nt":
            status = _MemoryStatus()
            status.length = ctypes.sizeof(status)
            query = ctypes.windll.kernel32.GlobalMemoryStatusEx
            query.argtypes = [ctypes.POINTER(_MemoryStatus)]
            query.restype = ctypes.c_int
            if not query(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            return HostMemory(status.total, status.available, status.commit_available)
        import psutil
        memory = psutil.virtual_memory()
        return HostMemory(int(memory.total), int(memory.available))
    except Exception:
        logger.debug("Cannot query host memory", exc_info=True)
        return None


class HostMemoryPressure(MemoryError):
    """A resumable job error; do not fall back to unguarded image decoding."""


def is_host_memory_error(error):
    if isinstance(error, MemoryError):
        return True
    message = str(error).lower()
    return any(token in message for token in (
        "insufficient memory", "failed to allocate", "std::bad_alloc", "cannot allocate memory",
        "can't allocate memory", "not enough memory",
    ))


class PhotoMemoryGuard:
    """Job-local image-header cache and conservative shared decode budget.

    Twelve bytes/pixel cover two uint8 color copies plus decoder/crop scratch.
    This is an estimate, not an exact bound for arbitrary model/native work.
    Keep additional system reserves and never use all available commit space.
    """

    def __init__(self, snapshot=None):
        self._snapshot = snapshot or host_memory
        self._sizes = OrderedDict()
        self._uncertain = set()
        self._last_plan = None
        self._last_memory = None

    def estimate(self, path):
        key = str(path)
        if key in self._sizes:
            self._sizes.move_to_end(key)
            return self._sizes[key]
        from PIL import Image
        try:
            with Image.open(path) as image:
                width, height = image.size
            size = max(4 * MIB, int(width) * int(height) * 12)
        except Image.DecompressionBombError as exc:
            raise HostMemoryPressure(f"图片像素数超过安全解码上限：{path}") from exc
        except Exception:
            # Unsupported/missing headers retain the decoder's per-file error
            # behavior, with a conservative estimate and no multi-file probe.
            size = 768 * MIB
            self._uncertain.add(key)
        self._sizes[key] = size
        if len(self._sizes) > 512:
            removed, _ = self._sizes.popitem(last=False)
            self._uncertain.discard(removed)
        return size

    def budget(self, reserved=0):
        memory = self._snapshot()
        self._last_memory = memory
        if memory is None:
            return max(0, GIB - reserved)
        physical_reserve = max(512 * MIB, min(2 * GIB, int(memory.total * .08)))
        commit_reserve = max(512 * MIB, min(2 * GIB, int(memory.total * .04)))
        limits = [max(0, min(8 * GIB, int(memory.total * .15)) - reserved),
                  max(0, (memory.available - physical_reserve) // 2)]
        if memory.commit_available is not None:
            limits.append(max(0, (memory.commit_available - commit_reserve) // 2))
        return min(limits)

    def choose(self, paths, maximum, *, reserved=0, optional=False, leave_lookahead=False):
        budget = self.budget(reserved)
        target = int(budget * .65) if leave_lookahead else budget
        count, cost = 0, 0
        for path in paths[:max(1, maximum)]:
            estimate = self.estimate(path)
            if str(path) in self._uncertain:
                if count == 0 and estimate <= budget:
                    count, cost = 1, estimate
                break
            if cost + estimate > target:
                if count == 0 and estimate <= budget:
                    count, cost = 1, estimate
                break
            count += 1
            cost += estimate
        if count == 0 and paths and not optional:
            raise HostMemoryPressure(
                "主内存或系统提交额度不足，已在解码前停止本任务；"
                "请释放内存后继续任务。"
                f" 单张预计需要 {self.estimate(paths[0])/MIB:.0f} MiB，"
                f"当前安全预算 {budget/MIB:.0f} MiB。"
            )
        decision = (maximum, count, bool(reserved))
        if count and decision != self._last_plan:
            logger.info("Photo host-memory plan: requested=%d selected=%d estimated=%.1f MiB "
                        "reserved=%.1f MiB available_budget=%.1f MiB physical_free=%s commit_free=%s",
                        maximum, count, cost/MIB, reserved/MIB, budget/MIB,
                        self._last_memory.available if self._last_memory else None,
                        self._last_memory.commit_available if self._last_memory else None)
            self._last_plan = decision
        return count, cost

    def ensure_decode(self, paths):
        cost = sum(self.estimate(path) for path in paths)
        if cost > self.budget():
            raise HostMemoryPressure("可用主内存/提交额度已下降，解码批次需要重新缩小。")

    def observe_decode(self, paths, loaded):
        if not loaded:
            return
        indices, bgr, rgb = loaded
        for index, left, right in zip(indices, bgr, rgb):
            measured = 2 * (getattr(left, "nbytes", 0) + getattr(right, "nbytes", 0))
            key = str(paths[index])
            self._sizes[key] = max(self.estimate(paths[index]), measured)

    def decode_workers(self, paths):
        largest = max((self.estimate(path) for path in paths), default=GIB)
        return max(1, min(8, len(paths), self.budget() // max(1, 4 * largest)))
