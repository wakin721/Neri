"""Bounded background JPEG writes; database updates stay on the job thread."""
from __future__ import annotations

from pathlib import Path
import errno
from queue import Queue
import threading
from typing import Callable

import numpy as np


class ExampleWriter:
    def __init__(self, write: Callable, *, max_items: int = 64, max_bytes: int = 16 * 1024**2):
        self._write = write
        self._max_items = max(1, max_items)
        self._max_bytes = max(1, max_bytes)
        self._queue: Queue = Queue()
        self._condition = threading.Condition()
        self._pending: set[Path] = set()
        self._sources: dict[Path, str | None] = {}
        self._failed_sources: set[str] = set()
        self._bytes = 0
        self._error: OSError | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    def _check_error(self, source_path=None):
        if self._error is not None:
            if source_path is not None:
                self._failed_sources.add(str(source_path))
            raise self._error

    @property
    def failed_sources(self):
        with self._condition:
            return frozenset(self._failed_sources)

    def submit(self, path: Path, crop: np.ndarray, *, source_path=None) -> None:
        # Only a 320-pixel crop is retained, never an entire decoded frame or
        # its context-variable cache. Own the pixels before returning to inference.
        with self._condition:
            if self._closed:
                raise RuntimeError("Example writer is closed")
            while self._pending and (
                len(self._pending) >= self._max_items
                or self._bytes + crop.nbytes > self._max_bytes
            ):
                if self._closed:
                    raise RuntimeError("Example writer is closed")
                self._check_error(source_path)
                if path in self._pending:
                    return
                self._condition.wait()
            if self._closed:
                raise RuntimeError("Example writer is closed")
            self._check_error(source_path)
            if path in self._pending:
                return
            owned = np.ascontiguousarray(crop).copy()
            self._pending.add(path)
            self._sources[path] = str(source_path) if source_path is not None else None
            self._bytes += owned.nbytes
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="neri-example-writer", daemon=True)
                self._thread.start()
            self._queue.put((path, owned))

    def _run(self):
        while True:
            value = self._queue.get()
            if value is None:
                return
            path, crop = value
            try:
                if not self._write(path, crop):
                    raise OSError("无法保存示例图片")
            except Exception as exc:
                with self._condition:
                    source = self._sources.get(path)
                    if source is not None:
                        self._failed_sources.add(source)
                    if self._error is None:
                        self._error = OSError(
                            getattr(exc, "errno", None) or errno.EIO,
                            f"保存示例图片失败: {exc}", str(path),
                        )
            finally:
                with self._condition:
                    self._pending.discard(path)
                    self._sources.pop(path, None)
                    self._bytes -= crop.nbytes
                    self._condition.notify_all()
                # Drop this crop before blocking on the next queue item.
                del crop, value

    def flush(self):
        with self._condition:
            while self._pending:
                self._condition.wait()
            self._check_error()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        try:
            self.flush()
        finally:
            if self._thread is not None:
                self._queue.put(None)
                self._thread.join()
                self._thread = None
