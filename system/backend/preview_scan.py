from __future__ import annotations

import threading


class PreviewScanCoordinator:
    """Let the newest preview request cooperatively cancel older scans."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0

    def begin(self) -> int:
        with self._lock:
            self._generation += 1
            return self._generation

    def is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation
