"""Group camera-trap images into trigger events using capture timestamps."""
from __future__ import annotations

import hashlib
import os
import threading
from datetime import datetime
from pathlib import Path

from PIL import Image

from .policy import IMAGE_SUFFIXES

EVENT_GAP_SECONDS = 10
_MAX_FILE_TIME_CACHE = 50000
_MAX_DIR_EVENT_CACHE = 512


class TriggerEventResolver:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._file_time_cache: dict[tuple[str, int, int], datetime | None] = {}
        self._dir_event_cache: dict[
            tuple[str, str],
            tuple[int, dict[str, tuple[tuple[int, int], str]]],
        ] = {}

    @staticmethod
    def _capture_time(path: Path) -> datetime | None:
        try:
            with Image.open(path) as image:
                exif = image.getexif()
                raw = exif.get(36867) or exif.get(306)
        except (OSError, ValueError):
            return None
        if not raw:
            return None
        text = str(raw).strip()
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    def _cached_capture_time(
        self,
        path: Path,
        stat_sig: tuple[int, int] | None,
    ) -> datetime | None:
        if stat_sig is None:
            return self._capture_time(path)
        cache_key = (str(path), stat_sig[0], stat_sig[1])
        with self._lock:
            if cache_key in self._file_time_cache:
                return self._file_time_cache[cache_key]
        captured = self._capture_time(path)
        with self._lock:
            if len(self._file_time_cache) >= _MAX_FILE_TIME_CACHE:
                self._file_time_cache.pop(next(iter(self._file_time_cache)))
            self._file_time_cache[cache_key] = captured
        return captured

    def event_key(self, path: Path, *, folder_key: str) -> str:
        path = Path(path)
        try:
            resolved_target = path.resolve()
            target_stat = resolved_target.stat()
            target_sig: tuple[int, int] | None = (
                target_stat.st_size,
                target_stat.st_mtime_ns,
            )
        except OSError:
            resolved_target = path.resolve()
            target_sig = None
        resolved_target_key = str(resolved_target)
        parent_dir = resolved_target.parent

        try:
            dir_mtime_ns = parent_dir.stat().st_mtime_ns
        except OSError:
            dir_mtime_ns = None

        dir_cache_key = (str(parent_dir), folder_key)
        if dir_mtime_ns is not None and target_sig is not None:
            with self._lock:
                cached_entry = self._dir_event_cache.get(dir_cache_key)
                if cached_entry is not None and cached_entry[0] == dir_mtime_ns:
                    cached_target = cached_entry[1].get(resolved_target_key)
                    if cached_target is not None and cached_target[0] == target_sig:
                        return cached_target[1]

        target_time = self._cached_capture_time(resolved_target, target_sig)
        if target_time is None:
            identity = f"{folder_key}|file|{resolved_target_key}"
            return hashlib.sha256(identity.encode("utf-8")).hexdigest()

        timed: list[tuple[datetime, Path, tuple[int, int] | None]] = []
        try:
            with os.scandir(parent_dir) as entries:
                for entry in entries:
                    if os.path.splitext(entry.name)[1].lower() not in IMAGE_SUFFIXES:
                        continue
                    try:
                        if not entry.is_file():
                            continue
                        entry_stat = entry.stat()
                        entry_sig: tuple[int, int] | None = (
                            entry_stat.st_size,
                            entry_stat.st_mtime_ns,
                        )
                    except OSError:
                        continue
                    candidate_path = Path(entry.path)
                    captured = self._cached_capture_time(candidate_path, entry_sig)
                    if captured is not None:
                        timed.append((captured, candidate_path, entry_sig))
        except OSError:
            timed = [(target_time, resolved_target, target_sig)]

        timed.sort(key=lambda item: (item[0], item[1].name.casefold()))

        event_map: dict[str, tuple[tuple[int, int], str]] = {}
        cluster_start = ""
        previous_time: datetime | None = None
        target_event_key: str | None = None
        for captured, candidate, entry_sig in timed:
            if (
                previous_time is None
                or (captured - previous_time).total_seconds() > EVENT_GAP_SECONDS
            ):
                cluster_start = captured.isoformat()
            identity = f"{folder_key}|event|{cluster_start}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            candidate_key = str(candidate.resolve())
            if entry_sig is not None:
                event_map[candidate_key] = (entry_sig, digest)
            if candidate_key == resolved_target_key:
                target_event_key = digest
            previous_time = captured

        if dir_mtime_ns is not None and event_map:
            with self._lock:
                if len(self._dir_event_cache) >= _MAX_DIR_EVENT_CACHE:
                    self._dir_event_cache.pop(next(iter(self._dir_event_cache)))
                self._dir_event_cache[dir_cache_key] = (dir_mtime_ns, event_map)

        if target_event_key is None:
            identity = f"{folder_key}|time|{target_time.isoformat()}"
            return hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return target_event_key
