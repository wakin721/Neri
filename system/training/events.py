"""Group camera-trap images into trigger events using capture timestamps."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from PIL import Image

from .policy import IMAGE_SUFFIXES

EVENT_GAP_SECONDS = 10


class TriggerEventResolver:
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

    def event_key(self, path: Path, *, folder_key: str) -> str:
        path = Path(path)
        target_time = self._capture_time(path)
        if target_time is None:
            identity = f"{folder_key}|file|{path.resolve()}"
            return hashlib.sha256(identity.encode("utf-8")).hexdigest()

        timed: list[tuple[datetime, Path]] = []
        try:
            candidates = list(path.parent.iterdir())
        except OSError:
            candidates = [path]
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            captured = self._capture_time(candidate)
            if captured is not None:
                timed.append((captured, candidate))
        timed.sort(key=lambda item: (item[0], item[1].name.casefold()))

        target_index = next(
            (index for index, (_, candidate) in enumerate(timed) if candidate.resolve() == path.resolve()),
            None,
        )
        if target_index is None:
            identity = f"{folder_key}|time|{target_time.isoformat()}"
            return hashlib.sha256(identity.encode("utf-8")).hexdigest()

        start = target_index
        while start > 0:
            gap = (timed[start][0] - timed[start - 1][0]).total_seconds()
            if gap > EVENT_GAP_SECONDS:
                break
            start -= 1

        end = target_index
        while end + 1 < len(timed):
            gap = (timed[end + 1][0] - timed[end][0]).total_seconds()
            if gap > EVENT_GAP_SECONDS:
                break
            end += 1

        cluster_start = timed[start][0].isoformat()
        cluster_end = timed[end][0].isoformat()
        identity = f"{folder_key}|event|{cluster_start}|{cluster_end}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()
