"""Three-level video reference time resolver (own metadata, burst anchor, or none)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Any

from system.backend.media_time import camera_clock_time
from system.config import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS

_IMAGE_EXTENSIONS = frozenset(ext.lower() for ext in SUPPORTED_IMAGE_EXTENSIONS)
_VIDEO_EXTENSIONS = frozenset(ext.lower() for ext in SUPPORTED_VIDEO_EXTENSIONS)
_COUNTER_STEM_RE = re.compile(r"^(.*?)([0-9]+)$")


@dataclass(frozen=True)
class _ParsedItem:
    kind: str
    path_str: str
    path_key: str
    parent_key: str
    prefix: str | None
    counter: int | None
    counter_width: int | None
    digits: str | None
    photo_time: datetime | None
    own_video_time: datetime | None
    explicit_group_id: str | None


@dataclass
class _PhotoRun:
    photos: list[_ParsedItem]
    tainted: bool = False


def _none_result() -> dict[str, Any]:
    return {
        "source": "none",
        "reference_time": None,
        "precision": "unknown",
        "burst_group_id": None,
        "group_members": [],
        "group_end_time": None,
    }


def _video_result(dt_utc: datetime) -> dict[str, Any]:
    return {
        "source": "video",
        "reference_time": dt_utc.isoformat(),
        "precision": "exact",
        "burst_group_id": None,
        "group_members": [],
        "group_end_time": None,
    }


def _burst_result(
    burst_group_id: str,
    members: list[str],
    start_utc: datetime,
    end_utc: datetime,
) -> dict[str, Any]:
    return {
        "source": "burst_group",
        "reference_time": start_utc.isoformat(),
        "precision": "group_anchor",
        "burst_group_id": burst_group_id,
        "group_members": members,
        "group_end_time": end_utc.isoformat(),
    }


def _get_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _normalize_camera_clock(value: Any) -> datetime | None:
    if isinstance(value, bool) or not isinstance(value, (str, datetime)):
        return None
    return camera_clock_time(value)


def _extract_own_video_time(detection_data: Mapping[str, Any]) -> datetime | None:
    capture_time = detection_data.get("capture_time")
    if not isinstance(capture_time, Mapping):
        return None
    if capture_time.get("timestamp_reliable") is not True:
        return None
    return _normalize_camera_clock(capture_time.get("date_taken"))


def _extract_photo_time(raw_item: Any, detection_data: Mapping[str, Any]) -> datetime | None:
    capture_time = detection_data.get("capture_time")
    if isinstance(capture_time, Mapping):
        reliable = capture_time.get("timestamp_reliable")
        if reliable is False:
            return None
        if reliable is True and capture_time.get("date_taken") is not None:
            return _normalize_camera_clock(capture_time.get("date_taken"))
    return _normalize_camera_clock(_get_field(raw_item, "date_taken"))


def _normalize_parent_and_path(path_obj: Path) -> tuple[str, str]:
    try:
        resolved_parent = path_obj.expanduser().parent.resolve()
    except (OSError, RuntimeError, ValueError):
        resolved_parent = Path(os.path.abspath(str(path_obj.parent)))
    parent_key = os.path.normcase(str(resolved_parent)).replace("\\", "/")
    path_key = os.path.normcase(str(resolved_parent / path_obj.name)).replace("\\", "/")
    return parent_key, path_key


def _parse_item(raw_item: Any) -> _ParsedItem | None:
    raw_path = _get_field(raw_item, "path")
    if isinstance(raw_path, bool) or not isinstance(raw_path, (str, Path)):
        return None
    path_str = str(raw_path)
    if not path_str.strip():
        return None
    path_obj = Path(path_str)
    suffix = path_obj.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        kind = "photo"
    elif suffix in _VIDEO_EXTENSIONS:
        kind = "video"
    else:
        return None

    raw_data = _get_field(raw_item, "detection_data")
    data = raw_data if isinstance(raw_data, Mapping) else {}
    raw_gid = data.get("burst_group_id")
    explicit_group_id = raw_gid if isinstance(raw_gid, str) and raw_gid.strip() else None

    parent_key, path_key = _normalize_parent_and_path(path_obj)
    match = _COUNTER_STEM_RE.match(path_obj.stem)
    if match is not None:
        prefix, digits = match.groups()
        counter: int | None = int(digits)
        counter_width: int | None = len(digits)
    else:
        prefix = None
        digits = None
        counter = None
        counter_width = None

    return _ParsedItem(
        kind=kind,
        path_str=path_str,
        path_key=path_key,
        parent_key=parent_key,
        prefix=prefix,
        counter=counter,
        counter_width=counter_width,
        digits=digits,
        photo_time=_extract_photo_time(raw_item, data) if kind == "photo" else None,
        own_video_time=_extract_own_video_time(data) if kind == "video" else None,
        explicit_group_id=explicit_group_id,
    )


def _is_valid_gap_seconds(gap_seconds: Any) -> bool:
    if isinstance(gap_seconds, bool) or not isinstance(gap_seconds, (int, float)):
        return False
    return math.isfinite(gap_seconds) and gap_seconds >= 0


def _is_valid_burst_size(burst_size: Any) -> bool:
    return not isinstance(burst_size, bool) and isinstance(burst_size, int) and burst_size >= 1


def _resolve_explicit_groups(
    parsed_items: list[_ParsedItem],
    results: dict[str, dict[str, Any]],
    *,
    gap_seconds: float,
) -> None:
    """Resolve fallback time for validated explicit persisted burst_group_id groups."""
    groups: dict[str, list[_ParsedItem]] = {}
    for item in parsed_items:
        if item.explicit_group_id is not None:
            groups.setdefault(item.explicit_group_id, []).append(item)

    for gid, members in groups.items():
        if len({member.parent_key for member in members}) != 1:
            continue
        videos = [member for member in members if member.kind == "video"]
        photos = [member for member in members if member.kind == "photo"]
        if len(videos) != 1 or not photos:
            continue
        video = videos[0]
        if results[video.path_str]["source"] == "video":
            continue
        if any(photo.photo_time is None for photo in photos):
            continue
        ordered_photos = sorted(
            photos,
            key=lambda photo: (photo.photo_time, photo.path_key, photo.path_str),  # type: ignore[arg-type]
        )
        consistent = True
        for prev_photo, curr_photo in zip(ordered_photos, ordered_photos[1:]):
            assert prev_photo.photo_time is not None and curr_photo.photo_time is not None
            if (curr_photo.photo_time - prev_photo.photo_time).total_seconds() > gap_seconds:
                consistent = False
                break
        if not consistent:
            continue
        assert ordered_photos[0].photo_time is not None and ordered_photos[-1].photo_time is not None
        group_members = [photo.path_str for photo in ordered_photos] + [video.path_str]
        results[video.path_str] = _burst_result(
            gid,
            group_members,
            ordered_photos[0].photo_time,
            ordered_photos[-1].photo_time,
        )


def _auto_burst_id(
    parent_key: str,
    prefix: str,
    counter_width: int,
    start_digits: str,
    end_digits: str,
    video_digits: str,
) -> str:
    payload = f"{parent_key}\0{prefix}\0{counter_width}\0{start_digits}\0{end_digits}\0{video_digits}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"auto:{prefix}{start_digits}-{end_digits}:{digest}"


def _detect_completed_burst_size(
    completed_runs: list[_PhotoRun],
    *,
    default_burst_size: int,
    detect_burst: bool,
) -> int:
    if not detect_burst:
        return default_burst_size
    counts: dict[int, int] = {}
    for run in completed_runs:
        if not run.tainted and run.photos:
            length = len(run.photos)
            counts[length] = counts.get(length, 0) + 1
    if not counts:
        return default_burst_size
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], -pair[0]))
    best_len, best_count = ranked[0]
    if best_count < 2:
        return default_burst_size
    if len(ranked) > 1 and ranked[1][1] == best_count:
        return default_burst_size
    return best_len


def _resolve_stream_counters(
    stream_items: list[_ParsedItem],
    results: dict[str, dict[str, Any]],
    *,
    burst_size: int,
    gap_seconds: float,
    detect_burst: bool,
) -> None:
    """Resolve counter-adjacent bursts within a single (parent, prefix, width) stream."""
    by_counter: dict[int, list[_ParsedItem]] = {}
    for item in stream_items:
        assert item.counter is not None
        by_counter.setdefault(item.counter, []).append(item)

    sorted_counters = sorted(by_counter.keys())
    slot_kinds: dict[int, str] = {}
    for counter in sorted_counters:
        bucket = by_counter[counter]
        if len(bucket) > 1:
            slot_kinds[counter] = "ambiguous"
        else:
            single = bucket[0]
            if single.kind == "video":
                slot_kinds[counter] = "video"
            elif single.photo_time is None:
                slot_kinds[counter] = "no_time"
            else:
                slot_kinds[counter] = "photo"

    completed_runs: list[_PhotoRun] = []
    preceding_run_by_video_counter: dict[int, _PhotoRun] = {}
    current_run: _PhotoRun | None = None
    prev_counter: int | None = None
    prev_kind: str | None = None

    def close_current_run(*, right_tainted: bool = False) -> None:
        nonlocal current_run
        if current_run is not None:
            if right_tainted:
                current_run.tainted = True
            completed_runs.append(current_run)
            current_run = None

    for counter in sorted_counters:
        kind = slot_kinds[counter]
        is_contiguous = prev_counter is not None and counter == prev_counter + 1
        if kind == "photo":
            photo = by_counter[counter][0]
            if current_run is not None and is_contiguous and prev_kind == "photo":
                prev_photo = current_run.photos[-1]
                assert prev_photo.photo_time is not None and photo.photo_time is not None
                delta = (photo.photo_time - prev_photo.photo_time).total_seconds()
                if delta < 0:
                    current_run.tainted = True
                    current_run.photos.append(photo)
                elif delta > gap_seconds:
                    close_current_run()
                    current_run = _PhotoRun([photo], tainted=False)
                else:
                    current_run.photos.append(photo)
            else:
                close_current_run()
                left_tainted = is_contiguous and prev_kind in ("ambiguous", "no_time")
                current_run = _PhotoRun([photo], tainted=left_tainted)
        elif kind in ("ambiguous", "no_time"):
            close_current_run(right_tainted=is_contiguous)
        elif kind == "video":
            if current_run is not None and is_contiguous:
                preceding_run_by_video_counter[counter] = current_run
            close_current_run()

        prev_counter = counter
        prev_kind = kind

    close_current_run()

    effective_burst_size = _detect_completed_burst_size(
        completed_runs,
        default_burst_size=burst_size,
        detect_burst=detect_burst,
    )

    for counter in sorted_counters:
        if slot_kinds[counter] != "video":
            continue
        video = by_counter[counter][0]
        if results[video.path_str]["source"] != "none":
            continue
        if slot_kinds.get(counter + 1) in ("video", "ambiguous"):
            continue
        run = preceding_run_by_video_counter.get(counter)
        if run is None or run.tainted or len(run.photos) != effective_burst_size:
            continue
        first_photo = run.photos[0]
        last_photo = run.photos[-1]
        assert (
            first_photo.photo_time is not None
            and last_photo.photo_time is not None
            and video.prefix is not None
            and video.counter_width is not None
            and first_photo.digits is not None
            and last_photo.digits is not None
            and video.digits is not None
        )
        gid = _auto_burst_id(
            video.parent_key,
            video.prefix,
            video.counter_width,
            first_photo.digits,
            last_photo.digits,
            video.digits,
        )
        members = [photo.path_str for photo in run.photos] + [video.path_str]
        results[video.path_str] = _burst_result(
            gid,
            members,
            first_photo.photo_time,
            last_photo.photo_time,
        )


def resolve_video_references(
    items: Iterable[Any],
    *,
    burst_size: int = 3,
    gap_seconds: float = 30,
    detect_burst: bool = True,
) -> dict[str, dict[str, Any]]:
    """Resolve each video item's time reference: own metadata > burst group > none."""
    parsed_items: list[_ParsedItem] = []
    for raw_item in items:
        parsed = _parse_item(raw_item)
        if parsed is not None:
            parsed_items.append(parsed)

    parsed_items.sort(key=lambda item: (item.path_key, item.path_str))

    videos_by_path: dict[str, list[_ParsedItem]] = {}
    for item in parsed_items:
        if item.kind == "video":
            videos_by_path.setdefault(item.path_str, []).append(item)

    results: dict[str, dict[str, Any]] = {}
    for path_str, video_entries in videos_by_path.items():
        own_times = {entry.own_video_time for entry in video_entries}
        if len(own_times) == 1:
            own_dt = next(iter(own_times))
            results[path_str] = _video_result(own_dt) if own_dt is not None else _none_result()
        else:
            results[path_str] = _none_result()

    if not _is_valid_gap_seconds(gap_seconds):
        return results

    _resolve_explicit_groups(parsed_items, results, gap_seconds=float(gap_seconds))

    if not _is_valid_burst_size(burst_size):
        return results

    streams: dict[tuple[str, str, int], list[_ParsedItem]] = {}
    for item in parsed_items:
        if (
            item.explicit_group_id is None
            and item.prefix is not None
            and item.counter_width is not None
            and item.counter is not None
        ):
            streams.setdefault((item.parent_key, item.prefix, item.counter_width), []).append(item)

    for stream_items in streams.values():
        _resolve_stream_counters(
            stream_items,
            results,
            burst_size=burst_size,
            gap_seconds=float(gap_seconds),
            detect_burst=bool(detect_burst),
        )

    return results
