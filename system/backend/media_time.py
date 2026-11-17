"""Bounded sidecar extraction of trusted embedded video capture timestamps."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
from typing import Any


CAMERA_LOCAL_UTC_OFFSET = timezone(timedelta(hours=8))
CAMERA_LOCAL_OFFSET_STR = "+08:00"

FFPROBE_TIMEOUT_SECONDS = 3.0
MAX_FFPROBE_JSON_BYTES = 256 * 1024
MAX_CONTAINER_BYTES_READ = 512 * 1024
MAX_CONTAINER_STEPS = 256

_EPOCH_1904 = datetime(1904, 1, 1, tzinfo=timezone.utc)
_MIN_VALID_YEAR = 1980
_MAX_VALID_YEAR = 2100

_SENTINEL_STRINGS = frozenset({"", "none", "null", "nan", "n/a", "unknown", "-", "0"})

_ISO_DATETIME_RE = re.compile(
    r"^(\d{4})[-:/](\d{2})[-:/](\d{2})[T ](\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,6})\d*)?(?:\s*(Z|[+-]\d{2}:?\d{2}))?$",
    re.IGNORECASE,
)

_ASCTIME_RE = re.compile(
    r"^(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+)?"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(\d{1,2})\s+(\d{2}):(\d{2}):(\d{2})\s+(\d{4})$",
    re.IGNORECASE,
)

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_MP4_TOP_LEVEL_BOXES = frozenset(
    {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot", b"uuid", b"pdin", b"styp", b"jP  "}
)

_FFPROBE_TAG_PRIORITY = (
    "com.apple.quicktime.creationdate",
    "creation_time",
    "datetime_original",
    "date_recorded",
    "creation_date",
    "idit",
    "icrd",
    "date",
)


def _unknown_result() -> dict[str, Any]:
    return {
        "date_taken": None,
        "timestamp_source": "unknown",
        "timestamp_reliable": False,
        "timezone_aware": False,
        "timezone_offset": None,
        "timezone_assumed": False,
        "timezone_assumption": "unknown",
        "assumed_utc_offset": None,
    }


def _format_utc_offset(offset: timedelta) -> str:
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_minutes = abs(total_seconds) // 60
    hours, minutes = divmod(total_minutes, 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def _is_valid_capture_datetime(dt: datetime) -> bool:
    if dt.year < _MIN_VALID_YEAR or dt.year > _MAX_VALID_YEAR:
        return False
    if dt.tzinfo is not None:
        offset = dt.utcoffset()
        if offset is None or abs(offset.total_seconds()) > 14 * 3600:
            return False
        utc_dt = dt.astimezone(timezone.utc)
        if utc_dt.year < _MIN_VALID_YEAR or utc_dt.year > _MAX_VALID_YEAR:
            return False
    return True


def _parse_tz_suffix(tz_raw: str | None) -> timezone | None:
    if not tz_raw:
        return None
    text = tz_raw.strip().upper()
    if text == "Z":
        return timezone.utc
    sign = 1 if text[0] == "+" else -1
    digits = text[1:].replace(":", "")
    hours = int(digits[:2])
    minutes = int(digits[2:4])
    if hours > 14 or minutes >= 60 or (hours == 14 and minutes > 0):
        raise ValueError("UTC offset out of range")
    return timezone(timedelta(hours=sign * hours, minutes=sign * minutes))


def parse_embedded_capture_time(value: Any) -> datetime | None:
    """Parse a strict embedded capture timestamp, rejecting sentinels and bare dates."""
    if not isinstance(value, str):
        return None
    text = value.replace("\x00", "").strip()
    if text.lower() in _SENTINEL_STRINGS:
        return None

    match = _ISO_DATETIME_RE.match(text)
    if match is not None:
        year, month, day, hour, minute, second, frac, tz_raw = match.groups()
        microsecond = int(frac.ljust(6, "0")) if frac else 0
        try:
            tzinfo = _parse_tz_suffix(tz_raw)
            dt = datetime(
                int(year), int(month), int(day),
                int(hour), int(minute), int(second),
                microsecond, tzinfo=tzinfo,
            )
        except ValueError:
            return None
        return dt if _is_valid_capture_datetime(dt) else None

    asc = _ASCTIME_RE.match(text)
    if asc is not None:
        mon_str, day, hour, minute, second, year = asc.groups()
        month = _MONTHS.get(mon_str.lower())
        if month is None:
            return None
        try:
            dt = datetime(int(year), month, int(day), int(hour), int(minute), int(second))
        except ValueError:
            return None
        return dt if _is_valid_capture_datetime(dt) else None

    return None


def normalize_capture_time_utc(
    value: Any,
    *,
    naive_tz: timezone = CAMERA_LOCAL_UTC_OFFSET,
) -> datetime | None:
    """Normalize an embedded ISO/asctime capture timestamp to UTC for event grouping."""
    if isinstance(value, datetime):
        dt = value if _is_valid_capture_datetime(value) else None
    else:
        dt = parse_embedded_capture_time(value)
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=naive_tz).astimezone(timezone.utc)
    return dt.astimezone(timezone.utc)


def camera_clock_time(value: Any) -> datetime | None:
    """Compare photos/videos on their recorded camera clock, without shifting.

    Preserve original offset-bearing metadata separately. UTC tzinfo here is
    only a common arithmetic/serialization carrier for the literal clock fields,
    not a claim that the camera clock represents actual UTC.
    """
    dt = value if isinstance(value, datetime) else parse_embedded_capture_time(value)
    if dt is None or not _is_valid_capture_datetime(dt):
        return None
    return dt.replace(tzinfo=timezone.utc)


def _result_from_datetime(dt: datetime, source: str) -> dict[str, Any]:
    if not _is_valid_capture_datetime(dt):
        return _unknown_result()
    offset = dt.utcoffset() if dt.tzinfo is not None else None
    if offset is not None:
        offset_str = _format_utc_offset(offset)
        is_utc = offset == timedelta(0)
        return {
            "date_taken": dt.isoformat(),
            "timestamp_source": source,
            "timestamp_reliable": True,
            "timezone_aware": True,
            "timezone_offset": offset_str,
            "timezone_assumed": False,
            "timezone_assumption": "explicit_utc" if is_utc else "explicit",
            "assumed_utc_offset": offset_str,
        }
    return {
        "date_taken": dt.isoformat(),
        "timestamp_source": source,
        "timestamp_reliable": True,
        "timezone_aware": False,
        "timezone_offset": None,
        "timezone_assumed": False,
        "timezone_assumption": "camera_clock",
        "assumed_utc_offset": None,
    }


def _probe_with_ffprobe(file_path: Path) -> tuple[str, datetime] | None:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format_tags:stream_tags",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=FFPROBE_TIMEOUT_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if proc.returncode != 0 or not proc.stdout or len(proc.stdout) > MAX_FFPROBE_JSON_BYTES:
        return None
    try:
        payload = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    tag_sources: list[tuple[str, dict[str, Any]]] = []
    fmt = payload.get("format")
    if isinstance(fmt, dict) and isinstance(fmt.get("tags"), dict):
        tag_sources.append(("format.tags", {str(k).lower(): v for k, v in fmt["tags"].items()}))
    streams = payload.get("streams")
    if isinstance(streams, list):
        for stream in streams:
            if isinstance(stream, dict) and isinstance(stream.get("tags"), dict):
                tag_sources.append(("stream.tags", {str(k).lower(): v for k, v in stream["tags"].items()}))

    for key in _FFPROBE_TAG_PRIORITY:
        for scope, tags in tag_sources:
            if key in tags:
                dt = parse_embedded_capture_time(tags[key])
                if dt is not None:
                    return f"ffprobe:{scope}.{key}", dt
    return None


class _BoundedReader:
    """Read-budgeted, step-budgeted binary file reader that skips media payloads via seek."""

    def __init__(self, handle: Any, file_size: int) -> None:
        self._handle = handle
        self.file_size = file_size
        self.bytes_read = 0
        self.steps = 0

    def next_step(self) -> bool:
        self.steps += 1
        return self.steps <= MAX_CONTAINER_STEPS

    def read_at(self, offset: int, size: int) -> bytes:
        if offset < 0 or size <= 0 or offset >= self.file_size:
            return b""
        allowed = min(size, self.file_size - offset, MAX_CONTAINER_BYTES_READ - self.bytes_read)
        if allowed <= 0:
            return b""
        self._handle.seek(offset)
        data = self._handle.read(allowed)
        self.bytes_read += len(data)
        return data


def _read_mp4_box_header(reader: _BoundedReader, offset: int, limit: int) -> tuple[bytes, int, int] | None:
    if offset + 8 > limit:
        return None
    header = reader.read_at(offset, 8)
    if len(header) < 8:
        return None
    size32, box_type = struct.unpack(">I4s", header)
    if size32 == 1:
        if offset + 16 > limit:
            return None
        ext = reader.read_at(offset + 8, 8)
        if len(ext) < 8:
            return None
        box_size = struct.unpack(">Q", ext)[0]
        header_len = 16
    elif size32 == 0:
        box_size = limit - offset
        header_len = 8
    else:
        box_size = size32
        header_len = 8
    if box_size < header_len or offset + box_size > limit:
        return None
    return box_type, offset + header_len, offset + box_size


def _parse_mp4_mvhd(reader: _BoundedReader, start: int, end: int) -> datetime | None:
    data = reader.read_at(start, min(end - start, 128))
    if len(data) < 8:
        return None
    version = data[0]
    if version == 0:
        seconds = struct.unpack(">I", data[4:8])[0]
    elif version == 1 and len(data) >= 12:
        seconds = struct.unpack(">Q", data[4:12])[0]
    else:
        return None
    if seconds == 0:
        return None
    try:
        dt = _EPOCH_1904 + timedelta(seconds=seconds)
    except OverflowError:
        return None
    return dt if _is_valid_capture_datetime(dt) else None


def _decode_mp4_day_payload(payload: bytes) -> datetime | None:
    if len(payload) >= 4:
        str_len = struct.unpack(">H", payload[:2])[0]
        if 4 <= str_len <= len(payload) - 4:
            text = payload[4 : 4 + str_len].decode("utf-8", errors="ignore")
            dt = parse_embedded_capture_time(text)
            if dt is not None:
                return dt
    return parse_embedded_capture_time(payload.decode("utf-8", errors="ignore"))


def _parse_mp4_ilst_data(reader: _BoundedReader, start: int, end: int) -> datetime | None:
    cursor = start
    while cursor < end and reader.next_step():
        box = _read_mp4_box_header(reader, cursor, end)
        if box is None:
            break
        box_type, payload_start, box_end = box
        if box_type == b"data" and box_end - payload_start > 8:
            raw = reader.read_at(payload_start + 8, min(box_end - payload_start - 8, 256))
            dt = parse_embedded_capture_time(raw.decode("utf-8", errors="ignore"))
            if dt is not None:
                return dt
        cursor = box_end
    return None


def _parse_mp4_meta(reader: _BoundedReader, start: int, end: int) -> tuple[str, datetime] | None:
    if end - start < 8:
        return None
    probe = reader.read_at(start, 12)
    if len(probe) >= 8 and probe[4:8] in (b"hdlr", b"keys", b"ilst"):
        cursor = start
    else:
        cursor = start + 4

    creationdate_index: int | None = None
    while cursor < end and reader.next_step():
        box = _read_mp4_box_header(reader, cursor, end)
        if box is None:
            break
        box_type, payload_start, box_end = box
        if box_type == b"keys" and box_end - payload_start >= 8:
            keys_data = reader.read_at(payload_start, min(box_end - payload_start, 4096))
            if len(keys_data) >= 8:
                entry_count = struct.unpack(">I", keys_data[4:8])[0]
                pos = 8
                for idx in range(1, min(entry_count, 64) + 1):
                    if pos + 8 > len(keys_data):
                        break
                    k_size, _ns = struct.unpack(">I4s", keys_data[pos : pos + 8])
                    if k_size < 8 or pos + k_size > len(keys_data):
                        break
                    key_name = keys_data[pos + 8 : pos + k_size].decode("utf-8", errors="ignore")
                    if key_name == "com.apple.quicktime.creationdate":
                        creationdate_index = idx
                    pos += k_size
        elif box_type == b"ilst":
            ilst_cursor = payload_start
            while ilst_cursor < box_end and reader.next_step():
                item = _read_mp4_box_header(reader, ilst_cursor, box_end)
                if item is None:
                    break
                item_type, item_start, item_end = item
                if item_type == b"\xa9day":
                    dt = _parse_mp4_ilst_data(reader, item_start, item_end)
                    if dt is not None:
                        return "mp4:udta.day", dt
                elif creationdate_index is not None:
                    key_idx = struct.unpack(">I", item_type)[0]
                    if key_idx == creationdate_index:
                        dt = _parse_mp4_ilst_data(reader, item_start, item_end)
                        if dt is not None:
                            return "mp4:udta.creationdate", dt
                ilst_cursor = item_end
        cursor = box_end
    return None


def _parse_mp4_udta(reader: _BoundedReader, start: int, end: int) -> tuple[str, datetime] | None:
    cursor = start
    while cursor < end and reader.next_step():
        box = _read_mp4_box_header(reader, cursor, end)
        if box is None:
            break
        box_type, payload_start, box_end = box
        if box_type == b"\xa9day":
            payload = reader.read_at(payload_start, min(box_end - payload_start, 512))
            dt = _decode_mp4_day_payload(payload)
            if dt is not None:
                return "mp4:udta.day", dt
        elif box_type == b"meta":
            found = _parse_mp4_meta(reader, payload_start, box_end)
            if found is not None:
                return found
        cursor = box_end
    return None


def _parse_mp4_container(reader: _BoundedReader) -> tuple[str, datetime] | None:
    first = _read_mp4_box_header(reader, 0, reader.file_size)
    if first is None or first[0] not in _MP4_TOP_LEVEL_BOXES:
        return None

    mvhd_dt: datetime | None = None
    udta_result: tuple[str, datetime] | None = None

    cursor = 0
    while cursor < reader.file_size and reader.next_step():
        box = _read_mp4_box_header(reader, cursor, reader.file_size)
        if box is None:
            break
        box_type, payload_start, box_end = box
        if box_type == b"moov":
            moov_cursor = payload_start
            while moov_cursor < box_end and reader.next_step():
                child = _read_mp4_box_header(reader, moov_cursor, box_end)
                if child is None:
                    break
                c_type, c_start, c_end = child
                if c_type == b"mvhd" and mvhd_dt is None:
                    mvhd_dt = _parse_mp4_mvhd(reader, c_start, c_end)
                elif c_type == b"udta" and udta_result is None:
                    udta_result = _parse_mp4_udta(reader, c_start, c_end)
                moov_cursor = c_end
            break
        cursor = box_end

    if udta_result is not None and udta_result[1].tzinfo is not None:
        return udta_result
    if mvhd_dt is not None:
        return "mp4:mvhd", mvhd_dt
    return udta_result


def _walk_riff_chunks(reader: _BoundedReader, start: int, end: int) -> tuple[str, datetime] | None:
    cursor = start
    icrd_candidate: tuple[str, datetime] | None = None
    while cursor + 8 <= end and reader.next_step():
        header = reader.read_at(cursor, 8)
        if len(header) < 8:
            break
        fourcc = header[:4]
        chunk_size = struct.unpack("<I", header[4:8])[0]
        payload_start = cursor + 8
        payload_end = payload_start + chunk_size
        if payload_end > end:
            break

        if fourcc == b"IDIT":
            raw = reader.read_at(payload_start, min(chunk_size, 256))
            dt = parse_embedded_capture_time(raw.decode("utf-8", errors="ignore"))
            if dt is not None:
                return "avi:idit", dt
        elif fourcc == b"ICRD" and icrd_candidate is None:
            raw = reader.read_at(payload_start, min(chunk_size, 256))
            dt = parse_embedded_capture_time(raw.decode("utf-8", errors="ignore"))
            if dt is not None:
                icrd_candidate = ("avi:icrd", dt)
        elif fourcc == b"LIST" and chunk_size >= 4:
            list_type = reader.read_at(payload_start, 4)
            if list_type in (b"hdrl", b"INFO"):
                found = _walk_riff_chunks(reader, payload_start + 4, payload_end)
                if found is not None:
                    if found[0] == "avi:idit":
                        return found
                    if icrd_candidate is None:
                        icrd_candidate = found

        padded = chunk_size + (chunk_size & 1)
        next_cursor = payload_start + padded
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return icrd_candidate


def _parse_avi_container(reader: _BoundedReader) -> tuple[str, datetime] | None:
    header = reader.read_at(0, 12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] not in (b"AVI ", b"AVIX"):
        return None
    riff_size = struct.unpack("<I", header[4:8])[0]
    riff_end = min(reader.file_size, 8 + riff_size)
    return _walk_riff_chunks(reader, 12, riff_end)


def _probe_builtin_container(file_path: Path) -> tuple[str, datetime] | None:
    try:
        file_size = file_path.stat().st_size
        if file_size < 12:
            return None
        with file_path.open("rb") as handle:
            reader = _BoundedReader(handle, file_size)
            head = reader.read_at(0, 12)
            if len(head) < 12:
                return None
            if head[:4] == b"RIFF" and head[8:12] in (b"AVI ", b"AVIX"):
                return _parse_avi_container(reader)
            if head[4:8] in _MP4_TOP_LEVEL_BOXES:
                return _parse_mp4_container(reader)
            return None
    except OSError:
        return None


def video_capture_time(path: str | Path) -> dict[str, Any]:
    """Extract trusted embedded capture time and timezone metadata for a video file."""
    file_path = Path(path)
    try:
        if not file_path.is_file():
            return _unknown_result()
    except OSError:
        return _unknown_result()

    ffprobe_hit = _probe_with_ffprobe(file_path)
    if ffprobe_hit is not None:
        source, dt = ffprobe_hit
        return _result_from_datetime(dt, source)

    builtin_hit = _probe_builtin_container(file_path)
    if builtin_hit is not None:
        source, dt = builtin_hit
        return _result_from_datetime(dt, source)

    return _unknown_result()
