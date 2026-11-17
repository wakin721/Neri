"""Tests for bounded sidecar video capture timestamp extraction."""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import struct
import subprocess

import pytest


def _mp4_box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), box_type) + payload


def _mp4_mvhd_v0(dt_utc: datetime) -> bytes:
    epoch_1904 = datetime(1904, 1, 1, tzinfo=timezone.utc)
    seconds = int((dt_utc - epoch_1904).total_seconds())
    payload = struct.pack(">4BIIII", 0, 0, 0, 0, seconds, seconds, 1000, 5000) + (b"\x00" * 76)
    return _mp4_box(b"mvhd", payload)


def _mp4_mvhd_v1(dt_utc: datetime) -> bytes:
    epoch_1904 = datetime(1904, 1, 1, tzinfo=timezone.utc)
    seconds = int((dt_utc - epoch_1904).total_seconds())
    payload = struct.pack(">4BQQIQ", 1, 0, 0, 0, seconds, seconds, 1000, 5000) + (b"\x00" * 72)
    return _mp4_box(b"mvhd", payload)


def _mp4_udta_day_direct(text: str) -> bytes:
    encoded = text.encode("utf-8")
    day_payload = struct.pack(">HH", len(encoded), 0x15C7) + encoded
    return _mp4_box(b"udta", _mp4_box(b"\xa9day", day_payload))


def _mp4_udta_meta_ilst_day(text: str) -> bytes:
    encoded = text.encode("utf-8")
    data_box = _mp4_box(b"data", struct.pack(">II", 1, 0) + encoded)
    day_box = _mp4_box(b"\xa9day", data_box)
    ilst_box = _mp4_box(b"ilst", day_box)
    hdlr_box = _mp4_box(b"hdlr", b"\x00" * 4 + b"\x00" * 4 + b"mdir" + b"\x00" * 12)
    meta_box = _mp4_box(b"meta", b"\x00" * 4 + hdlr_box + ilst_box)
    return _mp4_box(b"udta", meta_box)


def _riff_chunk(fourcc: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if (len(payload) & 1) else b""
    return fourcc + struct.pack("<I", len(payload)) + payload + pad


def _riff_list(list_type: bytes, children: bytes) -> bytes:
    return _riff_chunk(b"LIST", list_type + children)


def test_missing_and_non_file_paths_return_unknown(tmp_path: Path):
    from system.backend.media_time import video_capture_time

    missing = video_capture_time(tmp_path / "does_not_exist.mp4")
    assert missing["date_taken"] is None
    assert missing["timestamp_source"] == "unknown"
    assert missing["timestamp_reliable"] is False
    assert missing["timezone_aware"] is False
    assert missing["timezone_offset"] is None
    assert missing["timezone_assumed"] is False
    assert missing["timezone_assumption"] == "unknown"
    assert missing["assumed_utc_offset"] is None

    directory = video_capture_time(tmp_path)
    assert directory["date_taken"] is None
    assert directory["timestamp_reliable"] is False
    assert directory["timestamp_source"] == "unknown"


def test_never_uses_filesystem_mtime_or_filenames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from system.backend import media_time

    monkeypatch.setattr(media_time.shutil, "which", lambda _name: None)

    for name in ("123456.mp4", "VID_20261116_102030.mp4", "2026-11-16_10-20-30.avi"):
        path = tmp_path / name
        path.write_bytes(b"\x00" * 64)
        result = media_time.video_capture_time(path)
        assert result["date_taken"] is None
        assert result["timestamp_reliable"] is False
        assert result["timestamp_source"] == "unknown"


@pytest.mark.parametrize(
    "bad_value",
    [
        "",
        "   ",
        "null",
        "None",
        "nan",
        "0000-00-00 00:00:00",
        "0000-00-00T00:00:00Z",
        "0000:00:00 00:00:00",
        "1904-01-01T00:00:00.000000Z",
        "1904-01-01T08:00:00+08:00",
        "1970-01-01T00:00:00Z",
        "1970-01-01T08:00:00+08:00",
        "1969-12-31T23:59:59Z",
        "2026",
        "2026-11-16",
        "2026-02-30T10:00:00Z",
        "not-a-timestamp",
    ],
)
def test_ffprobe_sentinel_and_invalid_dates_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: str
):
    from system.backend import media_time

    video = tmp_path / "clip.mkv"
    video.write_bytes(b"fake-mkv-header")

    monkeypatch.setattr(media_time.shutil, "which", lambda _name: "ffprobe")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"format": {"tags": {"creation_time": bad_value}}, "streams": []}),
            stderr="",
        )

    monkeypatch.setattr(media_time.subprocess, "run", fake_run)

    result = media_time.video_capture_time(video)
    assert result["date_taken"] is None
    assert result["timestamp_reliable"] is False
    assert result["timestamp_source"] == "unknown"


def test_ffprobe_invocation_flags_timeout_and_preserves_timezones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from system.backend import media_time

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-a-real-mp4")
    monkeypatch.setattr(media_time.shutil, "which", lambda _name: "C:/ffmpeg/bin/ffprobe.exe")

    captured_calls = []

    def make_runner(payload):
        def fake_run(cmd, **kwargs):
            captured_calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

        return fake_run

    monkeypatch.setattr(
        media_time.subprocess,
        "run",
        make_runner({"format": {"tags": {"creation_time": "2026-11-16T04:30:00.000000Z"}}, "streams": []}),
    )
    utc_res = media_time.video_capture_time(video)
    assert utc_res["date_taken"] == "2026-11-16T04:30:00+00:00"
    assert utc_res["timestamp_reliable"] is True
    assert utc_res["timestamp_source"] == "ffprobe:format.tags.creation_time"
    assert utc_res["timezone_aware"] is True
    assert utc_res["timezone_offset"] == "+00:00"
    assert utc_res["timezone_assumed"] is False
    assert utc_res["timezone_assumption"] == "explicit_utc"
    assert utc_res["assumed_utc_offset"] == "+00:00"

    cmd, kwargs = captured_calls[-1]
    assert "-print_format" in cmd or "-of" in cmd
    assert "json" in cmd
    assert kwargs["timeout"] <= 5
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)

    monkeypatch.setattr(
        media_time.subprocess,
        "run",
        make_runner(
            {
                "format": {
                    "tags": {
                        "creation_time": "2026-11-16T04:30:00Z",
                        "com.apple.quicktime.creationdate": "2026-11-16T12:30:00+0800",
                    }
                },
                "streams": [],
            }
        ),
    )
    local_res = media_time.video_capture_time(video)
    assert local_res["date_taken"] == "2026-11-16T12:30:00+08:00"
    assert local_res["timestamp_reliable"] is True
    assert local_res["timestamp_source"] == "ffprobe:format.tags.com.apple.quicktime.creationdate"
    assert local_res["timezone_aware"] is True
    assert local_res["timezone_offset"] == "+08:00"
    assert local_res["timezone_assumed"] is False
    assert local_res["timezone_assumption"] == "explicit"

    monkeypatch.setattr(
        media_time.subprocess,
        "run",
        make_runner({"format": {"tags": {}}, "streams": [{"tags": {"creation_time": "2026-11-16 12:30:00"}}]}),
    )
    naive_res = media_time.video_capture_time(video)
    assert naive_res["date_taken"] == "2026-11-16T12:30:00"
    assert naive_res["timestamp_reliable"] is True
    assert naive_res["timestamp_source"] == "ffprobe:stream.tags.creation_time"
    assert naive_res["timezone_aware"] is False
    assert naive_res["timezone_offset"] is None
    assert naive_res["timezone_assumed"] is False
    assert naive_res["timezone_assumption"] == "camera_clock"
    assert naive_res["assumed_utc_offset"] is None


def test_ffprobe_timeout_and_bad_json_fallback_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from system.backend import media_time

    monkeypatch.setattr(media_time.shutil, "which", lambda _name: "ffprobe")

    dummy = tmp_path / "broken.mp4"
    dummy.write_bytes(b"corrupt-bytes")

    def raise_timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 3))

    monkeypatch.setattr(media_time.subprocess, "run", raise_timeout)
    res_timeout = media_time.video_capture_time(dummy)
    assert res_timeout["date_taken"] is None
    assert res_timeout["timestamp_reliable"] is False
    assert res_timeout["timestamp_source"] == "unknown"

    def bad_json(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="{not-valid-json", stderr="")

    monkeypatch.setattr(media_time.subprocess, "run", bad_json)
    res_bad_json = media_time.video_capture_time(dummy)
    assert res_bad_json["date_taken"] is None
    assert res_bad_json["timestamp_reliable"] is False
    assert res_bad_json["timestamp_source"] == "unknown"

    mp4 = tmp_path / "valid.mp4"
    dt = datetime(2026, 11, 16, 6, 15, 0, tzinfo=timezone.utc)
    mp4.write_bytes(_mp4_box(b"ftyp", b"isom\x00\x00\x02\x00isom") + _mp4_box(b"moov", _mp4_mvhd_v0(dt)))
    monkeypatch.setattr(media_time.subprocess, "run", raise_timeout)
    res_fallback = media_time.video_capture_time(mp4)
    assert res_fallback["date_taken"] == "2026-11-16T06:15:00+00:00"
    assert res_fallback["timestamp_reliable"] is True
    assert res_fallback["timestamp_source"] == "mp4:mvhd"


def test_builtin_mp4_mvhd_and_udta_without_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from system.backend import media_time

    monkeypatch.setattr(media_time.shutil, "which", lambda _name: None)

    dt = datetime(2026, 11, 16, 9, 45, 12, tzinfo=timezone.utc)
    mp4_v0 = tmp_path / "v0.mp4"
    mp4_v0.write_bytes(_mp4_box(b"ftyp", b"isom\x00\x00\x02\x00") + _mp4_box(b"moov", _mp4_mvhd_v0(dt)))
    res_v0 = media_time.video_capture_time(mp4_v0)
    assert res_v0["date_taken"] == "2026-11-16T09:45:12+00:00"
    assert res_v0["timestamp_source"] == "mp4:mvhd"
    assert res_v0["timestamp_reliable"] is True
    assert res_v0["timezone_aware"] is True
    assert res_v0["timezone_offset"] == "+00:00"

    mp4_v1 = tmp_path / "v1.mp4"
    mp4_v1.write_bytes(_mp4_box(b"ftyp", b"isom\x00\x00\x02\x00") + _mp4_box(b"moov", _mp4_mvhd_v1(dt)))
    res_v1 = media_time.video_capture_time(mp4_v1)
    assert res_v1["date_taken"] == "2026-11-16T09:45:12+00:00"
    assert res_v1["timestamp_source"] == "mp4:mvhd"

    mov_udta = tmp_path / "udta.mov"
    mov_udta.write_bytes(
        _mp4_box(b"ftyp", b"qt  \x00\x00\x00\x00")
        + _mp4_box(b"moov", _mp4_mvhd_v0(dt) + _mp4_udta_day_direct("2026-11-16T17:45:12+08:00"))
    )
    res_udta = media_time.video_capture_time(mov_udta)
    assert res_udta["date_taken"] == "2026-11-16T17:45:12+08:00"
    assert res_udta["timestamp_source"] == "mp4:udta.day"
    assert res_udta["timezone_aware"] is True
    assert res_udta["timezone_offset"] == "+08:00"
    assert res_udta["timezone_assumed"] is False

    mp4_ilst = tmp_path / "ilst.mp4"
    mp4_ilst.write_bytes(
        _mp4_box(b"ftyp", b"mp42\x00\x00\x00\x00")
        + _mp4_box(b"moov", _mp4_udta_meta_ilst_day("2026-11-16T17:45:12+08:00"))
    )
    res_ilst = media_time.video_capture_time(mp4_ilst)
    assert res_ilst["date_taken"] == "2026-11-16T17:45:12+08:00"
    assert res_ilst["timestamp_source"] == "mp4:udta.day"

    mp4_zero = tmp_path / "zero.mp4"
    epoch_1904 = datetime(1904, 1, 1, tzinfo=timezone.utc)
    mp4_zero.write_bytes(_mp4_box(b"ftyp", b"isom\x00\x00\x02\x00") + _mp4_box(b"moov", _mp4_mvhd_v0(epoch_1904)))
    res_zero = media_time.video_capture_time(mp4_zero)
    assert res_zero["date_taken"] is None
    assert res_zero["timestamp_reliable"] is False
    assert res_zero["timestamp_source"] == "unknown"


def test_builtin_avi_idit_without_ffprobe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from system.backend import media_time

    monkeypatch.setattr(media_time.shutil, "which", lambda _name: None)

    avi = tmp_path / "trailcam.avi"
    hdrl = _riff_list(b"hdrl", _riff_chunk(b"avih", b"\x00" * 56))
    idit = _riff_chunk(b"IDIT", b"Mon Nov 16 20:30:05 2026\n\x00")
    movi = _riff_list(b"movi", _riff_chunk(b"00dc", b"\xff" * 64))
    avi_body = b"AVI " + hdrl + idit + movi
    avi.write_bytes(b"RIFF" + struct.pack("<I", len(avi_body)) + avi_body)

    res = media_time.video_capture_time(avi)
    assert res["date_taken"] == "2026-11-16T20:30:05"
    assert res["timestamp_source"] == "avi:idit"
    assert res["timestamp_reliable"] is True
    assert res["timezone_aware"] is False
    assert res["timezone_offset"] is None
    assert res["timezone_assumed"] is False
    assert res["timezone_assumption"] == "camera_clock"
    assert res["assumed_utc_offset"] is None


def test_bounded_reads_never_scan_entire_large_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from system.backend import media_time

    monkeypatch.setattr(media_time.shutil, "which", lambda _name: None)

    large_mp4 = tmp_path / "huge.mp4"
    ftyp = _mp4_box(b"ftyp", b"isom\x00\x00\x02\x00")
    mdat_payload_size = 512 * 1024 * 1024
    mdat_header = struct.pack(">I4s", 8 + mdat_payload_size, b"mdat")
    dt = datetime(2026, 11, 16, 11, 22, 33, tzinfo=timezone.utc)
    moov = _mp4_box(b"moov", _mp4_mvhd_v0(dt))

    with large_mp4.open("wb") as f:
        f.write(ftyp)
        f.write(mdat_header)
        f.seek(mdat_payload_size, io.SEEK_CUR)
        f.write(moov)

    bytes_read = 0
    real_open = Path.open

    class CountingFile:
        def __init__(self, inner):
            self._inner = inner

        def read(self, n=-1):
            nonlocal bytes_read
            chunk = self._inner.read(n)
            bytes_read += len(chunk)
            return chunk

        def seek(self, *args, **kwargs):
            return self._inner.seek(*args, **kwargs)

        def tell(self):
            return self._inner.tell()

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

    def tracked_open(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        if "b" in mode and self == large_mp4:
            return CountingFile(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracked_open)

    res = media_time.video_capture_time(large_mp4)
    assert res["date_taken"] == "2026-11-16T11:22:33+00:00"
    assert res["timestamp_reliable"] is True
    assert bytes_read < 4096

    large_junk = tmp_path / "garbage.mp4"
    with large_junk.open("wb") as f:
        f.write(b"NOT_A_MEDIA_CONTAINER")
        f.seek(64 * 1024 * 1024)
        f.write(b"\x00")

    bytes_read = 0
    monkeypatch.setattr(
        Path,
        "open",
        lambda self, mode="r", *a, **kw: CountingFile(real_open(self, mode, *a, **kw))
        if ("b" in mode and self == large_junk)
        else real_open(self, mode, *a, **kw),
    )
    junk_res = media_time.video_capture_time(large_junk)
    assert junk_res["date_taken"] is None
    assert junk_res["timestamp_reliable"] is False
    assert bytes_read <= 64


@pytest.mark.parametrize('suffix', ['', 'Z', '+00:00', '+08:00', '-05:00'])
def test_camera_clock_time_does_not_convert_recorded_fields(suffix):
    from system.backend.media_time import camera_clock_time

    assert camera_clock_time(f'2025-11-22T16:32:22{suffix}') == datetime(
        2025, 11, 22, 16, 32, 22, tzinfo=timezone.utc)
    assert camera_clock_time(None) is None
    assert camera_clock_time('1970-01-01T00:00:00Z') is None


def test_normalize_capture_time_utc_helper():
    from system.backend.media_time import normalize_capture_time_utc

    assert normalize_capture_time_utc("2026-11-16T20:30:00") == datetime(
        2026, 11, 16, 12, 30, 0, tzinfo=timezone.utc
    )
    assert normalize_capture_time_utc("2026-11-16T07:30:00-05:00") == datetime(
        2026, 11, 16, 12, 30, 0, tzinfo=timezone.utc
    )
    assert normalize_capture_time_utc("2026-11-16T12:30:00+00:00") == datetime(
        2026, 11, 16, 12, 30, 0, tzinfo=timezone.utc
    )
    assert normalize_capture_time_utc(None) is None
    assert normalize_capture_time_utc("1970-01-01T00:00:00Z") is None


def test_builtin_mp4_meta_keys_ilst_creationdate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from system.backend import media_time

    monkeypatch.setattr(media_time.shutil, "which", lambda _name: None)

    key_name = b"com.apple.quicktime.creationdate"
    key_entry = struct.pack(">I4s", 8 + len(key_name), b"mdta") + key_name
    keys_box = _mp4_box(b"keys", struct.pack(">II", 0, 1) + key_entry)
    encoded_val = b"2026-11-16T20:45:00+0800"
    data_box = _mp4_box(b"data", struct.pack(">II", 1, 0) + encoded_val)
    ilst_item = _mp4_box(struct.pack(">I", 1), data_box)
    ilst_box = _mp4_box(b"ilst", ilst_item)
    meta_box = _mp4_box(b"meta", b"\x00" * 4 + keys_box + ilst_box)
    udta_box = _mp4_box(b"udta", meta_box)

    mov = tmp_path / "iphone.mov"
    mov.write_bytes(_mp4_box(b"ftyp", b"qt  \x00\x00\x00\x00") + _mp4_box(b"moov", udta_box))

    res = media_time.video_capture_time(mov)
    assert res["date_taken"] == "2026-11-16T20:45:00+08:00"
    assert res["timestamp_source"] == "mp4:udta.creationdate"
    assert res["timestamp_reliable"] is True
    assert res["timezone_aware"] is True
    assert res["timezone_offset"] == "+08:00"


def test_bounded_avi_skips_huge_movi_and_finds_trailing_idit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from system.backend import media_time

    monkeypatch.setattr(media_time.shutil, "which", lambda _name: None)

    avi = tmp_path / "huge.avi"
    hdrl = _riff_list(b"hdrl", _riff_chunk(b"avih", b"\x00" * 56))
    movi_payload_size = 256 * 1024 * 1024
    movi_header = b"LIST" + struct.pack("<I4s", 4 + movi_payload_size, b"movi")
    info = _riff_list(b"INFO", _riff_chunk(b"IDIT", b"2026/11/16 20:35:00\x00"))
    total_avi_payload = 4 + len(hdrl) + 12 + movi_payload_size + len(info)

    with avi.open("wb") as f:
        f.write(b"RIFF" + struct.pack("<I4s", total_avi_payload, b"AVI "))
        f.write(hdrl)
        f.write(movi_header)
        f.seek(movi_payload_size, io.SEEK_CUR)
        f.write(info)

    bytes_read = 0
    real_open = Path.open

    class CountingFile:
        def __init__(self, inner):
            self._inner = inner

        def read(self, n=-1):
            nonlocal bytes_read
            chunk = self._inner.read(n)
            bytes_read += len(chunk)
            return chunk

        def seek(self, *args, **kwargs):
            return self._inner.seek(*args, **kwargs)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

    monkeypatch.setattr(
        Path,
        "open",
        lambda self, mode="r", *a, **kw: CountingFile(real_open(self, mode, *a, **kw))
        if ("b" in mode and self == avi)
        else real_open(self, mode, *a, **kw),
    )

    res = media_time.video_capture_time(avi)
    assert res["date_taken"] == "2026-11-16T20:35:00"
    assert res["timestamp_source"] == "avi:idit"
    assert res["timestamp_reliable"] is True
    assert bytes_read < 4096


def test_corrupt_box_chain_terminates_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from system.backend import media_time

    monkeypatch.setattr(media_time.shutil, "which", lambda _name: None)
    corrupt = tmp_path / "loop.mp4"
    corrupt.write_bytes(_mp4_box(b"ftyp", b"isom") + (_mp4_box(b"free", b"") * 400))
    res = media_time.video_capture_time(corrupt)
    assert res["date_taken"] is None
    assert res["timestamp_source"] == "unknown"
    assert res["timestamp_reliable"] is False
