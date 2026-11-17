from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from system.backend.models import DetectionItem
from system.dinov3.video_reference import resolve_video_references


def _photo(
    path: Path | str,
    date_taken: str | None = None,
    *,
    modified_at: str | None = None,
    detection_data: dict | None = None,
) -> DetectionItem:
    p = Path(path)
    return DetectionItem(
        filename=p.name,
        path=str(path),
        file_type=p.suffix.lower().lstrip("."),
        date_taken=date_taken,
        modified_at=modified_at,
        detection_data=dict(detection_data or {}),
    )


def _video(
    path: Path | str,
    *,
    date_taken: str | None = None,
    modified_at: str | None = None,
    reliable: bool | None = None,
    capture_date: str | None = None,
    burst_group_id: str | None = None,
    extra_data: dict | None = None,
) -> DetectionItem:
    p = Path(path)
    data: dict = dict(extra_data or {})
    if reliable is not None or capture_date is not None:
        ct: dict = {}
        if reliable is not None:
            ct["timestamp_reliable"] = reliable
        if capture_date is not None:
            ct["date_taken"] = capture_date
        data["capture_time"] = ct
    if burst_group_id is not None:
        data["burst_group_id"] = burst_group_id
    return DetectionItem(
        filename=p.name,
        path=str(path),
        file_type=p.suffix.lower().lstrip("."),
        date_taken=date_taken,
        modified_at=modified_at,
        detection_data=data,
    )


def test_practical_three_photos_plus_video_burst_fallback(tmp_path: Path):
    p1 = _photo(tmp_path / "IMG_0001.JPG", "2026-11-16T08:00:00")
    p2 = _photo(tmp_path / "IMG_0002.JPG", "2026-11-16T08:00:01")
    p3 = _photo(tmp_path / "IMG_0003.JPG", "2026-11-16T08:00:02")
    v1 = _video(tmp_path / "IMG_0004.MP4", reliable=False)

    result = resolve_video_references([p1, p2, p3, v1])

    assert set(result.keys()) == {str(tmp_path / "IMG_0004.MP4")}
    ref = result[str(tmp_path / "IMG_0004.MP4")]
    assert ref["source"] == "burst_group"
    assert ref["precision"] == "group_anchor"
    assert ref["reference_time"] == "2026-11-16T08:00:00+00:00"
    assert ref["group_end_time"] == "2026-11-16T08:00:02+00:00"
    assert isinstance(ref["burst_group_id"], str) and ref["burst_group_id"]
    assert ref["group_members"] == [
        str(tmp_path / "IMG_0001.JPG"),
        str(tmp_path / "IMG_0002.JPG"),
        str(tmp_path / "IMG_0003.JPG"),
        str(tmp_path / "IMG_0004.MP4"),
    ]


def test_own_video_time_has_priority_over_burst_and_preserves_own_date(tmp_path: Path):
    p1 = _photo(tmp_path / "IMG_0001.JPG", "2026-11-16T08:00:00", detection_data={"burst_group_id": "grp-1"})
    p2 = _photo(tmp_path / "IMG_0002.JPG", "2026-11-16T08:00:01", detection_data={"burst_group_id": "grp-1"})
    p3 = _photo(tmp_path / "IMG_0003.JPG", "2026-11-16T08:00:02", detection_data={"burst_group_id": "grp-1"})
    v1 = _video(
        tmp_path / "IMG_0004.MP4",
        date_taken="2026-11-16T09:30:00",
        reliable=True,
        capture_date="2026-11-16T09:15:00",
        burst_group_id="grp-1",
    )

    ref = resolve_video_references([p1, p2, p3, v1])[str(tmp_path / "IMG_0004.MP4")]
    assert ref == {
        "source": "video",
        "reference_time": "2026-11-16T09:15:00+00:00",
        "precision": "exact",
        "burst_group_id": None,
        "group_members": [],
        "group_end_time": None,
    }


def test_unreliable_or_invalid_own_video_metadata_does_not_use_top_level_date_taken(tmp_path: Path):
    v_unreliable = _video(
        tmp_path / "clip1.mp4",
        date_taken="2026-11-16T08:00:00",
        reliable=False,
        capture_date="2026-11-16T08:00:00",
    )
    v_bad_date = _video(
        tmp_path / "clip2.mp4",
        date_taken="2026-11-16T08:00:00",
        reliable=True,
        capture_date="2026-11-16",
    )
    v_truthy_str = _video(
        tmp_path / "clip3.mp4",
        date_taken="2026-11-16T08:00:00",
        extra_data={"capture_time": {"timestamp_reliable": "true", "date_taken": "2026-11-16T08:00:00"}},
    )

    result = resolve_video_references([v_unreliable, v_bad_date, v_truthy_str])
    for key in result:
        assert result[key] == {
            "source": "none",
            "reference_time": None,
            "precision": "unknown",
            "burst_group_id": None,
            "group_members": [],
            "group_end_time": None,
        }


def test_photo_and_video_offsets_do_not_shift_the_shared_camera_clock(tmp_path: Path):
    # Literal times share the camera clock regardless of container offset tags.
    p1 = _photo(tmp_path / "IMG_0001.JPG", "2026:11:16 08:00:00")
    p2 = _photo(tmp_path / "IMG_0002.JPG", "2026-11-16T08:00:01+02:00")
    p3 = _photo(tmp_path / "IMG_0003.JPG", "2026-11-16T08:00:02Z")
    v1 = _video(tmp_path / "IMG_0004.MP4")

    v_explicit = _video(
        tmp_path / "IMG_0010.MP4",
        reliable=True,
        capture_date="2026-11-15T19:00:05-05:00",
    )

    result = resolve_video_references([p1, p2, p3, v1, v_explicit])
    assert result[str(tmp_path / "IMG_0004.MP4")]["reference_time"] == "2026-11-16T08:00:00+00:00"
    assert result[str(tmp_path / "IMG_0004.MP4")]["group_end_time"] == "2026-11-16T08:00:02+00:00"
    assert result[str(tmp_path / "IMG_0010.MP4")]["reference_time"] == "2026-11-15T19:00:05+00:00"


def test_missing_photo_timestamps_and_filename_or_mtime_never_used_as_anchor(tmp_path: Path):
    p1 = _photo(tmp_path / "20261116_080000_0001.JPG", None, modified_at="2026-11-16T08:00:00")
    p2 = _photo(tmp_path / "20261116_080000_0002.JPG", "2026-11-16T08:00:01", modified_at="2026-11-16T08:00:01")
    p3 = _photo(tmp_path / "20261116_080000_0003.JPG", "2026-11-16T08:00:02", modified_at="2026-11-16T08:00:02")
    v1 = _video(tmp_path / "20261116_080000_0004.MP4", modified_at="2026-11-16T08:00:03")

    ref = resolve_video_references([p1, p2, p3, v1])[str(tmp_path / "20261116_080000_0004.MP4")]
    assert ref["source"] == "none"
    assert ref["reference_time"] is None

    # Unreliable photo capture_time metadata must also block anchoring
    p1_bad = _photo(
        tmp_path / "IMG_0001.JPG",
        "2026-11-16T08:00:00",
        detection_data={"capture_time": {"timestamp_reliable": False, "date_taken": "2026-11-16T08:00:00"}},
    )
    p2_ok = _photo(tmp_path / "IMG_0002.JPG", "2026-11-16T08:00:01")
    p3_ok = _photo(tmp_path / "IMG_0003.JPG", "2026-11-16T08:00:02")
    v2 = _video(tmp_path / "IMG_0004.MP4")
    assert resolve_video_references([p1_bad, p2_ok, p3_ok, v2])[str(tmp_path / "IMG_0004.MP4")]["source"] == "none"


def test_rejects_mixed_prefix_counter_width_mismatch_and_cross_camera(tmp_path: Path):
    # Prefix mismatch between photos and video
    items_prefix = [
        _photo(tmp_path / "cam1" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "cam1" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "cam1" / "IMG_0003.JPG", "2026-11-16T08:00:02"),
        _video(tmp_path / "cam1" / "VID_0004.MP4"),
    ]
    assert resolve_video_references(items_prefix)[str(tmp_path / "cam1" / "VID_0004.MP4")]["source"] == "none"

    # Counter width mismatch
    items_width = [
        _photo(tmp_path / "cam2" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "cam2" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "cam2" / "IMG_0003.JPG", "2026-11-16T08:00:02"),
        _video(tmp_path / "cam2" / "IMG_004.MP4"),
    ]
    assert resolve_video_references(items_width)[str(tmp_path / "cam2" / "IMG_004.MP4")]["source"] == "none"

    # Cross-camera directory split
    items_cross = [
        _photo(tmp_path / "camA" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "camA" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "camA" / "IMG_0003.JPG", "2026-11-16T08:00:02"),
        _video(tmp_path / "camB" / "IMG_0004.MP4"),
    ]
    assert resolve_video_references(items_cross)[str(tmp_path / "camB" / "IMG_0004.MP4")]["source"] == "none"

    # Interleaved different prefix breaks contiguous counter run
    items_interleaved = [
        _photo(tmp_path / "cam3" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "cam3" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "cam3" / "ALT_0003.JPG", "2026-11-16T08:00:02"),
        _video(tmp_path / "cam3" / "IMG_0004.MP4"),
    ]
    assert resolve_video_references(items_interleaved)[str(tmp_path / "cam3" / "IMG_0004.MP4")]["source"] == "none"


def test_rejects_duplicate_counters_gaps_time_reversal_and_competing_videos(tmp_path: Path):
    # Duplicate photo counter
    dup_photo = [
        _photo(tmp_path / "d1" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "d1" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "d1" / "IMG_0002.PNG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "d1" / "IMG_0003.JPG", "2026-11-16T08:00:02"),
        _video(tmp_path / "d1" / "IMG_0004.MP4"),
    ]
    assert resolve_video_references(dup_photo)[str(tmp_path / "d1" / "IMG_0004.MP4")]["source"] == "none"

    # Duplicate video counter
    dup_video = [
        _photo(tmp_path / "d2" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "d2" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "d2" / "IMG_0003.JPG", "2026-11-16T08:00:02"),
        _video(tmp_path / "d2" / "IMG_0004.MP4"),
        _video(tmp_path / "d2" / "IMG_0004.AVI"),
    ]
    res_dv = resolve_video_references(dup_video)
    assert res_dv[str(tmp_path / "d2" / "IMG_0004.MP4")]["source"] == "none"
    assert res_dv[str(tmp_path / "d2" / "IMG_0004.AVI")]["source"] == "none"

    # Adjacent competing videos
    competing_videos = [
        _photo(tmp_path / "d3" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "d3" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "d3" / "IMG_0003.JPG", "2026-11-16T08:00:02"),
        _video(tmp_path / "d3" / "IMG_0004.MP4"),
        _video(tmp_path / "d3" / "IMG_0005.MP4"),
    ]
    res_cv = resolve_video_references(competing_videos)
    assert res_cv[str(tmp_path / "d3" / "IMG_0004.MP4")]["source"] == "none"
    assert res_cv[str(tmp_path / "d3" / "IMG_0005.MP4")]["source"] == "none"

    # Counter gap
    counter_gap = [
        _photo(tmp_path / "d4" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "d4" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "d4" / "IMG_0004.JPG", "2026-11-16T08:00:02"),
        _video(tmp_path / "d4" / "IMG_0005.MP4"),
    ]
    assert resolve_video_references(counter_gap)[str(tmp_path / "d4" / "IMG_0005.MP4")]["source"] == "none"

    # Time gap exceeding gap_seconds
    time_gap = [
        _photo(tmp_path / "d5" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "d5" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "d5" / "IMG_0003.JPG", "2026-11-16T08:00:45"),
        _video(tmp_path / "d5" / "IMG_0004.MP4"),
    ]
    assert resolve_video_references(time_gap, gap_seconds=30)[str(tmp_path / "d5" / "IMG_0004.MP4")]["source"] == "none"

    # Time reversal along increasing counter
    reversed_time = [
        _photo(tmp_path / "d6" / "IMG_0001.JPG", "2026-11-16T08:00:10"),
        _photo(tmp_path / "d6" / "IMG_0002.JPG", "2026-11-16T08:00:05"),
        _photo(tmp_path / "d6" / "IMG_0003.JPG", "2026-11-16T08:00:06"),
        _video(tmp_path / "d6" / "IMG_0004.MP4"),
    ]
    assert resolve_video_references(reversed_time)[str(tmp_path / "d6" / "IMG_0004.MP4")]["source"] == "none"


def test_detect_burst_and_incomplete_or_oversized_runs(tmp_path: Path):
    # Single incomplete 2-photo run must NOT override burst_size=3 even when detect_burst=True
    incomplete = [
        _photo(tmp_path / "inc" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "inc" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _video(tmp_path / "inc" / "IMG_0003.MP4"),
    ]
    assert resolve_video_references(incomplete, burst_size=3, detect_burst=True)[
        str(tmp_path / "inc" / "IMG_0003.MP4")
    ]["source"] == "none"

    # Multiple completed 5-photo runs auto-detected when detect_burst=True, rejected when detect_burst=False
    five_bursts = [
        _photo(tmp_path / "f5" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "f5" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "f5" / "IMG_0003.JPG", "2026-11-16T08:00:02"),
        _photo(tmp_path / "f5" / "IMG_0004.JPG", "2026-11-16T08:00:03"),
        _photo(tmp_path / "f5" / "IMG_0005.JPG", "2026-11-16T08:00:04"),
        _video(tmp_path / "f5" / "IMG_0006.MP4"),
        _photo(tmp_path / "f5" / "IMG_0007.JPG", "2026-11-16T08:10:00"),
        _photo(tmp_path / "f5" / "IMG_0008.JPG", "2026-11-16T08:10:01"),
        _photo(tmp_path / "f5" / "IMG_0009.JPG", "2026-11-16T08:10:02"),
        _photo(tmp_path / "f5" / "IMG_0010.JPG", "2026-11-16T08:10:03"),
        _photo(tmp_path / "f5" / "IMG_0011.JPG", "2026-11-16T08:10:04"),
        _video(tmp_path / "f5" / "IMG_0012.MP4"),
    ]
    res_auto = resolve_video_references(five_bursts, burst_size=3, detect_burst=True)
    assert res_auto[str(tmp_path / "f5" / "IMG_0006.MP4")]["source"] == "burst_group"
    assert len(res_auto[str(tmp_path / "f5" / "IMG_0006.MP4")]["group_members"]) == 6
    assert res_auto[str(tmp_path / "f5" / "IMG_0012.MP4")]["source"] == "burst_group"

    res_fixed = resolve_video_references(five_bursts, burst_size=3, detect_burst=False)
    assert res_fixed[str(tmp_path / "f5" / "IMG_0006.MP4")]["source"] == "none"
    assert res_fixed[str(tmp_path / "f5" / "IMG_0012.MP4")]["source"] == "none"


def test_explicit_burst_group_validation(tmp_path: Path):
    # Valid explicit group (even non-numeric filenames)
    valid_items = [
        _photo(tmp_path / "ok" / "b.jpg", "2026-11-16T08:00:05", detection_data={"burst_group_id": "cap-1"}),
        _photo(tmp_path / "ok" / "a.jpg", "2026-11-16T08:00:01", detection_data={"burst_group_id": "cap-1"}),
        _video(tmp_path / "ok" / "clip.avi", burst_group_id="cap-1"),
    ]
    ref = resolve_video_references(valid_items)[str(tmp_path / "ok" / "clip.avi")]
    assert ref["source"] == "burst_group"
    assert ref["precision"] == "group_anchor"
    assert ref["burst_group_id"] == "cap-1"
    assert ref["reference_time"] == "2026-11-16T08:00:01+00:00"
    assert ref["group_end_time"] == "2026-11-16T08:00:05+00:00"
    assert ref["group_members"] == [
        str(tmp_path / "ok" / "a.jpg"),
        str(tmp_path / "ok" / "b.jpg"),
        str(tmp_path / "ok" / "clip.avi"),
    ]

    # Cross-camera explicit group rejected
    cross = [
        _photo(tmp_path / "cam1" / "a.jpg", "2026-11-16T08:00:01", detection_data={"burst_group_id": "cap-x"}),
        _video(tmp_path / "cam2" / "clip.avi", burst_group_id="cap-x"),
    ]
    assert resolve_video_references(cross)[str(tmp_path / "cam2" / "clip.avi")]["source"] == "none"

    # Competing videos in explicit group rejected
    competing = [
        _photo(tmp_path / "comp" / "a.jpg", "2026-11-16T08:00:01", detection_data={"burst_group_id": "cap-c"}),
        _video(tmp_path / "comp" / "v1.avi", burst_group_id="cap-c"),
        _video(tmp_path / "comp" / "v2.avi", burst_group_id="cap-c"),
    ]
    res_comp = resolve_video_references(competing)
    assert res_comp[str(tmp_path / "comp" / "v1.avi")]["source"] == "none"
    assert res_comp[str(tmp_path / "comp" / "v2.avi")]["source"] == "none"

    # Missing photo timestamp in explicit group rejected
    missing_photo_ts = [
        _photo(tmp_path / "mt" / "a.jpg", "2026-11-16T08:00:01", detection_data={"burst_group_id": "cap-m"}),
        _photo(tmp_path / "mt" / "b.jpg", None, detection_data={"burst_group_id": "cap-m"}),
        _video(tmp_path / "mt" / "v1.avi", burst_group_id="cap-m"),
    ]
    assert resolve_video_references(missing_photo_ts)[str(tmp_path / "mt" / "v1.avi")]["source"] == "none"

    # Inconsistent photo gap in explicit group rejected
    inconsistent_gap = [
        _photo(tmp_path / "ig" / "a.jpg", "2026-11-16T08:00:00", detection_data={"burst_group_id": "cap-g"}),
        _photo(tmp_path / "ig" / "b.jpg", "2026-11-16T09:00:00", detection_data={"burst_group_id": "cap-g"}),
        _video(tmp_path / "ig" / "v1.avi", burst_group_id="cap-g"),
    ]
    assert resolve_video_references(inconsistent_gap, gap_seconds=30)[str(tmp_path / "ig" / "v1.avi")]["source"] == "none"

    # Video with explicit group missing photos must NOT fall back to unlabelled counter burst
    orphaned_explicit = [
        _photo(tmp_path / "oe" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "oe" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "oe" / "IMG_0003.JPG", "2026-11-16T08:00:02"),
        _video(tmp_path / "oe" / "IMG_0004.MP4", burst_group_id="missing-grp"),
    ]
    assert resolve_video_references(orphaned_explicit)[str(tmp_path / "oe" / "IMG_0004.MP4")]["source"] == "none"


def test_multi_camera_multi_prefix_and_sequential_triggers(tmp_path: Path):
    items = [
        # Camera A, prefix IMG_, two back-to-back triggers separated by first video
        _photo(tmp_path / "camA" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "camA" / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "camA" / "IMG_0003.JPG", "2026-11-16T08:00:02"),
        _video(tmp_path / "camA" / "IMG_0004.MP4"),
        _photo(tmp_path / "camA" / "IMG_0005.JPG", "2026-11-16T08:00:05"),
        _photo(tmp_path / "camA" / "IMG_0006.JPG", "2026-11-16T08:00:06"),
        _photo(tmp_path / "camA" / "IMG_0007.JPG", "2026-11-16T08:00:07"),
        _video(tmp_path / "camA" / "IMG_0008.MP4"),
        # Camera A, independent prefix RCNX in same folder
        _photo(tmp_path / "camA" / "RCNX0001.JPG", "2026-11-16T09:00:00"),
        _photo(tmp_path / "camA" / "RCNX0002.JPG", "2026-11-16T09:00:01"),
        _photo(tmp_path / "camA" / "RCNX0003.JPG", "2026-11-16T09:00:02"),
        _video(tmp_path / "camA" / "RCNX0004.AVI"),
        # Camera B with identical filenames
        _photo(tmp_path / "camB" / "IMG_0001.JPG", "2026-11-16T10:00:00"),
        _photo(tmp_path / "camB" / "IMG_0002.JPG", "2026-11-16T10:00:01"),
        _photo(tmp_path / "camB" / "IMG_0003.JPG", "2026-11-16T10:00:02"),
        _video(tmp_path / "camB" / "IMG_0004.MP4"),
        # Non-media file should be ignored
        {"path": str(tmp_path / "camA" / "notes.txt"), "date_taken": "2026-11-16T08:00:00"},
    ]

    res = resolve_video_references(items)
    assert res[str(tmp_path / "camA" / "IMG_0004.MP4")]["reference_time"] == "2026-11-16T08:00:00+00:00"
    assert res[str(tmp_path / "camA" / "IMG_0008.MP4")]["reference_time"] == "2026-11-16T08:00:05+00:00"
    assert res[str(tmp_path / "camA" / "RCNX0004.AVI")]["reference_time"] == "2026-11-16T09:00:00+00:00"
    assert res[str(tmp_path / "camB" / "IMG_0004.MP4")]["reference_time"] == "2026-11-16T10:00:00+00:00"
    assert (
        res[str(tmp_path / "camA" / "IMG_0004.MP4")]["burst_group_id"]
        != res[str(tmp_path / "camB" / "IMG_0004.MP4")]["burst_group_id"]
    )


def test_gap_boundary_non_numeric_and_invalid_parameters(tmp_path: Path):
    # Exact gap_seconds=30 allowed; 31 rejected
    boundary_ok = [
        _photo(tmp_path / "b_ok" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "b_ok" / "IMG_0002.JPG", "2026-11-16T08:00:30"),
        _photo(tmp_path / "b_ok" / "IMG_0003.JPG", "2026-11-16T08:01:00"),
        _video(tmp_path / "b_ok" / "IMG_0004.MP4"),
    ]
    assert resolve_video_references(boundary_ok, gap_seconds=30)[
        str(tmp_path / "b_ok" / "IMG_0004.MP4")
    ]["source"] == "burst_group"

    boundary_fail = [
        _photo(tmp_path / "b_fail" / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "b_fail" / "IMG_0002.JPG", "2026-11-16T08:00:31"),
        _photo(tmp_path / "b_fail" / "IMG_0003.JPG", "2026-11-16T08:01:00"),
        _video(tmp_path / "b_fail" / "IMG_0004.MP4"),
    ]
    assert resolve_video_references(boundary_fail, gap_seconds=30)[
        str(tmp_path / "b_fail" / "IMG_0004.MP4")
    ]["source"] == "none"

    # Non-numeric names without explicit burst_group_id -> none
    non_numeric = [
        _photo(tmp_path / "nn" / "alpha.jpg", "2026-11-16T08:00:00"),
        _photo(tmp_path / "nn" / "beta.jpg", "2026-11-16T08:00:01"),
        _photo(tmp_path / "nn" / "gamma.jpg", "2026-11-16T08:00:02"),
        _video(tmp_path / "nn" / "delta.mp4"),
    ]
    assert resolve_video_references(non_numeric)[str(tmp_path / "nn" / "delta.mp4")]["source"] == "none"

    # Invalid burst_size / gap_seconds still preserve own video time and reject fallback
    own_v = _video(tmp_path / "inv" / "own.mp4", reliable=True, capture_date="2026-11-16T08:00:00")
    no_v = _video(tmp_path / "inv" / "IMG_0004.MP4")
    res_bad_burst = resolve_video_references([own_v, no_v], burst_size=0)
    assert res_bad_burst[str(tmp_path / "inv" / "own.mp4")]["source"] == "video"
    assert res_bad_burst[str(tmp_path / "inv" / "IMG_0004.MP4")]["source"] == "none"
    res_bad_gap = resolve_video_references([own_v, no_v], gap_seconds=-1)
    assert res_bad_gap[str(tmp_path / "inv" / "own.mp4")]["source"] == "video"
    assert res_bad_gap[str(tmp_path / "inv" / "IMG_0004.MP4")]["source"] == "none"


def test_input_order_independent_missing_fields_and_no_file_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _forbid_open(*args, **kwargs):
        raise AssertionError("video_reference must not open or read files")

    monkeypatch.setattr(Path, "open", _forbid_open)

    base_items = [
        _photo(tmp_path / "IMG_0001.JPG", "2026-11-16T08:00:00"),
        _photo(tmp_path / "IMG_0002.JPG", "2026-11-16T08:00:01"),
        _photo(tmp_path / "IMG_0003.JPG", "2026-11-16T08:00:02"),
        _video(tmp_path / "IMG_0004.MP4"),
        {"path": str(tmp_path / "IMG_0005.JPG"), "date_taken": "2026-11-16T08:10:00"},
        SimpleNamespace(path=str(tmp_path / "IMG_0006.JPG"), date_taken="2026-11-16T08:10:01", detection_data=None),
        SimpleNamespace(path=str(tmp_path / "IMG_0007.JPG"), date_taken="2026-11-16T08:10:02", detection_data="bad"),
        {"path": str(tmp_path / "IMG_0008.MP4"), "detection_data": None},
        {"path": None},
        {"path": ""},
        SimpleNamespace(date_taken="2026-11-16T08:00:00"),
        _video(tmp_path / "bad_meta.mp4", extra_data={"capture_time": "invalid"}),
    ]

    expected = resolve_video_references(base_items)
    rng = random.Random(42)
    shuffled = list(base_items)
    rng.shuffle(shuffled)
    actual = resolve_video_references(shuffled)

    assert actual == expected
    assert actual[str(tmp_path / "IMG_0004.MP4")]["source"] == "burst_group"
    assert actual[str(tmp_path / "IMG_0008.MP4")]["source"] == "burst_group"
    assert actual[str(tmp_path / "bad_meta.mp4")]["source"] == "none"
