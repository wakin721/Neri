from contextlib import closing
import io
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from system.backend.models import DetectionItem
from system.dinov3.job_cache import DinoV3JobCache, file_stamp
import system.dinov3.media_cache as media_cache
from system.dinov3.media_cache import DinoV3VideoCache
from system.dinov3.processor import DinoV3CropCollector, DinoV3CropRecord
from system.dinov3.video import video_frame_key


def _make_video_item(path: Path, **extra_detection_data) -> DetectionItem:
    return DetectionItem(
        filename=path.name,
        path=str(path),
        file_type="mp4",
        date_taken="2026-11-16 12:34:56",
        size_bytes=path.stat().st_size if path.exists() else None,
        detection_data={
            "source_timestamp": "2026-11-16T12:34:56Z",
            "timestamp_source": "quicktime_creation_time",
            **extra_detection_data,
        },
    )


def _make_embedding(seed: float, *, dtype=np.float64) -> np.ndarray:
    vec = np.linspace(seed, seed + 1.0, 768, dtype=dtype)
    vec[0] = seed + 0.123456789012345
    return vec


def _make_crop(
    video_path: Path,
    frame_index: int,
    *,
    box_index: int = 0,
    seed: float = 1.0,
    dtype=np.float64,
    use_frame_key: bool = True,
    accepted: bool = True,
) -> DinoV3CropRecord:
    image_path = video_frame_key(video_path, frame_index) if use_frame_key else str(video_path)
    return DinoV3CropRecord(
        image_path=image_path,
        box_index=box_index,
        bbox=(1.5 + box_index, 2.5, 30.25, 40.75),
        detection_confidence=0.92 - box_index * 0.1,
        raw_embedding=_make_embedding(seed, dtype=dtype),
        initial_candidate={
            "name": "fox" if accepted else "Unknown",
            "accepted": accepted,
            "sequence_pending": True,
            "sequence_status": "pending",
        },
    )


def _make_frame_data(
    box_indices: list[int],
    *,
    frame_digest: str = "abc123digest",
    source_timestamp: str = "2026-11-16T12:34:57Z",
    completed: bool = True,
    error: str | None = None,
) -> dict:
    return {
        "detection_completed": completed,
        "error": error,
        "detection_box_count": len(box_indices),
        "detection_box_indices": list(box_indices),
        "frame_digest": frame_digest,
        "source_timestamp": source_timestamp,
        "检测框": [
            {"边界框": [1.5 + idx, 2.5, 30.25, 40.75], "置信度": 0.9}
            for idx in box_indices
        ],
    }


def _make_info(frame_points: list[int], **extra) -> dict:
    return {
        "frame_points": list(frame_points),
        "fps": 25.0,
        "width": 1920,
        "height": 1080,
        "total_frames": 100,
        "timestamp_source": "mp4_mvhd",
        **extra,
    }


def test_video_cache_lives_beside_photo_cache_and_removes_cleanly(tmp_path):
    state_path = tmp_path / "state.json"
    photo_cache = DinoV3JobCache(state_path, "job-1", {"model": "v1"})
    video_cache = DinoV3VideoCache(state_path, "job-1", {"model": "v1"})

    assert video_cache.path.parent == photo_cache.path.parent
    assert video_cache.path.parent.name == "dinov3_features"
    assert video_cache.path.name.endswith("-video.sqlite3")
    assert video_cache.path != photo_cache.path
    assert video_cache.path.exists()
    assert photo_cache.path.exists()

    DinoV3VideoCache.remove(state_path, "job-1")
    assert not video_cache.path.exists()
    assert photo_cache.path.exists()


def test_roundtrip_preserves_metadata_precision_order_and_progress(tmp_path):
    state_path = tmp_path / "state.json"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video-bytes-001")
    stamp = file_stamp(video_path)
    signature = {"model": "v1", "vid_stride": 3, "confidence": 0.25}

    item = _make_video_item(video_path)
    frame0 = (
        0,
        10,
        _make_frame_data([0, 1], frame_digest="digest-f10", source_timestamp="2026-11-16T12:34:56.400Z"),
        [
            _make_crop(video_path, 10, box_index=0, seed=1.0, dtype=np.float64, accepted=True),
            _make_crop(video_path, 10, box_index=1, seed=2.0, dtype=np.float64, accepted=False),
        ],
    )
    frame1 = (
        1,
        25,
        _make_frame_data([0], frame_digest="digest-f25", source_timestamp="2026-11-16T12:34:57.000Z"),
        [_make_crop(video_path, 25, box_index=0, seed=3.0, dtype=np.float32, accepted=False)],
    )
    info = _make_info([10, 25], camera_clock="utc")

    cache = DinoV3VideoCache(state_path, "job-1", signature)
    # Pass frames out of order to ensure restore returns sorted by frame_order.
    assert cache.save(item, [frame1, frame0], info, stamp) is True

    reopened = DinoV3VideoCache(state_path, "job-1", signature)
    progress_calls = []
    restored = reopened.restore([video_path], progress_fn=lambda done, total: progress_calls.append((done, total)))

    assert progress_calls == [(0, 1), (1, 1)]
    assert list(restored.keys()) == [str(video_path)]
    r_item, r_frames, r_info = restored[str(video_path)]

    assert r_item == item
    assert r_item.detection_data["source_timestamp"] == "2026-11-16T12:34:56Z"
    assert r_item.detection_data["timestamp_source"] == "quicktime_creation_time"
    assert r_info == info
    assert [(f[0], f[1]) for f in r_frames] == [(0, 10), (1, 25)]

    assert r_frames[0][2]["frame_digest"] == "digest-f10"
    assert r_frames[0][2]["source_timestamp"] == "2026-11-16T12:34:56.400Z"
    assert r_frames[1][2]["frame_digest"] == "digest-f25"
    assert r_frames[0][3][0].initial_candidate["accepted"] is True
    assert r_frames[0][3][1].initial_candidate["accepted"] is False
    assert r_frames[0][3][0].raw_embedding.dtype == np.float64
    np.testing.assert_array_equal(r_frames[0][3][0].raw_embedding, frame0[3][0].raw_embedding)
    np.testing.assert_array_equal(r_frames[0][3][1].raw_embedding, frame0[3][1].raw_embedding)
    assert r_frames[1][3][0].raw_embedding.dtype == np.float32
    np.testing.assert_array_equal(r_frames[1][3][0].raw_embedding, frame1[3][0].raw_embedding)

    # Job isolation
    other = DinoV3VideoCache(state_path, "job-2", signature)
    assert other.restore([video_path]) == {}

    # No raw image pixel payloads stored in SQLite
    with closing(sqlite3.connect(reopened.path)) as conn:
        frame_cols = {row[1] for row in conn.execute("PRAGMA table_info(video_frames)")}
        assert frame_cols.isdisjoint({"bgr", "rgb", "pixels", "image", "frame_bytes"})


def test_empty_zero_box_frames_valid_and_zero_frame_video_rejected(tmp_path):
    state_path = tmp_path / "state.json"
    empty_video = tmp_path / "empty.mp4"
    empty_video.write_bytes(b"empty-video")
    stamp = file_stamp(empty_video)

    cache = DinoV3VideoCache(state_path, "job-1", {"model": "v1"})
    item = _make_video_item(empty_video)
    frames = [
        (0, 5, _make_frame_data([], frame_digest="d5"), []),
        (1, 15, _make_frame_data([], frame_digest="d15"), []),
    ]
    info = _make_info([5, 15])
    assert cache.save(item, frames, info, stamp) is True

    restored = DinoV3VideoCache(state_path, "job-1", {"model": "v1"}).restore([empty_video])
    assert str(empty_video) in restored
    _, r_frames, r_info = restored[str(empty_video)]
    assert r_info["frame_points"] == [5, 15]
    assert [f[3] for f in r_frames] == [[], []]

    # Video with 0 sampled frames cannot be cached as completed.
    zero_frames_video = tmp_path / "zero.mp4"
    zero_frames_video.write_bytes(b"zero")
    assert cache.save(
        _make_video_item(zero_frames_video), [], _make_info([]), file_stamp(zero_frames_video),
    ) is False

    # Zero-box frame without completed empty 检测框 list is rejected.
    bad_empty_data = _make_frame_data([])
    del bad_empty_data["检测框"]
    assert cache.save(
        item, [(0, 5, bad_empty_data, [])], _make_info([5]), stamp,
    ) is False


def test_source_stamp_mismatch_and_file_change_invalidate_cache(tmp_path):
    state_path = tmp_path / "state.json"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"original-content")
    stamp = file_stamp(video_path)
    item = _make_video_item(video_path)
    frames = [(0, 10, _make_frame_data([0]), [_make_crop(video_path, 10)])]
    info = _make_info([10])

    cache = DinoV3VideoCache(state_path, "job-1", {"model": "v1"})
    assert cache.save(item, frames, info, [stamp[0] + 1, stamp[1]]) is False
    assert cache.restore([video_path]) == {}

    assert cache.save(item, frames, info, stamp) is True
    video_path.write_bytes(b"modified-content-longer")
    assert cache.restore([video_path]) == {}

    # Invalid entry should have been discarded from SQLite.
    with closing(sqlite3.connect(cache.path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM video_frames").fetchone()[0] == 0


def test_model_signature_invalidation_discards_cached_videos(tmp_path):
    state_path = tmp_path / "state.json"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    stamp = file_stamp(video_path)
    item = _make_video_item(video_path)
    frames = [(0, 10, _make_frame_data([0]), [_make_crop(video_path, 10)])]
    info = _make_info([10])

    cache = DinoV3VideoCache(state_path, "job-1", {"model": "v1", "vid_stride": 3})
    assert cache.save(item, frames, info, stamp) is True

    reopened = DinoV3VideoCache(state_path, "job-1", {"model": "v2", "vid_stride": 3})
    assert reopened.restore([video_path]) == {}
    with closing(sqlite3.connect(reopened.path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM video_frames").fetchone()[0] == 0


@pytest.mark.parametrize("case", [
    "item_error",
    "incomplete_frame",
    "frame_error",
    "crop_count_mismatch",
    "crop_index_mismatch",
    "bad_embedding_dim",
    "int_embedding",
    "nan_embedding",
    "nan_bbox",
    "nan_confidence",
    "wrong_crop_path",
    "non_increasing_frame_points",
    "duplicate_frame_order",
    "frame_points_mismatch",
])
def test_save_rejects_invalid_inputs(tmp_path, case):
    state_path = tmp_path / "state.json"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    stamp = file_stamp(video_path)
    cache = DinoV3VideoCache(state_path, "job-1", {"model": "v1"})

    item = _make_video_item(video_path)
    data = _make_frame_data([0])
    crop = _make_crop(video_path, 10, box_index=0)
    frames = [(0, 10, data, [crop])]
    info = _make_info([10])

    if case == "item_error":
        item = item.model_copy(update={"error": "decode failed"})
    elif case == "incomplete_frame":
        data["detection_completed"] = False
    elif case == "frame_error":
        data["error"] = "frame error"
    elif case == "crop_count_mismatch":
        frames = [(0, 10, data, [])]
    elif case == "crop_index_mismatch":
        crop = _make_crop(video_path, 10, box_index=1)
        frames = [(0, 10, data, [crop])]
    elif case == "bad_embedding_dim":
        crop = DinoV3CropRecord(
            crop.image_path, 0, crop.bbox, 0.9, np.ones(512, dtype=np.float32),
        )
        frames = [(0, 10, data, [crop])]
    elif case == "int_embedding":
        crop = DinoV3CropRecord(
            crop.image_path, 0, crop.bbox, 0.9, np.ones(768, dtype=np.int32),
        )
        frames = [(0, 10, data, [crop])]
    elif case == "nan_embedding":
        emb = _make_embedding(1.0)
        emb[10] = np.nan
        crop = DinoV3CropRecord(crop.image_path, 0, crop.bbox, 0.9, emb)
        frames = [(0, 10, data, [crop])]
    elif case == "nan_bbox":
        crop = DinoV3CropRecord(
            crop.image_path, 0, (1.0, np.inf, 10.0, 20.0), 0.9, _make_embedding(1.0),
        )
        frames = [(0, 10, data, [crop])]
    elif case == "nan_confidence":
        crop = DinoV3CropRecord(
            crop.image_path, 0, crop.bbox, float("nan"), _make_embedding(1.0),
        )
        frames = [(0, 10, data, [crop])]
    elif case == "wrong_crop_path":
        crop = _make_crop(tmp_path / "other.mp4", 10, box_index=0)
        frames = [(0, 10, data, [crop])]
    elif case == "non_increasing_frame_points":
        frames = [
            (0, 20, _make_frame_data([0]), [_make_crop(video_path, 20)]),
            (1, 10, _make_frame_data([0]), [_make_crop(video_path, 10)]),
        ]
        info = _make_info([20, 10])
    elif case == "duplicate_frame_order":
        frames = [
            (0, 10, _make_frame_data([0]), [_make_crop(video_path, 10)]),
            (0, 20, _make_frame_data([0]), [_make_crop(video_path, 20)]),
        ]
        info = _make_info([10, 20])
    elif case == "frame_points_mismatch":
        info = _make_info([10, 20])

    assert cache.save(item, frames, info, stamp) is False
    assert cache.restore([video_path]) == {}


@pytest.mark.parametrize("corruption", [
    "corrupt_blob",
    "pickled_blob",
    "wrong_shape_blob",
    "nan_in_blob",
    "missing_frame_row",
    "corrupt_frame_json",
    "corrupt_info_json",
    "missing_box_index_in_crops_meta",
])
def test_restore_discards_corrupt_or_incomplete_records(tmp_path, corruption):
    state_path = tmp_path / "state.json"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    stamp = file_stamp(video_path)
    cache = DinoV3VideoCache(state_path, "job-1", {"model": "v1"})

    item = _make_video_item(video_path)
    frames = [
        (0, 10, _make_frame_data([0]), [_make_crop(video_path, 10, seed=1.0)]),
        (1, 20, _make_frame_data([0]), [_make_crop(video_path, 20, seed=2.0)]),
    ]
    info = _make_info([10, 20])
    assert cache.save(item, frames, info, stamp) is True

    with closing(sqlite3.connect(cache.path)) as conn, conn:
        if corruption == "corrupt_blob":
            conn.execute("UPDATE video_frames SET features=? WHERE frame_order=0", (b"broken",))
        elif corruption == "pickled_blob":
            buf = io.BytesIO()
            np.save(buf, np.array([{"evil": 1}], dtype=object), allow_pickle=True)
            conn.execute("UPDATE video_frames SET features=? WHERE frame_order=0", (buf.getvalue(),))
        elif corruption == "wrong_shape_blob":
            buf = io.BytesIO()
            np.save(buf, np.ones((1, 512), dtype=np.float32), allow_pickle=False)
            conn.execute("UPDATE video_frames SET features=? WHERE frame_order=0", (buf.getvalue(),))
        elif corruption == "nan_in_blob":
            buf = io.BytesIO()
            arr = np.ones((1, 768), dtype=np.float32)
            arr[0, 5] = np.nan
            np.save(buf, arr, allow_pickle=False)
            conn.execute("UPDATE video_frames SET features=? WHERE frame_order=0", (buf.getvalue(),))
        elif corruption == "missing_frame_row":
            conn.execute("DELETE FROM video_frames WHERE frame_order=1")
        elif corruption == "corrupt_frame_json":
            conn.execute("UPDATE video_frames SET data=? WHERE frame_order=0", ("{not-json",))
        elif corruption == "corrupt_info_json":
            conn.execute("UPDATE videos SET info=?", (json.dumps({"frame_points": [10]}),))
        elif corruption == "missing_box_index_in_crops_meta":
            conn.execute(
                "UPDATE video_frames SET crops=? WHERE frame_order=0",
                (json.dumps([{"box_index": 2, "bbox": [1, 2, 3, 4], "detection_confidence": 0.9}]),),
            )

    assert cache.restore([video_path]) == {}
    with closing(sqlite3.connect(cache.path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM video_frames").fetchone()[0] == 0


def test_atomic_failed_save_retains_older_complete_record(tmp_path):
    state_path = tmp_path / "state.json"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    stamp = file_stamp(video_path)
    cache = DinoV3VideoCache(state_path, "job-1", {"model": "v1"})

    old_item = _make_video_item(video_path, tag="old")
    old_frames = [
        (0, 10, _make_frame_data([0], frame_digest="old-d10"), [_make_crop(video_path, 10, seed=1.0)]),
        (1, 20, _make_frame_data([0], frame_digest="old-d20"), [_make_crop(video_path, 20, seed=2.0)]),
    ]
    old_info = _make_info([10, 20])
    assert cache.save(old_item, old_frames, old_info, stamp) is True

    # Validation failure does not disturb older record.
    new_item = _make_video_item(video_path, tag="new")
    assert cache.save(new_item, [(0, 10, _make_frame_data([0]), [])], _make_info([10]), stamp) is False

    # SQLite mid-transaction abort rolls back DELETE/REPLACE so older record remains intact.
    with closing(sqlite3.connect(cache.path)) as conn, conn:
        conn.execute(
            "CREATE TRIGGER abort_second_frame BEFORE INSERT ON video_frames "
            "WHEN NEW.frame_order = 1 "
            "BEGIN SELECT RAISE(ABORT, 'simulated failure'); END"
        )

    new_frames = [
        (0, 10, _make_frame_data([0], frame_digest="new-d10"), [_make_crop(video_path, 10, seed=9.0)]),
        (1, 20, _make_frame_data([0], frame_digest="new-d20"), [_make_crop(video_path, 20, seed=10.0)]),
    ]
    with pytest.raises(sqlite3.IntegrityError, match="simulated failure"):
        cache.save(new_item, new_frames, old_info, stamp)

    with closing(sqlite3.connect(cache.path)) as conn, conn:
        conn.execute("DROP TRIGGER abort_second_frame")

    restored = cache.restore([video_path])
    assert str(video_path) in restored
    r_item, r_frames, r_info = restored[str(video_path)]
    assert r_item.detection_data["tag"] == "old"
    assert [f[2]["frame_digest"] for f in r_frames] == ["old-d10", "old-d20"]
    np.testing.assert_array_equal(r_frames[0][3][0].raw_embedding, old_frames[0][3][0].raw_embedding)
    np.testing.assert_array_equal(r_frames[1][3][0].raw_embedding, old_frames[1][3][0].raw_embedding)
    assert r_info == old_info


def test_partial_restore_keeps_valid_video_discards_corrupt_and_propagates_cancel(tmp_path):
    state_path = tmp_path / "state.json"
    good_video = tmp_path / "good.mp4"
    bad_video = tmp_path / "bad.mp4"
    good_video.write_bytes(b"good-video")
    bad_video.write_bytes(b"bad-video")
    cache = DinoV3VideoCache(state_path, "job-1", {"model": "v1"})

    good_crop = _make_crop(good_video, 10, use_frame_key=False)
    assert cache.save(
        _make_video_item(good_video),
        [(0, 10, _make_frame_data([0]), [good_crop])],
        _make_info([10]),
        file_stamp(good_video),
    ) is True
    assert cache.save(
        _make_video_item(bad_video),
        [(0, 10, _make_frame_data([0]), [_make_crop(bad_video, 10)])],
        _make_info([10]),
        file_stamp(bad_video),
    ) is True

    bad_video.write_bytes(b"bad-video-modified")
    restored = cache.restore([good_video, bad_video])
    assert list(restored.keys()) == [str(good_video)]
    assert restored[str(good_video)][1][0][3][0].image_path == str(good_video)

    def cancel_on_second(done, total):
        if done == 1:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        cache.restore([good_video, bad_video], progress_fn=cancel_on_second)

    # Tampered metadata signature row invalidates restore.
    with closing(sqlite3.connect(cache.path)) as conn, conn:
        conn.execute("UPDATE metadata SET signature=?", (json.dumps({"version": 999}),))
    assert cache.restore([good_video]) == {}
    with closing(sqlite3.connect(cache.path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0


@pytest.mark.parametrize("payload_mode", [
    "short_header",
    "garbage_bytes",
    "truncated_db_50",
    "truncated_db_page",
])
def test_recovers_from_truncated_sqlite_header_and_preserves_siblings(tmp_path, payload_mode):
    state_path = tmp_path / "state.json"
    state_path.write_text('{"user": "keep"}', encoding="utf-8")
    photo_path = tmp_path / "photo.jpg"
    photo_path.write_bytes(b"photo-bytes")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model-bytes")
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    stamp = file_stamp(video_path)

    photo_cache = DinoV3JobCache(state_path, "job-1", {"model": "v1"})
    photo_collector = DinoV3CropCollector()
    photo_collector.record_crop(
        DinoV3CropRecord(str(photo_path), 0, (1, 2, 10, 20), 0.9, _make_embedding(4.0)),
    )
    photo_item = DetectionItem(
        filename=photo_path.name,
        path=str(photo_path),
        file_type="jpg",
        detection_data={
            "detection_completed": True,
            "detection_box_count": 1,
            "detection_box_indices": [0],
            "检测框": [{"边界框": [1, 2, 10, 20]}],
        },
    )
    photo_cache.save_batch([photo_item], photo_collector, {str(photo_path): file_stamp(photo_path)})

    signature = {"model": "v1"}
    seed_cache = DinoV3VideoCache(state_path, "job-1", signature)
    assert seed_cache.save(
        _make_video_item(video_path),
        [(0, 10, _make_frame_data([0]), [_make_crop(video_path, 10)])],
        _make_info([10]),
        stamp,
    ) is True
    raw_db = seed_cache.path.read_bytes()

    if payload_mode == "short_header":
        corrupt_bytes = b"SQLite format 3\x00"[:8]
    elif payload_mode == "garbage_bytes":
        corrupt_bytes = b"corrupt-non-sqlite-payload"
    elif payload_mode == "truncated_db_50":
        corrupt_bytes = raw_db[:50]
    else:
        corrupt_bytes = raw_db[:1000]

    seed_cache.path.write_bytes(corrupt_bytes)
    for suffix in ("-wal", "-shm", "-journal"):
        seed_cache.path.with_name(seed_cache.path.name + suffix).write_bytes(b"stale-sidecar")

    recovered = DinoV3VideoCache(state_path, "job-1", signature)
    assert recovered.restore([video_path]) == {}
    for suffix in ("-wal", "-shm", "-journal"):
        assert not recovered.path.with_name(recovered.path.name + suffix).exists()

    assert recovered.save(
        _make_video_item(video_path),
        [(0, 10, _make_frame_data([0]), [_make_crop(video_path, 10)])],
        _make_info([10]),
        stamp,
    ) is True
    assert str(video_path) in DinoV3VideoCache(state_path, "job-1", signature).restore([video_path])

    # Sibling photo cache, userdata, photos, and model files remain intact.
    assert state_path.read_text(encoding="utf-8") == '{"user": "keep"}'
    assert photo_path.read_bytes() == b"photo-bytes"
    assert model_path.read_bytes() == b"model-bytes"
    verify_collector = DinoV3CropCollector()
    restored_photos = photo_cache.restore([photo_path], verify_collector)
    assert [item.path for item in restored_photos] == [str(photo_path)]
    assert len(verify_collector.get_crops_for_path(photo_path)) == 1


@pytest.mark.parametrize("legacy_schema", [
    "metadata_missing_signature",
    "videos_missing_info",
    "video_frames_missing_features",
    "videos_missing_primary_key",
])
def test_recovers_from_missing_columns_in_legacy_tables_with_data(tmp_path, legacy_schema):
    state_path = tmp_path / "state.json"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    stamp = file_stamp(video_path)
    signature = {"model": "v1"}
    expected_sig_json = json.dumps({"version": DinoV3VideoCache.VERSION, **signature}, sort_keys=True, ensure_ascii=False)

    cache_path = DinoV3VideoCache.path_for(state_path, "job-1")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(cache_path)) as conn, conn:
        if legacy_schema == "metadata_missing_signature":
            conn.execute("CREATE TABLE metadata(version INTEGER NOT NULL)")
            conn.execute("INSERT INTO metadata VALUES(1)")
            conn.execute(
                "CREATE TABLE videos(path TEXT PRIMARY KEY, stamp TEXT NOT NULL, item TEXT NOT NULL, info TEXT NOT NULL)"
            )
            conn.execute("INSERT INTO videos VALUES('k', '[1,2]', '{}', '{}')")
            conn.execute(
                "CREATE TABLE video_frames(path TEXT NOT NULL, frame_order INTEGER NOT NULL, "
                "frame_index INTEGER NOT NULL, data TEXT NOT NULL, crops TEXT NOT NULL, features BLOB NOT NULL, "
                "PRIMARY KEY(path, frame_order))"
            )
            conn.execute("INSERT INTO video_frames VALUES('k', 0, 10, '{}', '[]', X'00')")
        elif legacy_schema == "videos_missing_info":
            conn.execute("CREATE TABLE metadata(signature TEXT NOT NULL)")
            conn.execute("INSERT INTO metadata VALUES(?)", (expected_sig_json,))
            conn.execute("CREATE TABLE videos(path TEXT PRIMARY KEY, stamp TEXT NOT NULL, item TEXT NOT NULL)")
            conn.execute("INSERT INTO videos VALUES('k', '[1,2]', '{}')")
            conn.execute(
                "CREATE TABLE video_frames(path TEXT NOT NULL, frame_order INTEGER NOT NULL, "
                "frame_index INTEGER NOT NULL, data TEXT NOT NULL, crops TEXT NOT NULL, features BLOB NOT NULL, "
                "PRIMARY KEY(path, frame_order))"
            )
            conn.execute("INSERT INTO video_frames VALUES('k', 0, 10, '{}', '[]', X'00')")
        elif legacy_schema == "video_frames_missing_features":
            conn.execute("CREATE TABLE metadata(signature TEXT NOT NULL)")
            conn.execute("INSERT INTO metadata VALUES(?)", (expected_sig_json,))
            conn.execute(
                "CREATE TABLE videos(path TEXT PRIMARY KEY, stamp TEXT NOT NULL, item TEXT NOT NULL, info TEXT NOT NULL)"
            )
            conn.execute("INSERT INTO videos VALUES('k', '[1,2]', '{}', '{}')")
            conn.execute(
                "CREATE TABLE video_frames(path TEXT NOT NULL, frame_order INTEGER NOT NULL, "
                "frame_index INTEGER NOT NULL, data TEXT NOT NULL, crops TEXT NOT NULL, "
                "PRIMARY KEY(path, frame_order))"
            )
            conn.execute("INSERT INTO video_frames VALUES('k', 0, 10, '{}', '[]')")
        elif legacy_schema == "videos_missing_primary_key":
            conn.execute("CREATE TABLE metadata(signature TEXT NOT NULL)")
            conn.execute("INSERT INTO metadata VALUES(?)", (expected_sig_json,))
            conn.execute("CREATE TABLE videos(path TEXT, stamp TEXT NOT NULL, item TEXT NOT NULL, info TEXT NOT NULL)")
            conn.execute("INSERT INTO videos VALUES('k', '[1,2]', '{}', '{}')")
            conn.execute(
                "CREATE TABLE video_frames(path TEXT NOT NULL, frame_order INTEGER NOT NULL, "
                "frame_index INTEGER NOT NULL, data TEXT NOT NULL, crops TEXT NOT NULL, features BLOB NOT NULL, "
                "PRIMARY KEY(path, frame_order))"
            )
            conn.execute("INSERT INTO video_frames VALUES('k', 0, 10, '{}', '[]', X'00')")

    cache = DinoV3VideoCache(state_path, "job-1", signature)
    assert cache.restore([video_path]) == {}

    with closing(sqlite3.connect(cache.path)) as conn:
        assert {r[1] for r in conn.execute("PRAGMA table_info(metadata)")} == {"signature"}
        assert {r[1] for r in conn.execute("PRAGMA table_info(videos)")} == {"path", "stamp", "item", "info"}
        assert {r[1] for r in conn.execute("PRAGMA table_info(video_frames)")} == {
            "path", "frame_order", "frame_index", "data", "crops", "features",
        }
        assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM video_frames").fetchone()[0] == 0

    assert cache.save(
        _make_video_item(video_path),
        [(0, 10, _make_frame_data([0]), [_make_crop(video_path, 10)])],
        _make_info([10]),
        stamp,
    ) is True
    assert str(video_path) in cache.restore([video_path])
    assert DinoV3VideoCache(state_path, "job-1", {"model": "v2"}).restore([video_path]) == {}


@pytest.mark.parametrize("failure_mode", [
    "drop_video_frames_column",
    "drop_videos_column",
    "drop_metadata_table",
    "truncate_file_after_init",
    "truncate_overflow_after_init",
    "query_malformed_error",
])
def test_restore_recovers_from_query_corruption_or_schema_changed_after_constructor(
    tmp_path, monkeypatch, failure_mode,
):
    state_path = tmp_path / "state.json"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    stamp = file_stamp(video_path)
    signature = {"model": "v1"}
    cache = DinoV3VideoCache(state_path, "job-1", signature)

    assert cache.save(
        _make_video_item(video_path),
        [(0, 10, _make_frame_data([0]), [_make_crop(video_path, 10)])],
        _make_info([10]),
        stamp,
    ) is True

    if failure_mode == "drop_video_frames_column":
        with closing(sqlite3.connect(cache.path)) as conn, conn:
            conn.execute("ALTER TABLE video_frames DROP COLUMN features")
    elif failure_mode == "drop_videos_column":
        with closing(sqlite3.connect(cache.path)) as conn, conn:
            conn.execute("ALTER TABLE videos DROP COLUMN info")
    elif failure_mode == "drop_metadata_table":
        with closing(sqlite3.connect(cache.path)) as conn, conn:
            conn.execute("DROP TABLE metadata")
    elif failure_mode == "truncate_file_after_init":
        cache.path.write_bytes(cache.path.read_bytes()[:64])
    elif failure_mode == "truncate_overflow_after_init":
        raw = cache.path.read_bytes()
        cache.path.write_bytes(raw[: len(raw) - 1024])
    elif failure_mode == "query_malformed_error":
        real_connect = sqlite3.connect
        state = {"failed": False}

        class FailingCursorConn:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *args, **kwargs):
                if not state["failed"] and "FROM video_frames" in sql:
                    state["failed"] = True
                    raise sqlite3.DatabaseError("database disk image is malformed")
                return self._inner.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                return self._inner.__exit__(exc_type, exc, tb)

        monkeypatch.setattr(
            media_cache.sqlite3,
            "connect",
            lambda *args, **kwargs: FailingCursorConn(real_connect(*args, **kwargs)),
        )

    progress: list[tuple[int, int]] = []
    assert cache.restore([video_path], progress_fn=lambda d, t: progress.append((d, t))) == {}
    assert progress[-1] == (1, 1)

    assert cache.save(
        _make_video_item(video_path),
        [(0, 10, _make_frame_data([0]), [_make_crop(video_path, 10)])],
        _make_info([10]),
        stamp,
    ) is True
    assert str(video_path) in cache.restore([video_path])


def test_real_lock_permission_and_io_errors_surface_without_destroying_cache(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    stamp = file_stamp(video_path)
    cache = DinoV3VideoCache(state_path, "job-1", {"model": "v1"})
    assert cache.save(
        _make_video_item(video_path),
        [(0, 10, _make_frame_data([0]), [_make_crop(video_path, 10)])],
        _make_info([10]),
        stamp,
    ) is True

    # Locked database surfaces OperationalError and preserves cached records.
    real_connect = sqlite3.connect
    lock_conn = real_connect(cache.path)
    try:
        lock_conn.execute("BEGIN EXCLUSIVE")
        with monkeypatch.context() as mp:
            mp.setattr(
                media_cache.sqlite3,
                "connect",
                lambda path, *args, **kwargs: real_connect(path, *args, timeout=0.05, **kwargs),
            )
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                DinoV3VideoCache(state_path, "job-1", {"model": "v1"})
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                cache.restore([video_path])
        lock_conn.rollback()
    finally:
        lock_conn.close()
    assert str(video_path) in cache.restore([video_path])

    # Simulated disk I/O error surfaces without resetting cache.
    class IoErrorConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            if "FROM videos" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._inner.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._inner.__exit__(exc_type, exc, tb)

    with monkeypatch.context() as mp:
        mp.setattr(
            media_cache.sqlite3,
            "connect",
            lambda *args, **kwargs: IoErrorConn(real_connect(*args, **kwargs)),
        )
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            cache.restore([video_path])
    assert str(video_path) in cache.restore([video_path])

    # Read-only cache surfaces OperationalError on write rather than deleting or chmodding.
    os.chmod(cache.path, 0o444)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            DinoV3VideoCache(state_path, "job-1", {"model": "v2"})
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            cache.save(
                _make_video_item(video_path),
                [(0, 10, _make_frame_data([0]), [_make_crop(video_path, 10)])],
                _make_info([10]),
                stamp,
            )
        assert cache.path.exists()
    finally:
        os.chmod(cache.path, 0o666)
    assert str(video_path) in cache.restore([video_path])

    # Access denied during recovery unlink surfaces PermissionError.
    cache.path.write_bytes(b"corrupt-header")
    real_unlink = Path.unlink

    def deny_cache_unlink(self, *args, **kwargs):
        if self == cache.path:
            raise PermissionError("Access is denied")
        return real_unlink(self, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(Path, "unlink", deny_cache_unlink)
        with pytest.raises(PermissionError, match="Access is denied"):
            DinoV3VideoCache(state_path, "job-1", {"model": "v1"})
