from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from system.dinov3 import processor
from tests.test_dinov3_mixed_sequences import fixture, finalize


@pytest.mark.parametrize("recipe", ["production_dual", "experiment_full"])
def test_video_tracks_never_share_classification_sequence_with_photos_or_other_targets(tmp_path, recipe):
    detector, items, videos = fixture(tmp_path, count=2)
    detector.dinov3_classifier.model = SimpleNamespace(inference_recipe=recipe)
    output, saved = finalize(detector, items, videos, tmp_path)
    calls = detector.dinov3_classifier.calls
    assert len(calls) == 1
    raw, context = calls[0]
    photo_ids = set(context["sequence_ids"][:2])
    video_ids = context["sequence_ids"][2:]
    assert len(set(video_ids)) == 2 and photo_ids.isdisjoint(video_ids)
    assert "evidence_weights" not in context
    assert len({box["track_id"] for box in output[1].detection_data["检测框"]}) == 2
    assert output[1].detection_data["video_reference"]["source"] == "video"


def test_video_without_any_reference_time_keeps_its_own_tracks_and_no_history(tmp_path, monkeypatch):
    detector, items, videos = fixture(tmp_path, start=None, count=2)
    detector.dinov3_classifier.model = SimpleNamespace(inference_recipe="experiment_full")
    history_rows = []
    def adapt(raw, history, **kwargs):
        history_rows.append(history.copy())
        return raw.copy()
    monkeypatch.setattr(processor, "adapt_history_for_job", adapt)
    output, _ = finalize(detector, items, videos, tmp_path)
    _, context = detector.dinov3_classifier.calls[0]
    assert len(set(context["sequence_ids"][2:])) == 2
    assert set(context["sequence_ids"][:2]).isdisjoint(context["sequence_ids"][2:])
    assert not (history_rows[0][2:] >= 0).any()
    assert output[1].date_taken is None
    assert output[1].detection_data["video_reference"]["source"] == "none"
    assert output[1].detection_data["sequence_completed"] is True


def test_burst_time_is_an_approximate_cutoff_and_does_not_fabricate_capture_time(tmp_path, monkeypatch):
    detector, items, videos = fixture(tmp_path, start=None)
    items[0].detection_data["burst_group_id"] = "capture-1"
    items[1].detection_data["burst_group_id"] = "capture-1"
    history_rows = []
    def adapt(raw, history, **kwargs):
        history_rows.append(history.copy())
        return raw.copy()
    monkeypatch.setattr(processor, "adapt_history_for_job", adapt)
    output, _ = finalize(detector, items, videos, tmp_path)
    assert output[1].date_taken is None
    reference = output[1].detection_data["video_reference"]
    assert reference["source"] == "burst_group" and reference["precision"] == "group_anchor"
    assert reference["reference_time"] == "2026-11-16T08:00:00+00:00"
    assert not (history_rows[0][1:] >= 0).any(), "same-group photos must not be treated as earlier history"


def test_own_video_time_can_use_earlier_photos_but_never_future_photos_or_its_own_frames(tmp_path, monkeypatch):
    detector, items, videos = fixture(tmp_path)
    past = items[0]
    future_path = tmp_path / "future.jpg"
    future_path.write_bytes(b"future")
    future = past.model_copy(update={"path": str(future_path), "filename": future_path.name,
                                     "date_taken": "2026-11-16T08:00:30"})
    crop = detector._dinov3_collector.get_crops_for_path(past.path)[0]
    from dataclasses import replace
    detector._dinov3_collector.record_crop(replace(crop, image_path=str(future_path)))
    captured = []
    def adapt(raw, history, **kwargs):
        captured.append(history.copy())
        return raw.copy()
    monkeypatch.setattr(processor, "adapt_history_for_job", adapt)
    finalize(detector, [past, future, items[1]], videos, tmp_path)
    video_history = captured[0][2:]
    assert all(0 in row for row in video_history)
    assert not (video_history == 1).any()
    assert not (video_history >= 2).any()


def test_approximate_burst_video_cannot_become_history_for_later_photos(tmp_path, monkeypatch):
    detector, items, videos = fixture(tmp_path, start=None)
    items[0].detection_data["burst_group_id"] = "capture-1"
    items[1].detection_data["burst_group_id"] = "capture-1"
    future_path = tmp_path / "later.jpg"
    future_path.write_bytes(b"later")
    future = items[0].model_copy(update={"path": str(future_path), "date_taken": "2026-11-16T09:00:00",
                                       "detection_data": dict(items[0].detection_data)})
    future.detection_data.pop("burst_group_id")
    from dataclasses import replace
    detector._dinov3_collector.record_crop(replace(
        detector._dinov3_collector.get_crops_for_path(items[0].path)[0], image_path=str(future_path)))
    captured = []
    monkeypatch.setattr(processor, "adapt_history_for_job", lambda raw, history, **kw: captured.append(history.copy()) or raw)
    finalize(detector, [items[0], future, items[1]], videos, tmp_path)
    assert 0 in captured[0][1]
    assert not (captured[0][1] >= 2).any(), "approximate time is not reliable historical ordering"


@pytest.mark.parametrize("timed", [False, True])
def test_rejection_regrouping_preserves_video_tracks_across_nested_directories(tmp_path, timed):
    photo_path = tmp_path / "CamA" / "sub1" / "photo.jpg"
    video_path = tmp_path / "CamA" / "sub2" / "clip.avi"
    records = [dict(crop_idx=0, camera_id="camera-a", path=str(photo_path), box_index=0,
                    date_taken="2026-11-16T00:00:00+00:00", event_id="event:camera-a:time",
                    track_id="event:camera-a:time", sequence_id="event:camera-a:time")]
    for index in (1, 2):
        records.append(dict(crop_idx=index, camera_id="camera-a", path=f"{video_path}.neri-frame-{index:012d}",
            box_index=0, date_taken="2026-11-16T00:00:10+00:00" if timed else None,
            event_id="video:clip-a", track_id="video:clip-a:track_00", sequence_id="video:clip-a:track_00"))
    grouped = processor._rejection_event_groups(records)
    video_groups = [row for row in grouped if row["crop_idx"] > 0]
    assert {row["track_id"] for row in video_groups} == {"video:clip-a:track_00"}
    assert {row["event_id"] for row in video_groups} == {"video:clip-a"}


def test_explicit_burst_membership_survives_final_persistence(tmp_path):
    detector, items, videos = fixture(tmp_path, start=None)
    for item in items:
        item.detection_data["burst_group_id"] = "capture-persistent"
    output, _ = finalize(detector, items, videos, tmp_path)
    assert output[-1].detection_data["burst_group_id"] == "capture-persistent"
