from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from system.backend import services
from system.backend.models import DetectionItem
from system.dinov3.classifier import DinoV3Prediction
from system.dinov3.observation_support import parse_dinov3_observation_id
from system.dinov3.processor import DinoV3CropCollector, DinoV3CropRecord
from system.dinov3 import mixed_media
from system.dinov3.video import video_frame_key


class Classifier:
    backend = "dinov3"
    model = SimpleNamespace(inference_recipe="production_dual")
    encoder = SimpleNamespace(device="cpu")

    def __init__(self):
        self.calls = []

    def classify_features(self, raw, **kwargs):
        self.calls.append((raw.copy(), kwargs))
        return [DinoV3Prediction(
            species="fox", accepted=True, best_known_species="fox", known_score=0.8,
            threshold=0.2, embedding=row.copy(), candidates=(), ridge_score=0.9, within_score=0.8,
        ) for row in raw]


def record(path, box_index=0, axis=0):
    feature = np.zeros(768, np.float32)
    feature[axis] = 1
    return DinoV3CropRecord(str(path), box_index, (4 + 30 * box_index, 4, 24 + 30 * box_index, 24), 0.9, feature)


def frame_data(count=1, digest="frame"):
    return dict(detection_completed=True, detection_box_count=count,
                detection_box_indices=list(range(count)),
                检测框=[{"边界框": [4 + 30 * i, 4, 24 + 30 * i, 24]} for i in range(count)],
                _frame_digest=digest)


def fixture(tmp_path, *, start="2026-11-16T08:00:10+00:00", count=1, frames=3):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"photo")
    video = tmp_path / "clip.avi"
    video.write_bytes(b"video")
    classifier = Classifier()
    collector = DinoV3CropCollector()
    photo_crops = [record(photo, i, i) for i in range(count)]
    collector.record_crops(photo_crops)
    photo_item = DetectionItem(filename=photo.name, path=str(photo), file_type="jpg",
        date_taken="2026-11-16T08:00:00", detection_data=frame_data(count))
    video_item = DetectionItem(filename=video.name, path=str(video), file_type="avi", date_taken=start,
        detection_data={"capture_time": {"date_taken": start, "timestamp_reliable": bool(start),
                                       "timestamp_source": "embedded"},
                        "sequence_completed": False, "sequence_pending": True})
    frame_records = [(i, 25 * i, frame_data(count, f"frame-{i}"),
                      [record(video_frame_key(video, 25 * i), j, j) for j in range(count)]) for i in range(frames)]
    info = dict(fps=25, width=96, height=64, total_frames=100, frame_points=[25 * i for i in range(frames)])
    detector = SimpleNamespace(dinov3_classifier=classifier, _dinov3_collector=collector)
    return detector, [photo_item, video_item], {str(video): (video_item, frame_records, info)}


def finalize(detector, items, videos, root, **kwargs):
    saved = []
    output = mixed_media.finalize_mixed_media_job(
        detector, items, root, video_sequences=videos,
        save_batch_fn=lambda pairs, root: saved.extend(pairs),
        apply_data_fn=services._legacy._apply_detection_data,
        build_video_data_fn=lambda path, info, frames: services._legacy._build_fast_video_detection_data(
            detector, path, info, frames),
        **kwargs,
    )
    return output, saved


def test_utc_video_and_local_photo_keep_independent_classification_sequences(tmp_path):
    detector, items, videos = fixture(tmp_path)
    output, saved = finalize(detector, items, videos, tmp_path)
    assert len(detector.dinov3_classifier.calls) == 1
    raw, context = detector.dinov3_classifier.calls[0]
    assert raw.shape == (4, 768)
    assert len(set(context["sequence_ids"])) == 2
    assert context["sequence_ids"][0] not in context["sequence_ids"][1:]
    assert "evidence_weights" not in context
    assert [item.path for item in output] == [item.path for item in items]
    assert len(saved) == 2 and all(path in [Path(i.path) for i in items] for path, _ in saved)
    assert all(item.detection_data["sequence_completed"] for item in output)
    video_boxes = output[1].detection_data["检测框"]
    assert [box["frame_index"] for box in video_boxes] == [0, 25, 50]
    assert [box["timestamp"] for box in video_boxes] == [0, 1, 2]
    for index, box in enumerate(video_boxes):
        source, box_index, _ = parse_dinov3_observation_id(box["observation_id"])
        assert source == Path(items[1].path) and box_index == index
    assert items[0].date_taken == output[0].date_taken
    assert detector._dinov3_collector.get_all_crops() == []


def test_same_frame_targets_keep_independent_video_and_photo_tracks(tmp_path):
    detector, items, videos = fixture(tmp_path, count=2)
    output, _ = finalize(detector, items, videos, tmp_path)
    context = detector.dinov3_classifier.calls[0][1]
    assert len(set(context["sequence_ids"])) == 4
    assert "evidence_weights" not in context
    assert len(output[1].detection_boxes) == 6
    assert output[1].detection_data["物种数量"] == "2"


@pytest.mark.parametrize("start", ["2026-11-16T01:00:00+00:00", "2026-11-15T23:00:00+00:00"])
def test_real_time_gap_separates_events_independent_of_file_order(tmp_path, start):
    detector, items, videos = fixture(tmp_path, start=start)
    output, _ = finalize(detector, items[::-1], videos, tmp_path)
    assert len({seq for _, context in detector.dinov3_classifier.calls for seq in context["sequence_ids"]}) == 2
    assert [item.path for item in output] == [item.path for item in items[::-1]]


def test_missing_video_time_refinalizes_cached_tracks_without_photo_time_context(tmp_path):
    detector, items, videos = fixture(tmp_path, start=None)
    items[1] = items[1].model_copy(update={"detection_data": {"sequence_completed": True,
        "sequence_pending": False, "sequence_status": "complete", "物种名称": "deer"}})
    videos[str(items[1].path)] = (items[1], videos[str(items[1].path)][1], videos[str(items[1].path)][2])
    output, saved = finalize(detector, items, videos, tmp_path)
    raw, context = detector.dinov3_classifier.calls[0]
    assert len(raw) == 4 and "evidence_weights" not in context
    assert output[1].detection_data["video_reference"]["source"] == "none"
    assert output[1].date_taken is None
    assert [path for path, _ in saved] == [Path(item.path) for item in items]


def test_other_camera_never_merges_with_photo(tmp_path):
    camera_a, camera_b = tmp_path / "camera-a", tmp_path / "camera-b"
    camera_a.mkdir()
    camera_b.mkdir()
    detector, items, videos = fixture(camera_a)
    moved = camera_b / "clip.avi"
    moved.write_bytes(Path(items[1].path).read_bytes())
    video_item, frames, info = videos.pop(items[1].path)
    items[1] = video_item.model_copy(update={"path": str(moved)})
    videos[str(moved)] = (items[1], frames, info)
    finalize(detector, items, videos, tmp_path)
    assert len({cam for _, context in detector.dinov3_classifier.calls for cam in context["camera_ids"]}) == 2


def test_cancelled_joint_summary_never_saves_final_results(tmp_path):
    detector, items, videos = fixture(tmp_path)
    output, saved = finalize(detector, items, videos, tmp_path, is_cancelled=lambda: True)
    assert saved == [] and detector.dinov3_classifier.calls == []
    assert all(not item.detection_data.get("sequence_completed") for item in output)


def test_past_video_trajectory_is_only_one_history_observation(tmp_path, monkeypatch):
    detector, items, videos = fixture(tmp_path, start="2026-11-15T23:00:00+00:00")
    captured = []
    def adapt(raw, history, **kwargs):
        captured.append(history.copy())
        return raw.copy()
    monkeypatch.setattr("system.dinov3.processor.adapt_history_for_job", adapt)
    finalize(detector, items, videos, tmp_path)
    assert (captured[0][0] >= 0).sum() == 1
    assert not (captured[0][1:] == 0).any(), "a later photo must not enter earlier video context"


def test_photo_only_event_in_a_mixed_job_keeps_unweighted_classifier_protocol(tmp_path):
    detector, items, videos = fixture(tmp_path, start="2026-11-16T01:00:00+00:00")
    finalize(detector, items, videos, tmp_path)
    assert len(detector.dinov3_classifier.calls) == 1
    raw, context = detector.dinov3_classifier.calls[0]
    assert len(raw) == 4 and "evidence_weights" not in context
    assert context["sequence_ids"][0] not in context["sequence_ids"][1:]


def test_changed_video_source_is_rejected_before_joint_classification(tmp_path):
    detector, items, videos = fixture(tmp_path)
    path = Path(items[1].path)
    info = videos[items[1].path][2]
    stamp = path.stat()
    info["source_stamp"] = [stamp.st_size, stamp.st_mtime_ns]
    path.write_bytes(b"replacement video")
    with pytest.raises(Exception, match="源文件.*变化"):
        finalize(detector, items, videos, tmp_path)
    assert detector.dinov3_classifier.calls == []


def test_photo_exif_offset_is_preserved_but_comparison_uses_recorded_clock():
    from system.metadata_extractor import ImageMetadataExtractor
    captured = ImageMetadataExtractor._get_date_from_exif(
        {36867: "2026:11:16 02:00:00", 36881: "+02:00"}, "photo.jpg")
    assert captured.utcoffset() == timedelta(hours=2)
    assert mixed_media.capture_time_camera_clock(captured).hour == 2


@pytest.mark.parametrize("when", ["before_summary", "after_digest_lookup"])
def test_deleted_photo_is_not_used_as_evidence_or_saved(tmp_path, monkeypatch, when):
    detector, items, videos = fixture(tmp_path)
    photo = Path(items[0].path)
    if when == "before_summary":
        photo.unlink()
    else:
        def digest(path):
            photo.unlink()
            return "old-content-digest"
        monkeypatch.setattr(detector._dinov3_collector, "image_digest", digest)
    output, saved = finalize(detector, items, videos, tmp_path)
    assert [item.path for item in output] == [items[1].path]
    assert detector.dinov3_classifier.calls[0][0].shape == (3, 768)
    assert [path for path, _ in saved] == [Path(items[1].path)]


def test_standalone_video_resume_requires_matching_source_and_inference_signature(tmp_path):
    path = tmp_path / "clip.avi"
    path.write_bytes(b"video")
    stamp = path.stat()
    item = DetectionItem(filename=path.name, path=str(path), file_type="avi", detection_data={
        "sequence_completed": True, "capture_time": {"timestamp_reliable": False},
        "video_source_stamp": [stamp.st_size, stamp.st_mtime_ns], "video_inference_signature": "model-and-settings-A",
    })
    assert mixed_media.completed_video_is_current(item, "model-and-settings-A")
    assert not mixed_media.completed_video_is_current(item, "model-and-settings-B")
    path.write_bytes(b"replaced video with new capture metadata")
    assert not mixed_media.completed_video_is_current(item, "model-and-settings-A")


@pytest.mark.parametrize("corruption", ["missing_frame", "nan_fps", "duplicate_frame"])
def test_incomplete_sampled_video_never_publishes_complete_joint_results(tmp_path, corruption):
    detector, items, videos = fixture(tmp_path)
    item, frames, info = videos[items[1].path]
    if corruption == "missing_frame":
        frames.pop()
    elif corruption == "nan_fps":
        info["fps"] = float("nan")
    else:
        frames[1] = frames[0]
    with pytest.raises(Exception, match="video|视频|FPS"):
        finalize(detector, items, videos, tmp_path)
    assert detector.dinov3_classifier.calls == []
