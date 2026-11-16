from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from system.backend import services
from system.backend.models import CreateJobRequest, ProcessingOptions
from system.dino.image_processor import ImageProcessor
from system.dinov3.classifier import DinoV3Prediction


class Encoder:
    device = "cpu"

    def __init__(self):
        self.batch_sizes = []

    def encode(self, crops, *, array_color):
        assert array_color == "rgb"
        self.batch_sizes.append(len(crops))
        features = np.zeros((len(crops), 768), np.float32)
        for i, crop in enumerate(crops):
            features[i, 0] = 1
            features[i, 1] = float(crop[0, 0, 0]) / 255
        return features / np.linalg.norm(features, axis=1, keepdims=True)


class Classifier:
    backend = "dinov3"
    names = {0: "fox"}

    def __init__(self, accepted=True):
        self.encoder = Encoder()
        self.calls = []
        self.accepted = accepted

    def classify_crops(self, *args, **kwargs):
        pytest.fail("Video crops must be encoded without a redundant per-batch classifier pass")

    def classify_features(self, features, **kwargs):
        self.calls.append((features.copy(), kwargs))
        return [DinoV3Prediction(
            species="fox", accepted=self.accepted, best_known_species="fox",
            known_score=0.8 if self.accepted else -0.5, threshold=0.2,
            embedding=row.copy(), candidates=(), ridge_score=0.9, within_score=0.8,
        ) for row in features]


def make_detector(classifier, *, boxes=1):
    detector = ImageProcessor(None)
    detector.load_dino_classifier(classifier)
    def preload(frames):
        return list(range(len(frames))), frames, [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    detector._legacy = SimpleNamespace(
        preload_frame_data=preload, translation_dict={},
        _determine_device=lambda half: ("cpu", False), _sync_device=lambda device: None,
    )
    def detect(frames, **kwargs):
        return [SimpleNamespace(names={0: "animal"}, boxes=[SimpleNamespace(
            xyxy=np.asarray([[4. + i * 30, 4., 24. + i * 30, 24.]]),
            conf=np.asarray(0.9), cls=np.asarray(0),
        ) for i in range(boxes)]) for frame in frames]
    detector.model = detect
    return detector


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (96, 64))
    assert writer.isOpened()
    try:
        for i in range(100):
            writer.write(np.full((64, 96, 3), [20, 50, i * 2], np.uint8))
    finally:
        writer.release()
    return path


def run_video(detector, paths, monkeypatch, *, batch_size=2, cancelled=None):
    saved = []
    published = []
    monkeypatch.setattr(services._legacy, "_save_detection_data_batch", lambda pairs, root: saved.extend(pairs))
    request = CreateJobRequest(input_dir=str(paths[0].parent), options=ProcessingOptions(
        batch_size=batch_size, vid_stride=3, video_mode="fast", use_augment=False,
    ))
    results = services._legacy._detect_video_fast_batch(
        detector, paths, [services._build_fast_metadata_item(p) for p in paths],
        request, paths[0].parent, cancelled=cancelled,
        on_item_ready=lambda index, result: published.append(result),
    )
    return results, saved, published


@pytest.mark.parametrize("accepted", [True, False])
def test_sampled_video_encodes_once_and_finalizes_all_frames_without_jpegs(video, monkeypatch, accepted):
    classifier = Classifier(accepted)
    detector = make_detector(classifier)
    monkeypatch.setattr(cv2, "imread", lambda *a: pytest.fail("intermediate JPEG read"))
    monkeypatch.setattr(cv2, "imwrite", lambda *a: pytest.fail("intermediate JPEG write"))
    results, saved, published = run_video(detector, [video], monkeypatch)
    assert results[0].error is None, results[0].error
    assert len(classifier.calls) == 1
    features, context = classifier.calls[0]
    assert features.shape == (3, 768)
    assert np.all(np.diff(features[:, 1]) > 0), "earlier frame features were overwritten"
    assert sum(classifier.encoder.batch_sizes) == 3
    assert len(set(context["sequence_ids"])) == 1
    data = results[0].detection_data
    assert data["sampled_frames"] == [25, 50, 75]
    assert [box["frame_index"] for box in data["检测框"]] == [25, 50, 75]
    assert [box["timestamp"] for box in data["检测框"]] == [1., 2., 3.]
    assert results[0].species == (["fox"] if accepted else [])
    assert data["物种名称"] == ("fox" if accepted else "Unknown")
    assert [box["物种"] for box in data["检测框"]] == ["fox" if accepted else "Unknown"] * 3
    assert data["sequence_completed"] is True and data["sequence_pending"] is False
    assert all(box["sequence_status"] == "complete" for box in data["检测框"])
    assert len(saved) == len(published) == 1
    assert published[0].detection_data["sequence_completed"] is True
    assert detector._dinov3_collector.get_all_crops() == []


def test_same_frame_targets_and_different_videos_keep_separate_tracks(video, monkeypatch):
    second = video.with_name("second.avi")
    second.write_bytes(video.read_bytes())
    classifier = Classifier()
    detector = make_detector(classifier, boxes=2)
    results, saved, _ = run_video(detector, [video, second], monkeypatch, batch_size=4)
    assert all(item.error is None for item in results), [item.error for item in results]
    assert len(saved) == len(classifier.calls) == 2
    first_tracks, second_tracks = [set(call[1]["sequence_ids"]) for call in classifier.calls]
    assert len(first_tracks) == len(second_tracks) == 2
    assert first_tracks.isdisjoint(second_tracks)
    assert [len(item.detection_boxes) for item in results] == [6, 6]
    assert all(item.detection_data["物种数量"] == "2" for item in results)


def test_empty_video_finishes_without_invoking_encoder_or_head(video, monkeypatch):
    classifier = Classifier()
    detector = make_detector(classifier, boxes=0)
    results, saved, _ = run_video(detector, [video], monkeypatch)
    assert results[0].error is None
    assert results[0].detection_data["sequence_completed"] is True
    assert results[0].species == [] and results[0].detection_boxes == []
    assert classifier.calls == [] and classifier.encoder.batch_sizes == []
    assert len(saved) == 1


def test_cancelled_video_discards_unsaved_features(video, monkeypatch):
    classifier = Classifier()
    detector = make_detector(classifier)
    saved = []
    monkeypatch.setattr(services._legacy, "_save_detection_data_batch", lambda pairs, root: saved.extend(pairs))
    request = CreateJobRequest(input_dir=str(video.parent), options=ProcessingOptions(batch_size=1, vid_stride=3))
    with pytest.raises(RuntimeError, match="用户取消"):
        services._legacy._detect_video_fast_batch(
            detector, [video], [services._build_fast_metadata_item(video)], request, video.parent,
            cancelled=lambda: bool(classifier.encoder.batch_sizes),
        )
    assert saved == [] and classifier.calls == []
    assert detector._dinov3_collector.get_all_crops() == []


def test_bad_video_and_failed_head_do_not_lose_other_completed_videos(video, monkeypatch):
    bad = video.with_name("bad.avi")
    bad.write_bytes(b"broken video")
    classifier = Classifier()
    detector = make_detector(classifier)
    results, saved, _ = run_video(detector, [bad, video], monkeypatch)
    assert results[0].error and results[1].error is None
    assert [path for path, _ in saved] == [video]
    assert detector._dinov3_collector.get_all_crops() == []

    second = video.with_name("second.avi")
    second.write_bytes(video.read_bytes())
    original = classifier.classify_features
    def fail_one(features, **kwargs):
        if not classifier.calls:
            classifier.calls.append((features, kwargs))
            raise RuntimeError("head failed")
        return original(features, **kwargs)
    classifier.calls.clear()
    classifier.classify_features = fail_one
    results, saved, _ = run_video(detector, [video, second], monkeypatch)
    assert sum(item.error is None for item in results) == 1
    assert len(saved) == 1
    assert detector._dinov3_collector.get_all_crops() == []


def test_automatic_oom_retry_keeps_each_frame_once(video, monkeypatch):
    from system.backend.inference_batching import InferenceBatcher
    from system.dinov3.processor import DinoV3CropRecord
    monkeypatch.setattr("system.backend.inference_batching._automatic_batch_size", lambda options: 4)
    classifier = Classifier()
    detector = make_detector(classifier)
    encode = classifier.encoder.encode
    attempts = []
    def encode_with_oom(crops, **kwargs):
        attempts.append(len(crops))
        if len(crops) > 1:
            raise RuntimeError("CUDA out of memory")
        return encode(crops, **kwargs)
    classifier.encoder.encode = encode_with_oom
    # A pending photo's records must not be removed by video cleanup.
    photo = DinoV3CropRecord("photo.jpg", 0, (1, 2, 3, 4), 0.9, np.ones(768, np.float32))
    detector._dinov3_collector.record_crop(photo)
    saved = []
    monkeypatch.setattr(services._legacy, "_save_detection_data_batch", lambda pairs, root: saved.extend(pairs))
    request = CreateJobRequest(input_dir=str(video.parent), options=ProcessingOptions(batch_size=0, vid_stride=3))
    results = services._legacy._detect_video_fast_batch(
        detector, [video], [services._build_fast_metadata_item(video)], request, video.parent,
        inference_batcher=InferenceBatcher(request.options),
    )
    assert results[0].error is None, results[0].error
    assert max(attempts) > 1 and attempts[-1] == 1
    assert classifier.calls[0][0].shape == (3, 768)
    assert [box["frame_index"] for box in results[0].detection_data["检测框"]] == [25, 50, 75]
    assert len(saved) == 1
    assert detector._dinov3_collector.get_all_crops() == [photo]


def test_missing_crop_fails_instead_of_publishing_partial_sequence(video, monkeypatch):
    classifier = Classifier()
    detector = make_detector(classifier)
    detector._crop_single_box = lambda task: None
    results, saved, _ = run_video(detector, [video], monkeypatch)
    assert results[0].error and "特征不完整" in results[0].error
    assert saved == [] and classifier.calls == []


def test_video_observation_preview_reads_the_correct_sampled_frame(video, monkeypatch):
    from system.dinov3.observation_support import make_dinov3_observation_id, render_dinov3_observation_example
    observation = make_dinov3_observation_id(video, 1, (4, 4, 24, 24))
    monkeypatch.setattr(services, "_load_detection_data_for_path", lambda *args: {
        "检测框": [{"frame_index": 25}, {"frame_index": 75, "边界框": [4, 4, 24, 24]}],
    })
    encoded = render_dinov3_observation_example(observation)
    crop = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    assert float(crop[:, :, 2].mean()) == pytest.approx(150, abs=5)


def test_litert_detector_fast_video_needs_no_torch_import(video, monkeypatch):
    import builtins
    class LiteDetector:
        def __init__(self, *args, **kwargs):
            self.names = {0: "animal"}
        def predict(self, image, **kwargs):
            return [SimpleNamespace(x1=4, y1=4, x2=24, y2=24, confidence=0.9, class_id=0)]
    monkeypatch.setattr("system.dino.litert_image_processor.LiteRtDetector", LiteDetector)
    path = video.with_suffix(".tflite")
    path.write_bytes(b"model boundary double")
    classifier = Classifier()
    detector = ImageProcessor(str(path))
    detector.load_dino_classifier(classifier)
    original_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "ultralytics"}:
            raise ModuleNotFoundError(name=name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    results, saved, _ = run_video(detector, [video], monkeypatch)
    assert results[0].error is None, results[0].error
    assert len(results[0].detection_boxes) == 3
    assert len(classifier.calls) == 1 and len(saved) == 1
    assert results[0].detection_data["sequence_completed"] is True
