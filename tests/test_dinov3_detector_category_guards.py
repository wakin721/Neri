from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from system.backend import services
from system.backend.models import DetectionItem
from system.dino.image_processor import ImageProcessor
from system.dinov3.classifier import DinoV3Prediction
from system.dinov3.processor import DinoV3CropCollector, DinoV3CropRecord, finalize_photo_job


def legacy_data(two_boxes=True):
    boxes = [dict(物种="家羊", 置信度=0.92598, 边界框=[0, 1, 20, 30], accepted=True,
        known_score=0.137, threshold=-1.757, 候选项=[dict(name="家羊", conf=0.92598,
        source="dinov3_ridge_within", gate_mode="sequence", accepted=True, known_score=0.137, threshold=-1.757)])]
    person = dict(物种="家羊", 置信度=0.00124, 边界框=[50, 1, 70, 30], accepted=True,
        known_score=-0.719, threshold=-1.757, observation_id="dinov3:unchanged-identity",
        候选项=[dict(name="家羊", conf=0.00124, source="dinov3_ridge_within", gate_mode="sequence",
                    accepted=True, known_score=-0.719, threshold=-1.757)])
    boxes = [*boxes, person] if two_boxes else [person]
    return {"sequence_completed": True, "sequence_status": "complete", "检测框": boxes,
        "names_map": {"0": "animal", "1": "person", "2": "vehicle"},
        "all_classes": [0.0, 1.0] if two_boxes else [1.0],
        "all_confidences": [0.95459, 0.70166] if two_boxes else [0.83984],
        "物种名称": "家羊", "物种数量": "2" if two_boxes else "1", "最低置信度": "0.001"}


@pytest.mark.parametrize("two_boxes", [True, False])
def test_legacy_screenshot_result_preserves_detector_person_without_mutating_source(two_boxes):
    data = legacy_data(two_boxes)
    before = deepcopy(data)
    item = DetectionItem(filename="photo.jpg", path="photo.jpg", file_type="jpg")
    updated = services._legacy._apply_detection_data(item, data)
    person = updated.detection_boxes[-1]
    assert person["species"] == "人员"
    assert person["confidence"] == pytest.approx(0.70166 if two_boxes else 0.83984)
    assert updated.detection_data["检测框"][-1]["observation_id"] == "dinov3:unchanged-identity"
    assert updated.detection_data["物种名称"] == ("家羊,人员" if two_boxes else "人员")
    assert updated.detection_data["物种数量"] == ("1,1" if two_boxes else "1")
    assert data == before


@pytest.mark.parametrize("protection", ["box", "image", "validated"])
def test_category_compatibility_never_overwrites_human_labels(protection):
    data = legacy_data()
    if protection == "box":
        data["检测框"][-1]["feedback_status"] = "confirmed"
    elif protection == "image":
        data["最低置信度"] = "人工校验"
    item = DetectionItem(filename="photo.jpg", path="photo.jpg", file_type="jpg", validated=protection == "validated")
    result = services._legacy._apply_detection_data(item, data)
    assert result.detection_boxes[-1]["species"] == "家羊"


def test_species_detector_numeric_class_one_is_not_assumed_person():
    data = legacy_data()
    data["names_map"] = {"0": "fox", "1": "sheep"}
    result = services._legacy._apply_detection_data(
        DetectionItem(filename="photo.jpg", path="photo.jpg", file_type="jpg"), data)
    assert result.detection_boxes[-1]["species"] == "家羊"


def test_human_image_annotation_is_not_reinterpreted_as_detector_output():
    data = legacy_data(False)
    data["image_annotation"] = {"source": "human", "species_name": "工作人员"}
    result = services._legacy._apply_detection_data(
        DetectionItem(filename="photo.jpg", path="photo.jpg", file_type="jpg"), data)
    assert result.detection_boxes[0]["species"] == "家羊"


def test_legacy_vehicle_category_and_confidence_are_retained():
    data = legacy_data(False)
    data["all_classes"] = [2.0]
    result = services._legacy._apply_detection_data(
        DetectionItem(filename="photo.jpg", path="photo.jpg", file_type="jpg"), data)
    assert result.species == ["车辆"]
    assert result.detection_boxes[0]["confidence"] == pytest.approx(.83984)


def test_collected_crop_keeps_detector_category_in_deferred_mode():
    encoder = SimpleNamespace(encode=lambda crops, **kw: np.ones((len(crops), 768), np.float32))
    detector = ImageProcessor(None)
    detector.load_dino_classifier(SimpleNamespace(backend="dinov3", encoder=encoder))
    detector._dinov3_current_paths = ("one.jpg",)
    detector._crop_single_box = lambda task: (task[0], task[1], task[3])
    boxes = [SimpleNamespace(cls=np.asarray(i), conf=np.asarray(.9), xyxy=np.asarray([[1, 1, 10, 10]])) for i in (0, 1, 2)]
    detector._classify_dino_crops([SimpleNamespace(names={0:"animal",1:"person",2:"vehicle"}, boxes=boxes)],
                                [np.zeros((20, 20, 3), np.uint8)], defer_sequence=True)
    assert [crop.detector_category for crop in detector._dinov3_collector.get_all_crops()] == ["animal", "person", "vehicle"]


@pytest.mark.parametrize("animal_category", ["animal", None])
def test_experiment_photo_targets_and_detector_categories_cannot_borrow_sheep_vote(tmp_path, animal_category):
    paths = [tmp_path / "sheep.jpg", tmp_path / "person.jpg"]
    collector = DinoV3CropCollector()
    items = []
    for index, path in enumerate(paths):
        path.write_bytes(b"photo")
        raw = np.zeros(768, np.float32); raw[0] = 1
        crop = DinoV3CropRecord(str(path), 0, (1, 1, 20, 30), .84, raw,
                                detector_category=animal_category if index == 0 else "person")
        collector.record_crop(crop)
        items.append(DetectionItem(filename=path.name, path=str(path), file_type="jpg", date_taken=f"2026-01-08T17:31:2{index}",
            detection_data={"detection_completed":True,"detection_box_indices":[0],"detection_box_count":1,"检测框":[{}]}))
    calls = []
    def classify(raw, **kwargs):
        calls.append(kwargs)
        # Deliberately wrong classifier: detector person must still remain person.
        return [DinoV3Prediction("家羊", True, "家羊", .2, -1.7, row, (), .8, .8) for row in raw]
    classifier = SimpleNamespace(backend="dinov3", encoder=SimpleNamespace(device="cpu"),
        model=SimpleNamespace(inference_recipe="experiment_full"), classify_features=classify)
    detector = SimpleNamespace(dinov3_classifier=classifier, _dinov3_collector=collector)
    output = finalize_photo_job(detector, items, tmp_path, apply_data_fn=services._legacy._apply_detection_data)
    assert calls[0]["sequence_ids"][0] != calls[0]["sequence_ids"][1]
    assert [item.species for item in output] == [["家羊"], ["人员"]]
    assert output[1].detection_boxes[0]["confidence"] == pytest.approx(.84)


def test_video_cache_roundtrip_preserves_detector_category(tmp_path):
    from system.dinov3.media_cache import DinoV3VideoCache, file_stamp
    from system.dinov3.video import video_frame_key
    path = tmp_path / "clip.avi"; path.write_bytes(b"video")
    item = DetectionItem(filename=path.name, path=str(path), file_type="avi")
    data = dict(detection_completed=True, detection_box_count=1, detection_box_indices=[0], 检测框=[{}])
    crop = DinoV3CropRecord(video_frame_key(path, 10), 0, (1, 2, 20, 30), .9,
        np.ones(768, np.float32), detector_category="person")
    cache = DinoV3VideoCache(tmp_path / "state.json", "one", {"model":"v1"})
    assert cache.save(item, [(0, 10, data, [crop])], dict(frame_points=[10], fps=25), file_stamp(path))
    assert cache.restore([path])[str(path)][1][0][3][0].detector_category == "person"


@pytest.mark.parametrize("human_validated", [False, True])
def test_source_local_loader_repairs_only_automatic_records(tmp_path, monkeypatch, human_validated):
    import json, sqlite3
    path = tmp_path / "photo.jpg"; path.write_bytes(b"photo")
    database = tmp_path / "detections.db"
    original = json.dumps(legacy_data(False), ensure_ascii=False)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE detections(base_name TEXT,image_filename TEXT,detection_json TEXT)")
        connection.execute("CREATE TABLE validation(image_filename TEXT,is_validated INTEGER)")
        connection.execute("INSERT INTO detections VALUES(?,?,?)", ("photo", "photo.jpg", original))
        connection.execute("INSERT INTO validation VALUES(?,?)", ("photo.jpg", int(human_validated)))
    monkeypatch.setattr(services._legacy, "_candidate_detection_dbs_for_roots", lambda *a, **kw: [database])
    loaded = services._legacy._load_detection_data_for_path(path, [tmp_path])
    assert loaded["检测框"][0]["物种"] == ("家羊" if human_validated else "人员")
    from system.backend.models import ValidationMarkRequest
    for action in ("correct", "unverified"):
        request = ValidationMarkRequest(input_path=str(tmp_path), file_path=str(path), action=action)
        update, _ = services._legacy._build_validation_update(request, path, legacy_data(False))
        assert update["检测框"][0]["物种"] == ("家羊" if human_validated else "人员")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT detection_json FROM detections").fetchone()[0] == original


def test_job_history_snapshot_normalizes_legacy_people_without_rewriting_saved_job():
    from system.backend.models import JobSummary, JobState
    item = DetectionItem(filename="photo.jpg", path="photo.jpg", file_type="jpg", detection_data=legacy_data(False))
    job = JobSummary(id="one", state=JobState.COMPLETED, input_dir=".", created_at="2026-11-17T00:00:00Z",
                     updated_at="2026-11-17T00:00:00Z", results=[item])
    manager = services.ProcessingJobManager.__new__(services.ProcessingJobManager)
    manager._jobs = {"one": job}
    manager._active_job_ids = set()
    snapshot = manager._snapshot_unlocked("one")
    assert snapshot.results[0].species == ["人员"]
    assert job.results[0].detection_data["检测框"][0]["物种"] == "家羊"


def test_legacy_video_quantity_counts_targets_per_frame_and_ignores_confirmed_empty_boxes():
    from system.dinov3.detection_categories import normalize_detector_categories
    data = legacy_data(False)
    prototype = data["检测框"][0]
    data["检测框"] = [{**deepcopy(prototype), "frame_index": index} for index in (10, 20, 30)]
    data["all_classes"] = [1, 1, 1]; data["all_confidences"] = [.8, .8, .8]
    normalized = normalize_detector_categories(data)
    assert normalized["物种数量"] == "1" and normalized["物种名称"] == "人员"
    data = legacy_data()
    data["检测框"][0].update(物种="空", feedback_status="confirmed")
    normalized = normalize_detector_categories(data)
    assert normalized["物种名称"] == "人员" and normalized["物种数量"] == "1"
