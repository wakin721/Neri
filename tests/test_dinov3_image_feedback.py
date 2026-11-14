"""Whole-image human counts and crop learning must remain distinct."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from system.backend import services
from system.backend.dinov3_image_feedback import prepare_image_feedback
from system.backend import dinov3_learning as learning
from system.backend.models import ValidationMarkRequest
from system.dinov3.observation_support import (
    apply_dinov3_box_feedback, make_dinov3_observation_id, _confirmed_empty_feedback,
)
from system.confidence import dinov3_final_species
from tests.test_dinov3_box_explanation_and_feedback import deployed as box_deployed, _box, _write_image, _write_detection_db
from tests.test_dinov3_deployment import deployed
from tests.test_dinov3_learning import ready, loaded


@pytest.fixture(autouse=True)
def isolate_workers(monkeypatch):
    monkeypatch.setattr(learning, "schedule_learning", lambda *args: None)
    import system.training
    monkeypatch.setattr(system.training, "get_queue", lambda: SimpleNamespace(
        enqueue_many_async=lambda values: None, enqueue_async=lambda *args, **kwargs: None))


def data(species="C", count="4", boxes=2):
    return {"物种名称": species, "物种数量": count, "最低置信度": "人工校验",
            "sequence_completed": True, "检测框": [_box("Unknown", accepted=False, bbox=(10+i*60, 10, 70+i*60, 80)) for i in range(boxes)]}


@pytest.mark.parametrize("species,count,box_count,status,learned", [
    ("C", "4", 2, "eligible", 2), ("C", "2", 2, "eligible", 2),
    ("C", "1", 2, "count_exceeded", 0), ("C,D", "1,1", 2, "multiple_species", 0),
    ("C、D", "2", 2, "multiple_species", 0), ("C", "4", 0, "no_boxes", 0),
    ("C", "", 2, "invalid_count", 0), ("C", "-1", 2, "invalid_count", 0),
    ("Unknown", "4", 2, "unknown_species", 0), ("C", "0", 2, "empty", 0),
])
def test_bulk_admission_table(tmp_path, species, count, box_count, status, learned):
    original = data(species, count, box_count)
    result = deepcopy(original)
    staged = prepare_image_feedback(result, tmp_path/"photo.jpg", "update", "bulk")
    assert result["learning_feedback"]["status"] == status
    assert sum(action == "update" for box, action in staged) == learned
    assert result["物种数量"] == count
    if status in {"eligible", "count_exceeded", "invalid_count"}:
        assert all(b["物种"] == species for b in result["检测框"])
    if status == "count_exceeded":
        assert all(b["feedback_status"] == "image_label" and b["learning_eligible"] is False for b in result["检测框"])
        assert all(dinov3_final_species(b) == "C" for b in result["检测框"])
    if status == "multiple_species":
        assert all(b["物种"] == "Unknown" for b in result["检测框"])
    assert original == data(species, count, box_count)


def test_bulk_opt_out_and_false_positive_are_not_added_back(tmp_path):
    result = data(count="2", boxes=3)
    result["检测框"][0].update(物种="空", feedback_status="confirmed", label_source="box_annotation")
    staged = prepare_image_feedback(result, tmp_path/"photo.jpg", "update", "bulk")
    assert result["learning_feedback"]["box_count"] == 2
    assert sum(action == "update" for _, action in staged) == 2
    assert result["检测框"][0]["物种"] == "空"


def test_unverify_then_confirm_again_can_learn(tmp_path):
    result = data()
    prepare_image_feedback(result, tmp_path/"photo.jpg", "update", "first")
    prepare_image_feedback(result, tmp_path/"photo.jpg", "unverified", "undo")
    assert "image_annotation" not in result
    assert all(b["物种"] == "Unknown" for b in result["检测框"])
    result.update(物种名称="C", 物种数量="4", 最低置信度="人工校验")
    assert all(action == "update" for _, action in prepare_image_feedback(result, tmp_path/"photo.jpg", "update", "again"))


def test_multi_species_retracts_inferred_labels(tmp_path):
    result = data()
    prepare_image_feedback(result, tmp_path/"photo.jpg", "update", "first")
    result.update(物种名称="C,D", 物种数量="1,3")
    staged = prepare_image_feedback(result, tmp_path/"photo.jpg", "update", "multi")
    assert all(action == "unverified" for _, action in staged)
    assert all(b["物种"] == "Unknown" for b in result["检测框"])
    assert result["物种数量"] == "1,3"


def test_invalid_crop_never_enters_bulk_learning(tmp_path):
    result = data()
    result["检测框"][0]["边界框"] = [1, 1, 1, 5]
    staged = prepare_image_feedback(result, tmp_path/"photo.jpg", "update", "invalid")
    assert result["learning_feedback"]["status"] == "invalid_boxes"
    assert all(action == "unverified" for _, action in staged)


def test_explicit_box_opt_out_survives_bulk_label(tmp_path):
    result = data()
    result["检测框"][0]["learning_excluded"] = True
    result["检测框"][0]["label_source"] = "box_annotation"
    staged = prepare_image_feedback(result, tmp_path/"photo.jpg", "update", "bulk")
    assert sum(action == "update" for _, action in staged) == 1
    assert result["检测框"][0]["learning_eligible"] is False


def test_learning_opt_out_does_not_reduce_detection_count_for_admission(tmp_path):
    result = data(count="1", boxes=2)
    result["检测框"][0]["learning_excluded"] = True
    result["检测框"][0]["label_source"] = "box_annotation"
    staged = prepare_image_feedback(result, tmp_path/"photo.jpg", "update", "bulk")
    assert result["learning_feedback"]["box_count"] == 2
    assert result["learning_feedback"]["status"] == "count_exceeded"
    assert all(action == "unverified" for _, action in staged)


def _photo(tmp_path):
    source = tmp_path/"photo.jpg"
    _write_image(source, 130)
    _write_detection_db(tmp_path, [(source, data("A", "2"))])
    return source


def _mark(manifest, source, species, count, operation="bulk", action="update"):
    return services.mark_validation_item(ValidationMarkRequest(input_path=str(source.parent), file_path=str(source),
        classification_model_path=str(manifest), species_name=species, species_count=count,
        feedback_operation_id=operation, action=action))


def test_image_count_survives_per_box_feedback(box_deployed, tmp_path, monkeypatch):
    monkeypatch.setenv("NERI_DINO_STATE_DIR", str(tmp_path/"state"))
    source = _photo(tmp_path)
    marked = _mark(box_deployed, source, "C", "4")
    assert learning.learning_status(box_deployed)["pending_count"] == 2
    raw_box = marked.detection_data["检测框"][0]
    result = apply_dinov3_box_feedback(SimpleNamespace(input_path=str(tmp_path), file_path=str(source),
        classification_model_path=str(box_deployed), observation_id=raw_box["observation_id"],
        action="empty", species_name=None, feedback_operation_id="exclude"))
    assert result["item"].detection_data["物种数量"] == "4"
    assert result["item"].detection_data["物种名称"] == "C"
    assert learning.learning_status(box_deployed)["pending_count"] == 1
    assert not _confirmed_empty_feedback({"image_annotation": {"source":"human", "species":"C", "count":"4"},
        "检测框": [{"物种":"空", "feedback_status":"confirmed"}]})


@pytest.mark.parametrize("redetected_empty", [False, True])
def test_smaller_manual_count_withdraws_already_learned_image_crops(ready, tmp_path, redetected_empty):
    manifest, *_ = ready
    source = _photo(tmp_path)
    _mark(manifest, source, "C", "4")
    learning.update_model(manifest)
    assert "C" in loaded(manifest).classes
    assert learning.learning_status(manifest)["learned_count"] > 0
    if redetected_empty:
        _write_detection_db(tmp_path, [(source, data("C", "0", boxes=0))])
    count = "0" if redetected_empty else "1"
    changed = _mark(manifest, source, "C", count, "correct-count")
    assert changed.detection_data["物种数量"] == count
    assert changed.detection_data["learning_feedback"]["status"] == ("empty" if redetected_empty else "count_exceeded")
    learning.update_model(manifest)
    assert learning.learning_status(manifest)["learned_count"] == 0
    assert set(loaded(manifest).classes) == {"A", "B"}


def test_batch_rules_are_applied_per_photo_not_to_combined_counts(box_deployed, tmp_path, monkeypatch):
    from system.backend.models import ValidationBatchMarkRequest
    monkeypatch.setenv("NERI_DINO_STATE_DIR", str(tmp_path/"state"))
    a = _photo(tmp_path)
    b = tmp_path/"other.jpg"
    _write_image(b)
    _write_detection_db(tmp_path, [(b, data("A", "1", boxes=1))])
    results = services.mark_validation_items(ValidationBatchMarkRequest(input_path=str(tmp_path), file_paths=[str(a),str(b)],
        classification_model_path=str(box_deployed), action="update", species_name="C", species_count="1", feedback_operation_id="batch"))
    assert [r.detection_data["learning_feedback"]["status"] for r in results] == ["count_exceeded", "eligible"]
    assert learning.learning_status(box_deployed)["species_counts"] == {"C":1}
