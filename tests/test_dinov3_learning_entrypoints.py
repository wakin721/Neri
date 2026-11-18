"""Check the actual annotation and HTTP entrypoints, not only learning helpers."""
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from system.backend import services
from system.backend.dino_feedback_api import dino_feedback_router, DinoV3LearningRequest, DinoV3BoxFeedbackRequest
from system.backend.dinov3_learning import learning_status
from system.backend.models import ValidationMarkRequest, ValidationBatchMarkRequest
from tests.test_dinov3_box_explanation_and_feedback import (
    deployed, _box, _write_image, _write_detection_db,
)


@pytest.fixture
def media(deployed, tmp_path, monkeypatch):
    monkeypatch.setenv("NERI_DINO_STATE_DIR", str(tmp_path / "state"))
    import system.training
    monkeypatch.setattr(system.training, "get_queue", lambda: SimpleNamespace(enqueue_many_async=lambda items: None))
    folder = tmp_path / "media"
    folder.mkdir()
    source = folder / "single.jpg"
    multi = folder / "multi.jpg"
    _write_image(source)
    _write_image(multi, 90)
    _write_detection_db(folder, [
        (source, {"检测框": [_box()], "物种名称": "A", "物种数量": "1", "sequence_completed": True}),
        (multi, {"检测框": [_box(), _box(bbox=(80, 10, 150, 80))], "物种名称": "A", "物种数量": "2", "sequence_completed": True}),
    ])
    return deployed, folder, source, multi


def test_single_file_http_mark_stages_new_species_and_unverified_clears_it(media):
    manifest, folder, source, _multi = media
    # services.mark_validation_item used to call the unpatched legacy batch
    # function, silently bypassing learning even though bulk marks worked.
    result = services.mark_validation_item(ValidationMarkRequest(
        input_path=str(folder), file_path=str(source), classification_model_path=str(manifest),
        action="update", species_name="new_species", species_count="1", feedback_operation_id="single-op"))
    assert result.detection_data["检测框"][0]["物种"] == "new_species"
    status = learning_status(manifest)
    assert status["species_counts"] == {"new_species": 1}
    services.mark_validation_item(ValidationMarkRequest(
        input_path=str(folder), file_path=str(source), classification_model_path=str(manifest),
        action="unverified", feedback_operation_id="unverify"))
    assert learning_status(manifest)["pending_count"] == 0


def test_excess_boxes_get_display_labels_without_training_qualification(media):
    manifest, folder, source, multi = media
    results = services.mark_validation_items(ValidationBatchMarkRequest(
        input_path=str(folder), file_paths=[str(source), str(multi)], classification_model_path=str(manifest),
        action="update", species_name="B", species_count="1", feedback_operation_id="bulk-op"))
    assert len(results) == 2
    assert learning_status(manifest)["species_counts"] == {"B": 1}
    assert [box["物种"] for box in results[1].detection_data["检测框"]] == ["B", "B"]
    assert all(box["feedback_status"] == "image_label" for box in results[1].detection_data["检测框"])
    assert results[1].detection_data["learning_feedback"]["status"] == "count_exceeded"


def test_learning_http_old_model_can_save_labels_but_update_requires_calibration(media):
    manifest, folder, source, _multi = media
    routes = {route.path: route.endpoint for route in dino_feedback_router().routes}
    model = DinoV3LearningRequest(classification_model_path=str(manifest))
    assert routes["/api/dino/learning/status"](model)["calibration_available"] is False
    from system.dinov3.observation_support import make_dinov3_observation_id
    routes["/api/dino/feedback/box"](DinoV3BoxFeedbackRequest(
        classification_model_path=str(manifest), input_path=str(folder), file_path=str(source),
        observation_id=make_dinov3_observation_id(source, 0, [10, 10, 80, 80]),
        action="update", species_name="new_species", feedback_operation_id="box-http"))
    before = manifest.read_bytes()
    with pytest.raises(HTTPException) as error:
        routes["/api/dino/learning/update"](model)
    assert error.value.status_code == 400
    assert "校准" in error.value.detail
    assert manifest.read_bytes() == before
    assert routes["/api/dino/learning/status"](model)["pending_count"] == 1
    details = routes["/api/dino/learning/status"](model.model_copy(update={"include_species": True}))
    assert "new_species" not in details["original_species"]
    assert details["species_counts"] == {"new_species": 1}
    assert routes["/api/dino/learning/discard"](model)["pending_count"] == 0


@pytest.mark.parametrize("action", ["empty", "unverified"])
def test_whole_file_cancellation_removes_previously_confirmed_individual_boxes(media, action):
    manifest, folder, _source, multi = media
    from system.dinov3.observation_support import apply_dinov3_box_feedback, make_dinov3_observation_id
    for index, bbox in enumerate(([10, 10, 80, 80], [80, 10, 150, 80])):
        apply_dinov3_box_feedback(SimpleNamespace(
            input_path=str(folder), file_path=str(multi), classification_model_path=str(manifest),
            observation_id=make_dinov3_observation_id(multi, index, bbox),
            action="update", species_name="B", feedback_operation_id=f"box-{index}"))
    assert learning_status(manifest)["pending_count"] == 2
    services.mark_validation_item(ValidationMarkRequest(
        input_path=str(folder), file_path=str(multi), classification_model_path=str(manifest),
        action=action, feedback_operation_id="cancel-all"))
    assert learning_status(manifest)["pending_count"] == 0
