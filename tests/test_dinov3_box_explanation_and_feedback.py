"""Neri1.1 box explanations, crop examples, and local feedback persistence."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from system.backend import services
from system.backend.models_legacy import DetectionItem
from system.detection_db import get_detection, init_db, upsert_detections_bulk
from system.dinov3.observation_support import (
    apply_dinov3_box_feedback,
    explain_dinov3_observation,
    is_dinov3_box_mapping,
    is_dinov3_observation_id,
    make_dinov3_observation_id,
    parse_dinov3_observation_id,
    render_dinov3_observation_example,
    revert_dinov3_feedback_operation,
)


@pytest.fixture(autouse=True)
def contribution_queue(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from system.training.queue import TrainingQueue

    queue = TrainingQueue(
        tmp_path / "training-state", transport=Mock(), debounce_seconds=0,
        privacy_blurrer=lambda image: (image, 0),
    )
    monkeypatch.setattr("system.training.get_queue", lambda: queue)
    yield queue
    queue.stop()


def _candidate(name: str, within: float, ridge: float = 0.5, **extra):
    return {"name": name, "within_score": within, "ridge_score": ridge, **extra}


def _box(
    species: str = "A",
    bbox=(10, 10, 80, 80),
    *,
    accepted: bool = True,
    best: str = "A",
    within: float = 0.8,
    candidates=None,
    **extra,
):
    return {
        "物种": species,
        "置信度": 0.9,
        "边界框": list(bbox),
        "accepted": accepted,
        "best_known_species": best,
        "known_score": 1.5,
        "threshold": 0.5,
        "within_score": within,
        "ridge_score": 0.65,
        "候选项": candidates
        if candidates is not None
        else [_candidate(best, within, best_known_species=best)],
        **extra,
    }


def _write_image(path: Path, shade: int = 40) -> None:
    import cv2

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[20:100, 20:140] = (shade, shade, shade)
    assert cv2.imwrite(str(path), image)


def _write_detection_db(
    directory: Path,
    rows: list[tuple[Path, dict]],
) -> Path:
    db_path = directory / "detections.db"
    init_db(str(db_path))
    upsert_detections_bulk(
        str(db_path),
        [(path.stem, path.name, data) for path, data in rows],
    )
    return db_path


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    from system.dinov3.deployment import publish_model
    from system.dinov3.encoder import PREPROCESSING, sha256
    from system.dinov3.model import DinoV3RidgeWithinModel

    root = tmp_path_factory.mktemp("dinov3-box-model")
    rng = np.random.default_rng(20261108)
    dim = 768
    features: list[np.ndarray] = []
    labels: list[str] = []
    cameras: list[str] = []
    sequences: list[str] = []
    for axis, name in enumerate(("A", "B", "C")):
        for index in range(8):
            vector = np.zeros(dim, dtype=np.float64)
            vector[axis] = 1.0
            vector += 0.02 * rng.standard_normal(dim)
            vector /= np.linalg.norm(vector)
            features.append(vector)
            labels.append(name)
            cameras.append(f"cam{index % 4}")
            sequences.append(f"seq-{name}-{index // 2}")

    unknown = rng.standard_normal((8, dim))
    unknown /= np.linalg.norm(unknown, axis=1, keepdims=True)
    calibration_count = len(features) + len(unknown)

    weights = root / "source.pth"
    weights.write_bytes(b"test-only-dinov3-encoder")
    code = root / "source-code"
    (code / "dinov3/hub").mkdir(parents=True)
    (code / "dinov3/hub/backbones.py").write_text("# test-only DINOv3 source", "utf8")
    (code / "LICENSE.md").write_text("test-only license", "utf8")

    with threadpool_limits(limits=1):
        model = DinoV3RidgeWithinModel().fit(
            np.stack(features),
            labels,
            cameras,
            sequences,
            origin=np.zeros(dim),
            transform=np.eye(dim),
            metadata={
                "encoder_sha256": sha256(weights),
                "preprocessing": PREPROCESSING,
            },
        )
        model.calibrate(
            np.r_[np.stack(features), unknown],
            labels + ["U"] * 8,
            ["cal"] * calibration_count,
            [f"cal-{index}" for index in range(calibration_count)],
        )
        manifest = publish_model(
            model,
            root / "joint.neri.json",
            encoder_weights=weights,
            encoder_code=code,
        )
    return manifest


def test_observation_id_round_trip_and_box_recognition(tmp_path):
    image = tmp_path / "a.jpg"
    observation_id = make_dinov3_observation_id(
        image,
        3,
        [1.234, 5.678, 9.1011, 12.1314],
    )
    assert is_dinov3_observation_id(observation_id)
    path, box_index, bbox = parse_dinov3_observation_id(observation_id)
    assert path == image.resolve()
    assert box_index == 3
    assert bbox == [1.23, 5.68, 9.1, 12.13]

    assert not is_dinov3_observation_id("obs-1")
    assert not is_dinov3_observation_id(None)
    with pytest.raises(KeyError):
        parse_dinov3_observation_id("dinov3:not-base64!!")
    with pytest.raises(KeyError):
        parse_dinov3_observation_id("obs-1")

    assert is_dinov3_box_mapping({"ridge_score": 0.4})
    assert is_dinov3_box_mapping({"raw_ridge_score": 0.4})
    assert is_dinov3_box_mapping({"within_score": 0.4})
    assert is_dinov3_box_mapping({"box_known_score": 0.4})
    assert is_dinov3_box_mapping({"sequence_known_score": 0.4})
    assert is_dinov3_box_mapping({"候选项": [{"within_score": 0.4}]})
    assert is_dinov3_box_mapping({"candidates": [{"ridge_score": 0.4}]})
    assert is_dinov3_box_mapping({}, {"sequence_completed": True})
    assert not is_dinov3_box_mapping({"物种": "A"}, {"sequence_completed": False})


def test_detection_normalization_synthesizes_ids_only_for_dinov3_boxes(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"image")
    data = {
        "检测框": [
            {
                "物种": "A",
                "边界框": [10, 10, 80, 80],
                "within_score": 0.8,
                "best_known_species": "A",
                "候选项": [{"name": "A", "within_score": 0.8}],
            }
        ],
        "sequence_completed": True,
    }
    item = DetectionItem(filename=image.name, path=str(image), file_type="image")
    applied = services._apply_detection_data(item, data)
    box = applied.detection_boxes[0]
    assert is_dinov3_observation_id(box["observation_id"])
    assert box["predicted_species"] == "A"
    path, box_index, bbox = parse_dinov3_observation_id(box["observation_id"])
    assert path == image.resolve()
    assert box_index == 0
    assert bbox == [10.0, 10.0, 80.0, 80.0]

    plain_data = {
        "检测框": [{"物种": "A", "边界框": [1, 1, 10, 10], "候选项": []}],
    }
    plain_item = services._apply_detection_data(
        DetectionItem(filename=image.name, path=str(image), file_type="image"),
        plain_data,
    )
    assert "observation_id" not in plain_item.detection_boxes[0]


@dataclass
class _Prediction:
    species: str
    accepted: bool
    best_known_species: str
    known_score: float
    threshold: float
    embedding: np.ndarray
    confidence: float = 0.8
    ridge_score: float = 0.7
    within_score: float = 0.75
    candidates: tuple = ()

    def as_candidate(self, *, detection_confidence=None):
        return {
            "name": self.best_known_species,
            "accepted": self.accepted,
            "known_score": self.known_score,
            "threshold": self.threshold,
            "best_known_species": self.best_known_species,
            "predicted_species": self.best_known_species,
            "ridge_score": self.ridge_score,
            "within_score": self.within_score,
        }


def test_finalize_photo_job_attaches_deterministic_observation_ids(tmp_path):
    from unittest.mock import MagicMock

    from system.dinov3.processor import (
        DinoV3CropCollector,
        DinoV3CropRecord,
        finalize_photo_job,
    )

    image = tmp_path / "IMG_0001.jpg"
    image.write_bytes(b"image-bytes")
    collector = DinoV3CropCollector()
    collector.record_crop(
        DinoV3CropRecord(
            str(image),
            0,
            (10.0, 20.0, 60.0, 70.0),
            0.9,
            np.ones(768, dtype=np.float32) / np.sqrt(768),
        )
    )
    detector = MagicMock()
    detector.dinov3_classifier = MagicMock()
    detector.dinov3_classifier.backend = "dinov3"
    detector._dinov3_collector = collector
    detector.dinov3_classifier.classify_features.return_value = [
        _Prediction("A", True, "A", 1.2, 0.5, np.ones(768, dtype=np.float32))
    ]

    item = DetectionItem(
        filename=image.name,
        path=str(image),
        file_type="image",
        date_taken="2026-11-01 10:00:00",
    )
    updated = finalize_photo_job(detector, [item], tmp_path)
    box = updated[0].detection_data["检测框"][0]
    assert box["observation_id"].startswith("dinov3:")
    assert box["候选项"][0]["observation_id"] == box["observation_id"]
    path, box_index, bbox = parse_dinov3_observation_id(box["observation_id"])
    assert path == image.resolve()
    assert box_index == 0
    assert bbox == [10.0, 20.0, 60.0, 70.0]


def _explanation_fixture(deployed, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    source = media / "a.jpg"
    other = media / "b.jpg"
    _write_image(source, 60)
    _write_image(other, 120)

    source_box = _box(
        candidates=[
            _candidate("A", 0.8, best_known_species="A"),
            _candidate("B", 0.6, best_known_species="B"),
        ],
    )
    other_box = _box("A", bbox=(30, 25, 110, 95))
    _write_detection_db(
        media,
        [
            (source, {"检测框": [source_box], "sequence_completed": True}),
            (other, {"检测框": [other_box], "sequence_completed": True}),
        ],
    )
    observation_id = make_dinov3_observation_id(source, 0, source_box["边界框"])
    return media, source, other, observation_id


def test_explain_dinov3_observation_uses_cached_bank_without_encoder(
    deployed,
    tmp_path,
    monkeypatch,
):
    from system.dinov3 import runtime

    monkeypatch.setattr(
        runtime,
        "load_dinov3_model",
        lambda *args, **kwargs: pytest.fail("explanation must not load the ViT encoder"),
    )
    _media, source, other, observation_id = _explanation_fixture(deployed, tmp_path)

    result = explain_dinov3_observation(str(deployed), observation_id)
    assert result["species"] == "A"
    assert result["accepted"] is True
    assert result["best_known_species"] == "A"
    assert result["known_score"] == pytest.approx(1.5)
    assert result["threshold"] == pytest.approx(0.5)
    assert result["rejection"] == {"mode": "ridge_within"}
    assert result["current_example_available"] is True

    nearest = result["nearest_species"]
    assert [item["name"] for item in nearest] == ["A", "B"]
    assert nearest[0]["cosine_score"] == pytest.approx(0.8)
    assert nearest[1]["cosine_score"] == pytest.approx(0.6)
    assert nearest[0]["squared_distance"] == pytest.approx(0.4)
    assert nearest[1]["squared_distance"] == pytest.approx(0.8)
    assert all(item["source"] == "base" for item in nearest)

    projection = result["projection"]
    assert projection["method"] == "nearest_two_species_axis"
    assert projection["species"] == ["A", "B"]
    prototype_points = [point for point in projection["points"] if point["kind"] == "prototype"]
    current_points = [point for point in projection["points"] if point["kind"] == "current"]
    assert len(prototype_points) == 2
    assert len(current_points) == 1
    assert prototype_points[0]["y"] == pytest.approx(0.0, abs=1e-6)
    assert prototype_points[1]["y"] == pytest.approx(0.0, abs=1e-6)
    assert abs(current_points[0]["y"]) > 1e-4

    example = result["nearest_example"]
    assert example is not None
    assert example["kind"] == "observation"
    assert example["species"] == "A"
    example_path, example_index, _bbox = parse_dinov3_observation_id(example["observation_id"])
    assert example_path == other.resolve()
    assert example_index == 0


def test_explain_locates_legacy_boxes_without_observation_id(deployed, tmp_path):
    """Existing detections.db rows predate ids; bbox/index fallback must work."""
    media = tmp_path / "media"
    media.mkdir()
    source = media / "legacy.jpg"
    _write_image(source, 90)
    raw_box = _box(
        "B",
        bbox=(12, 14, 90, 100),
        best="B",
        candidates=[_candidate("B", 0.7, best_known_species="B")],
    )
    _write_detection_db(
        media,
        [(source, {"检测框": [raw_box], "sequence_completed": True})],
    )
    observation_id = make_dinov3_observation_id(source, 0, [12, 14, 90, 100])

    result = explain_dinov3_observation(str(deployed), observation_id)
    assert result["nearest_species"][0]["name"] == "B"
    assert result["current_example_available"] is True


def test_render_dinov3_observation_example_uses_id_and_bbox_fallback(deployed, tmp_path):
    media, source, _other, observation_id = _explanation_fixture(deployed, tmp_path)

    with_bbox = render_dinov3_observation_example(observation_id)
    assert with_bbox.startswith(b"\xff\xd8")
    assert with_bbox.endswith(b"\xff\xd9")

    without_bbox = make_dinov3_observation_id(source, 0)
    from_db = render_dinov3_observation_example(without_bbox)
    assert from_db.startswith(b"\xff\xd8")
    assert from_db.endswith(b"\xff\xd9")


def _feedback_request(
    deployed,
    source: Path,
    observation_id: str,
    action: str,
    *,
    species_name: str | None = None,
    operation_id: str = "op-1",
):
    return SimpleNamespace(
        input_path=str(source.parent),
        file_path=str(source),
        classification_model_path=str(deployed),
        observation_id=observation_id,
        action=action,
        species_name=species_name,
        feedback_operation_id=operation_id,
    )


def test_dinov3_box_feedback_journals_and_reverts(
    deployed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NERI_DINO_STATE_DIR", str(tmp_path / "state"))
    media = tmp_path / "media"
    media.mkdir()
    source = media / "a.jpg"
    _write_image(source)
    raw_box = _box("A", candidates=[_candidate("A", 0.8, best_known_species="A")])
    db_path = _write_detection_db(
        media,
        [(source, {"检测框": [raw_box], "sequence_completed": True})],
    )
    observation_id = make_dinov3_observation_id(source, 0, raw_box["边界框"])

    result = apply_dinov3_box_feedback(
        _feedback_request(
            deployed,
            source,
            observation_id,
            "update",
            species_name="B",
            operation_id="op-update",
        )
    )
    box = result["item"].detection_data["检测框"][0]
    assert box["物种"] == "B"
    assert box["feedback_status"] == "confirmed"
    assert box["候选项"][0]["name"] == "B"
    assert box["候选项"][0]["observation_id"] == observation_id
    assert result["affected_species"] == []
    assert result["registry_id"] is None

    persisted = get_detection(str(db_path), source.stem)
    assert persisted["检测框"][0]["物种"] == "B"
    journal = tmp_path / "state" / "dinov3_feedback_journal.sqlite3"
    assert journal.is_file()
    with sqlite3.connect(str(journal)) as conn:
        rows = conn.execute(
            "SELECT operation_id, observation_id FROM feedback_ecological_operations"
        ).fetchall()
    assert rows == [("op-update", observation_id)]

    revert = revert_dinov3_feedback_operation(
        SimpleNamespace(feedback_operation_id="op-update")
    )
    assert revert == {"operation_id": "op-update", "affected_species": []}
    restored = get_detection(str(db_path), source.stem)
    assert restored["检测框"][0]["物种"] == "A"
    assert "feedback_status" not in restored["检测框"][0]
    with sqlite3.connect(str(journal)) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM feedback_ecological_operations"
        ).fetchone()[0] == 0


def test_dinov3_box_feedback_empty_correct_and_unverified(
    deployed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NERI_DINO_STATE_DIR", str(tmp_path / "state"))
    media = tmp_path / "media"
    media.mkdir()
    source = media / "a.jpg"
    _write_image(source)
    raw_box = _box(
        "Unknown",
        accepted=False,
        best="A",
        candidates=[_candidate("Unknown", 0.2, best_known_species="A")],
    )
    _write_detection_db(
        media,
        [(source, {"检测框": [raw_box], "sequence_completed": True})],
    )
    observation_id = make_dinov3_observation_id(source, 0, raw_box["边界框"])

    empty = apply_dinov3_box_feedback(
        _feedback_request(deployed, source, observation_id, "empty", operation_id="op-empty")
    )
    assert empty["item"].detection_data["检测框"][0]["物种"] == "空"

    correct = apply_dinov3_box_feedback(
        _feedback_request(
            deployed,
            source,
            observation_id,
            "correct",
            operation_id="op-correct",
        )
    )
    corrected_box = correct["item"].detection_data["检测框"][0]
    assert corrected_box["物种"] == "A"
    assert corrected_box["feedback_status"] == "confirmed"
    assert corrected_box["候选项"][0]["name"] == "A"

    unverified = apply_dinov3_box_feedback(
        _feedback_request(
            deployed,
            source,
            observation_id,
            "unverified",
            operation_id="op-unverified",
        )
    )
    assert "feedback_status" not in unverified["item"].detection_data["检测框"][0]


def _empty_feedback_photo(tmp_path, monkeypatch, box_count=1):
    monkeypatch.setenv("NERI_DINO_STATE_DIR", str(tmp_path / "state"))
    media = tmp_path / "media"
    media.mkdir()
    source = media / "false-positive.jpg"
    _write_image(source)
    boxes = [_box("A", bbox=(10 + index, 10, 80, 80)) for index in range(box_count)]
    _write_detection_db(media, [(source, {"检测框": boxes, "sequence_completed": True})])
    ids = [make_dinov3_observation_id(source, index, box["边界框"])
           for index, box in enumerate(boxes)]
    return source, ids


@pytest.mark.parametrize("enabled", [False, True])
def test_empty_box_feedback_uploads_only_after_all_boxes_confirmed(
    deployed, tmp_path, monkeypatch, contribution_queue, enabled,
):
    from system.training.policy import AGREEMENT_VERSION

    queue = contribution_queue
    queue.set_consent(AGREEMENT_VERSION, enabled)
    source, ids = _empty_feedback_photo(tmp_path, monkeypatch, box_count=2)
    original = source.read_bytes()
    for index, observation_id in enumerate(ids):
        apply_dinov3_box_feedback(_feedback_request(
            deployed, source, observation_id, "empty", operation_id=f"empty-{index}",
        ))
        assert queue.status()["stats"]["pending"] == int(enabled and index == 1)

    assert queue.process_once() is enabled
    if enabled:
        queue.transport.upload.assert_called_once()
        package = json.loads(queue.transport.upload.call_args.args[2])
        assert package["species"] == ["空照片"]
        assert package["confirmed_empty"] is True
        assert package["model_predictions"] == []
        assert not package["yolo_labels"]["空照片"].strip()
    else:
        queue.transport.upload.assert_not_called()
    assert source.read_bytes() == original


@pytest.mark.parametrize("withdrawal", ["undo", "unverified", "update"])
@pytest.mark.parametrize("uploaded", [False, True])
def test_empty_box_feedback_withdraws_pending_or_uploaded_contribution(
    deployed, tmp_path, monkeypatch, contribution_queue, withdrawal, uploaded,
):
    from system.training.policy import AGREEMENT_VERSION

    queue = contribution_queue
    queue.set_consent(AGREEMENT_VERSION, True)
    source, ids = _empty_feedback_photo(tmp_path, monkeypatch)
    apply_dinov3_box_feedback(_feedback_request(
        deployed, source, ids[0], "empty", operation_id="empty",
    ))
    assert queue.status()["stats"]["pending"] == 1
    if uploaded:
        assert queue.process_once()
    if withdrawal == "undo":
        revert_dinov3_feedback_operation(SimpleNamespace(feedback_operation_id="empty"))
    else:
        apply_dinov3_box_feedback(_feedback_request(
            deployed, source, ids[0], withdrawal, species_name="A", operation_id="withdraw",
        ))
    assert queue.process_once() is uploaded
    assert queue.status()["stats"]["pending"] == 0
    assert queue.transport.upload.call_count == int(uploaded)
    assert queue.transport.delete.call_count == int(uploaded)


def test_failed_empty_feedback_save_does_not_enqueue(
    deployed, tmp_path, monkeypatch, contribution_queue,
):
    from system.training.policy import AGREEMENT_VERSION

    contribution_queue.set_consent(AGREEMENT_VERSION, True)
    source, ids = _empty_feedback_photo(tmp_path, monkeypatch)

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(services, "_persist_validation_update", fail_save)
    with pytest.raises(OSError, match="disk full"):
        apply_dinov3_box_feedback(_feedback_request(deployed, source, ids[0], "empty"))
    assert contribution_queue.status()["stats"]["pending"] == 0


def test_dinov3_box_feedback_rejects_foreign_observation_id(
    deployed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NERI_DINO_STATE_DIR", str(tmp_path / "state"))
    media = tmp_path / "media"
    media.mkdir()
    source = media / "a.jpg"
    foreign = media / "foreign.jpg"
    _write_image(source)
    _write_image(foreign)
    raw_box = _box()
    _write_detection_db(
        media,
        [(source, {"检测框": [raw_box], "sequence_completed": True})],
    )
    foreign_id = make_dinov3_observation_id(foreign, 0, raw_box["边界框"])
    with pytest.raises(ValueError, match="selected file"):
        apply_dinov3_box_feedback(
            _feedback_request(deployed, source, foreign_id, "update", species_name="B")
        )

    with pytest.raises(ValueError, match="species_name"):
        observation_id = make_dinov3_observation_id(source, 0, raw_box["边界框"])
        apply_dinov3_box_feedback(
            _feedback_request(deployed, source, observation_id, "update")
        )

    with pytest.raises(ValueError, match="Unsupported"):
        apply_dinov3_box_feedback(
            _feedback_request(
                deployed,
                source,
                observation_id,
                "not-an-action",
            )
        )


def test_dino_feedback_entrypoints_delegate_for_joint_models(
    deployed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NERI_DINO_STATE_DIR", str(tmp_path / "state"))
    _media, source, _other, observation_id = _explanation_fixture(deployed, tmp_path)

    from system.backend import dino_feedback_service

    explanation = dino_feedback_service.explain_feedback_observation(
        str(deployed),
        observation_id,
    )
    assert explanation["rejection"] == {"mode": "ridge_within"}
    assert explanation["nearest_species"][0]["name"] == "A"

    image = dino_feedback_service.render_feedback_observation_example(
        str(deployed),
        observation_id,
    )
    assert image.startswith(b"\xff\xd8")

    result = dino_feedback_service.apply_box_feedback(
        _feedback_request(
            deployed,
            source,
            observation_id,
            "update",
            species_name="C",
            operation_id="op-delegate",
        )
    )
    assert result["item"].detection_data["检测框"][0]["物种"] == "C"
    assert result["affected_species"] == []

    reverted = dino_feedback_service.revert_feedback_operation(
        SimpleNamespace(
            classification_model_path=str(deployed),
            feedback_operation_id="op-delegate",
        )
    )
    assert reverted == {"operation_id": "op-delegate", "affected_species": []}
