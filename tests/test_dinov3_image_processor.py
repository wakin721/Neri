from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import torch

from system.dinov3.classifier import DinoV3Prediction


def _image_processor_module():
    if "ultralytics" not in sys.modules:
        module = types.ModuleType("ultralytics")
        module.YOLO = object
        sys.modules["ultralytics"] = module
    return importlib.import_module("system.image_processor")


class _FakeBox:
    def __init__(self, xyxy, conf=0.9, cls=0):
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.names = {0: "animal"}


class _FakeDetector:
    def __init__(self, results):
        self.results = results

    def __call__(self, *_args, **_kwargs):
        return self.results


class _FakeDinoClassifier:
    backend = "dinov3"

    def __init__(self, predictions):
        self.predictions = predictions
        self.seen_crops = None

    def classify_crops(self, crops, *, array_color="rgb"):
        assert array_color == "rgb"
        self.seen_crops = list(crops)
        return list(self.predictions)


def _prediction(
    *,
    species,
    accepted,
    embedding_value,
    source="checkpoint",
    registry_id=None,
    registration_status=None,
    score=0.9,
    threshold=0.4,
):
    embedding = np.zeros(768, dtype=np.float32)
    embedding[embedding_value] = 1.0
    return DinoV3Prediction(
        species=species,
        accepted=accepted,
        best_known_species="animal",
        head_species="animal",
        prototype_species="animal",
        head_prototype_consistent=True,
        known_score=score,
        threshold=threshold,
        candidates=({"name": "animal", "logit": 1.0, "prototype_score": score},),
        embedding=embedding,
        source=source,
        registry_id=registry_id,
        registration_status=registration_status,
    )


def _processor_with_predictions(predictions, boxes):
    module = _image_processor_module()
    processor = module.ImageProcessor(None)
    processor.model = _FakeDetector([_FakeResult(boxes)])
    processor.translation_dict = {}
    processor.load_dinov3_classifier(_FakeDinoClassifier(predictions))
    return processor


def test_detect_batch_species_routes_yolo_crops_to_dinov3_and_buffers_observations():
    boxes = [
        _FakeBox([10, 10, 30, 40], conf=0.8),
        _FakeBox([50, 50, 85, 80], conf=0.7),
    ]
    predictions = [
        _prediction(species="Leopard", accepted=True, embedding_value=0),
        _prediction(
            species="Unknown #7",
            accepted=False,
            embedding_value=1,
            source="registry",
            registry_id=7,
            registration_status="candidate",
            score=0.84,
            threshold=0.82,
        ),
    ]
    processor = _processor_with_predictions(predictions, boxes)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    result = processor.detect_batch_species(
        ["camera-a/image.jpg"],
        preloaded_data=([0], [rgb.copy()], [rgb]),
        conf=0.25,
    )[0]

    assert result["物种名称"] == "Leopard"
    detection_result = result["detect_results"][0]
    assert 1 in detection_result.classification_filtered_boxes
    assert detection_result.candidates_data[0][0]["source"] == "checkpoint"
    assert detection_result.candidates_data[1][0]["registration_status"] == "candidate"

    classifier = processor.dinov3_classifier
    assert len(classifier.seen_crops) == 2
    assert all(crop.shape[0] == crop.shape[1] for crop in classifier.seen_crops)

    observations = processor.drain_dinov3_observations()
    assert [(item.result_index, item.box_index) for item in observations] == [(0, 0), (0, 1)]
    assert observations[1].registry_id == 7
    assert observations[1].registration_status == "candidate"
    assert observations[1].embedding.shape == (768,)
    assert processor.drain_dinov3_observations() == ()


def test_provisional_registry_match_is_emitted_as_official_species():
    boxes = [_FakeBox([5, 5, 40, 45], conf=0.75)]
    prediction = _prediction(
        species="Clouded leopard",
        accepted=True,
        embedding_value=2,
        source="registry",
        registry_id=12,
        registration_status="provisional",
        score=0.86,
        threshold=0.78,
    )
    processor = _processor_with_predictions([prediction], boxes)
    rgb = np.zeros((60, 60, 3), dtype=np.uint8)

    result = processor.detect_batch_species(
        ["camera-b/image.jpg"],
        preloaded_data=([0], [rgb.copy()], [rgb]),
        conf=0.25,
    )[0]

    assert result["物种名称"] == "Clouded leopard"
    selected = result["detect_results"][0].selected_candidates_data[0]
    assert selected["registry_id"] == 12
    assert selected["registration_status"] == "provisional"


def test_dinov3_candidate_and_runtime_observation_share_observation_id():
    boxes = [_FakeBox([10, 12, 30, 40], conf=0.81)]
    prediction = _prediction(
        species="Leopard",
        accepted=True,
        embedding_value=3,
        source="checkpoint",
    )
    processor = _processor_with_predictions([prediction], boxes)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    result = processor.detect_batch_species(
        ["camera-c/image.jpg"],
        preloaded_data=([0], [rgb.copy()], [rgb]),
        conf=0.25,
    )[0]
    candidate = result["detect_results"][0].candidates_data[0][0]
    observations = processor.drain_dinov3_observations()

    assert isinstance(candidate["observation_id"], str)
    assert candidate["observation_id"]
    assert candidate["predicted_species"] == "Leopard"
    assert observations[0].observation_id == candidate["observation_id"]
    assert observations[0].best_known_species == "animal"
