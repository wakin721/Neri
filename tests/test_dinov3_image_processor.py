from types import SimpleNamespace

import numpy as np
import pytest

from system.dino.image_processor import ImageProcessor
from system.dinov3.classifier import DinoV3Prediction


@pytest.mark.parametrize("accepted", [True, False])
def test_joint_crops_collect_raw_features_without_v2_observations(accepted):
    raw = np.zeros(768, dtype=np.float32)
    raw[2] = 1
    prediction = DinoV3Prediction(species="A", accepted=accepted, best_known_species="A",
        known_score=.5 if accepted else -.4, threshold=.2, embedding=raw,
        candidates=(), ridge_score=.8, within_score=.6)
    classifier = SimpleNamespace(backend="dinov3", classify_crops=lambda crops, **kw: [prediction])
    detector = ImageProcessor(None)
    detector.load_dino_classifier(classifier)
    detector._dinov3_current_paths = ("unreadable.jpg", "valid.jpg")
    detector._crop_single_box = lambda task: (task[0], task[1], task[3])
    box = SimpleNamespace(conf=np.asarray(.9), xyxy=np.asarray([[1., 2., 10., 20.]]))
    maps, selected = detector._classify_dino_crops(
        [SimpleNamespace(boxes=[box])], [np.zeros((25, 25, 3), np.uint8)], source_indices=[1])
    records = detector._dinov3_collector.get_crops_for_path("valid.jpg")
    assert len(records) == 1 and records[0].box_index == 0
    np.testing.assert_array_equal(records[0].raw_embedding, raw)
    assert maps[0][0][0]["sequence_pending"] is True
    assert maps[0][0][0]["sequence_status"] == "pending"
    assert "observation_id" not in maps[0][0][0]
    assert maps[0][0][0]["raw_det_conf"] == pytest.approx(.9)
    assert (selected[0][0] is not None) == accepted
