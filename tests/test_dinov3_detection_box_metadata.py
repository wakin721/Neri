from system.backend import services
from system.backend.models import DetectionItem


def _raw_dino_detection_data():
    return {
        "物种名称": "Known",
        "物种数量": "1",
        "最低置信度": "0.910",
        "检测框": [
            {
                "物种": "Known",
                "置信度": 0.91,
                "边界框": [1.0, 2.0, 30.0, 40.0],
                "候选项": [
                    {
                        "name": "Known",
                        "conf": 0.91,
                        "observation_id": "obs-clickable",
                        "registry_id": 7,
                        "predicted_species": "Known",
                        "registration_status": "confirmed",
                    }
                ],
            }
        ],
    }


def test_normalized_dinov3_box_promotes_feedback_identity_from_candidate():
    boxes = services._normalize_detection_boxes(_raw_dino_detection_data())

    assert len(boxes) == 1
    assert boxes[0]["observation_id"] == "obs-clickable"
    assert boxes[0]["registry_id"] == 7
    assert boxes[0]["predicted_species"] == "Known"


def test_detection_item_exposes_promoted_dinov3_observation_id():
    item = DetectionItem(filename="image.jpg", path="image.jpg", file_type="jpg")

    updated = services._apply_detection_data(item, _raw_dino_detection_data())

    assert updated.detection_boxes[0]["observation_id"] == "obs-clickable"
