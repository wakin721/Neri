from types import SimpleNamespace

from system.backend.runtime_patches import (
    REJECTED_UNKNOWN_LABEL,
    preserve_dinov3_rejections,
)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCoords:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return [self.values]


class FakeBox:
    def __init__(self):
        self.conf = FakeScalar(0.88)
        self.xyxy = FakeCoords([1, 2, 30, 40])


def test_preserves_rejected_dinov3_box_as_unknown_instead_of_empty():
    rejected = {
        'name': 'Unknown',
        'conf': 0.61,
        'accepted': False,
        'observation_id': 'obs-1',
        'predicted_species': 'Unknown',
        'registry_id': None,
        'known_score': 0.61,
        'threshold': 0.72,
        'squared_distance': 0.44,
    }
    result = SimpleNamespace(
        boxes=[FakeBox()],
        candidates_data={0: [rejected]},
        classification_filtered_boxes={0},
    )
    detection = {'detect_results': [result]}
    payload = {
        '物种名称': '空',
        '物种数量': '空',
        '检测框': [],
        '最低置信度': None,
    }

    updated = preserve_dinov3_rejections(payload, detection)

    assert updated['物种名称'] == REJECTED_UNKNOWN_LABEL
    assert updated['物种数量'] == '1'
    assert updated['拒识数量'] == 1
    assert len(updated['检测框']) == 1
    box = updated['检测框'][0]
    assert box['物种'] == REJECTED_UNKNOWN_LABEL
    assert box['置信度'] is None
    assert box['候选项'] == []
    assert box['observation_id'] == 'obs-1'
    assert box['predicted_species'] == 'Unknown'
    assert box['known_score'] == 0.61
    assert box['threshold'] == 0.72


def test_does_not_relabel_normal_low_confidence_classifier_filter_as_unknown():
    low_confidence = {'name': '豹猫', 'conf': 0.20}
    result = SimpleNamespace(
        boxes=[FakeBox()],
        candidates_data={0: [low_confidence]},
        classification_filtered_boxes={0},
    )
    detection = {'detect_results': [result]}
    payload = {'物种名称': '空', '物种数量': '空', '检测框': []}

    updated = preserve_dinov3_rejections(payload, detection)

    assert updated == payload


def test_install_runtime_patches_is_idempotent_and_wraps_serializer():
    from system.backend.runtime_patches import install_runtime_patches

    calls = []

    def original_serializer(detector, detection):
        calls.append((detector, detection))
        return {'物种名称': '空', '物种数量': '空', '检测框': []}

    services = SimpleNamespace(
        _serialize_detector_output=original_serializer,
        preview_media_items=lambda *args, **kwargs: [],
    )
    install_runtime_patches(services, patch_preview=False)
    first = services._serialize_detector_output
    install_runtime_patches(services, patch_preview=False)

    assert services._serialize_detector_output is first
    result = SimpleNamespace(
        boxes=[FakeBox()],
        candidates_data={
            0: [
                {
                    'name': 'Unknown',
                    'accepted': False,
                    'observation_id': 'obs-2',
                    'predicted_species': 'Unknown',
                }
            ]
        },
        classification_filtered_boxes={0},
    )
    payload = services._serialize_detector_output(object(), {'detect_results': [result]})
    assert payload['物种名称'] == REJECTED_UNKNOWN_LABEL
    assert len(calls) == 1


def test_mixed_known_and_rejected_boxes_keep_species_counts_aligned():
    rejected = {
        'name': 'Unknown',
        'accepted': False,
        'observation_id': 'obs-mixed',
        'predicted_species': 'Unknown',
    }
    result = SimpleNamespace(
        boxes=[FakeBox()],
        candidates_data={0: [rejected]},
        classification_filtered_boxes={0},
    )
    payload = {
        '物种名称': '豹猫',
        '物种数量': '2',
        '检测框': [
            {'物种': '豹猫', '置信度': 0.9, '边界框': [1, 1, 2, 2], '候选项': []},
            {'物种': '豹猫', '置信度': 0.8, '边界框': [2, 2, 3, 3], '候选项': []},
        ],
    }

    updated = preserve_dinov3_rejections(payload, {'detect_results': [result]})

    assert updated['物种名称'] == '豹猫,拒识/Unknown'
    assert updated['物种数量'] == '2,1'
