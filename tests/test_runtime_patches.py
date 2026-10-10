import json
import sqlite3
from types import SimpleNamespace

from system.backend.runtime_patches import (
    REJECTED_UNKNOWN_LABEL,
    preserve_dinov2_rejections,
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


def test_preserves_rejected_dinov2_box_as_unknown_instead_of_empty():
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
        'class_margin': 0.10,
        'adjusted_distance_score': -0.34,
        'score_threshold': -0.25,
        'registry_action': 'new_mode_candidate',
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

    updated = preserve_dinov2_rejections(payload, detection)

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
    assert box['class_margin'] == 0.10
    assert box['adjusted_distance_score'] == -0.34
    assert box['score_threshold'] == -0.25
    assert box['registry_action'] == 'new_mode_candidate'


def test_does_not_relabel_normal_low_confidence_classifier_filter_as_unknown():
    low_confidence = {'name': '豹猫', 'conf': 0.20}
    result = SimpleNamespace(
        boxes=[FakeBox()],
        candidates_data={0: [low_confidence]},
        classification_filtered_boxes={0},
    )
    detection = {'detect_results': [result]}
    payload = {'物种名称': '空', '物种数量': '空', '检测框': []}

    updated = preserve_dinov2_rejections(payload, detection)

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


def test_real_batch_single_image_and_video_use_rejection_preserving_serializer():
    from system.backend import services

    serializer = services._serialize_detector_output
    assert getattr(serializer, '_neri_preserves_open_set_rejections', False)
    assert services._legacy._serialize_detector_output is serializer
    # These functions keep their defining module's globals when copied to the
    # facade. Verify the serializer actually used by all processing paths.
    for function in (services._legacy._detect_image_batch,
                     services._legacy._detect_image,
                     services._legacy._detect_video_fast_batch):
        assert function.__globals__['_serialize_detector_output'] is serializer


def test_actual_image_batch_returns_unknown_box_instead_of_empty(tmp_path, monkeypatch):
    from system.backend import services
    from system.backend.models import CreateJobRequest, DetectionItem

    path = tmp_path / 'animal.jpg'
    path.write_bytes(b'test')
    rejected = {
        'name': 'Unknown', 'accepted': False,
        'observation_id': 'batch-rejected', 'predicted_species': 'Unknown',
        'known_score': 0.3, 'threshold': 0.5,
    }
    result = SimpleNamespace(
        boxes=[FakeBox()], candidates_data={0: [rejected]},
        classification_filtered_boxes={0}, names={0: 'animal'},
    )
    detector = SimpleNamespace(detect_batch_species=lambda *args, **kwargs: [{
        '物种名称': '空', '物种数量': '空', 'detect_results': [result],
    }])
    monkeypatch.setattr(services._legacy, '_selected_inference_class_ids', lambda *args: None)
    monkeypatch.setattr(services._legacy, '_persist_dinov2_observations', lambda *args, **kwargs: None)
    monkeypatch.setattr(services._legacy, '_save_detection_data_batch', lambda *args: None)
    item = DetectionItem(filename=path.name, path=str(path), file_type='jpg')
    output = services._detect_image_batch(
        detector, [path], [item], CreateJobRequest(input_dir=str(tmp_path)), tmp_path,
    )
    assert output[0].species == [REJECTED_UNKNOWN_LABEL]
    assert len(output[0].detection_boxes) == 1
    assert output[0].detection_boxes[0]['observation_id'] == 'batch-rejected'


def test_loader_patches_use_targeted_sql_and_keep_legacy_full_fallback(tmp_path):
    from system.backend.runtime_patches import install_runtime_patches

    db_path = tmp_path / 'detections.db'
    conn = sqlite3.connect(db_path)
    conn.executescript('''
        CREATE TABLE detections (
            base_name TEXT PRIMARY KEY,
            image_filename TEXT NOT NULL,
            detection_json TEXT NOT NULL
        );
        CREATE INDEX idx_det_imgfile ON detections(image_filename);
        CREATE TABLE validation (
            image_filename TEXT PRIMARY KEY,
            is_validated INTEGER NOT NULL
        );
    ''')
    conn.execute(
        'INSERT INTO detections(base_name,image_filename,detection_json) VALUES(?,?,?)',
        ('a', 'a.jpg', json.dumps({'物种名称': '豹猫'})),
    )
    conn.execute(
        'INSERT INTO validation(image_filename,is_validated) VALUES(?,?)',
        ('a.jpg', 1),
    )
    conn.commit()
    conn.close()

    legacy_calls = []

    def legacy_detection(roots, recursive=False, filenames=None):
        legacy_calls.append(('detection', filenames))
        return {'legacy': {'物种名称': 'legacy'}}

    def legacy_validation(roots, recursive=False):
        legacy_calls.append(('validation', None))
        return {'legacy.jpg': True}

    services = SimpleNamespace(
        _serialize_detector_output=lambda *_args: {
            '物种名称': '空',
            '物种数量': '空',
            '检测框': [],
        },
        preview_media_items=lambda *args, **kwargs: [],
        _load_detection_index=legacy_detection,
        _load_validation_index=legacy_validation,
        _candidate_detection_dbs_for_roots=lambda roots, recursive=False: [db_path],
    )

    install_runtime_patches(services)

    targeted_detection = services._load_detection_index(
        [tmp_path],
        recursive=False,
        filenames={'a.jpg'},
    )
    targeted_validation = services._load_validation_index(
        [tmp_path],
        recursive=False,
        filenames={'a.jpg'},
    )
    assert targeted_detection['a']['物种名称'] == '豹猫'
    assert targeted_validation == {'a.jpg': True}
    assert legacy_calls == []

    assert services._load_detection_index(
        [tmp_path],
        recursive=False,
        filenames=None,
    ) == {'legacy': {'物种名称': 'legacy'}}
    assert services._load_validation_index(
        [tmp_path],
        recursive=False,
        filenames=None,
    ) == {'legacy.jpg': True}
    assert legacy_calls == [('detection', None), ('validation', None)]


def test_installer_activates_all_performance_fast_paths():
    from system.backend import dinov2_feedback_service
    from system.backend.runtime_patches import install_runtime_patches

    services = SimpleNamespace(
        _serialize_detector_output=lambda *_args: {},
        _load_detection_index=lambda *args, **kwargs: {},
        _load_validation_index=lambda *args, **kwargs: {},
        _candidate_detection_dbs_for_roots=lambda *args, **kwargs: [],
        preview_media_items=lambda *args, **kwargs: [],
        preview_media_item=lambda *args, **kwargs: None,
        _reload_validation_item=lambda *args, **kwargs: None,
        mark_validation_items=lambda *args, **kwargs: [],
        _learnable_observations_for_file=lambda *args, **kwargs: [],
    )

    install_runtime_patches(services)

    assert getattr(
        services.preview_media_item,
        '_neri_filtered_single_preview_sql',
        False,
    )
    assert getattr(
        services._reload_validation_item,
        '_neri_filtered_validation_reload_sql',
        False,
    )
    assert getattr(
        services.mark_validation_items,
        '_neri_batched_dinov2_feedback',
        False,
    )
    assert getattr(
        services._learnable_observations_for_file,
        '_neri_indexed_observation_lookup',
        False,
    )
    assert getattr(
        dinov2_feedback_service.persist_runtime_observations,
        '_neri_batched_runtime_persistence',
        False,
    )


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

    updated = preserve_dinov2_rejections(payload, {'detect_results': [result]})

    assert updated['物种名称'] == '豹猫,拒识/Unknown'
    assert updated['物种数量'] == '2,1'
