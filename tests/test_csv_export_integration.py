"""Exercise real annotation persistence, reload and CSV serialization together."""
import csv
from contextlib import closing
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from PIL import Image
import pytest

from system import data_processor
from system.backend import services
from system.backend import services_legacy as legacy
from system.backend.models import (
    ValidationBatchMarkRequest, ValidationExportRequest, ValidationMarkRequest,
)


@pytest.fixture
def media(tmp_path, monkeypatch):
    catalogue = tmp_path / 'species.db'
    with closing(sqlite3.connect(catalogue)) as connection:
        connection.execute(
            'CREATE TABLE species (中文名, 学名, 纲, 目中文名, 目拉丁名, '
            '科中文名, 科拉丁名, 属中文名, 属拉丁名, 物种类型)'
        )
        connection.executemany(
            'INSERT INTO species (中文名, 物种类型) VALUES (?, ?)',
            [('fox', '兽'), ('cat', '兽'), ('bird', '鸟')],
        )
        connection.commit()
    monkeypatch.setattr(data_processor, 'resource_path', lambda _: str(catalogue))
    monkeypatch.setattr(legacy, 'resource_path', lambda _: str(catalogue))
    # These tests exercise annotation/CSV integration, not legacy config loading.
    monkeypatch.setattr(data_processor, '_resolve_confidence_settings', lambda values: dict(values or {}))
    monkeypatch.setenv('NERI_DINO_STATE_DIR', str(tmp_path / 'state'))
    import system.training
    monkeypatch.setattr(system.training, 'get_queue', lambda: SimpleNamespace(
        enqueue_many_async=lambda values: None,
        enqueue_async=lambda *args, **kwargs: None,
    ))

    def create(relative, payload=None):
        path = tmp_path / 'media' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == '.mp4':
            path.write_bytes(b'test-video')
        else:
            exif = Image.Exif()
            exif[36867] = '2026:11:13 12:00:00'
            with Image.new('RGB', (20, 20)) as image:
                image.save(path, exif=exif)
        if payload is not None:
            legacy._persist_validation_updates([(path, payload, None)], path.parent)
        return path

    return tmp_path / 'media', create


def export_rows(root, **kwargs):
    response = services.export_validation_data(ValidationExportRequest(
        input_path=str(root), file_format='csv', **kwargs,
    ))
    rows = []
    for output in response.output_paths:
        with Path(output).open(encoding='utf-8-sig', newline='') as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def box(species, confidence, **kwargs):
    return {'物种': species, '置信度': confidence, '边界框': [0, 0, 8, 8], **kwargs}


@pytest.mark.parametrize('batch', [False, True])
def test_confirm_filters_reach_persisted_csv_for_single_and_batch(media, batch):
    root, create = media
    paths = []
    for count in (1, 2):
        paths.append(create(f'{count}.jpg', {
            '物种名称': 'fox,cat', '物种数量': f'{count},1',
            '检测框': [box('fox', 0.9) for _ in range(count)] + [box('cat', 0.3)],
        }))
    options = dict(input_path=str(root), action='correct',
                   confidence_settings={'global': 0.8}, min_frame_ratio=0.2)
    if batch:
        services.mark_validation_items(ValidationBatchMarkRequest(
            file_paths=list(map(str, paths)), **options,
        ))
    else:
        for path in paths:
            services.mark_validation_item(ValidationMarkRequest(file_path=str(path), **options))
    for path, count in zip(paths, (1, 2)):
        reloaded = services.preview_media_item(str(path), str(root))
        assert reloaded.detection_data['物种名称'] == 'fox'
        assert reloaded.detection_data['物种数量'] == str(count)
    rows = export_rows(root, confidence_settings={'global': 0.99})
    assert [(row['物种名称'], row['物种数量']) for row in rows] == [('fox', '1'), ('fox', '2')]
    assert all(row['最低置信度'] == '人工校验' for row in rows)


def test_video_confirmation_uses_filtered_tracks_then_preserves_human_count(media):
    root, create = media
    path = create('clip.mp4', {
        'total_frames_processed': 10,
        'tracks': {
            'fox': [{'species': 'fox', 'confidence': 0.9, 'bbox': [0, 0, 8, 8],
                     'frame_index': i} for i in range(5)],
            'bird': [{'species': 'bird', 'confidence': 0.9, 'bbox': [0, 0, 8, 8],
                      'frame_index': 0}],
        },
    })
    services.mark_validation_item(ValidationMarkRequest(
        input_path=str(root), file_path=str(path), action='correct',
        confidence_settings={'global': 0.8}, min_frame_ratio=0.2,
    ))
    row = export_rows(root)[0]
    assert (row['物种名称'], row['物种数量'], row['格式']) == ('fox', '1', 'MP4')
    services.mark_validation_item(ValidationMarkRequest(
        input_path=str(root), file_path=str(path), action='update',
        species_name='fox', species_count='4',
    ))
    services.mark_validation_item(ValidationMarkRequest(
        input_path=str(root), file_path=str(path), action='correct',
        confidence_settings={'global': 0.99}, min_frame_ratio=1.0,
    ))
    assert export_rows(root)[0]['物种数量'] == '4'


def test_species_change_invalidates_old_manual_type_across_reload(media):
    root, create = media
    path = create('photo.jpg', {
        '物种名称': 'bird', '物种数量': '1', '物种类型': '鸟', '最低置信度': '人工校验',
    })
    services.mark_validation_item(ValidationMarkRequest(
        input_path=str(root), file_path=str(path), action='update',
        species_name='fox', species_count='1',
    ))
    row = export_rows(root)[0]
    assert (row['物种名称'], row['物种类型']) == ('fox', '兽')


def test_combined_export_retains_camera_identity_status_and_tie_order(media):
    root, create = media
    for name in ('cam-a/IMG_10.JPG', 'cam-a/IMG_2.JPG', 'cam-b/IMG_2.JPG'):
        create(name, {'检测框': [box('fox', 0.9)]})
    create('cam-b/new.JPG')
    create('cam-b/empty.JPG', {'检测框': []})
    rows = export_rows(root)
    indexed = {row['相对路径'].replace('\\', '/'): row for row in rows}
    assert len(indexed) == 5
    assert indexed['cam-a/IMG_2.JPG']['独立探测首只'] == '1'
    assert indexed['cam-a/IMG_10.JPG']['独立探测首只'] == ''
    assert indexed['cam-b/IMG_2.JPG']['独立探测首只'] == '1'
    assert indexed['cam-b/new.JPG']['检测状态'] == '未检测'
    assert indexed['cam-b/new.JPG']['物种名称'] == ''
    assert indexed['cam-b/empty.JPG']['检测状态'] == '已检测'
    assert indexed['cam-b/empty.JPG']['物种名称'] == '空'
    assert {row['格式'] for row in rows} == {'JPG'}


def test_export_uses_current_source_identity_instead_of_stale_payload(media):
    root, create = media
    create('cam/photo.JPG', {
        '文件名': 'old-name.mp4', '格式': 'mp4', '相对路径': 'old-camera/old-name.mp4',
        '_source_path': 'old-source', '检测框': [],
    })
    row = export_rows(root)[0]
    assert row['文件名'] == 'photo.JPG'
    assert row['格式'] == 'JPG'
    assert row['相对路径'].replace('\\', '/') == 'cam/photo.JPG'


@pytest.mark.parametrize('other_automatic_box', [False, True])
def test_real_dinov3_box_feedback_keeps_raw_score_out_of_csv(
    media, monkeypatch, other_automatic_box,
):
    from system.backend import dinov3_learning as learning
    from system.dinov3.observation_support import apply_dinov3_box_feedback, make_dinov3_observation_id

    root, create = media
    boxes = [box('Unknown', 0.01, accepted=False, within_score=-3)]
    if other_automatic_box:
        boxes.append(box('cat', 0.85, accepted=True, within_score=0.9))
    path = create('feedback.jpg', {'检测框': boxes, 'sequence_completed': True})
    # Crop learning is an independent subsystem; retain real annotation/journal/DB writes.
    monkeypatch.setattr(learning, 'stage_feedback', lambda *args, **kwargs: None)
    monkeypatch.setattr(learning, 'schedule_learning', lambda *args: None)
    apply_dinov3_box_feedback(SimpleNamespace(
        input_path=str(root), file_path=str(path), classification_model_path=str(root / 'model.neri.json'),
        observation_id=make_dinov3_observation_id(path, 0, boxes[0]['边界框']),
        action='update', species_name='fox', feedback_operation_id='csv-feedback',
    ))
    saved = services.preview_media_item(str(path), str(root)).detection_data
    assert saved['检测框'][0]['置信度'] == 0.01
    row = export_rows(root)[0]
    assert 'fox' in row['物种名称']
    if other_automatic_box:
        assert float(row['最低置信度']) == pytest.approx(0.85)
    else:
        assert row['最低置信度'] == '人工校验'
