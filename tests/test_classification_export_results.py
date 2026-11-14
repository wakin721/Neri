"""Final classifier decisions must survive export and event counting."""
import csv
from datetime import datetime, timedelta
import json
import sqlite3

import pandas as pd
import pytest

from system import data_processor
from system.data_processor import DataProcessor


def detector_data(**updates):
    data = {
        '文件名': 'animal.jpg',
        '物种名称': '空',
        '物种数量': '空',
        '最低置信度': None,
        'all_confidences': [0.99],
        'all_classes': [0],
        'names_map': {'0': 'detector-animal'},
    }
    data.update(updates)
    return data


@pytest.fixture
def export_rows(tmp_path, monkeypatch):
    # Keep the tests independent of the installed species catalogue and conf.json.
    database = tmp_path / 'species.db'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE species (中文名, 学名, 纲, 目中文名, '
                           '目拉丁名, 科中文名, 科拉丁名, 属中文名, 属拉丁名, 物种类型)')
    monkeypatch.setattr(data_processor, 'resource_path', lambda _: str(database))
    monkeypatch.chdir(tmp_path)

    def export(rows, settings=None):
        output = tmp_path / 'results.csv'
        assert DataProcessor.export_to_excel(
            rows, str(output), settings or {'global': 0.25}, file_format='csv',
        )
        return pd.read_csv(output, keep_default_na=False).to_dict('records')

    return export


def test_empty_final_boxes_do_not_restore_detector_species(export_rows):
    row = export_rows([detector_data(检测框=[], 分类候选项=[])])[0]
    assert row['物种名称'] == '空'
    assert row['物种数量'] == '空'


def test_dinov3_rejection_is_not_filtered_into_empty_by_legacy_slider(export_rows):
    boxes = [{
        '物种': 'Unknown', '置信度': score, 'accepted': False,
        '候选项': [{'name': 'Unknown', 'conf': score, 'accepted': False,
                  'box_known_score': -2, 'box_threshold': -1},
                 {'name': 'horse', 'conf': 0.999}],
    } for score in (0.171, 0.116, 0.086)]
    row = export_rows([detector_data(检测框=boxes)], {'global': 0.95})[0]
    assert row['物种名称'] == 'Unknown'
    assert str(row['物种数量']) == '3'
    assert row['最低置信度'] == ''


def test_dinov3_accepted_species_survives_high_legacy_display_threshold(export_rows):
    row = export_rows([detector_data(检测框=[{
        '物种': 'horse', '置信度': 0.72, 'accepted': True,
        '候选项': [{'name': 'horse', 'conf': 0.72, 'accepted': True,
                  'sequence_known_score': 1.2, 'sequence_threshold': -0.3}],
    }])], {'global': 0.95})[0]
    assert row['物种名称'] == 'horse'
    assert str(row['物种数量']) == '1'


def test_manual_dinov3_correction_overrides_original_rejection(export_rows):
    row = export_rows([detector_data(检测框=[{
        '物种': 'horse', 'feedback_status': 'confirmed', 'accepted': False,
        '置信度': 0.01, 'box_known_score': -3,
        '候选项': [{'name': 'Unknown', 'conf': 0.01, 'accepted': False}],
    }])], {'global': 0.95})[0]
    assert row['物种名称'] == 'horse'


def test_dinov3_unknown_does_not_start_an_independent_species_event():
    rows = [detector_data(检测框=[{
        '物种': 'Unknown', '置信度': 0.01, 'box_known_score': -3,
        'accepted': False,
    }], 拍摄日期对象=datetime(2026, 11, 13, 12))]
    assert DataProcessor.process_independent_detection(rows, {'global': 0.95})[0]['独立探测首只'] == ''


def test_rejected_unknown_with_null_confidence_survives_export(export_rows):
    row = export_rows([detector_data(检测框=[{
        '物种': '拒识/Unknown', '置信度': None, '候选项': [],
        '边界框': [1, 2, 30, 40], 'observation_id': 'rejected-1',
    }])])[0]
    assert row['物种名称'] == '拒识/Unknown'
    assert str(row['物种数量']) == '1'
    assert row['最低置信度'] == ''


def test_mixed_known_and_unknown_keep_counts_and_numeric_confidence(export_rows):
    row = export_rows([detector_data(检测框=[
        {'物种': 'cat', '置信度': 0.8, '候选项': [{'name': 'cat', 'conf': 0.8}]},
        {'物种': '拒识/Unknown', '置信度': None, '候选项': []},
    ])])[0]
    assert row['物种名称'] == 'cat,拒识/Unknown'
    assert row['物种数量'] == '1,1'
    assert row['最低置信度'] == pytest.approx(0.8)


@pytest.mark.parametrize('candidate', [
    {'name': 'cat', 'conf': 0.1},
    {'name': 'cat', 'conf': 0.99, 'accepted': False},
])
def test_filtered_candidates_do_not_restore_detector_species(export_rows, candidate):
    row = export_rows([detector_data(检测框=[{
        '物种': 'cat', '置信度': 0.99, '候选项': [candidate],
    }])])[0]
    assert row['物种名称'] == '空'


def test_classifier_only_candidates_work_with_empty_boxes(export_rows):
    row = export_rows([detector_data(检测框=[], 分类候选项=[
        {'name': 'cat', 'conf': 0.6}, {'name': 'fox', 'conf': 0.5},
    ])], {'global': 0.25, 'cat': 0.7})[0]
    assert row['物种名称'] == 'fox'
    assert row['最低置信度'] == pytest.approx(0.5)


def test_legacy_detector_data_still_exports(export_rows):
    row = export_rows([detector_data()])[0]
    assert row['物种名称'] == 'detector-animal'
    assert row['最低置信度'] == pytest.approx(0.99)


def test_manual_empty_result_is_authoritative(export_rows):
    row = export_rows([detector_data(检测框=[], 最低置信度='人工校验')])[0]
    assert row['物种名称'] == '空'
    assert row['最低置信度'] == '人工校验'


def test_independent_events_use_classifier_species_instead_of_detector():
    start = datetime(2026, 10, 19, 12)
    rows = [detector_data(**{
        '物种名称': species,
        '检测框': [{'物种': species, '置信度': 0.9}],
        '拍摄日期对象': start + timedelta(seconds=index),
    }) for index, species in enumerate(['cat', 'fox', 'cat'])]
    DataProcessor.process_independent_detection(rows, {'global': 0.25})
    assert [row['独立探测首只'] for row in rows] == ['1', '1', '']


@pytest.mark.parametrize('boxes', [
    [],
    [{'物种': 'cat', '置信度': 0.1}],
    [{'物种': '拒识/Unknown', '置信度': None, '候选项': []}],
])
def test_empty_filtered_and_unknown_do_not_count_as_species_events(boxes):
    row = detector_data(检测框=boxes, 拍摄日期对象=datetime(2026, 10, 19, 12))
    DataProcessor.process_independent_detection([row], {'global': 0.25})
    assert row['独立探测首只'] == ''


def test_independent_events_use_same_candidate_thresholds_as_export(export_rows):
    row = detector_data(检测框=[{
        '物种': 'cat', '置信度': 0.6,
        '候选项': [{'name': 'cat', 'conf': 0.6}, {'name': 'fox', 'conf': 0.5}],
    }], 拍摄日期对象=datetime(2026, 10, 19, 12))
    settings = {'global': 0.25, 'cat': 0.7}
    DataProcessor.process_independent_detection([row], settings)
    exported = export_rows([row], settings)[0]
    assert exported['物种名称'] == 'fox'
    assert str(exported['独立探测首只']) == '1'


def test_csv_export_normalizes_db_nulls_and_integer_working_days(tmp_path, monkeypatch):
    database = tmp_path / 'species.db'
    with sqlite3.connect(database) as connection:
        connection.execute(
            'CREATE TABLE species (中文名, 学名, 纲, 目中文名, '
            '目拉丁名, 科中文名, 科拉丁名, 属中文名, 属拉丁名, 物种类型)'
        )
        connection.execute(
            'INSERT INTO species (中文名, 物种类型) VALUES (?, ?)',
            ('家牛', '家畜'),
        )
    monkeypatch.setattr(data_processor, 'resource_path', lambda _: str(database))
    monkeypatch.chdir(tmp_path)

    rows = [
        detector_data(
            文件名='a.jpg',
            工作天数=1,
            检测框=[{'物种': '家牛', '置信度': 0.9}],
        ),
        detector_data(
            文件名='b.jpg',
            工作天数=None,
            检测框=[],
        ),
    ]
    output = tmp_path / 'results.csv'
    assert DataProcessor.export_to_excel(
        rows, str(output), {'global': 0.25}, file_format='csv',
    )
    with output.open(encoding='utf-8-sig', newline='') as handle:
        exported = list(csv.DictReader(handle))
    assert exported[0]['工作天数'] == '1'
    assert exported[1]['工作天数'] == ''
    assert exported[0]['物种类型'] == '家畜'
    for column in ('学名', '目名', '目拉丁名', '科名', '科拉丁名', '属名', '属拉丁名'):
        assert exported[0][column] == ''


def test_csv_export_preserves_manual_type_and_chinese_comma_events(tmp_path, monkeypatch):
    database = tmp_path / 'species.db'
    with sqlite3.connect(database) as connection:
        connection.execute(
            'CREATE TABLE species (中文名, 学名, 纲, 目中文名, '
            '目拉丁名, 科中文名, 科拉丁名, 属中文名, 属拉丁名, 物种类型)'
        )
        connection.execute(
            'INSERT INTO species (中文名, 物种类型) VALUES (?, ?)',
            ('赤狐', '兽'),
        )
    monkeypatch.setattr(data_processor, 'resource_path', lambda _: str(database))
    monkeypatch.chdir(tmp_path)

    start = datetime(2026, 10, 19, 12)
    rows = [
        detector_data(
            文件名='manual_custom.jpg',
            物种名称='未入库物种',
            物种数量='1',
            物种类型='兽',
            最低置信度='人工校验',
            拍摄日期对象=start,
        ),
        detector_data(
            文件名='manual_multi.jpg',
            物种名称='赤狐，豹猫',
            物种数量='1,1',
            最低置信度='人工校验',
            拍摄日期对象=start + timedelta(seconds=10),
        ),
    ]
    DataProcessor.process_independent_detection(rows, {'global': 0.25})
    output = tmp_path / 'results.csv'
    assert DataProcessor.export_to_excel(
        rows, str(output), {'global': 0.25}, file_format='csv',
    )
    with output.open(encoding='utf-8-sig', newline='') as handle:
        exported = list(csv.DictReader(handle))
    assert exported[0]['物种类型'] == '兽'
    assert exported[0]['独立探测首只'] == '1'
    assert exported[1]['物种类型'] == '兽'
    assert exported[1]['独立探测首只'] == '1'


def test_independent_events_and_export_share_conf_json_thresholds(tmp_path, monkeypatch):
    database = tmp_path / 'species.db'
    with sqlite3.connect(database) as connection:
        connection.execute(
            'CREATE TABLE species (中文名, 学名, 纲, 目中文名, '
            '目拉丁名, 科中文名, 科拉丁名, 属中文名, 属拉丁名, 物种类型)'
        )
    monkeypatch.setattr(data_processor, 'resource_path', lambda _: str(database))
    (tmp_path / 'temp').mkdir()
    (tmp_path / 'temp' / 'conf.json').write_text(
        json.dumps({'fox': 0.8}), encoding='utf-8',
    )
    monkeypatch.setattr(data_processor, '_confidence_config_path', lambda: tmp_path / 'temp' / 'conf.json')
    monkeypatch.chdir(tmp_path)

    row = detector_data(
        检测框=[{'物种': 'fox', '置信度': 0.5}],
        拍摄日期对象=datetime(2026, 10, 19, 12),
    )
    DataProcessor.process_independent_detection([row], {'global': 0.25})
    output = tmp_path / 'results.csv'
    assert DataProcessor.export_to_excel(
        [row], str(output), {'global': 0.25}, file_format='csv',
    )
    with output.open(encoding='utf-8-sig', newline='') as handle:
        exported = list(csv.DictReader(handle))[0]
    assert exported['物种名称'] == '空'
    assert exported['独立探测首只'] == ''


def test_fast_video_boxes_export_max_per_frame_count(export_rows):
    row = detector_data(
        文件名='clip.mp4',
        格式='mp4',
        video_mode='fast',
        检测框=[
            {'物种': 'fox', '置信度': 0.8, 'frame_index': 25},
            {'物种': 'fox', '置信度': 0.85, 'frame_index': 50},
            {'物种': 'fox', '置信度': 0.9, 'frame_index': 50},
            {'物种': 'fox', '置信度': 0.75, 'frame_index': 75},
        ],
    )
    exported = export_rows([row])[0]
    assert exported['物种名称'] == 'fox'
    assert str(exported['物种数量']) == '2'


@pytest.mark.parametrize('file_format', ['csv', 'excel'])
def test_export_without_species_database(tmp_path, monkeypatch, file_format):
    database = tmp_path / 'missing.db'
    monkeypatch.setattr(data_processor, 'resource_path', lambda _: str(database))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / ('results.csv' if file_format == 'csv' else 'results.xlsx')
    rows = [detector_data(检测框=[{'物种': 'fox', '置信度': 0.9}])]

    assert DataProcessor.export_to_excel(
        rows, str(output), {'global': 0.25}, file_format=file_format,
    )
    if file_format == 'csv':
        exported = pd.read_csv(output, keep_default_na=False)
    else:
        exported = pd.read_excel(output, keep_default_na=False)
    assert exported.loc[0, '物种名称'] == 'fox'
    assert exported.loc[0, '物种数量'] == 1
    assert exported.loc[0, '学名'] == ''
    assert not database.exists()


@pytest.mark.parametrize('manual, existing_type, expected_type', [
    (True, '自定义类型', '自定义类型'),
    (True, '兽,自定义类型', '兽,自定义类型'),
    (True, '', '兽'),
    (True, '   ', '兽'),
    (False, '旧类型', '兽'),
])
def test_export_manual_type_takes_priority_over_catalogue(
    export_rows, tmp_path, manual, existing_type, expected_type,
):
    with sqlite3.connect(tmp_path / 'species.db') as connection:
        connection.execute(
            'INSERT INTO species (中文名, 物种类型) VALUES (?, ?)', ('赤狐', '兽'),
        )
    row = detector_data(
        物种名称='赤狐', 物种数量='1', 物种类型=existing_type,
        最低置信度='人工校验' if manual else 0.9,
        检测框=[{'物种': '赤狐', '置信度': 0.9}],
    )

    exported = export_rows([row])[0]
    assert exported['物种类型'] == expected_type
    assert exported['物种名称'] == '赤狐'
    assert str(exported['物种数量']) == '1'


@pytest.mark.parametrize('source', ['manual', 'tracks', 'boxes'])
@pytest.mark.parametrize('unknown', ['Unknown', '拒识/Unknown'])
def test_unknowns_do_not_create_independent_events_in_any_source(
    export_rows, source, unknown,
):
    start = datetime(2026, 11, 13, 12)
    rows = []
    for index, species_names in enumerate((['fox'], [unknown], ['fox', unknown])):
        row = detector_data(
            文件名=f'{index}.jpg', 拍摄日期对象=start + timedelta(seconds=index),
        )
        if source == 'manual':
            row.update({
                '物种名称': '，'.join(species_names),
                '物种数量': ','.join('1' for _ in species_names),
                '最低置信度': '人工校验',
            })
        elif source == 'tracks':
            row['tracks'] = {
                str(track_id): [{'species': species, 'confidence': 0.9}]
                for track_id, species in enumerate(species_names)
            }
        else:
            row['检测框'] = [
                {'物种': species, '置信度': 0.9} for species in species_names
            ]
        rows.append(row)

    DataProcessor.process_independent_detection(rows, {'global': 0.25})
    exported = export_rows(rows)
    assert [str(row['独立探测首只']) for row in exported] == ['1', '', '']
    # Unknown results remain visible in the CSV even though they are not events.
    assert exported[1]['物种名称'] == unknown
    assert unknown in exported[2]['物种名称']


@pytest.mark.parametrize('invalid', [None, 'invalid', float('nan'), float('inf'), float('-inf'), True])
def test_tracks_ignore_invalid_confidences_and_exclude_losing_votes(export_rows, invalid):
    row = detector_data(
        tracks={'1': [
            {'species': 'fox', 'confidence': 0.9},
            {'species': 'fox', 'confidence': 0.8},
            {'species': 'fox', 'confidence': invalid},
            {'species': 'cat', 'confidence': 0.3},
        ]}, 拍摄日期对象=datetime(2026, 11, 13, 12),
    )
    DataProcessor.process_independent_detection([row], {'global': 0.25})
    exported = export_rows([row])[0]
    assert exported['物种名称'] == 'fox'
    assert str(exported['物种数量']) == '1'
    assert exported['最低置信度'] == pytest.approx(0.8)
    assert str(exported['独立探测首只']) == '1'


def test_all_invalid_track_confidences_export_detected_empty(export_rows):
    row = detector_data(
        tracks={'1': [{'species': 'fox', 'confidence': None}]},
        拍摄日期对象=datetime(2026, 11, 13, 12),
    )
    DataProcessor.process_independent_detection([row], {'global': 0.25})
    exported = export_rows([row])[0]
    assert exported['物种名称'] == '空'
    assert exported['最低置信度'] == ''
    assert exported['独立探测首只'] == ''
    assert exported['检测状态'] == '已检测'


def test_legacy_thresholds_use_application_root_not_current_directory(tmp_path, monkeypatch):
    app = tmp_path / 'app'
    (app / 'temp').mkdir(parents=True)
    (app / 'temp/conf.json').write_text(json.dumps({'fox': 0.8}), encoding='utf-8')
    unrelated = tmp_path / 'other'
    (unrelated / 'temp').mkdir(parents=True)
    (unrelated / 'temp/conf.json').write_text(json.dumps({'fox': 0.1}), encoding='utf-8')
    monkeypatch.setattr(data_processor, '__file__', str(app / 'system/data_processor.py'))
    monkeypatch.chdir(unrelated)
    assert data_processor._resolve_confidence_settings({'global': 0.25})['fox'] == 0.8
    assert data_processor._resolve_confidence_settings({'fox': 0.7})['fox'] == 0.7


def test_human_image_annotation_is_authoritative_for_events_and_export(export_rows):
    start = datetime(2026, 11, 13, 12)
    rows = [
        detector_data(
            image_annotation={'source': 'human', 'species': 'fox', 'count': '4'},
            检测框=[{'物种': 'bird', '置信度': 0.9}], 拍摄日期对象=start,
        ),
        detector_data(
            检测框=[{'物种': 'fox', '置信度': 0.99}],
            拍摄日期对象=start + timedelta(seconds=1),
        ),
    ]
    DataProcessor.process_independent_detection(rows, {'global': 0.95})
    exported = export_rows(rows, {'global': 0.95})
    assert exported[0]['物种名称'] == 'fox'
    assert str(exported[0]['物种数量']) == '4'
    assert exported[0]['最低置信度'] == '人工校验'
    assert [str(row['独立探测首只']) for row in exported] == ['1', '']
