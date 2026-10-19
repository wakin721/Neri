"""Final classifier decisions must survive export and event counting."""
from datetime import datetime, timedelta
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
