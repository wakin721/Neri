import json
import sqlite3
from pathlib import Path

from system.backend.preview_fast import (
    make_preview_media_item,
    make_reload_validation_item,
)


class _Item:
    def __init__(self, path, data=None, validated=None):
        self.path = str(path)
        self.data = data
        self.validated = validated


class _Services:
    def __init__(self, db_path):
        self.db_path = db_path

    def _build_metadata_item(self, path):
        return _Item(path)

    def _detection_db_search_roots(self, input_path, output_dir):
        return [input_path]

    def _load_detection_data_for_path(self, path, roots, recursive=False):
        return {'物种名称': '豹猫'}

    def _apply_detection_data(self, item, data):
        return _Item(item.path, data=data, validated=item.validated)

    def _apply_validation_state(self, item, filename, validation_index):
        return _Item(
            item.path,
            data=item.data,
            validated=validation_index.get(filename),
        )

    def _candidate_detection_dbs_for_roots(self, roots, recursive=False):
        return [self.db_path]


def _database(path: Path):
    conn = sqlite3.connect(path)
    conn.executescript('''
        CREATE TABLE detections(
            base_name TEXT PRIMARY KEY,
            image_filename TEXT NOT NULL,
            detection_json TEXT NOT NULL
        );
        CREATE INDEX idx_det_imgfile ON detections(image_filename);
        CREATE TABLE validation(
            image_filename TEXT PRIMARY KEY,
            is_validated INTEGER NOT NULL
        );
    ''')
    conn.executemany(
        'INSERT INTO detections(base_name,image_filename,detection_json) VALUES(?,?,?)',
        [
            ('a', 'a.jpg', json.dumps({'物种名称': '豹猫'})),
            ('noise', 'noise.jpg', json.dumps({'物种名称': '噪声'})),
        ],
    )
    conn.executemany(
        'INSERT INTO validation(image_filename,is_validated) VALUES(?,?)',
        [('a.jpg', 1), ('noise.jpg', 0)],
    )
    conn.commit()
    conn.close()


def test_single_preview_and_reload_use_targeted_validation_lookup(tmp_path):
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    path = input_dir / 'a.jpg'
    path.write_bytes(b'x')
    db_path = input_dir / 'detections.db'
    _database(db_path)
    services = _Services(db_path)

    preview = make_preview_media_item(services)
    preview_item = preview(str(path), input_dir=str(input_dir))
    assert preview_item.validated is True
    assert preview_item.data == {'物种名称': '豹猫'}

    reload_item = make_reload_validation_item(services)
    reloaded = reload_item(path, input_dir)
    assert reloaded.validated is True
    assert reloaded.data == {'物种名称': '豹猫'}
