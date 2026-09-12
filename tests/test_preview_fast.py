import json
import sqlite3
from pathlib import Path

from system.backend.preview_fast import load_preview_indexes


def _make_db(path: Path):
    conn = sqlite3.connect(path)
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
    return conn


def test_preview_index_queries_only_requested_filenames(tmp_path, monkeypatch):
    db_path = tmp_path / 'detections.db'
    conn = _make_db(db_path)
    conn.executemany(
        'INSERT INTO detections(base_name, image_filename, detection_json) VALUES(?,?,?)',
        [
            ('wanted1', 'wanted1.jpg', json.dumps({'物种名称': '豹猫'})),
            ('wanted2', 'wanted2.jpg', json.dumps({'物种名称': '拒识/Unknown'})),
            *[
                (f'noise{i}', f'noise{i}.jpg', json.dumps({'物种名称': '噪声'}))
                for i in range(200)
            ],
        ],
    )
    conn.executemany(
        'INSERT INTO validation(image_filename, is_validated) VALUES(?,?)',
        [
            ('wanted1.jpg', 1),
            ('wanted2.jpg', 0),
            *[(f'noise{i}.jpg', 0) for i in range(200)],
        ],
    )
    conn.commit()
    conn.close()

    statements = []
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        traced = real_connect(*args, **kwargs)
        traced.set_trace_callback(statements.append)
        return traced

    import system.backend.preview_fast as preview_fast

    monkeypatch.setattr(preview_fast.sqlite3, 'connect', traced_connect)

    detections, validations = load_preview_indexes(
        [db_path],
        {'wanted1.jpg', 'wanted2.jpg'},
    )

    assert set(detections) == {'wanted1', 'wanted2'}
    assert validations == {'wanted1.jpg': True, 'wanted2.jpg': False}
    select_statements = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith('SELECT')
    ]
    assert select_statements
    assert all(
        'WHERE IMAGE_FILENAME IN' in statement.upper()
        for statement in select_statements
    )


def test_preview_index_chunks_large_filename_sets(tmp_path):
    db_path = tmp_path / 'detections.db'
    conn = _make_db(db_path)
    rows = [
        (f'image{i}', f'image{i}.jpg', json.dumps({'物种名称': '豹猫'}))
        for i in range(1200)
    ]
    conn.executemany(
        'INSERT INTO detections(base_name, image_filename, detection_json) VALUES(?,?,?)',
        rows,
    )
    conn.commit()
    conn.close()

    detections, _ = load_preview_indexes(
        [db_path],
        {f'image{i}.jpg' for i in range(1200)},
    )

    assert len(detections) == 1200


class _FakeItem:
    def __init__(self, path, data=None, validated=None):
        self.path = str(path)
        self.data = data
        self.validated = validated


class _FakeServices:
    def __init__(self, files, db_path):
        self.files = files
        self.db_path = db_path
        self.detection_calls = 0
        self.validation_calls = 0

    def _resolve_supported_inputs(self, input_path):
        return iter(self.files)

    def _preview_detection_db_roots(self, input_path, output_dir, files):
        return [input_path]

    def _candidate_detection_dbs_for_roots(self, roots, recursive=False):
        assert recursive is False
        return [self.db_path]

    def _build_fast_metadata_item(self, path):
        return _FakeItem(path)

    def _apply_detection_data(self, item, data):
        self.detection_calls += 1
        return _FakeItem(item.path, data=data, validated=item.validated)

    def _apply_validation_state(self, item, filename, validation_index):
        self.validation_calls += 1
        return _FakeItem(
            item.path,
            data=item.data,
            validated=validation_index.get(filename),
        )


def test_fast_preview_uses_filtered_indexes_and_keeps_legacy_item_pipeline(tmp_path):
    from system.backend.preview_fast import make_preview_media_items

    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    files = [input_dir / 'a.jpg', input_dir / 'b.jpg']
    for path in files:
        path.write_bytes(b'x')

    db_path = input_dir / 'detections.db'
    conn = _make_db(db_path)
    conn.execute(
        'INSERT INTO detections(base_name, image_filename, detection_json) VALUES(?,?,?)',
        ('a', 'a.jpg', json.dumps({'物种名称': '拒识/Unknown'})),
    )
    conn.execute(
        'INSERT INTO validation(image_filename, is_validated) VALUES(?,?)',
        ('a.jpg', 1),
    )
    conn.commit()
    conn.close()

    services = _FakeServices(files, db_path)
    preview = make_preview_media_items(services)
    items = preview(str(input_dir), include_cached=True)

    assert [Path(item.path).name for item in items] == ['a.jpg', 'b.jpg']
    assert items[0].data == {'物种名称': '拒识/Unknown'}
    assert items[0].validated is True
    assert items[1].data is None
    assert items[1].validated is None
    assert services.detection_calls == 1
    assert services.validation_calls == 2
