import json
import sqlite3
from pathlib import Path

from system.backend.preview_fast import load_preview_indexes
from system.dinov3.feedback_index import (
    install_feedback_store_patches,
    normalize_observation_source_path,
)


def test_preview_20k_rows_returns_only_requested_filenames(tmp_path):
    db_path = tmp_path / 'detections.db'
    conn = sqlite3.connect(db_path)
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
        'INSERT INTO detections(base_name,image_filename,detection_json) '
        'VALUES(?,?,?)',
        (
            (f'image-{index}', f'image-{index}.jpg', json.dumps({'index': index}))
            for index in range(20000)
        ),
    )
    conn.executemany(
        'INSERT INTO validation(image_filename,is_validated) VALUES(?,?)',
        ((f'image-{index}.jpg', index % 2) for index in range(20000)),
    )
    conn.commit()
    conn.close()

    requested = {f'image-{index}.jpg' for index in range(100)}
    detections, validations = load_preview_indexes([db_path], requested)

    assert len(detections) == 100
    assert len(validations) == 100
    assert set(detections) == {f'image-{index}' for index in range(100)}


class _ScaleFeedbackStore:
    def __init__(self, path):
        self.path = Path(path)
        self.materialized_ids = []
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript('''
            CREATE TABLE IF NOT EXISTS metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observations(
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                embedding BLOB NOT NULL
            );
        ''')
        self._conn.commit()

    def get_observation(self, observation_id):
        self.materialized_ids.append(observation_id)
        return observation_id


def test_50k_observation_lookup_materializes_only_target_and_uses_index(tmp_path):
    install_feedback_store_patches(_ScaleFeedbackStore)
    store = _ScaleFeedbackStore(tmp_path / 'feedback.sqlite3')
    store._conn.executemany(
        'INSERT INTO observations(id,source_path,payload,embedding) VALUES(?,?,?,?)',
        (
            (
                f'noise-{index}',
                normalize_observation_source_path(tmp_path / f'noise-{index}.jpg'),
                '{}',
                b'x',
            )
            for index in range(50000)
        ),
    )
    target = normalize_observation_source_path(tmp_path / 'target.jpg')
    store._conn.execute(
        'INSERT INTO observations(id,source_path,payload,embedding) VALUES(?,?,?,?)',
        ('target', target, '{}', b'x'),
    )
    store._conn.commit()

    assert store.observations_for_source_path(tmp_path / 'target.jpg') == ['target']
    assert store.materialized_ids == ['target']
    plan = store._conn.execute(
        'EXPLAIN QUERY PLAN SELECT id FROM observations WHERE source_path=?',
        (target,),
    ).fetchall()
    assert any(
        'idx_observations_source_path' in str(row[3]).lower()
        for row in plan
    )
    store._conn.close()
