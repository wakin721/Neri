import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from system.dinov3.feedback_index import (
    install_feedback_store_patches,
    normalize_observation_source_path,
)


class _LegacyStore:
    def __init__(self, path):
        self.path = Path(path)
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
        return observation_id


def test_source_path_migration_backfills_valid_rows_and_is_idempotent(tmp_path):
    path = tmp_path / 'feedback.sqlite3'
    conn = sqlite3.connect(path)
    conn.executescript('''
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE observations(
            id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            embedding BLOB NOT NULL
        );
    ''')
    conn.executemany(
        'INSERT INTO observations(id,payload,embedding) VALUES(?,?,?)',
        [
            ('good', json.dumps({'source_path': str(tmp_path / 'A.JPG')}), b'x'),
            ('bad', '{not-json', b'x'),
            ('other', json.dumps({'source_path': str(tmp_path / 'B.JPG')}), b'x'),
        ],
    )
    conn.commit()
    conn.close()

    install_feedback_store_patches(_LegacyStore)
    store = _LegacyStore(path)
    columns = {
        row['name']
        for row in store._conn.execute('PRAGMA table_info(observations)')
    }
    indexes = {
        row['name']
        for row in store._conn.execute('PRAGMA index_list(observations)')
    }
    assert 'source_path' in columns
    assert 'idx_observations_source_path' in indexes
    expected = normalize_observation_source_path(tmp_path / 'A.JPG')
    assert store._conn.execute(
        'SELECT source_path FROM observations WHERE id="good"'
    ).fetchone()[0] == expected
    assert store._conn.execute(
        'SELECT source_path FROM observations WHERE id="bad"'
    ).fetchone()[0] is None
    assert store._conn.execute(
        "SELECT value FROM metadata "
        "WHERE key='migration_observations_source_path_v1'"
    ).fetchone()[0] == '1'
    store._conn.close()

    reopened = _LegacyStore(path)
    assert reopened._conn.execute(
        'SELECT source_path FROM observations WHERE id="good"'
    ).fetchone()[0] == expected
    reopened._conn.close()


def test_source_path_lookup_uses_index_and_materializes_only_target(tmp_path):
    install_feedback_store_patches(_LegacyStore)
    store = _LegacyStore(tmp_path / 'feedback.sqlite3')
    noise = [
        (
            f'n-{index}',
            normalize_observation_source_path(tmp_path / f'n-{index}.jpg'),
            '{}',
            b'x',
        )
        for index in range(3000)
    ]
    target = normalize_observation_source_path(tmp_path / 'target.jpg')
    store._conn.executemany(
        'INSERT INTO observations(id,source_path,payload,embedding) VALUES(?,?,?,?)',
        noise,
    )
    store._conn.execute(
        'INSERT INTO observations(id,source_path,payload,embedding) VALUES(?,?,?,?)',
        ('target', target, '{}', b'x'),
    )
    store._conn.commit()

    statements = []
    store._conn.set_trace_callback(statements.append)
    assert store.observations_for_source_path(tmp_path / 'target.jpg') == ['target']
    selects = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith('SELECT')
    ]
    assert any(
        'WHERESOURCE_PATH=' in statement.upper().replace(' ', '')
        for statement in selects
    )
    plan = store._conn.execute(
        'EXPLAIN QUERY PLAN SELECT id FROM observations WHERE source_path=?',
        (target,),
    ).fetchall()
    assert any(
        'idx_observations_source_path' in str(row[3]).lower()
        for row in plan
    )
    store._conn.close()


def test_persist_observations_writes_batch_and_source_projection(tmp_path):
    install_feedback_store_patches(_LegacyStore)
    store = _LegacyStore(tmp_path / 'feedback.sqlite3')
    observations = [
        SimpleNamespace(
            id=f'obs-{index}',
            source_path=str(tmp_path / f'{index}.JPG'),
            captured_at=None,
            embedding=np.eye(3, dtype=np.float32)[index],
            value=index,
        )
        for index in range(3)
    ]

    assert store.persist_observations(observations) == (
        'obs-0',
        'obs-1',
        'obs-2',
    )
    rows = store._conn.execute(
        'SELECT id,source_path,payload FROM observations ORDER BY id'
    ).fetchall()
    assert len(rows) == 3
    assert rows[0]['source_path'] == normalize_observation_source_path(
        tmp_path / '0.JPG'
    )
    assert json.loads(rows[0]['payload'])['source_path'] == str(tmp_path / '0.JPG')
    store._conn.close()
