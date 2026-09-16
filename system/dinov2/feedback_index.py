"""Indexed compatibility layer for persisted DINOv3 feedback observations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

_MIGRATION_KEY = 'migration_observations_source_path_v1'


def normalize_observation_source_path(value: str | Path) -> str:
    """Return the stable path identity used by validation feedback lookups."""
    text = str(value or '').strip()
    if not text:
        return ''
    try:
        text = str(Path(text).expanduser().resolve())
    except Exception:
        pass
    return text.replace('\\', '/').casefold()


def ensure_observation_source_index(store: Any) -> None:
    """Add and backfill the observation source-path projection once."""
    conn = store._conn  # noqa: SLF001 - same SQLite ownership boundary
    columns = {
        str(row['name'] if hasattr(row, 'keys') else row[1])
        for row in conn.execute('PRAGMA table_info(observations)').fetchall()
    }
    if 'source_path' not in columns:
        conn.execute('ALTER TABLE observations ADD COLUMN source_path TEXT')
    conn.execute(
        'CREATE INDEX IF NOT EXISTS idx_observations_source_path '
        'ON observations(source_path)'
    )

    marker = conn.execute(
        'SELECT value FROM metadata WHERE key=?',
        (_MIGRATION_KEY,),
    ).fetchone()
    if marker is None:
        rows = conn.execute(
            'SELECT id,payload FROM observations WHERE source_path IS NULL'
        ).fetchall()
        updates: list[tuple[str, str]] = []
        for row in rows:
            try:
                payload = json.loads(row['payload'])
            except (TypeError, json.JSONDecodeError):
                continue
            source_path = normalize_observation_source_path(
                payload.get('source_path', '')
            )
            if source_path:
                updates.append((source_path, str(row['id'])))
        if updates:
            conn.executemany(
                'UPDATE observations SET source_path=? WHERE id=?',
                updates,
            )
        conn.execute(
            'INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)',
            (_MIGRATION_KEY, '1'),
        )
    conn.commit()


def persist_observations(
    store: Any,
    observations: Iterable[Any],
) -> tuple[str, ...]:
    """Persist an observation batch with one atomic SQLite commit."""
    rows: list[tuple[str, str, str, bytes]] = []
    ids: list[str] = []
    for observation in observations:
        payload = {
            key: value
            for key, value in observation.__dict__.items()
            if key != 'embedding'
        }
        captured_at = getattr(observation, 'captured_at', None)
        payload['captured_at'] = captured_at.isoformat() if captured_at else None
        observation_id = str(observation.id)
        rows.append(
            (
                observation_id,
                normalize_observation_source_path(observation.source_path),
                json.dumps(payload, ensure_ascii=False),
                observation.embedding.astype('<f4').tobytes(),
            )
        )
        ids.append(observation_id)

    if not rows:
        return ()
    try:
        store._conn.executemany(  # noqa: SLF001 - same SQLite ownership boundary
            'INSERT OR REPLACE INTO observations(id,source_path,payload,embedding) '
            'VALUES(?,?,?,?)',
            rows,
        )
        store._conn.commit()  # noqa: SLF001
    except Exception:
        store._conn.rollback()  # noqa: SLF001
        raise
    return tuple(ids)


def observations_for_source_path(
    store: Any,
    source_path: str | Path,
) -> list[Any]:
    """Materialize only observations belonging to one normalized source path."""
    identity = normalize_observation_source_path(source_path)
    if not identity:
        return []
    rows = store._conn.execute(  # noqa: SLF001 - same SQLite ownership boundary
        'SELECT id FROM observations WHERE source_path=? ORDER BY id',
        (identity,),
    ).fetchall()
    return [
        store.get_observation(str(row['id']))
        for row in rows
        if row['id']
    ]


def install_feedback_store_patches(store_cls=None) -> None:
    """Install the additive projection on HumanFeedbackStore idempotently."""
    if store_cls is None:
        from .feedback import HumanFeedbackStore

        store_cls = HumanFeedbackStore
    if getattr(store_cls, '_neri_source_path_indexed', False):
        return

    original_init_schema = store_cls._init_schema

    def patched_init_schema(self):
        original_init_schema(self)
        ensure_observation_source_index(self)

    def patched_persist_observations(self, observations):
        return persist_observations(self, observations)

    def patched_persist_observation(self, observation):
        return patched_persist_observations(self, (observation,))[0]

    def patched_observations_for_source_path(self, source_path):
        return observations_for_source_path(self, source_path)

    store_cls._init_schema = patched_init_schema
    store_cls.persist_observations = patched_persist_observations
    store_cls.persist_observation = patched_persist_observation
    store_cls.observations_for_source_path = patched_observations_for_source_path
    store_cls._neri_source_path_indexed = True
