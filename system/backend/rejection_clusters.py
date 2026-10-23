"""Stable links from rejected observations to fingerprint-scoped Registry clusters."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np


def cluster_assignment(fingerprint, entry):
    # Runtime rejections only enter candidate registrations, whose prototype
    # budget is one. This is the same cluster 0 exposed by cluster_details().
    return {
        "cluster_id": f"registry:{fingerprint}:{entry.id}:0",
        "registry_id": int(entry.id),
        "candidate_number": int(entry.candidate_number),
        "prototype_index": 0,
        "model_fingerprint": fingerprint,
    }


def _schema(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rejection_cluster_assignments("
        "observation_id TEXT PRIMARY KEY,registration_id INTEGER NOT NULL,"
        "candidate_number INTEGER NOT NULL,model_fingerprint TEXT NOT NULL)"
    )


def persist_cluster_assignments(feedback, assignments):
    """One feedback-DB transaction; never invalidate the Registry match cache."""
    if not assignments or feedback is None or not hasattr(feedback, "_conn"):
        return
    conn = feedback._conn
    with conn:
        _schema(conn)
        conn.executemany(
            "INSERT INTO rejection_cluster_assignments("
            "observation_id,registration_id,candidate_number,model_fingerprint) VALUES(?,?,?,?) "
            "ON CONFLICT(observation_id) DO UPDATE SET registration_id=excluded.registration_id,"
            "candidate_number=excluded.candidate_number,model_fingerprint=excluded.model_fingerprint",
            [(oid, value["registry_id"], value["candidate_number"], value["model_fingerprint"])
             for oid, value in assignments.items()],
        )


def resolve_rejection_clusters(classification_model_path, observation_ids):
    from .dinov2_feedback_service import _open_feedback_state

    feedback, _ = _open_feedback_state(classification_model_path)
    try:
        path = Path(feedback.path).with_name("registry.sqlite3")
        if not path.exists():
            return {}
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            return _resolve(feedback, conn, list(dict.fromkeys(observation_ids)))
    finally:
        feedback.close()


def _resolve(feedback, conn, observation_ids):
    from types import SimpleNamespace
    from system.dinov2.registry_match_index import RegistryMatchIndex

    fingerprint = feedback.model_fingerprint
    row = conn.execute("SELECT value FROM metadata WHERE key='model_fingerprint'").fetchone()
    if row is None or row[0] != fingerprint:
        raise ValueError("Registry model fingerprint does not match feedback")
    _schema(feedback._conn)
    assignments = {}
    cached_ids = set()
    pending = []
    # Only fetch requested observations. Neither source media nor all historical
    # feedback rows are read to prepare this view.
    for start in range(0, len(observation_ids), 500):
        ids = observation_ids[start:start + 500]
        marks = ",".join("?" for _ in ids)
        rows = feedback._conn.execute(
            f"SELECT * FROM rejection_cluster_assignments WHERE observation_id IN ({marks})", ids,
        ).fetchall()
        for row in rows:
            oid = row["observation_id"]
            cached_ids.add(oid)
            # Deleted clusters stay unassigned rather than silently acquiring
            # an unrelated cluster's number.
            entry = conn.execute("SELECT id,candidate_number FROM registrations WHERE id=?",
                                 (row["registration_id"],)).fetchone()
            if entry is not None and row["model_fingerprint"] == fingerprint:
                assignments[oid] = cluster_assignment(fingerprint, SimpleNamespace(**dict(entry)))
        missing = [oid for oid in ids if oid not in cached_ids]
        if not missing:
            continue
        rows = feedback._conn.execute(
            f"SELECT id,payload,embedding FROM observations WHERE id IN ({','.join('?' for _ in missing)})",
            missing,
        ).fetchall()
        for row in rows:
            payload = json.loads(row["payload"])
            if payload.get("accepted") is not False:
                continue
            vector = np.frombuffer(row["embedding"], dtype="<f4").copy()
            norm = np.linalg.norm(vector)
            if vector.shape == (768,) and np.isfinite(vector).all() and norm > 1e-12:
                pending.append((row["id"], vector / norm))
    if pending:
        index = RegistryMatchIndex(conn, ("candidate",))
        # Historical rows lacked an explicit link. Match saved features to the
        # current candidate prototypes without creating or changing candidates.
        for offset in range(0, len(pending), 64):
            batch = pending[offset:offset + 64]
            for (oid, _), match in zip(batch, index.match_batch(np.stack([v for _, v in batch]), 0.82)):
                if match is not None:
                    assignments[oid] = cluster_assignment(fingerprint, SimpleNamespace(**match))
        persist_cluster_assignments(feedback, {oid: assignments[oid] for oid, _ in pending if oid in assignments})
    return assignments
