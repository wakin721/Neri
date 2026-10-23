"""Cached density verification of Registry candidate groups for human review.

Registry candidates are collection identities, not HDBSCAN cluster labels. A
candidate is displayable only when all of its events belong to one non-noise
density cluster and it has at least five events of its own. Ambiguous candidates
stay in the noise/insufficient-evidence bucket instead of presenting a mixture
as a coherent cluster. This conservative check never merges collection records
or propagates density labels into classifier training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from system.dinov2.rejection_clustering import (
    MIN_REJECTION_CLUSTER_EVENTS, REJECTION_DENSITY_POLICY, rejected_event_labels,
)

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def candidate_density_support(feedback, conn):
    # This lock is separate from Registry's write lock: a long density fit must
    # not hold up inference evidence writes. Concurrent review requests reuse
    # the first completed fit instead of duplicating CPU/memory work.
    key = Path(feedback.path).resolve()
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    with lock:
        return _candidate_density_support(feedback, conn)


def _candidate_density_support(feedback, conn):
    """Reuse a committed Registry revision; the feedback cache is disposable."""
    feedback._conn.execute(
        "CREATE TABLE IF NOT EXISTS rejection_density_cache("
        "model_fingerprint TEXT PRIMARY KEY,signature TEXT NOT NULL,result_json TEXT NOT NULL)"
    )
    revision = conn.execute("SELECT value FROM metadata WHERE key='rejection_density_revision_v1'").fetchone()
    signature = f"{REJECTION_DENSITY_POLICY}:{revision[0]}" if revision is not None else None
    cached = feedback._conn.execute(
        "SELECT signature,result_json FROM rejection_density_cache WHERE model_fingerprint=?",
        (feedback.model_fingerprint,),
    ).fetchone()
    if signature is not None and cached is not None and cached[0] == signature:
        return json.loads(cached[1])
    rows = conn.execute(
        "SELECT e.id,e.registration_id,e.embedding FROM events e "
        "JOIN registrations r ON r.id=e.registration_id WHERE r.status='candidate' ORDER BY e.id"
    ).fetchall()
    if signature is None:
        # Read-only compatibility with a Registry created by older versions.
        # Hash exact event data, never guess validity from counts or timestamps.
        digest = hashlib.sha256(REJECTION_DENSITY_POLICY.encode())
        for row in rows:
            digest.update(f"{row['id']}:{row['registration_id']}:".encode())
            digest.update(row['embedding'])
        signature = digest.hexdigest()
        if cached is not None and cached[0] == signature:
            return json.loads(cached[1])
    labels = rejected_event_labels(
        np.stack([np.frombuffer(row['embedding'], dtype='<f4') for row in rows])
    ) if rows else np.empty(0, dtype=np.int64)
    groups = {}
    for row, label in zip(rows, labels, strict=True):
        groups.setdefault(str(row['registration_id']), []).append(int(label))
    result = {}
    for entry_id, members in groups.items():
        supported = len(members) >= MIN_REJECTION_CLUSTER_EVENTS and len(set(members)) == 1 and members[0] >= 0
        result[entry_id] = {
            "event_count": len(members), "is_noise": not supported,
            "density_verified": supported,
            "density_reason": "supported" if supported else (
                "insufficient_events" if len(members) < MIN_REJECTION_CLUSTER_EVENTS else
                "density_noise" if -1 in members else "multiple_density_groups"
            ),
        }
    # Cache publication is atomic and never changes Registry matching state.
    with feedback._conn:
        feedback._conn.execute(
            "INSERT INTO rejection_density_cache VALUES(?,?,?) ON CONFLICT(model_fingerprint) "
            "DO UPDATE SET signature=excluded.signature,result_json=excluded.result_json",
            (feedback.model_fingerprint, signature, json.dumps(result)),
        )
    return result
