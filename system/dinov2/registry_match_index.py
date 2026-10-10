"""A filtered matching snapshot, refreshed as a runtime batch adds evidence."""
from __future__ import annotations

import numpy as np

from .simple_shot import build_prototype, normalize_embedding


class RegistryMatchIndex:
    def __init__(self, conn, statuses=None, candidate_kinds=None):
        self.conn = conn
        self.statuses = statuses
        self.candidate_kinds = candidate_kinds
        clauses = []
        parameters = []
        for column, values in (("status", statuses), ("candidate_kind", candidate_kinds)):
            if values is not None:
                clauses.append(f"r.{column} IN ({','.join('?' for _ in values)})")
                parameters.extend(values)
        self.where = " WHERE " + " AND ".join(clauses) if clauses else ""
        self.parameters = tuple(parameters)
        rows = conn.execute(
            "SELECT r.id,r.candidate_number,r.status,r.candidate_kind,r.common_name "
            "FROM registrations r" + self.where + " ORDER BY r.id", self.parameters,
        ).fetchall()
        prototypes = {}
        for row in conn.execute(
            "SELECT p.registration_id,p.embedding FROM prototypes p "
            "JOIN registrations r ON r.id=p.registration_id" + self.where
            + " ORDER BY p.registration_id,p.prototype_index", self.parameters,
        ):
            prototypes.setdefault(int(row[0]), []).append(row[1])
        events = {}
        for row in conn.execute(
            "SELECT e.registration_id,e.embedding FROM events e "
            "JOIN registrations r ON r.id=e.registration_id" + self.where
            + " ORDER BY e.registration_id,e.id", self.parameters,
        ):
            if int(row[0]) not in prototypes:
                events.setdefault(int(row[0]), []).append(row[1])
        self.entries = {}
        for row in rows:
            key = int(row["id"])
            self._set(row, prototypes.get(key, ()), events.get(key, ()))
        self._matrix = None

    def _set(self, row, prototypes, events):
        key = int(row["id"])
        blobs = prototypes or events
        if not blobs:
            self.entries.pop(key, None)
            return
        values = np.stack([
            normalize_embedding(np.frombuffer(blob, dtype="<f4").copy()) for blob in blobs
        ])
        if not prototypes:
            values = build_prototype(values)[None, :]
        # Match the historical cosine_similarity normalization once per snapshot.
        values = np.stack([normalize_embedding(value) for value in values])
        self.entries[key] = (dict(row), values)

    def refresh(self, entry_id):
        row = self.conn.execute(
            "SELECT id,candidate_number,status,candidate_kind,common_name FROM registrations WHERE id=?",
            (entry_id,),
        ).fetchone()
        if row is None or (
            self.statuses is not None and row["status"] not in self.statuses
        ) or (self.candidate_kinds is not None and row["candidate_kind"] not in self.candidate_kinds):
            self.entries.pop(entry_id, None)
        else:
            prototypes = [r[0] for r in self.conn.execute(
                "SELECT embedding FROM prototypes WHERE registration_id=? ORDER BY prototype_index", (entry_id,)
            )]
            events = [] if prototypes else [r[0] for r in self.conn.execute(
                "SELECT embedding FROM events WHERE registration_id=? ORDER BY id", (entry_id,)
            )]
            self._set(row, prototypes, events)
        self._matrix = None

    def scores(self, vector):
        if not self.entries:
            return ()
        if self._matrix is None:
            self._rows = [self.entries[key] for key in sorted(self.entries)]
            self._offsets = np.cumsum([0] + [len(values) for _, values in self._rows[:-1]])
            self._matrix = np.concatenate([values for _, values in self._rows])
        similarities = self._matrix @ vector
        scores = np.maximum.reduceat(similarities, self._offsets)
        return ((row, float(score)) for (row, _), score in zip(self._rows, scores))
