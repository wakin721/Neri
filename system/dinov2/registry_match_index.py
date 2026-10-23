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
        entry_id = int(entry_id)
        previous = self.entries.get(entry_id)
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
        if self._matrix is None:
            return
        updated = self.entries.get(entry_id)
        if previous is None and updated is None:
            return
        if previous is not None and updated is not None and len(previous[1]) == len(updated[1]):
            position = self._positions[entry_id]
            start = self._offsets[position]
            self._matrix[start:start + len(updated[1])] = updated[1]
            self._rows[position] = updated
        elif previous is None and updated is not None and entry_id > self._rows[-1][0]["id"]:
            # AUTOINCREMENT candidates can append to reserved matrix storage.
            # Do not copy/re-sort the entire bank after every observation.
            start = len(self._matrix)
            required = start + len(updated[1])
            if required > len(self._storage):
                storage = np.empty((max(required, len(self._storage) * 2), self._matrix.shape[1]), dtype=np.float32)
                storage[:start] = self._matrix
                self._storage = storage
            self._matrix = self._storage[:required]
            self._matrix[start:] = updated[1]
            self._positions[entry_id] = len(self._rows)
            self._rows.append(updated)
            self._offsets = np.append(self._offsets, start)
        else:
            # Deletions, older IDs entering a filter, and prototype count changes
            # require rebuilding offsets; ordinary candidate updates do not.
            self._matrix = None

    def scores(self, vector, *, minimum_score=-np.inf):
        if not self.entries:
            return ()
        if self._matrix is None:
            self._rows = [self.entries[key] for key in sorted(self.entries)]
            self._offsets = np.cumsum([0] + [len(values) for _, values in self._rows[:-1]])
            self._positions = {int(row["id"]): index for index, (row, _) in enumerate(self._rows)}
            count = sum(len(values) for _, values in self._rows)
            dimension = self._rows[0][1].shape[1]
            self._storage = np.empty((max(16, count + count // 4), dimension), dtype=np.float32)
            self._matrix = self._storage[:count]
            np.concatenate([values for _, values in self._rows], out=self._matrix)
        similarities = self._matrix @ vector
        scores = np.maximum.reduceat(similarities, self._offsets)
        # Match the previous Python-float threshold comparison, including scores
        # just below a threshold that would round upward in float32 comparisons.
        eligible = np.flatnonzero(scores.astype(np.float64) >= minimum_score)
        return ((self._rows[index][0], float(scores[index])) for index in eligible)
