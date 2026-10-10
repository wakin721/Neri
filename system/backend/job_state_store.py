"""Transactional job checkpoints that append only newly completed results."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import sqlite3


class JobStateStore:
    def __init__(self, legacy_path: Path):
        self.path = legacy_path.with_suffix(".sqlite3")
        self._saved: dict[str, tuple[object, object, tuple]] = {}

    def load(self):
        if not self.path.exists():
            return None
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute("BEGIN")
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
            if "metadata" not in tables or not conn.execute(
                "SELECT 1 FROM metadata WHERE key='initialized' AND value='1'"
            ).fetchone():
                return None
            jobs = {}
            requests = {}
            for job_id, payload, request in conn.execute("SELECT id,payload,request FROM jobs"):
                jobs[job_id] = json.loads(payload)
                jobs[job_id]["results"] = []
                if request is not None:
                    requests[job_id] = json.loads(request)
            for job_id, payload in conn.execute(
                "SELECT job_id,payload FROM results ORDER BY job_id,position"
            ):
                jobs[job_id]["results"].append(json.loads(payload))
        return {"jobs": jobs, "requests": requests}

    def remember(self, jobs, requests):
        self._saved = {
            key: (job, requests.get(key), tuple(job.results))
            for key, job in jobs.items()
        }

    def has_saved(self, job_id, job):
        previous = self._saved.get(job_id)
        return previous is not None and previous[0] is job

    def save(self, jobs, requests):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                conn.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,payload TEXT NOT NULL,request TEXT)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS results("
                    "job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,position INTEGER,payload TEXT NOT NULL,"
                    "PRIMARY KEY(job_id,position))"
                )
                # Also remove persisted rows after a clear/restart, when _saved can be empty.
                persisted_ids = {row[0] for row in conn.execute("SELECT id FROM jobs")}
                for key in persisted_ids:
                    if key not in jobs:
                        conn.execute("DELETE FROM jobs WHERE id=?", (key,))
                for key, job in jobs.items():
                    request = requests.get(key)
                    previous = self._saved.get(key) if key in persisted_ids else None
                    if previous is not None and previous[0] is job and previous[1] is request:
                        continue
                    payload = job.model_dump(mode="json", exclude={"results", "active"})
                    payload["active"] = False
                    conn.execute(
                        "INSERT INTO jobs VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET "
                        "payload=excluded.payload,request=excluded.request",
                        (key, json.dumps(payload, ensure_ascii=False),
                         json.dumps(request.model_dump(mode="json"), ensure_ascii=False) if request else None),
                    )
                    old_results = previous[2] if previous else ()
                    append = len(job.results) >= len(old_results) and all(
                        old is new for old, new in zip(old_results, job.results)
                    )
                    if previous is None or not append:
                        conn.execute("DELETE FROM results WHERE job_id=?", (key,))
                    start = len(old_results) if append else 0
                    conn.executemany(
                        "INSERT INTO results VALUES(?,?,?)",
                        ((key, index, json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
                         for index, item in enumerate(job.results[start:], start)),
                    )
                conn.execute("INSERT OR REPLACE INTO metadata VALUES('initialized','1')")
            # Advance the in-memory baseline only after a successful commit.
            self.remember(jobs, requests)
        finally:
            conn.close()
