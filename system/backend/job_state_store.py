"""Transactional job checkpoints that append only newly completed results."""
from __future__ import annotations

import json
import logging
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SavedJob:
    job: object
    request: object
    results: list | tuple | None
    result_count: int


class JobStateStore:
    def __init__(self, legacy_path: Path):
        self.path = legacy_path.with_suffix(".sqlite3")
        self._saved: dict[str, _SavedJob] = {}

    def load(self, *, include_results: bool = True):
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
            if include_results:
                for job_id, payload in conn.execute(
                    "SELECT job_id,payload FROM results ORDER BY job_id,position"
                ):
                    jobs[job_id]["results"].append(json.loads(payload))
        return {"jobs": jobs, "requests": requests}

    def load_results(self, job_id: str):
        """Read only the selected task, using the existing primary-key index."""
        with closing(sqlite3.connect(self.path)) as conn:
            return [json.loads(row[0]) for row in conn.execute(
                "SELECT payload FROM results WHERE job_id=? ORDER BY position", (job_id,)
            )]

    def remember_job(self, job_id, job, request):
        previous = self._saved.get(job_id)
        # Retain an older baseline after a failed metadata checkpoint, so
        # hydrating results does not accidentally mark unsaved state as saved.
        self._saved[job_id] = _SavedJob(
            previous.job if previous is not None else job,
            previous.request if previous is not None else request,
            job.results,
            len(job.results),
        )

    def remember(self, jobs, requests, *, unloaded_job_ids=frozenset(), append_only_result_lists=False):
        saved = {}
        for key, job in jobs.items():
            results = None
            if key not in unloaded_job_ids:
                # The manager holds its lock and replaces the list for corrections.
                results = job.results if append_only_result_lists else tuple(job.results)
            saved[key] = _SavedJob(job, requests.get(key), results, len(results) if results is not None else 0)
        self._saved = saved

    def has_saved(self, job_id, job):
        previous = self._saved.get(job_id)
        return previous is not None and previous.job is job

    def load_inputs(self, job_id: str, input_key: str):
        """Load a completed discovery snapshot without traversing source media."""
        if not self.path.exists():
            return None
        with closing(sqlite3.connect(self.path)) as conn:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='job_inputs'"
            ).fetchone():
                return None
            row = conn.execute(
                "SELECT root,paths FROM job_inputs WHERE job_id=? AND input_key=?",
                (job_id, input_key),
            ).fetchone()
        if row is None:
            return None
        paths = json.loads(row[1])
        if not isinstance(row[0], str) or not row[0] or not isinstance(paths, list) or not paths or not all(
            isinstance(path, str) and path for path in paths
        ):
            raise ValueError("Invalid cached job input snapshot")
        return Path(row[0]), [Path(path) for path in paths]

    def save_inputs(self, job_id: str, input_key: str, root: Path, paths: list[Path]):
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with conn:
                conn.execute(
                    "INSERT INTO job_inputs(job_id,input_key,root,paths) VALUES(?,?,?,?) "
                    "ON CONFLICT(job_id) DO UPDATE SET "
                    "input_key=excluded.input_key,root=excluded.root,paths=excluded.paths",
                    (job_id, input_key, str(root), json.dumps([str(path) for path in paths], ensure_ascii=False)),
                )

    def save(self, jobs, requests, *, unloaded_job_ids=frozenset(), append_only_result_lists=False):
        started = time.perf_counter()
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
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS job_inputs("
                    "job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,"
                    "input_key TEXT NOT NULL,root TEXT NOT NULL,paths TEXT NOT NULL)"
                )
                # Also remove persisted rows after a clear/restart, when _saved can be empty.
                persisted_ids = {row[0] for row in conn.execute("SELECT id FROM jobs")}
                for key in persisted_ids:
                    if key not in jobs:
                        conn.execute("DELETE FROM jobs WHERE id=?", (key,))
                for key, job in jobs.items():
                    request = requests.get(key)
                    previous = self._saved.get(key) if key in persisted_ids else None
                    if previous is not None and previous.job is job and previous.request is request:
                        continue
                    payload = job.model_dump(mode="json", exclude={"results", "active"})
                    payload["active"] = False
                    conn.execute(
                        "INSERT INTO jobs VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET "
                        "payload=excluded.payload,request=excluded.request",
                        (key, json.dumps(payload, ensure_ascii=False),
                         json.dumps(request.model_dump(mode="json"), ensure_ascii=False) if request else None),
                    )
                    # A summary's empty results are a placeholder, never a
                    # request to delete the task's persisted result rows.
                    if key in unloaded_job_ids:
                        continue
                    results = job.results
                    old_results = previous.results if previous and previous.results is not None else ()
                    old_count = previous.result_count if previous else 0
                    start = old_count if old_results is results else 0
                    # Corrections replace only their own positions. Finalization
                    # may replace every object, but must not delete/rebuild the
                    # task's result tree or rewrite byte-identical payloads.
                    conn.executemany(
                        "INSERT INTO results VALUES(?,?,?) "
                        "ON CONFLICT(job_id,position) DO UPDATE SET payload=excluded.payload "
                        "WHERE results.payload<>excluded.payload",
                        ((key, index, results[index].model_dump_json())
                         for index in range(start, len(results))
                         if index >= old_count or old_results[index] is not results[index]),
                    )
                    if previous is None or len(results) < old_count:
                        conn.execute("DELETE FROM results WHERE job_id=? AND position>=?", (key, len(results)))
                conn.execute("INSERT OR REPLACE INTO metadata VALUES('initialized','1')")
            # Advance the in-memory baseline only after a successful commit.
            self.remember(
                jobs, requests, unloaded_job_ids=unloaded_job_ids,
                append_only_result_lists=append_only_result_lists,
            )
        finally:
            conn.close()
            logger.info("Job state checkpoint timing: elapsed=%.3fs", time.perf_counter() - started)
