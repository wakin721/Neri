"""Durable opt-in upload journal, with one background compressor/uploader."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .media import prepare_sample
from .policy import AGREEMENT_VERSION, IMAGE_SUFFIXES, MAX_EMPTY_PER_FOLDER, contribution_payload
from .transport import HttpTransport, UploadCancelled


class TrainingQueue:
    def __init__(self, state_dir: Path, transport=None, clock=time.time, debounce_seconds=10):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            self.state_dir.chmod(0o700)
        self.db_path = self.state_dir / "contributions.sqlite3"
        self.transport = transport or HttpTransport()
        self.clock = clock
        self.debounce_seconds = debounce_seconds
        self._lock = threading.RLock()
        self._processing_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        with self._connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS consent (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    source_key TEXT PRIMARY KEY, source_path TEXT NOT NULL, folder_key TEXT NOT NULL,
                    sample_id TEXT NOT NULL, secret TEXT NOT NULL, revision INTEGER NOT NULL,
                    payload TEXT NOT NULL, stamp TEXT NOT NULL, state TEXT NOT NULL,
                    operation TEXT NOT NULL DEFAULT 'upload', attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt REAL NOT NULL, ever_empty INTEGER NOT NULL DEFAULT 0,
                    empty_reserved INTEGER NOT NULL DEFAULT 0, uploaded_revision INTEGER NOT NULL DEFAULT 0,
                    remote_possible INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '');
                CREATE INDEX IF NOT EXISTS jobs_folder ON jobs(folder_key);
                CREATE INDEX IF NOT EXISTS jobs_waiting ON jobs(state,next_attempt);
            ''')
        # executescript commits its transaction. Start a fresh BEGIN IMMEDIATE
        # through _connect so schema inspection and migration serialize across instances.
        with self._connect() as db:
            if 'remote_possible' not in {row['name'] for row in db.execute('PRAGMA table_info(jobs)')}:
                db.execute('ALTER TABLE jobs ADD COLUMN remote_possible INTEGER NOT NULL DEFAULT 0')
                # Older journals may have lost a completion response before upgrading.
                db.execute("UPDATE jobs SET remote_possible=1 WHERE uploaded_revision>0 OR attempts>0 OR state='uploading'")
        if os.name != "nt":
            self.db_path.chmod(0o600)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                yield db
        finally:
            db.close()

    @staticmethod
    def _key(path):
        stat = path.stat()
        # Device + file ID survive renames. Birth time prevents ID reuse after deletion
        # on Windows/macOS; filesystems without IDs fall back to the resolved path.
        birth = getattr(stat, 'st_birthtime_ns', 0)
        identity = f'{stat.st_dev}:{stat.st_ino}:{birth}' if stat.st_ino else os.path.normcase(str(path))
        return hashlib.sha256(identity.encode('utf-8')).hexdigest()

    @staticmethod
    def _stamp(path):
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _enabled(self, db):
        values = dict(db.execute("SELECT key,value FROM consent"))
        return values.get("version") == AGREEMENT_VERSION and values.get("enabled") == "true"

    def status(self):
        with self._lock, self._connect() as db:
            values = dict(db.execute("SELECT key,value FROM consent"))
            counts = dict(db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state"))
        return {"agreement_version": AGREEMENT_VERSION,
                "agreement_accepted": values.get("version") == AGREEMENT_VERSION,
                "participation_decided": values.get("version") == AGREEMENT_VERSION and "enabled" in values,
                "training_enabled": values.get("version") == AGREEMENT_VERSION and values.get("enabled") == "true",
                "stats": {key: counts.get(key, 0) for key in ("pending", "uploading", "uploaded", "failed", "skipped")}}

    def set_consent(self, version: str, enabled: bool):
        if version != AGREEMENT_VERSION or not isinstance(enabled, bool):
            raise ValueError("请阅读当前版本协议并明确选择是否参加。")
        with self._lock, self._connect() as db:
            db.executemany("INSERT OR REPLACE INTO consent(key,value) VALUES (?,?)", [
                ("version", version), ("enabled", "true" if enabled else "false"),
                ("decided_at", str(self.clock())),
            ])
            if not enabled:
                db.execute("UPDATE jobs SET state='cancelled',empty_reserved=0 WHERE state IN ('pending','failed','uploading')")
        self._wake.set()
        return self.status()

    def clear_pending(self):
        with self._lock, self._connect() as db:
            db.execute("UPDATE jobs SET state='cancelled',empty_reserved=0 WHERE state IN ('pending','failed','uploading')")
        return self.status()

    def enqueue(self, path, data: dict, validated=True):
        # Consent is checked BEFORE stat, parsing metadata, decoding or any network work.
        with self._lock, self._connect() as db:
            if not self._enabled(db):
                return False
            path = Path(path).expanduser().resolve()
            try:
                key, folder = self._key(path), self._key(path.parent)
                old = db.execute("SELECT * FROM jobs WHERE source_key=?", (key,)).fetchone()
            except OSError:
                old = db.execute("SELECT * FROM jobs WHERE source_path=?", (str(path),)).fetchone()
                if old is None:
                    return False
                key, folder = old['source_key'], old['folder_key']
            if old:
                db.execute('UPDATE jobs SET source_path=? WHERE source_key=?', (str(path), key))
            payload = contribution_payload(data) if validated and path.suffix.lower() in IMAGE_SUFFIXES else None
            if payload is None:
                if old:
                    if old["remote_possible"] or old["uploaded_revision"] or old["state"] == "uploading":
                        db.execute("UPDATE jobs SET revision=revision+1,state='pending',operation='delete',next_attempt=?,empty_reserved=0 WHERE source_key=?", (self.clock(), key))
                    else:
                        db.execute("UPDATE jobs SET state='cancelled',empty_reserved=0 WHERE source_key=?", (key,))
                self._wake.set()
                return False
            try:
                stamp = self._stamp(path)
            except OSError:
                return False
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if old and old["payload"] == encoded and old["stamp"] == stamp and old["state"] in ("pending", "failed", "uploading", "uploaded") and old["operation"] == "upload":
                return False
            empty = payload["empty"]
            owns_slot = old and (old["ever_empty"] or old["empty_reserved"])
            if empty and not owns_slot:
                count = db.execute("SELECT COUNT(*) FROM jobs WHERE folder_key=? AND (ever_empty=1 OR empty_reserved=1)", (folder,)).fetchone()[0]
                if count >= MAX_EMPTY_PER_FOLDER:
                    # A changed label invalidates any pending previous species submission too.
                    if old:
                        operation = "delete" if old["remote_possible"] or old["uploaded_revision"] or old["state"] == "uploading" else "upload"
                        state = "pending" if operation == "delete" else "skipped"
                        db.execute("UPDATE jobs SET revision=revision+1,state=?,operation=?,empty_reserved=0,next_attempt=? WHERE source_key=?", (state, operation, self.clock(), key))
                    self._wake.set()
                    return False
            if old:
                db.execute("UPDATE jobs SET revision=revision+1,payload=?,stamp=?,state='pending',operation='upload',attempts=0,next_attempt=?,empty_reserved=?,error='' WHERE source_key=?",
                           (encoded, stamp, self.clock() + self.debounce_seconds, int(empty), key))
            else:
                db.execute("INSERT INTO jobs(source_key,source_path,folder_key,sample_id,secret,revision,payload,stamp,state,next_attempt,empty_reserved) VALUES (?,?,?,?,?,1,?,?,'pending',?,?)",
                           (key, str(path), folder, uuid.uuid4().hex, secrets.token_urlsafe(32), encoded, stamp, self.clock() + self.debounce_seconds, int(empty)))
        self._wake.set()
        return True

    def _current(self, job):
        if self._stop.is_set():
            return False
        with self._lock, self._connect() as db:
            row = db.execute("SELECT revision,state FROM jobs WHERE source_key=?", (job["source_key"],)).fetchone()
            return self._enabled(db) and row is not None and row["revision"] == job["revision"] and row["state"] == "uploading"

    def process_once(self):
        if not self._processing_lock.acquire(blocking=False):
            return False
        try:
            # OS locks are released on crash; one worker owns the shared journal even
            # when multiple backend processes are open. Enqueuing remains available.
            with open(self.state_dir / 'worker.lock', 'a+b') as handle:
                handle.seek(0)
                try:
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return False
                try:
                    return self._process_once()
                finally:
                    handle.seek(0)
                    if os.name == 'nt':
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._processing_lock.release()

    def _process_once(self):
        with self._lock, self._connect() as db:
            if self._stop.is_set() or not self._enabled(db):
                return False
            # Only the exclusive worker may recover a claim left behind by a crash.
            db.execute("UPDATE jobs SET state='pending' WHERE state='uploading'")
            row = db.execute("SELECT * FROM jobs WHERE state IN ('pending','failed') AND next_attempt<=? ORDER BY next_attempt,rowid LIMIT 1", (self.clock(),)).fetchone()
            if row is None:
                return False
            job = dict(row)
            job["payload"] = json.loads(job["payload"])
            db.execute("UPDATE jobs SET state='uploading' WHERE source_key=?", (job["source_key"],))
        try:
            if job["operation"] == "delete":
                self.transport.delete(job, lambda: self._current(job))
            else:
                path = Path(job["source_path"])
                if self._stamp(path) != job["stamp"]:
                    raise ValueError("source_changed")
                photo, annotation = prepare_sample(path, job)
                if self._stamp(path) != job["stamp"]:
                    raise ValueError("source_changed")
                if not self._current(job):
                    raise UploadCancelled()
                if job["payload"]["empty"]:
                    with self._lock, self._connect() as db:
                        db.execute("UPDATE jobs SET ever_empty=1 WHERE source_key=? AND revision=?", (job["source_key"], job["revision"]))
                # Persist before sending: an exception/crash does not prove that the
                # server rejected the upload. Keep this across retries and relabels.
                with self._lock, self._connect() as db:
                    db.execute("UPDATE jobs SET remote_possible=1 WHERE source_key=? AND revision=? AND state='uploading'", (job["source_key"], job["revision"]))
                self.transport.upload(job, photo, annotation, lambda: self._current(job))
            with self._lock, self._connect() as db:
                if job["operation"] == "upload":
                    # Retain receipt even if a concurrent edit has queued a newer revision.
                    db.execute("UPDATE jobs SET uploaded_revision=MAX(uploaded_revision,?) WHERE source_key=?", (job["revision"], job["source_key"]))
                db.execute("UPDATE jobs SET state=?,uploaded_revision=?,remote_possible=?,empty_reserved=0,error='' WHERE source_key=? AND revision=? AND state='uploading'",
                           ("deleted" if job["operation"] == "delete" else "uploaded",
                            0 if job["operation"] == "delete" else job["revision"],
                            int(job["operation"] == "upload"), job["source_key"], job["revision"]))
        except UploadCancelled:
            with self._lock, self._connect() as db:
                db.execute("UPDATE jobs SET state='pending' WHERE source_key=? AND revision=? AND state='uploading'", (job["source_key"], job["revision"]))
        except (ValueError, FileNotFoundError, ImageDecodeError) as error:
            with self._lock, self._connect() as db:
                db.execute("UPDATE jobs SET state='skipped',empty_reserved=0,error=? WHERE source_key=? AND revision=? AND state='uploading'", (type(error).__name__, job["source_key"], job["revision"]))
        except Exception as error:
            # Persist only error type, never exception strings containing URLs, paths or tokens.
            with self._lock, self._connect() as db:
                delay = min(3600, 30 * 2 ** min(job["attempts"], 7))
                db.execute("UPDATE jobs SET state='failed',attempts=attempts+1,next_attempt=?,error=? WHERE source_key=? AND revision=? AND state='uploading'",
                           (self.clock() + delay, type(error).__name__, job["source_key"], job["revision"]))
        return True

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="neri-training-upload", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                if self.process_once():
                    continue
            except Exception:
                pass  # This optional worker must never crash local annotation/inference.
            self._wake.wait(2)
            self._wake.clear()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)


# Keep PIL imports out of annotation handlers; decoding runs only on the worker.
from PIL import UnidentifiedImageError as ImageDecodeError
