from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .storage import RemoteEntry

ROOT = "/Neri_Data/Model"


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ManifestSnapshot:
    manifest_id: str
    files: tuple[ManifestEntry, ...]


def _allowed(folder: str, name: str) -> bool:
    suffix = Path(name).suffix.lower()
    if folder == "detect":
        return suffix == ".pt"
    if folder == "cls":
        return suffix in {".pt", ".onnx", ".engine"}
    return False


class ManifestBuilder:
    def __init__(self, state_dir: Path, store):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.db_path = self.state_dir / "model_distribution.sqlite3"
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS hash_cache(
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    modified TEXT,
                    sha256 TEXT NOT NULL
                )
            """)

    def _hash_remote(self, logical: str, remote: str, entry: RemoteEntry) -> str:
        if entry.modified:
            with closing(sqlite3.connect(self.db_path)) as db, db:
                row = db.execute(
                    "SELECT size,modified,sha256 FROM hash_cache WHERE path=?",
                    (logical,),
                ).fetchone()
            if row and row[0] == entry.size and row[1] == entry.modified:
                return row[2]

        digest = hashlib.sha256()
        link = self.store.resolve_link(remote)
        received = 0
        for chunk in self.store.iter_bytes(link):
            received += len(chunk)
            if received > entry.size:
                raise ManifestError("drive_stream_size_mismatch")
            digest.update(chunk)
        if received != entry.size:
            raise ManifestError("drive_stream_size_mismatch")
        value = digest.hexdigest()
        if entry.modified:
            with closing(sqlite3.connect(self.db_path)) as db, db:
                db.execute(
                    "INSERT INTO hash_cache(path,size,modified,sha256) VALUES(?,?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET size=excluded.size,modified=excluded.modified,sha256=excluded.sha256",
                    (logical, entry.size, entry.modified, value),
                )
        return value

    def build(self) -> ManifestSnapshot:
        candidates: list[tuple[str, str, RemoteEntry]] = []
        for folder in ("detect", "cls"):
            remote_dir = f"{ROOT}/{folder}"
            for entry in self.store.list_dir(remote_dir):
                if entry.is_dir or not _allowed(folder, entry.name):
                    continue
                logical = f"{folder}/{entry.name}"
                remote = f"{remote_dir}/{entry.name}"
                candidates.append((logical, remote, entry))

        tracker = self.store.stat(f"{ROOT}/tracker.yaml")
        if tracker is not None and not tracker.is_dir:
            candidates.append(("tracker.yaml", f"{ROOT}/tracker.yaml", tracker))

        seen: set[str] = set()
        normalized: list[tuple[str, str, RemoteEntry]] = []
        for logical, remote, entry in candidates:
            key = unicodedata.normalize("NFC", logical).casefold()
            if key in seen:
                raise ManifestError("duplicate_model_path")
            seen.add(key)
            normalized.append((logical, remote, entry))

        files = tuple(
            ManifestEntry(logical, entry.size, self._hash_remote(logical, remote, entry))
            for logical, remote, entry in sorted(normalized, key=lambda item: item[0])
        )
        payload = json.dumps(
            [[item.path, item.size, item.sha256] for item in files],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return ManifestSnapshot(hashlib.sha256(payload).hexdigest(), files)
