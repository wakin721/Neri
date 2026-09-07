from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
import urllib.parse
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .manifest import ManifestEntry

SAFE_DIRECT_HEADER_NAMES = frozenset({"accept", "referer", "user-agent"})


class CapabilityError(RuntimeError):
    pass


def valid_direct_download_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or "").lower()
        return (
            parsed.scheme == "https"
            and not parsed.username
            and not parsed.password
            and parsed.port in (None, 443)
            and (
                host.endswith(".1drv.com")
                or host.endswith(".sharepoint.com")
                or host.endswith(".microsoftpersonalcontent.com")
            )
        )
    except (TypeError, ValueError):
        return False


def safe_direct_headers(headers: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in SAFE_DIRECT_HEADER_NAMES:
            result[str(key)] = str(value)
    return result


@dataclass(frozen=True)
class BoundCapability:
    path: str
    sha256: str
    size: int
    expires_at: float


class CapabilityStore:
    def __init__(self, state_dir: Path, *, ttl_seconds: int = 600, clock=time.time):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "model_distribution.sqlite3"
        self.ttl_seconds = int(ttl_seconds)
        self.clock = clock
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS capabilities(
                    token_hash TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    expires_at REAL NOT NULL,
                    uses_remaining INTEGER NOT NULL
                )
            """)

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("ascii")).hexdigest()

    def issue(self, entry: ManifestEntry) -> tuple[str, float]:
        token = secrets.token_urlsafe(32)
        expires_at = float(self.clock()) + self.ttl_seconds
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute(
                "INSERT INTO capabilities(token_hash,path,sha256,size,expires_at,uses_remaining) VALUES(?,?,?,?,?,4)",
                (self._hash(token), entry.path, entry.sha256, entry.size, expires_at),
            )
        return token, expires_at

    def consume(self, token: str) -> BoundCapability:
        if not isinstance(token, str) or not 20 <= len(token) <= 128:
            raise CapabilityError("invalid_capability")
        token_hash = self._hash(token)
        now = float(self.clock())
        with closing(sqlite3.connect(self.db_path)) as db, db:
            row = db.execute(
                "SELECT path,sha256,size,expires_at,uses_remaining FROM capabilities WHERE token_hash=?",
                (token_hash,),
            ).fetchone()
            if row is None or row[3] < now or row[4] <= 0:
                db.execute("DELETE FROM capabilities WHERE token_hash=?", (token_hash,))
                raise CapabilityError("expired_or_invalid_capability")
            db.execute(
                "UPDATE capabilities SET uses_remaining=uses_remaining-1 WHERE token_hash=?",
                (token_hash,),
            )
        return BoundCapability(row[0], row[1], int(row[2]), float(row[3]))
