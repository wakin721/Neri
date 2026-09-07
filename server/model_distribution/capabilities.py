from __future__ import annotations

import hashlib
import hmac
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


class BudgetError(RuntimeError):
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


class BudgetStore:
    def __init__(
        self,
        state_dir: Path,
        *,
        secret: bytes,
        requests_per_minute: int,
        daily_ip_bytes: int,
        daily_total_bytes: int,
        clock=time.time,
    ):
        if not isinstance(secret, (bytes, bytearray)) or not secret:
            raise ValueError("budget secret must be non-empty bytes")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "model_distribution.sqlite3"
        self.secret = bytes(secret)
        self.requests_per_minute = max(0, int(requests_per_minute))
        self.daily_ip_bytes = max(0, int(daily_ip_bytes))
        self.daily_total_bytes = max(0, int(daily_total_bytes))
        self.clock = clock
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS request_budgets(
                    ip_hash TEXT NOT NULL,
                    minute_bucket INTEGER NOT NULL,
                    request_count INTEGER NOT NULL,
                    PRIMARY KEY(ip_hash, minute_bucket)
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS proxy_ip_budgets(
                    ip_hash TEXT NOT NULL,
                    day_bucket INTEGER NOT NULL,
                    byte_count INTEGER NOT NULL,
                    PRIMARY KEY(ip_hash, day_bucket)
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS proxy_total_budgets(
                    day_bucket INTEGER PRIMARY KEY,
                    byte_count INTEGER NOT NULL
                )
            """)

    def _ip_hash(self, client_ip: str) -> str:
        normalized = str(client_ip or "unknown").strip().lower().encode("utf-8")
        return hmac.new(self.secret, normalized, hashlib.sha256).hexdigest()

    def check_request(self, client_ip: str) -> None:
        now = float(self.clock())
        minute_bucket = int(now // 60)
        ip_hash = self._ip_hash(client_ip)
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT request_count FROM request_budgets WHERE ip_hash=? AND minute_bucket=?",
                    (ip_hash, minute_bucket),
                ).fetchone()
                current = int(row[0]) if row else 0
                if current >= self.requests_per_minute:
                    raise BudgetError("request_rate_limited")
                db.execute(
                    """
                    INSERT INTO request_budgets(ip_hash,minute_bucket,request_count)
                    VALUES(?,?,1)
                    ON CONFLICT(ip_hash,minute_bucket)
                    DO UPDATE SET request_count=request_count+1
                    """,
                    (ip_hash, minute_bucket),
                )
                db.execute(
                    "DELETE FROM request_budgets WHERE minute_bucket < ?",
                    (minute_bucket - 2,),
                )
                db.commit()
            except Exception:
                db.rollback()
                raise

    def reserve_proxy_bytes(self, client_ip: str, byte_count: int) -> None:
        size = int(byte_count)
        if size < 0:
            raise BudgetError("invalid_proxy_size")
        day_bucket = int(float(self.clock()) // 86400)
        ip_hash = self._ip_hash(client_ip)
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                ip_row = db.execute(
                    "SELECT byte_count FROM proxy_ip_budgets WHERE ip_hash=? AND day_bucket=?",
                    (ip_hash, day_bucket),
                ).fetchone()
                total_row = db.execute(
                    "SELECT byte_count FROM proxy_total_budgets WHERE day_bucket=?",
                    (day_bucket,),
                ).fetchone()
                current_ip = int(ip_row[0]) if ip_row else 0
                current_total = int(total_row[0]) if total_row else 0
                if current_ip + size > self.daily_ip_bytes:
                    raise BudgetError("proxy_ip_budget_exceeded")
                if current_total + size > self.daily_total_bytes:
                    raise BudgetError("proxy_total_budget_exceeded")
                db.execute(
                    """
                    INSERT INTO proxy_ip_budgets(ip_hash,day_bucket,byte_count)
                    VALUES(?,?,?)
                    ON CONFLICT(ip_hash,day_bucket)
                    DO UPDATE SET byte_count=byte_count+excluded.byte_count
                    """,
                    (ip_hash, day_bucket, size),
                )
                db.execute(
                    """
                    INSERT INTO proxy_total_budgets(day_bucket,byte_count)
                    VALUES(?,?)
                    ON CONFLICT(day_bucket)
                    DO UPDATE SET byte_count=byte_count+excluded.byte_count
                    """,
                    (day_bucket, size),
                )
                db.execute(
                    "DELETE FROM proxy_ip_budgets WHERE day_bucket < ?",
                    (day_bucket - 2,),
                )
                db.execute(
                    "DELETE FROM proxy_total_budgets WHERE day_bucket < ?",
                    (day_bucket - 2,),
                )
                db.commit()
            except Exception:
                db.rollback()
                raise
