from __future__ import annotations

import hashlib
import os
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .layout import ModelLayout
from .protocol import ModelManifestEntry
from .state import SyncState, VerifiedFileState, load_state, save_state

SyncPhase = Literal['idle', 'checking', 'downloading', 'completed', 'failed']


@dataclass(frozen=True)
class ModelSyncStatus:
    state: SyncPhase = 'idle'
    run_id: str | None = None
    current_file: str | None = None
    total_files: int = 0
    completed_files: int = 0
    received_bytes: int = 0
    total_bytes: int | None = None
    last_successful_sync: str | None = None
    manifest_id: str | None = None
    error: str | None = None
    cloud_detect_count: int | None = None
    cloud_cls_count: int | None = None


class ModelSyncManager:
    def __init__(self, layout: ModelLayout, client):
        self.layout = layout
        self.client = client
        self._lock = threading.RLock()
        previous = load_state(layout.state_file)
        self._status = ModelSyncStatus(
            last_successful_sync=previous.last_successful_sync,
            manifest_id=previous.manifest_id,
        )
        self._worker: threading.Thread | None = None

    def status(self) -> ModelSyncStatus:
        with self._lock:
            return self._status

    def start(self) -> ModelSyncStatus:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return self._status
            run_id = uuid.uuid4().hex
            self._status = replace(self._status, state='checking', run_id=run_id, error=None)
            self._worker = threading.Thread(target=self._run_guarded, name='neri-model-sync', daemon=True)
            self._worker.start()
            return self._status

    def _run_guarded(self):
        try:
            self._sync_once()
        except Exception as exc:
            with self._lock:
                self._status = replace(self._status, state='failed', error=str(exc)[:300])

    def sync_once_for_test(self) -> ModelSyncStatus:
        with self._lock:
            self._status = replace(self._status, state='checking', run_id=uuid.uuid4().hex, error=None)
        self._run_guarded()
        return self.status()

    def _target(self, entry: ModelManifestEntry) -> Path:
        if entry.path == 'tracker.yaml':
            target = self.layout.tracker
            parent = self.layout.root
        elif entry.path.startswith('detect/'):
            target = self.layout.detect_sync / entry.path.split('/', 1)[1]
            parent = self.layout.detect_sync
        elif entry.path.startswith('cls/'):
            target = self.layout.cls_sync / entry.path.split('/', 1)[1]
            parent = self.layout.cls_sync
        else:
            raise ValueError('invalid_model_path')
        resolved_parent = parent.resolve()
        resolved_target = target.resolve()
        if resolved_target != resolved_parent and resolved_parent not in resolved_target.parents:
            raise ValueError('model_path_escape')
        return target

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _matches(self, target: Path, entry: ModelManifestEntry, cached: VerifiedFileState | None) -> bool:
        try:
            stat = target.stat()
        except OSError:
            return False
        if stat.st_size != entry.size:
            return False
        if cached and cached.size == stat.st_size and cached.mtime_ns == stat.st_mtime_ns and cached.sha256 == entry.sha256:
            return True
        return self._sha256(target) == entry.sha256

    def _progress(self, logical: str, received: int, total: int | None):
        with self._lock:
            self._status = replace(
                self._status,
                state='downloading',
                current_file=logical,
                received_bytes=received,
                total_bytes=total,
            )

    def _verify_candidate(self, part: Path, entry: ModelManifestEntry) -> None:
        stat = part.stat()
        if stat.st_size != entry.size or self._sha256(part) != entry.sha256:
            raise ValueError('download_integrity_mismatch')

    def _download_entry(self, manifest_id: str, entry: ModelManifestEntry, target: Path) -> None:
        capability = self.client.request_download(manifest_id, entry)
        part = Path(str(target) + '.part')
        part.parent.mkdir(parents=True, exist_ok=True)
        try:
            part.unlink(missing_ok=True)
            direct_error = None
            if capability.direct_url:
                try:
                    self.client.download_to(
                        capability.direct_url,
                        capability.direct_headers,
                        part,
                        lambda got, total: self._progress(entry.path, got, total),
                    )
                    self._verify_candidate(part, entry)
                except Exception as exc:
                    direct_error = exc
                    part.unlink(missing_ok=True)
            else:
                direct_error = RuntimeError('direct_unavailable')

            if direct_error is not None:
                proxy_url = self.client.proxy_url(capability.proxy_token)
                self.client.download_to(
                    proxy_url,
                    {},
                    part,
                    lambda got, total: self._progress(entry.path, got, total),
                )
                self._verify_candidate(part, entry)

            os.replace(part, target)
        finally:
            part.unlink(missing_ok=True)

    def _cleanup_parts(self):
        for directory in (self.layout.detect_sync, self.layout.cls_sync, self.layout.root):
            if not directory.is_dir():
                continue
            for path in directory.glob('*.part'):
                try:
                    path.unlink()
                except OSError:
                    pass

    def _delete_stale(self, wanted: set[str]):
        for prefix, directory in (('detect/', self.layout.detect_sync), ('cls/', self.layout.cls_sync)):
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                if not path.is_file() or path.name.endswith('.part'):
                    continue
                logical = prefix + path.name
                if logical not in wanted:
                    path.unlink()

    def _sync_once(self):
        self._cleanup_parts()
        manifest = self.client.fetch_manifest()
        detect_count = sum(1 for entry in manifest.files if entry.path.startswith('detect/'))
        cls_count = sum(1 for entry in manifest.files if entry.path.startswith('cls/'))
        previous = load_state(self.layout.state_file)
        cached_files = previous.normalized_files()
        required: list[tuple[ModelManifestEntry, Path]] = []
        for entry in manifest.files:
            target = self._target(entry)
            if not self._matches(target, entry, cached_files.get(entry.path)):
                required.append((entry, target))

        with self._lock:
            self._status = replace(
                self._status,
                state='checking' if not required else 'downloading',
                total_files=len(required),
                completed_files=0,
                current_file=None,
                received_bytes=0,
                total_bytes=None,
                cloud_detect_count=detect_count,
                cloud_cls_count=cls_count,
                manifest_id=manifest.manifest_id,
            )

        failures: list[str] = []
        for entry, target in required:
            try:
                self._download_entry(manifest.manifest_id, entry, target)
                with self._lock:
                    self._status = replace(self._status, completed_files=self._status.completed_files + 1)
            except Exception:
                failures.append(entry.path)

        if failures:
            with self._lock:
                self._status = replace(
                    self._status,
                    state='failed',
                    error='同步失败: ' + ', '.join(failures),
                    current_file=None,
                )
            return

        wanted = {entry.path for entry in manifest.files}
        self._delete_stale(wanted)
        verified: dict[str, VerifiedFileState] = {}
        for entry in manifest.files:
            target = self._target(entry)
            if target.is_file():
                stat = target.stat()
                verified[entry.path] = VerifiedFileState(stat.st_size, stat.st_mtime_ns, entry.sha256)
        now = datetime.now(timezone.utc).isoformat()
        save_state(self.layout.state_file, SyncState(now, manifest.manifest_id, verified))
        with self._lock:
            self._status = replace(
                self._status,
                state='completed',
                current_file=None,
                received_bytes=0,
                total_bytes=None,
                last_successful_sync=now,
                manifest_id=manifest.manifest_id,
                error=None,
            )
