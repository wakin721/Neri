from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerifiedFileState:
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class SyncState:
    last_successful_sync: str | None = None
    manifest_id: str | None = None
    files: dict[str, VerifiedFileState] | None = None

    def normalized_files(self) -> dict[str, VerifiedFileState]:
        return dict(self.files or {})


def load_state(path: Path) -> SyncState:
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        files = {
            str(key): VerifiedFileState(
                size=int(value['size']),
                mtime_ns=int(value['mtime_ns']),
                sha256=str(value['sha256']),
            )
            for key, value in dict(raw.get('files') or {}).items()
        }
        return SyncState(raw.get('last_successful_sync'), raw.get('manifest_id'), files)
    except Exception:
        return SyncState(files={})


def save_state(path: Path, state: SyncState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    payload = {
        'last_successful_sync': state.last_successful_sync,
        'manifest_id': state.manifest_id,
        'files': {
            key: asdict(value)
            for key, value in state.normalized_files().items()
        },
    }
    with open(temp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(temp, path)
