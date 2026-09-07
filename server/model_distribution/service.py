from __future__ import annotations

from pathlib import Path

from .capabilities import CapabilityStore, safe_direct_headers, valid_direct_download_url
from .manifest import ManifestBuilder, ManifestEntry, ManifestSnapshot


class DistributionError(RuntimeError):
    pass


class DistributionService:
    def __init__(self, state_dir: Path, store, *, ttl_seconds: int = 600):
        self.store = store
        self.builder = ManifestBuilder(Path(state_dir), store)
        self.capabilities = CapabilityStore(Path(state_dir), ttl_seconds=ttl_seconds)
        self._snapshot: ManifestSnapshot | None = None

    def manifest(self) -> ManifestSnapshot:
        self._snapshot = self.builder.build()
        return self._snapshot

    def _current_entry(self, manifest_id: str, path: str, sha256: str) -> ManifestEntry:
        snapshot = self._snapshot or self.manifest()
        if snapshot.manifest_id != manifest_id:
            raise DistributionError("stale_manifest")
        for entry in snapshot.files:
            if entry.path == path:
                if entry.sha256 != sha256:
                    raise DistributionError("checksum_mismatch")
                return entry
        raise DistributionError("model_not_in_manifest")

    @staticmethod
    def _remote_path(logical: str) -> str:
        return "/Neri_Data/Model/" + logical

    def direct(self, manifest_id: str, path: str, sha256: str) -> dict:
        entry = self._current_entry(manifest_id, path, sha256)
        proxy_token, expires_at = self.capabilities.issue(entry)
        direct_url = None
        direct_headers: dict[str, str] = {}
        try:
            link = self.store.resolve_link(self._remote_path(entry.path))
            if valid_direct_download_url(link.url):
                direct_url = link.url
                direct_headers = safe_direct_headers(link.headers)
        except Exception:
            direct_url = None
            direct_headers = {}
        return {
            "path": entry.path,
            "size": entry.size,
            "sha256": entry.sha256,
            "direct_url": direct_url,
            "direct_headers": direct_headers,
            "proxy_token": proxy_token,
            "expires_at": expires_at,
        }

    def consume_proxy(self, token: str):
        return self.capabilities.consume(token)
