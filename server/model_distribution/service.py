from __future__ import annotations

from pathlib import Path

from .capabilities import CapabilityStore, safe_direct_headers, valid_direct_download_url
from .manifest import DinoV2ManifestBuilder, DinoV3ManifestBuilder, DinoV3CpuManifestBuilder, ManifestBuilder, ManifestEntry, ManifestSnapshot


class DistributionError(RuntimeError):
    pass


class DistributionService:
    def __init__(self, state_dir: Path, store, *, ttl_seconds: int = 600):
        self.store = store
        state_dir = Path(state_dir)
        self.builder = ManifestBuilder(state_dir, store)
        self.dinov3_builder = DinoV3ManifestBuilder(state_dir, store)
        self.dinov3_cpu_builder = DinoV3CpuManifestBuilder(state_dir, store)
        self.dinov2_builder = DinoV2ManifestBuilder(state_dir, store)
        self.capabilities = CapabilityStore(state_dir, ttl_seconds=ttl_seconds)
        self._snapshot: ManifestSnapshot | None = None
        self._dinov3_snapshot: ManifestSnapshot | None = None
        self._dinov3_cpu_snapshot: ManifestSnapshot | None = None
        self._dinov2_snapshot: ManifestSnapshot | None = None

    def manifest(self) -> ManifestSnapshot:
        self._snapshot = self.builder.build()
        return self._snapshot

    def dinov3_manifest(self, runtime: str = 'gpu') -> ManifestSnapshot:
        if runtime == 'cpu':
            self._dinov3_cpu_snapshot = self.dinov3_cpu_builder.build()
            return self._dinov3_cpu_snapshot
        if runtime != 'gpu':
            raise DistributionError('invalid_runtime')
        self._dinov3_snapshot = self.dinov3_builder.build()
        return self._dinov3_snapshot

    def dinov2_manifest(self) -> ManifestSnapshot:
        self._dinov2_snapshot = self.dinov2_builder.build()
        return self._dinov2_snapshot

    @staticmethod
    def _entry_from_snapshot(
        snapshot: ManifestSnapshot,
        manifest_id: str,
        path: str,
        sha256: str,
    ) -> ManifestEntry:
        if snapshot.manifest_id != manifest_id:
            raise DistributionError("stale_manifest")
        for entry in snapshot.files:
            if entry.path == path:
                if entry.sha256 != sha256:
                    raise DistributionError("checksum_mismatch")
                return entry
        raise DistributionError("model_not_in_manifest")

    def _current_entry(self, manifest_id: str, path: str, sha256: str) -> ManifestEntry:
        return self._entry_from_snapshot(
            self._snapshot or self.manifest(), manifest_id, path, sha256
        )

    def _current_dinov3_entry(
        self,
        manifest_id: str,
        path: str,
        sha256: str,
    ) -> ManifestEntry:
        if path.startswith('DINOv3/cpu/'):
            return self._entry_from_snapshot(
                self._dinov3_cpu_snapshot or self.dinov3_manifest('cpu'),
                manifest_id, path, sha256,
            )
        return self._entry_from_snapshot(
            self._dinov3_snapshot or self.dinov3_manifest(),
            manifest_id,
            path,
            sha256,
        )

    @staticmethod
    def _remote_path(logical: str) -> str:
        return "/Neri_Data/Model/" + logical

    def _direct_for_entry(self, entry: ManifestEntry) -> dict:
        proxy_token, expires_at = self.capabilities.issue(entry)
        direct_url = None
        direct_headers: dict[str, str] = {}
        try:
            resolver = getattr(self.store, 'resolve_direct_link', self.store.resolve_link)
            link = resolver(self._remote_path(entry.path))
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

    def direct(self, manifest_id: str, path: str, sha256: str) -> dict:
        return self._direct_for_entry(
            self._current_entry(manifest_id, path, sha256)
        )

    def dinov3_direct(self, manifest_id: str, path: str, sha256: str) -> dict:
        return self._direct_for_entry(
            self._current_dinov3_entry(manifest_id, path, sha256)
        )

    def consume_proxy(self, token: str):
        return self.capabilities.consume(token)

    def dinov2_direct(self, manifest_id: str, path: str, sha256: str) -> dict:
        return self._direct_for_entry(self._entry_from_snapshot(
            self._dinov2_snapshot or self.dinov2_manifest(), manifest_id, path, sha256,
        ))
