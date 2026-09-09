"""Anonymous DINOv3 component downloads through Neri's model distribution service."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import unicodedata
import urllib.parse
import urllib.request
from typing import Any, Callable

from system.model_sync.protocol import valid_direct_url

DINO_DISTRIBUTION_URL_ENV = "NERI_DINOV3_DISTRIBUTION_URL"
DINO_DEFAULT_DISTRIBUTION_URL = "https://myneri.top/api/models/v1"
_HEX64_RE = re.compile(r"^[a-f0-9]{64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{20,128}$")


class DinoV3DistributionError(RuntimeError):
    """Raised when the public DINOv3 distribution service is unavailable or invalid."""


@dataclass(frozen=True)
class DinoV3DistributionEntry:
    path: str
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class DinoV3DistributionManifest:
    manifest_id: str
    files: tuple[DinoV3DistributionEntry, ...]


@dataclass(frozen=True)
class DinoV3DownloadCapability:
    path: str
    size: int
    sha256: str
    direct_url: str | None
    direct_headers: dict[str, str]
    proxy_token: str


def _relative_component_path(path: Any) -> str:
    if not isinstance(path, str) or not path.startswith("DINOv3/"):
        raise DinoV3DistributionError("invalid_dinov3_distribution_path")
    relative = path[len("DINOv3/") :]
    if not relative or "\\" in relative or "\x00" in relative:
        raise DinoV3DistributionError("invalid_dinov3_distribution_path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise DinoV3DistributionError("invalid_dinov3_distribution_path")
    return pure.as_posix()


class DinoV3DistributionClient:
    """Download the immutable DINOv3 tree without exposing OpenList credentials."""

    def __init__(self, origin: str | None = None, *, opener=None):
        configured = (
            origin
            or os.environ.get(DINO_DISTRIBUTION_URL_ENV)
            or DINO_DEFAULT_DISTRIBUTION_URL
        ).strip()
        parsed = urllib.parse.urlsplit(configured)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError("DINOv3 模型分发地址无效。")
        if (
            parsed.scheme != "https"
            and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("DINOv3 模型分发地址必须使用 HTTPS。")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("DINOv3 模型分发地址无效。")
        path = parsed.path.rstrip("/")
        self.origin = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, path, "", "")
        )
        self.opener = opener or urllib.request.urlopen

    def _json_request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json", "User-Agent": "Neri DINOv3 installer"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.origin + endpoint,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self.opener(request, timeout=30) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
        except Exception as exc:
            raise DinoV3DistributionError(
                f"DINOv3 模型分发请求失败: {endpoint}"
            ) from exc
        if len(raw) > 16 * 1024 * 1024:
            raise DinoV3DistributionError("DINOv3 模型分发响应过大。")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DinoV3DistributionError("DINOv3 模型分发返回了无效 JSON。") from exc
        if not isinstance(value, dict):
            raise DinoV3DistributionError("DINOv3 模型分发响应格式无效。")
        return value

    def fetch_manifest(self) -> DinoV3DistributionManifest:
        value = self._json_request("GET", "/dinov3/manifest")
        if value.get("schema_version") != 1:
            raise DinoV3DistributionError("unsupported_dinov3_distribution_manifest")
        manifest_id = value.get("manifest_id")
        if not isinstance(manifest_id, str) or not _HEX64_RE.fullmatch(manifest_id):
            raise DinoV3DistributionError("invalid_dinov3_distribution_manifest_id")
        raw_files = value.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise DinoV3DistributionError("invalid_dinov3_distribution_files")

        files: list[DinoV3DistributionEntry] = []
        seen: set[str] = set()
        for item in raw_files:
            if not isinstance(item, dict):
                raise DinoV3DistributionError("invalid_dinov3_distribution_entry")
            path = item.get("path")
            relative = _relative_component_path(path)
            size = item.get("size")
            sha256 = item.get("sha256")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise DinoV3DistributionError("invalid_dinov3_distribution_size")
            if not isinstance(sha256, str) or not _HEX64_RE.fullmatch(sha256):
                raise DinoV3DistributionError("invalid_dinov3_distribution_sha256")
            key = unicodedata.normalize("NFC", relative).casefold()
            if key in seen:
                raise DinoV3DistributionError("duplicate_dinov3_distribution_path")
            seen.add(key)
            files.append(DinoV3DistributionEntry(path, relative, size, sha256))
        return DinoV3DistributionManifest(manifest_id, tuple(files))

    def _capability(
        self,
        manifest: DinoV3DistributionManifest,
        entry: DinoV3DistributionEntry,
    ) -> DinoV3DownloadCapability:
        value = self._json_request(
            "POST",
            "/dinov3/direct",
            {
                "manifest_id": manifest.manifest_id,
                "path": entry.path,
                "sha256": entry.sha256,
            },
        )
        if (
            value.get("path") != entry.path
            or value.get("size") != entry.size
            or value.get("sha256") != entry.sha256
        ):
            raise DinoV3DistributionError("dinov3_capability_identity_mismatch")
        token = value.get("proxy_token")
        if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
            raise DinoV3DistributionError("invalid_dinov3_proxy_token")

        direct_url = value.get("direct_url")
        direct_headers = value.get("direct_headers")
        if direct_url is not None and valid_direct_url(direct_url):
            if not isinstance(direct_headers, dict):
                direct_headers = {}
            safe_headers = {
                str(key): str(item)
                for key, item in direct_headers.items()
                if str(key).lower() in {"accept", "referer", "user-agent"}
            }
        else:
            direct_url = None
            safe_headers = {}
        return DinoV3DownloadCapability(
            entry.path,
            entry.size,
            entry.sha256,
            direct_url,
            safe_headers,
            token,
        )

    def _stream_to_file(
        self,
        url: str,
        target: Path,
        *,
        expected_size: int,
        expected_sha256: str,
        headers: dict[str, str] | None = None,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> None:
        request_headers = {"User-Agent": "Neri DINOv3 installer"}
        request_headers.update(headers or {})
        request = urllib.request.Request(url, headers=request_headers, method="GET")
        temp = target.with_name(target.name + ".download")
        temp.unlink(missing_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        received = 0
        try:
            with self.opener(request, timeout=90) as response, temp.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > expected_size:
                        raise DinoV3DistributionError("dinov3_download_size_mismatch")
                    digest.update(chunk)
                    output.write(chunk)
                    if on_progress is not None:
                        on_progress(received, expected_size)
            if received != expected_size:
                raise DinoV3DistributionError("dinov3_download_size_mismatch")
            if digest.hexdigest() != expected_sha256:
                raise DinoV3DistributionError("dinov3_download_sha256_mismatch")
            temp.replace(target)
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    def _download_entry(
        self,
        manifest: DinoV3DistributionManifest,
        entry: DinoV3DistributionEntry,
        target: Path,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> None:
        capability = self._capability(manifest, entry)
        if capability.direct_url is not None:
            try:
                self._stream_to_file(
                    capability.direct_url,
                    target,
                    expected_size=entry.size,
                    expected_sha256=entry.sha256,
                    headers=capability.direct_headers,
                    on_progress=on_progress,
                )
                return
            except Exception:
                target.unlink(missing_ok=True)
        self._stream_to_file(
            self.origin + "/proxy/" + capability.proxy_token,
            target,
            expected_size=entry.size,
            expected_sha256=entry.sha256,
            on_progress=on_progress,
        )

    def download_tree(
        self,
        relative_path: str,
        target: Path,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> None:
        raw = str(relative_path or "").replace("\\", "/").strip("/")
        if raw:
            pure = PurePosixPath(raw)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
                raise ValueError("DINOv3 分发路径无效。")
            prefix = pure.as_posix().rstrip("/") + "/"
        else:
            prefix = ""
        manifest = self.fetch_manifest()
        selected = [
            entry
            for entry in manifest.files
            if not prefix or entry.relative_path == raw or entry.relative_path.startswith(prefix)
        ]
        if not selected:
            raise DinoV3DistributionError("DINOv3 分发清单中没有目标文件。")
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        base_parts = len(PurePosixPath(raw).parts) if raw else 0
        for entry in selected:
            parts = PurePosixPath(entry.relative_path).parts
            local_parts = parts[base_parts:] if raw else parts
            if not local_parts:
                continue
            self._download_entry(
                manifest,
                entry,
                target.joinpath(*local_parts),
                on_progress=on_progress,
            )


# Compatibility name used by environment-maintenance call sites.
DinoV3CloudClient = DinoV3DistributionClient
