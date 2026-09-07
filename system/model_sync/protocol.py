from __future__ import annotations

import re
import unicodedata
import urllib.parse
from dataclasses import dataclass

_HEX64 = re.compile(r'^[a-f0-9]{64}$')
_TOKEN = re.compile(r'^[A-Za-z0-9_-]{20,128}$')


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ModelManifestEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelManifest:
    manifest_id: str
    files: tuple[ModelManifestEntry, ...]


@dataclass(frozen=True)
class DownloadCapability:
    path: str
    size: int
    sha256: str
    direct_url: str | None
    direct_headers: dict[str, str]
    proxy_token: str
    expires_at: float


def valid_logical_path(path: str) -> bool:
    if not isinstance(path, str) or not path or '\\' in path or '\x00' in path:
        return False
    if path.startswith('/') or path.startswith('../') or '/..' in path:
        return False
    if path == 'tracker.yaml':
        return True
    parts = path.split('/')
    if len(parts) != 2:
        return False
    folder, name = parts
    if not name or name in {'.', '..'}:
        return False
    lower = name.lower()
    if folder == 'detect':
        return lower.endswith('.pt')
    if folder == 'cls':
        return lower.endswith(('.pt', '.onnx', '.engine'))
    return False


def valid_direct_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or '').lower()
        return (
            parsed.scheme == 'https'
            and not parsed.username
            and not parsed.password
            and parsed.port in (None, 443)
            and (
                host.endswith('.1drv.com')
                or host.endswith('.sharepoint.com')
                or host.endswith('.microsoftpersonalcontent.com')
            )
        )
    except (TypeError, ValueError):
        return False


def _parse_entry(value) -> ModelManifestEntry:
    if not isinstance(value, dict):
        raise ProtocolError('invalid_manifest_entry')
    path = value.get('path')
    size = value.get('size')
    sha256 = value.get('sha256')
    if not valid_logical_path(path):
        raise ProtocolError('invalid_model_path')
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ProtocolError('invalid_model_size')
    if not isinstance(sha256, str) or not _HEX64.fullmatch(sha256):
        raise ProtocolError('invalid_model_sha256')
    return ModelManifestEntry(path, size, sha256)


def parse_manifest(value) -> ModelManifest:
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        raise ProtocolError('unsupported_manifest')
    manifest_id = value.get('manifest_id')
    if not isinstance(manifest_id, str) or not _HEX64.fullmatch(manifest_id):
        raise ProtocolError('invalid_manifest_id')
    raw_files = value.get('files')
    if not isinstance(raw_files, list):
        raise ProtocolError('invalid_manifest_files')
    files = tuple(_parse_entry(item) for item in raw_files)
    seen = set()
    for item in files:
        key = unicodedata.normalize('NFC', item.path).casefold()
        if key in seen:
            raise ProtocolError('duplicate_model_path')
        seen.add(key)
    return ModelManifest(manifest_id, files)


def parse_capability(value) -> DownloadCapability:
    if not isinstance(value, dict):
        raise ProtocolError('invalid_capability')
    entry = _parse_entry(value)
    token = value.get('proxy_token')
    expires_at = value.get('expires_at')
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise ProtocolError('invalid_proxy_token')
    if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
        raise ProtocolError('invalid_expiry')
    url = value.get('direct_url')
    headers = value.get('direct_headers')
    if url is not None and not valid_direct_url(url):
        url = None
        headers = {}
    elif url is not None:
        if not isinstance(headers, dict):
            headers = {}
        headers = {
            str(key): str(item)
            for key, item in headers.items()
            if str(key).lower() in {'accept', 'referer', 'user-agent'}
        }
    else:
        headers = {}
    return DownloadCapability(entry.path, entry.size, entry.sha256, url, headers, token, float(expires_at))
