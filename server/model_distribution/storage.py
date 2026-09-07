from __future__ import annotations

import json
import posixpath
import re
import urllib.request
from dataclasses import dataclass
from typing import Iterator

from .config import DistributionConfig


class StorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteEntry:
    name: str
    size: int
    modified: str | None
    is_dir: bool


@dataclass(frozen=True)
class UpstreamLink:
    url: str
    headers: dict[str, str]


class OpenListModelStore:
    def __init__(self, config: DistributionConfig):
        self.config = config
        self.base_root = config.root_path.rstrip("/")

    def _call(self, endpoint: str, body: dict):
        request = urllib.request.Request(
            self.config.openlist_url.rstrip("/") + "/api/fs/" + endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": self.config.openlist_token,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(1024 * 1024 + 1)
        except OSError as exc:
            raise StorageError("drive_unavailable") from exc
        if len(raw) > 1024 * 1024:
            raise StorageError("drive_response_too_large")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise StorageError("invalid_drive_response") from exc
        if not isinstance(payload, dict):
            raise StorageError("invalid_drive_response")
        if payload.get("code") != 200:
            raise StorageError("drive_operation_failed")
        return payload.get("data")

    def _safe_path(self, path: str) -> str:
        if not isinstance(path, str) or "\\" in path or "\x00" in path:
            raise StorageError("invalid_model_path")
        normalized = posixpath.normpath(path)
        root = self.base_root
        if normalized != root and not normalized.startswith(root + "/"):
            raise StorageError("model_path_outside_root")
        return normalized

    def list_dir(self, path: str) -> list[RemoteEntry]:
        path = self._safe_path(path)
        data = self._call("list", {
            "path": path,
            "password": "",
            "page": 1,
            "per_page": 0,
            "refresh": True,
        })
        if not isinstance(data, dict) or "content" not in data:
            raise StorageError("invalid_drive_listing")
        content = data["content"]
        total = data.get("total")
        if type(total) is not int or total < 0:
            raise StorageError("invalid_drive_listing")
        if content is None and total == 0:
            content = []
        if not isinstance(content, list) or len(content) != total:
            raise StorageError("incomplete_drive_listing")
        result = []
        for item in content:
            if not isinstance(item, dict):
                raise StorageError("invalid_drive_listing")
            name, size, is_dir = item.get("name"), item.get("size"), item.get("is_dir")
            if (not isinstance(name, str) or not name or name in {".", ".."}
                    or any(char in name for char in "/\\\x00")
                    or type(size) is not int or size < 0 or type(is_dir) is not bool):
                raise StorageError("invalid_drive_listing")
            result.append(RemoteEntry(
                name=name,
                size=size,
                modified=(str(item.get("modified")) if item.get("modified") else None),
                is_dir=is_dir,
            ))
        return result

    def stat(self, path: str) -> RemoteEntry | None:
        path = self._safe_path(path)
        try:
            data = self._call("get", {"path": path, "password": ""}) or {}
        except StorageError as exc:
            if str(exc) == "drive_operation_failed":
                return None
            raise
        return RemoteEntry(
            name=posixpath.basename(path),
            size=int(data.get("size") or 0),
            modified=(str(data.get("modified")) if data.get("modified") else None),
            is_dir=bool(data.get("is_dir")),
        )

    def resolve_link(self, path: str) -> UpstreamLink:
        path = self._safe_path(path)
        data = self._call("link", {"path": path, "password": ""}) or {}
        url = str(data.get("url") or data.get("link") or data.get("raw_url") or "")
        headers = data.get("header") or data.get("headers") or {}
        if not url:
            raise StorageError("missing_drive_link")
        return UpstreamLink(url=url, headers={str(k): str(v) for k, v in dict(headers).items()})

    def iter_bytes(self, link: UpstreamLink, range_header: str | None = None) -> Iterator[bytes]:
        headers = dict(link.headers)
        if range_header:
            headers["Range"] = range_header
        request = urllib.request.Request(link.url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                expected = None
                if range_header:
                    requested = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
                    actual = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )
                    if response.status != 206 or requested is None or actual is None:
                        raise StorageError("drive_range_not_honored")
                    start, end, total = map(int, actual.groups())
                    requested_end = int(requested[2]) if requested[2] else total - 1
                    if (start != int(requested[1]) or end != min(requested_end, total - 1)
                            or not 0 <= start <= end < total):
                        raise StorageError("drive_range_not_honored")
                    expected = end - start + 1
                received = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if expected is not None and received > expected:
                        raise StorageError("drive_stream_size_mismatch")
                    yield chunk
                if expected is not None and received != expected:
                    raise StorageError("drive_stream_size_mismatch")
        except OSError as exc:
            raise StorageError("drive_stream_failed") from exc
