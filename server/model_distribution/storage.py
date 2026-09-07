from __future__ import annotations

import json
import posixpath
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
        }) or {}
        content = data.get("content") or []
        result = []
        for item in content:
            result.append(RemoteEntry(
                name=str(item.get("name", "")),
                size=int(item.get("size") or 0),
                modified=(str(item.get("modified")) if item.get("modified") else None),
                is_dir=bool(item.get("is_dir")),
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
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
        except OSError as exc:
            raise StorageError("drive_stream_failed") from exc
