"""Publish an immutable local DINOv3 release alongside the OpenList catalog.

Local links are private handles, never download URLs. Existing capability,
byte-budget, and checksum rules apply to every local file as well.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote, unquote

from .storage import RemoteEntry, StorageError, UpstreamLink

PREFIX = "/Neri_Data/Model/DINOv3"
LOCAL_SCHEME = "neri-local-dinov3:"


class LocalComponentOverlay:
    def __init__(self, upstream, root: Path, *, prefix: str = PREFIX):
        self.upstream = upstream
        if prefix not in {PREFIX, PREFIX + '/cpu'}:
            raise StorageError("invalid_local_component_prefix")
        self.prefix = prefix
        # Resolve once: a release change requires restarting the service, so
        # outstanding capabilities cannot silently switch to another release.
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise StorageError("invalid_local_component_root")

    def _is_local(self, path: str) -> bool:
        return path == self.prefix or path.startswith(self.prefix + "/")

    def _path(self, remote: str) -> Path:
        if not self._is_local(remote) or "\\" in remote or "\x00" in remote:
            raise StorageError("invalid_local_component_path")
        relative = remote[len(self.prefix) + 1:] if remote != self.prefix else ""
        parts = relative.split("/") if relative else []
        if any(part in {"", ".", ".."} or ":" in part for part in parts):
            raise StorageError("invalid_local_component_path")
        target = self.root.joinpath(*PurePosixPath(relative).parts)
        try:
            target.resolve().relative_to(self.root)
        except (OSError, ValueError):
            raise StorageError("local_component_path_outside_root") from None
        # Release trees contain ordinary files only, including inside the root.
        if any(node.is_symlink() for node in (target, *target.parents) if node != self.root and self.root in node.parents):
            raise StorageError("local_component_symlink")
        return target

    @staticmethod
    def _entry(path: Path) -> RemoteEntry:
        stat = path.stat()
        return RemoteEntry(path.name, stat.st_size if path.is_file() else 0,
                           f"{stat.st_mtime_ns}:{stat.st_ctime_ns}", path.is_dir())

    def list_dir(self, path: str):
        if not self._is_local(path):
            return self.upstream.list_dir(path)
        target = self._path(path)
        return [self._entry(self._path(path + "/" + child.name))
                for child in sorted(target.iterdir())]

    def stat(self, path: str):
        if not self._is_local(path):
            return self.upstream.stat(path)
        target = self._path(path)
        return self._entry(target) if target.exists() else None

    def resolve_link(self, path: str) -> UpstreamLink:
        if not self._is_local(path):
            return self.upstream.resolve_link(path)
        if not self._path(path).is_file():
            raise StorageError("local_component_file_missing")
        return UpstreamLink(LOCAL_SCHEME + quote(path, safe="/"), {})

    def iter_bytes(self, link: UpstreamLink, range_header: str | None = None):
        if (not link.url.startswith(LOCAL_SCHEME)
                or not self._is_local(unquote(link.url[len(LOCAL_SCHEME):]))):
            yield from self.upstream.iter_bytes(link, range_header=range_header)
            return
        target = self._path(unquote(link.url[len(LOCAL_SCHEME):]))
        size = target.stat().st_size
        start, end = 0, size - 1
        if range_header is not None:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
            if not match:
                raise StorageError("invalid_local_component_range")
            start = int(match[1])
            end = min(int(match[2]), size - 1) if match[2] else size - 1
            if not 0 <= start <= end < size:
                raise StorageError("invalid_local_component_range")
        remaining = end - start + 1
        with target.open("rb") as stream:
            stream.seek(start)
            while remaining:
                data = stream.read(min(1024 * 1024, remaining))
                if not data:
                    raise StorageError("local_component_short_stream")
                remaining -= len(data)
                yield data
