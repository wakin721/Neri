"""Anonymous DINOv3 component downloads through Neri model distribution."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import urllib.parse
import urllib.error
import urllib.request
from typing import Any, Callable

from system.model_sync.protocol import valid_direct_url
from .download_cache import (checkpoint_partial, discard_partial, partial_paths,
                             restore_partial)
from .download_diagnostics import SERVER_DETAILS, exception_details, redact

DINO_DISTRIBUTION_URL_ENV = "NERI_DINOV3_DISTRIBUTION_URL"
DINO_DEFAULT_DISTRIBUTION_URL = "https://myneri.top/api/models/v1"
_HEX64_RE = re.compile(r"^[a-f0-9]{64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{20,128}$")
PROXY_SEGMENT_BYTES = 1024 * 1024
PROXY_SEGMENT_THRESHOLD = 8 * 1024 * 1024
PROXY_SEGMENT_ATTEMPTS = 5
JSON_REQUEST_ATTEMPTS = 5
_JSON_LIMIT = 16 * 1024 * 1024


class DinoV3DistributionError(RuntimeError):
    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details or {}


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
    if (not relative or "\\" in relative or "\x00" in relative or ':' in relative
            or any(part in {'', '.', '..'} for part in relative.split('/'))):
        raise DinoV3DistributionError("invalid_dinov3_distribution_path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise DinoV3DistributionError("invalid_dinov3_distribution_path")
    return pure.as_posix()


class DinoV3DistributionClient:
    def __init__(self, origin: str | None = None, *, opener=None,
                 cache_dir: Path | None = None, on_event=None):
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
        self.origin = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
        )
        self.opener = opener or urllib.request.urlopen
        self._native_fallback = opener is None and os.name == "nt"
        self._json_native = False
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        self.on_event = on_event

    def _event(self, **event):
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                # Diagnostic output must not turn a good transfer into failure.
                pass

    def _native_json_response(self, request):
        executable = shutil.which('curl.exe')
        if executable is None:
            raise OSError('Windows curl is unavailable')
        with tempfile.TemporaryDirectory(prefix='neri-api-') as scratch:
            root = Path(scratch)
            response_file, header_file = root/'response', root/'headers'
            command = [executable, '--disable', '--silent', '--show-error',
                       '--connect-timeout', '15', '--max-time', '30',
                       '--max-filesize', str(_JSON_LIMIT + 1),
                       '--request', request.get_method(),
                       '--header', 'Accept: application/json',
                       '--dump-header', str(header_file),
                       '--output', str(response_file), '--write-out', '%{http_code}']
            if request.data is not None:
                body = root/'request'
                body.write_bytes(request.data)
                command += ['--header', 'Content-Type: application/json',
                            '--data-binary', '@'+str(body)]
            command.append(request.full_url)
            try:
                result = subprocess.run(command, stdin=subprocess.DEVNULL,
                    capture_output=True, timeout=40,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError('Windows metadata request timed out') from exc
            if result.returncode:
                raise OSError(f'Windows metadata request failed ({result.returncode}): '
                              + redact(result.stderr.decode('utf-8', errors='replace'))[:200])
            raw = response_file.read_bytes()
            status = int(result.stdout.strip())
            if status >= 300:
                import io
                import email.parser
                blocks = header_file.read_text('iso-8859-1').split('\n\n')
                final_headers = next(block for block in reversed(blocks)
                                     if block.startswith('HTTP/'))
                headers = email.parser.Parser().parsestr(final_headers.split('\n', 1)[-1])
                raise urllib.error.HTTPError(request.full_url, status,
                    'metadata request failed', headers, io.BytesIO(raw))
            return raw

    def _json_request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "User-Agent": "Neri DINOv3 installer",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.origin + endpoint,
            data=data,
            headers=headers,
            method=method,
        )
        phase = 'capability' if endpoint.endswith('/direct') else 'manifest'
        file = payload.get('path') if payload else None
        for attempt in range(JSON_REQUEST_ATTEMPTS):
            started = time.monotonic()
            self._event(phase=phase, endpoint=endpoint, file=file,
                        attempt=attempt + 1, state='requesting',
                        transport='windows_curl' if self._json_native else 'urllib')
            try:
                if self._json_native:
                    raw = self._native_json_response(request)
                else:
                    with self.opener(request, timeout=30) as response:
                        raw = response.read(_JSON_LIMIT + 1)
                break
            except Exception as exc:
                detail = exception_details(exc)
                status = detail.get('http_status')
                retryable = isinstance(exc, (OSError, http.client.HTTPException))
                delay = min(8, 2 ** attempt)
                if isinstance(exc, urllib.error.HTTPError):
                    retryable = exc.code in {408, 425, 429, 500, 502, 503, 504}
                    try:
                        body = json.loads(exc.read(4096))
                        code = body.get('detail') if isinstance(body, dict) else None
                        if isinstance(code, str) and code in SERVER_DETAILS:
                            detail['server_detail'] = code
                    except Exception:
                        pass
                    retry_after = exc.headers.get('Retry-After') if exc.headers else None
                    if retry_after:
                        try:
                            wait = (float(retry_after) if retry_after.strip().isdigit()
                                    else (parsedate_to_datetime(retry_after) -
                                          datetime.now(timezone.utc)).total_seconds())
                            detail['retry_after_seconds'] = max(0, wait)
                            if wait > 60:
                                retryable = False
                            else:
                                delay = max(delay, max(0, wait))
                        except (ValueError, TypeError, OverflowError):
                            pass
                    try:
                        exc.close()
                    except Exception:
                        pass
                elif retryable and self._native_fallback and shutil.which('curl.exe'):
                    self._json_native = True
                detail.update(phase=phase, endpoint=endpoint, file=file,
                              attempt=attempt + 1, max_attempts=JSON_REQUEST_ATTEMPTS,
                              elapsed_seconds=round(time.monotonic() - started, 3))
                retrying = retryable and attempt + 1 < JSON_REQUEST_ATTEMPTS
                self._event(**detail, state='retrying' if retrying else 'failed',
                            retry_delay_seconds=delay if retrying else 0)
                if not retrying:
                    suffix = f'HTTP {status}' if status else detail['exception_type']
                    if detail.get('server_detail'):
                        suffix += ': ' + detail['server_detail']
                    raise DinoV3DistributionError(
                        f'DINOv3 模型分发请求失败: {endpoint}'
                        f'（文件: {file or "清单"}；{suffix}；尝试 {attempt + 1} 次）',
                        details=detail) from exc
                time.sleep(delay)
        if len(raw) > _JSON_LIMIT:
            raise DinoV3DistributionError("DINOv3 模型分发响应过大。")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DinoV3DistributionError(
                "DINOv3 模型分发返回了无效 JSON。"
            ) from exc
        if not isinstance(value, dict):
            raise DinoV3DistributionError("DINOv3 模型分发响应格式无效。")
        return value

    def fetch_manifest(self) -> DinoV3DistributionManifest:
        value = self._json_request("GET", "/dinov3/manifest")
        if value.get("schema_version") != 1:
            raise DinoV3DistributionError(
                "unsupported_dinov3_distribution_manifest"
            )
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

    def _native_proxy_segment(self, url, start, end, size, directory):
        """Use Windows' TLS transport after Python's TLS stream fails."""
        executable = shutil.which("curl.exe")
        if executable is None:
            raise OSError("Windows curl is unavailable")
        with tempfile.TemporaryDirectory(prefix=".neri-range-", dir=directory) as scratch:
            body, headers = Path(scratch) / "body", Path(scratch) / "headers"
            try:
                result = subprocess.run([
                    executable, "--disable", "--fail", "--silent", "--show-error",
                    "--connect-timeout", "30", "--max-time", "90",
                    "--max-filesize", str(end - start + 1),
                    "--range", f"{start}-{end}", "--dump-header", str(headers),
                    "--output", str(body), "--write-out", "%{http_code}", url,
                ], stdin=subprocess.DEVNULL, capture_output=True, timeout=100,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except subprocess.TimeoutExpired as exc:
                raise OSError("Windows range download timed out") from exc
            if result.returncode:
                detail = result.stderr.decode("utf-8", errors="replace").replace(url, "[download URL]").strip()
                raise OSError(f"Windows range download failed ({result.returncode}): {detail[:200]}")
            ranges = [line.split(":", 1)[1].strip() for line in
                      headers.read_text("iso-8859-1").splitlines()
                      if line.lower().startswith("content-range:")]
            if result.stdout.strip() != b"206" or ranges != [f"bytes {start}-{end}/{size}"]:
                raise DinoV3DistributionError("dinov3_download_range_mismatch")
            data = body.read_bytes()
            if len(data) != end - start + 1:
                raise DinoV3DistributionError("dinov3_download_size_mismatch")
            return data

    def _stream_proxy_segments(
        self,
        manifest: DinoV3DistributionManifest,
        entry: DinoV3DistributionEntry,
        target: Path,
        capability: DinoV3DownloadCapability,
        on_progress=None,
    ) -> None:
        """Bound each TLS stream and retry a segment before committing its bytes."""
        target.parent.mkdir(parents=True, exist_ok=True)
        persistent = self.cache_dir is not None
        if persistent:
            temp, metadata, offset, digest = restore_partial(
                target, entry.size, entry.sha256, PROXY_SEGMENT_BYTES)
        else:
            temp, metadata = partial_paths(target)
            discard_partial(target)
            offset, digest = 0, hashlib.sha256()
        self._event(phase='download', file=entry.relative_path,
                    state='resuming' if offset else 'downloading', offset=offset, total=entry.size)
        if on_progress is not None and offset:
            on_progress(offset, entry.size)
        capability_uses = 0
        use_native = False
        try:
            with temp.open('r+b' if offset else 'wb') as output:
                output.seek(offset)
                for start in range(offset, entry.size, PROXY_SEGMENT_BYTES):
                    end = min(start + PROXY_SEGMENT_BYTES, entry.size) - 1
                    length = end - start + 1
                    for attempt in range(PROXY_SEGMENT_ATTEMPTS):
                        # Capabilities have a small, bounded use count and TTL.
                        # Renew within the use count, and on every retry.
                        try:
                            if capability_uses >= 3 or attempt:
                                capability = self._capability(manifest, entry)
                                capability_uses = 0
                            capability_uses += 1
                            request = urllib.request.Request(
                                self.origin + "/proxy/" + capability.proxy_token,
                                headers={"User-Agent": "Neri DINOv3 installer",
                                         "Range": f"bytes={start}-{end}"}, method="GET",
                            )
                            if use_native:
                                data = self._native_proxy_segment(
                                    request.full_url, start, end, entry.size, target.parent)
                            else:
                                with self.opener(request, timeout=90) as response:
                                    if (response.status != 206 or response.headers.get("Content-Range")
                                            != f"bytes {start}-{end}/{entry.size}"):
                                        raise DinoV3DistributionError("dinov3_download_range_mismatch")
                                    data = response.read(length + 1)
                                    if len(data) != length:
                                        raise DinoV3DistributionError("dinov3_download_size_mismatch")
                            break
                        except (OSError, http.client.HTTPException) as exc:
                            self._event(**exception_details(exc), phase='segment',
                                file=entry.relative_path, offset=start,
                                attempt=attempt + 1, state='retrying')
                            if self._native_fallback and shutil.which("curl.exe"):
                                use_native = True
                            if attempt + 1 == PROXY_SEGMENT_ATTEMPTS:
                                raise
                            time.sleep(min(8, 2 ** attempt))
                        except DinoV3DistributionError as exc:
                            # JSON requests already exhausted their own retry
                            # budget, or received a permanent HTTP error.
                            if exc.details.get('phase') == 'capability':
                                raise
                            self._event(**exception_details(exc), phase='segment',
                                file=entry.relative_path, offset=start,
                                attempt=attempt + 1, state='retrying')
                            if attempt + 1 == PROXY_SEGMENT_ATTEMPTS:
                                raise
                    output.write(data)
                    digest.update(data)
                    if persistent:
                        output.flush()
                        os.fsync(output.fileno())
                        checkpoint_partial(metadata, size=entry.size, sha256=entry.sha256,
                            segment_size=PROXY_SEGMENT_BYTES, offset=end + 1, digest=digest)
                    if on_progress is not None:
                        on_progress(end + 1, entry.size)
            if temp.stat().st_size != entry.size:
                raise DinoV3DistributionError("dinov3_download_size_mismatch")
            if digest.hexdigest() != entry.sha256:
                discard_partial(target)
                raise DinoV3DistributionError("dinov3_download_sha256_mismatch")
            temp.replace(target)
            metadata.unlink(missing_ok=True)
        except Exception:
            if not persistent:
                discard_partial(target)
            raise

    def _download_entry(
        self,
        manifest: DinoV3DistributionManifest,
        entry: DinoV3DistributionEntry,
        target: Path,
        on_progress=None,
    ) -> None:
        capability = self._capability(manifest, entry)
        if capability.direct_url is not None and not (
                self.cache_dir is not None and partial_paths(target)[0].exists()):
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
        if entry.size >= PROXY_SEGMENT_THRESHOLD:
            self._stream_proxy_segments(manifest, entry, target, capability, on_progress)
            return
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
        on_progress=None,
    ) -> None:
        raw = str(relative_path or "").replace("\\", "/").strip("/")
        if raw:
            pure = PurePosixPath(raw)
            if pure.is_absolute() or any(
                part in {"", ".", ".."} for part in pure.parts
            ):
                raise ValueError("DINOv3 分发路径无效。")
            prefix = pure.as_posix().rstrip("/") + "/"
        else:
            prefix = ""
        manifest = self.fetch_manifest()
        selected = [
            entry
            for entry in manifest.files
            if not prefix
            or entry.relative_path == raw
            or entry.relative_path.startswith(prefix)
        ]
        if not selected:
            raise DinoV3DistributionError("DINOv3 分发清单中没有目标文件。")
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        base_parts = len(PurePosixPath(raw).parts) if raw else 0
        total_bytes = sum(entry.size for entry in selected)
        completed_bytes = 0
        reported_percent = 0
        for entry in selected:
            parts = PurePosixPath(entry.relative_path).parts
            local_parts = parts[base_parts:] if raw else parts
            if local_parts:
                def report_bytes(received, _file_size):
                    nonlocal reported_percent
                    if on_progress is None:
                        return
                    percent = (
                        min(99, (completed_bytes + received) * 100 // total_bytes)
                        if total_bytes
                        else 0
                    )
                    reported_percent = max(reported_percent, percent)
                    on_progress(
                        reported_percent,
                        f"正在下载联合模型组件: {entry.relative_path}",
                    )

                destination = target.joinpath(*local_parts)
                if self.cache_dir is None:
                    self._download_entry(manifest, entry, destination, on_progress=report_bytes)
                else:
                    cache_root = self.cache_dir / manifest.manifest_id
                    cached = cache_root.joinpath(*PurePosixPath(entry.relative_path).parts)
                    cached.resolve().relative_to(cache_root.resolve())
                    valid = cached.is_file() and cached.stat().st_size == entry.size
                    if valid:
                        with cached.open('rb') as stream:
                            valid = hashlib.file_digest(stream, 'sha256').hexdigest() == entry.sha256
                    if valid:
                        self._event(phase='cache', file=entry.relative_path, state='verified')
                        report_bytes(entry.size, entry.size)
                    else:
                        if cached.exists():
                            self._event(phase='cache', file=entry.relative_path, state='invalid')
                        cached.unlink(missing_ok=True)
                        self._download_entry(manifest, entry, cached, on_progress=report_bytes)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cached, destination)
                completed_bytes += entry.size
        if on_progress is not None:
            on_progress(100, "联合模型组件下载完成，正在校验组件。")
