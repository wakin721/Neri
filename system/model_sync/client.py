from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Mapping

from .protocol import (
    DownloadCapability,
    ModelManifest,
    ModelManifestEntry,
    parse_capability,
    parse_manifest,
    valid_direct_url,
)

_TOKEN = re.compile(r'^[A-Za-z0-9_-]{20,128}$')


class ModelDistributionError(RuntimeError):
    pass


class SafeDirectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow redirects only while the direct capability stays on Microsoft HTTPS."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if not valid_direct_url(newurl):
            raise urllib.error.HTTPError(
                newurl, code, 'unsafe_direct_redirect', headers, fp
            )
        return super().redirect_request(request, fp, code, msg, headers, newurl)


class ModelDistributionClient:
    def __init__(self, base_url: str | None = None, opener=None):
        self.base_url = (base_url or os.environ.get('NERI_MODEL_SERVICE_URL') or 'https://myneri.top/api/models/v1').rstrip('/')
        self.opener = opener or urllib.request.urlopen
        self.direct_opener = (
            opener
            if opener is not None
            else urllib.request.build_opener(SafeDirectRedirectHandler()).open
        )

    def _read_json(self, request: urllib.request.Request, *, timeout: int = 20):
        try:
            with self.opener(request, timeout=timeout) as response:
                raw = response.read(1024 * 1024 + 1)
        except Exception as exc:
            raise ModelDistributionError('model_service_unavailable') from exc
        if len(raw) > 1024 * 1024:
            raise ModelDistributionError('model_service_response_too_large')
        try:
            return json.loads(raw)
        except ValueError as exc:
            if raw.lstrip().lower().startswith((b'<!doctype html', b'<html')):
                raise ModelDistributionError(
                    '模型同步服务尚未正确配置，服务器返回了网页，请联系维护者。'
                ) from exc
            raise ModelDistributionError('invalid_model_service_response') from exc

    def fetch_manifest(self) -> ModelManifest:
        request = urllib.request.Request(self.base_url + '/manifest', method='GET')
        # Uncached OpenList models must be hashed by the server before publishing.
        # Match the model service's nginx read timeout for this operation only.
        return parse_manifest(self._read_json(request, timeout=900))

    def request_download(self, manifest_id: str, entry: ModelManifestEntry) -> DownloadCapability:
        body = json.dumps({'manifest_id': manifest_id, 'path': entry.path, 'sha256': entry.sha256}).encode('utf-8')
        request = urllib.request.Request(
            self.base_url + '/direct',
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        capability = parse_capability(self._read_json(request))
        if capability.path != entry.path or capability.sha256 != entry.sha256 or capability.size != entry.size:
            raise ModelDistributionError('capability_identity_mismatch')
        return capability

    def proxy_url(self, token: str) -> str:
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            raise ModelDistributionError('invalid_proxy_token')
        return self.base_url + '/proxy/' + token

    def download_to(
        self,
        url: str,
        headers: Mapping[str, str],
        target: Path,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> None:
        request = urllib.request.Request(url, headers=dict(headers), method='GET')
        opener = self.direct_opener if valid_direct_url(url) else self.opener
        try:
            with opener(request, timeout=20) as response, open(target, 'wb') as handle:
                total = response.headers.get('Content-Length') if hasattr(response, 'headers') else None
                total_bytes = int(total) if total and str(total).isdigit() else None
                received = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
                    if on_progress:
                        on_progress(received, total_bytes)
        except Exception as exc:
            raise ModelDistributionError('model_download_failed') from exc
