"""Bounded broker calls and direct OneDrive uploads. Never log bearer URLs."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_CHUNK_BYTES = 5 * 1024 * 1024


class UploadCancelled(Exception):
    pass


class UploadError(Exception):
    pass


def valid_upload_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or "").lower()
        return (parsed.scheme == "https" and not parsed.username and not parsed.password
                and parsed.port in (None, 443)
                and (host == "my.microsoftpersonalcontent.com" or host.endswith(".1drv.com")
                     or host.endswith(".sharepoint.com")))
    except (TypeError, ValueError):
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpTransport:
    def __init__(self, base_url="https://myneri.top/api/training", timeout=180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(_NoRedirect)

    def _request(self, url, method, body=None, headers=None):
        request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                content = response.read(1024 * 1024 + 1)
                if len(content) > 1024 * 1024:
                    raise UploadError("response_too_large")
                return json.loads(content) if content else {}
        except urllib.error.HTTPError as error:
            raise UploadError(f"http_{error.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise UploadError("network_or_response_error") from None

    def _broker(self, path, method, body):
        return self._request(self.base_url + path, method, json.dumps(body).encode(),
                             {"Content-Type": "application/json"})

    def _valid_cancel(self, url):
        prefix = self.base_url + '/v1/uploads/'
        if not isinstance(url, str) or not url.startswith(prefix):
            return False
        import re
        return re.fullmatch(r'[A-Za-z0-9_-]{43}', url[len(prefix):]) is not None

    def upload(self, job, photo, annotation, is_current):
        if not is_current():
            raise UploadCancelled()
        response = self._broker("/v1/submissions", "POST", {
            "sample_id": job["sample_id"], "revision": job["revision"], "secret": job["secret"],
            "species": job["payload"]["species"], "image_bytes": len(photo),
            "annotation_bytes": len(annotation),
        })
        if not isinstance(response, dict) or response.get('sample_id') != job['sample_id'] or response.get('revision') != job['revision']:
            raise UploadError('invalid_submission_receipt')
        if response.get('already_complete') is True and response.get('targets') == []:
            return
        targets = response.get("targets", [])
        expected = {(kind, species) for kind in ("image", "annotation") for species in job["payload"]["species"]}
        actual = {(t.get("kind"), t.get("species")) for t in targets if isinstance(t, dict)}
        if actual != expected or len(targets) != len(expected):
            raise UploadError("invalid_upload_targets")
        if any(not valid_upload_url(t.get("upload_url", "")) or not self._valid_cancel(t.get('cancel_url', '')) for t in targets):
            raise UploadError("invalid_upload_host")
        complete = False
        try:
            for target in targets:
                content = photo if target["kind"] == "image" else annotation
                chunk_size = int(target.get("chunk_size", DEFAULT_CHUNK_BYTES))
                if chunk_size <= 0 or chunk_size > 10 * 1024 * 1024 or chunk_size % 327680:
                    raise UploadError("invalid_chunk_size")
                for offset in range(0, len(content), chunk_size):
                    if not is_current():
                        raise UploadCancelled()
                    chunk = content[offset:offset + chunk_size]
                    self._request(target["upload_url"], "PUT", chunk, {
                        "Content-Type": "application/octet-stream", "Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {offset}-{offset + len(chunk) - 1}/{len(content)}",
                    })
            if not is_current():
                raise UploadCancelled()
            self._broker(f'/v1/submissions/{job["sample_id"]}/complete', "POST",
                         {"revision": job["revision"], "secret": job["secret"]})
            complete = True
        finally:
            if not complete:
                for target in targets:
                    try:
                        self._request(target["cancel_url"], "DELETE")
                    except Exception:
                        pass

    def delete(self, job, is_current):
        if not is_current():
            raise UploadCancelled()
        self._broker(f'/v1/submissions/{job["sample_id"]}', "DELETE", {"secret": job["secret"]})
