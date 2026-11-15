"""Local download diagnostics which never include capabilities or signed URLs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import traceback
from typing import Any
import urllib.error

SERVER_DETAILS = frozenset({
    'stale_manifest', 'checksum_mismatch', 'model_not_in_manifest',
    'request_rate_limited', 'proxy_ip_budget_exceeded', 'proxy_total_budget_exceeded',
    'expired_or_invalid_capability', 'invalid_capability', 'drive_unavailable',
    'drive_operation_failed', 'invalid_range',
})


def redact(value: Any) -> str:
    text = str(value)
    text = re.sub(r'https?://[^\s<>\"\']+', '[URL hidden]', text, flags=re.I)
    text = re.sub(r'(?i)Bearer\s+[^\s\"\']+', 'Bearer [hidden]', text)
    text = re.sub(r'(?i)((?:proxy_token|access_token|authorization|signature|sig)'
                  r'[\"\']?\s*[:=]\s*[\"\']?)[^\s\"\',;}]+', r'\1[hidden]', text)
    text = re.sub(r'(?i)(/proxy/)[A-Za-z0-9_-]{20,128}', r'\1[hidden]', text)
    return text


def exception_details(exc: BaseException) -> dict:
    result: dict[str, Any] = {'exception_type': type(exc).__name__,
                              'reason': redact(exc)[:1000]}
    chain = []
    current: BaseException | None = exc
    seen = set()
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append({'type': type(current).__name__, 'message': redact(current)[:1000]})
        if isinstance(current, urllib.error.HTTPError):
            result['http_status'] = current.code
        for field in ('errno', 'winerror'):
            code = getattr(current, field, None)
            if isinstance(code, int):
                result[field] = code
        nested = getattr(current, 'reason', None)
        current = (current.__cause__ or current.__context__ or
                   (nested if isinstance(nested, BaseException) else None))
    result['causes'] = chain
    result['traceback'] = redact(''.join(traceback.format_exception(exc)))[:24000]
    return result


def sanitize_tree(value):
    if isinstance(value, dict):
        return {key: ('[hidden]' if str(key).lower() in {
            'proxy_token', 'access_token', 'authorization', 'signature', 'sig',
            'direct_url', 'download_url'} else sanitize_tree(item))
            for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_tree(item) for item in value]
    return redact(value) if isinstance(value, str) else value


def event_line(event: dict) -> str:
    return json.dumps({'time': datetime.now(timezone.utc).isoformat(), **sanitize_tree(event)},
                      ensure_ascii=False)
