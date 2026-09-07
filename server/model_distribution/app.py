from __future__ import annotations

import hashlib
import ipaddress
import re
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .capabilities import BudgetError, BudgetStore
from .config import DistributionConfig
from .service import DistributionError, DistributionService
from .storage import OpenListModelStore, StorageError

_RANGE_RE = re.compile(r"^bytes=\d+-\d*$")


def bounded_proxy_stream(chunks, expected_size: int):
    received = 0
    for chunk in chunks:
        received += len(chunk)
        if received > expected_size:
            raise StorageError("drive_stream_size_mismatch")
        yield chunk
    if received != expected_size:
        raise StorageError("drive_stream_size_mismatch")


def valid_range_header(value: str | None) -> bool:
    return value is None or bool(_RANGE_RE.fullmatch(value))


def proxy_reserved_bytes(file_size: int, range_header: str | None) -> int:
    size = int(file_size)
    if size < 0:
        raise ValueError("invalid_file_size")
    if range_header is None:
        return size
    if not valid_range_header(range_header):
        raise ValueError("invalid_range")

    start_text, end_text = range_header[6:].split("-", 1)
    start = int(start_text)
    if start >= size:
        raise ValueError("range_not_satisfiable")
    if not end_text:
        return size - start

    end = int(end_text)
    if end < start:
        raise ValueError("range_not_satisfiable")
    return min(end, size - 1) - start + 1


def request_client_ip(request: Request) -> str:
    peer = request.client.host if request.client is not None else "unknown"
    try:
        normalized_peer = str(ipaddress.ip_address(peer))
    except ValueError:
        normalized_peer = peer or "unknown"

    if normalized_peer not in {"127.0.0.1", "::1"}:
        return normalized_peer

    forwarded = request.headers.get("x-real-ip", "").strip()
    if not forwarded:
        return normalized_peer
    try:
        return str(ipaddress.ip_address(forwarded))
    except ValueError:
        return normalized_peer


class DirectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    manifest_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    path: str = Field(min_length=1, max_length=260)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _budget_secret(config: DistributionConfig) -> bytes:
    return hashlib.sha256(
        b"neri-model-budget-v1\0" + config.openlist_token.encode("utf-8")
    ).digest()


def create_app(
    config: DistributionConfig | None = None,
    service: DistributionService | None = None,
    budget: BudgetStore | None = None,
) -> FastAPI:
    config = config or DistributionConfig.from_env()
    service = service or DistributionService(
        config.state_dir,
        OpenListModelStore(config),
        ttl_seconds=config.capability_ttl_seconds,
    )
    budget = budget or BudgetStore(
        config.state_dir,
        secret=_budget_secret(config),
        requests_per_minute=config.requests_per_minute,
        daily_ip_bytes=config.daily_proxy_ip_bytes,
        daily_total_bytes=config.daily_proxy_total_bytes,
    )
    app = FastAPI(title="Neri Model Distribution", version="1")

    @app.exception_handler(DistributionError)
    async def handle_distribution_error(_request: Request, exc: DistributionError):
        code = str(exc)
        status = 409 if code == "stale_manifest" else 422
        return JSONResponse(status_code=status, content={"detail": code})

    @app.exception_handler(BudgetError)
    async def handle_budget_error(_request: Request, exc: BudgetError):
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    def consume_request_budget(request: Request) -> str:
        client_ip = request_client_ip(request)
        budget.check_request(client_ip)
        return client_ip

    @app.get("/health")
    def health():
        return {"status": "ok", "schema_version": 1}

    @app.get("/v1/manifest")
    def manifest(request: Request):
        consume_request_budget(request)
        snapshot = service.manifest()
        return {
            "schema_version": 1,
            "manifest_id": snapshot.manifest_id,
            "files": [
                {"path": item.path, "size": item.size, "sha256": item.sha256}
                for item in snapshot.files
            ],
        }

    @app.post("/v1/direct")
    def direct(payload: DirectRequest, request: Request):
        consume_request_budget(request)
        return service.direct(payload.manifest_id, payload.path, payload.sha256)

    @app.get("/v1/proxy/{token}")
    def proxy(token: str, request: Request):
        client_ip = consume_request_budget(request)
        range_header = request.headers.get("range")
        if not valid_range_header(range_header):
            raise HTTPException(status_code=416, detail="invalid_range")
        bound = service.consume_proxy(token)
        try:
            reserved_bytes = proxy_reserved_bytes(bound.size, range_header)
        except ValueError as exc:
            raise HTTPException(status_code=416, detail="invalid_range") from exc
        budget.reserve_proxy_bytes(client_ip, reserved_bytes)

        remote = "/Neri_Data/Model/" + bound.path
        link = service.store.resolve_link(remote)
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(reserved_bytes),
            "Content-Disposition": "attachment; filename*=UTF-8''" + quote(bound.path.rsplit("/", 1)[-1], safe=""),
        }
        if range_header:
            start = int(range_header[6:].split("-", 1)[0])
            headers["Content-Range"] = f"bytes {start}-{start + reserved_bytes - 1}/{bound.size}"
        status_code = 206 if range_header else 200
        return StreamingResponse(
            bounded_proxy_stream(service.store.iter_bytes(link, range_header=range_header), reserved_bytes),
            media_type="application/octet-stream",
            status_code=status_code,
            headers=headers,
        )

    return app
