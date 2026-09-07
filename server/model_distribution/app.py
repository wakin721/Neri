from __future__ import annotations

import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import DistributionConfig
from .service import DistributionError, DistributionService
from .storage import OpenListModelStore

_RANGE_RE = re.compile(r"^bytes=\d+-\d*$")


def valid_range_header(value: str | None) -> bool:
    return value is None or bool(_RANGE_RE.fullmatch(value))


class DirectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    manifest_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    path: str = Field(min_length=1, max_length=260)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def create_app(
    config: DistributionConfig | None = None,
    service: DistributionService | None = None,
) -> FastAPI:
    config = config or DistributionConfig.from_env()
    service = service or DistributionService(
        config.state_dir,
        OpenListModelStore(config),
        ttl_seconds=config.capability_ttl_seconds,
    )
    app = FastAPI(title="Neri Model Distribution", version="1")

    @app.exception_handler(DistributionError)
    async def handle_distribution_error(_request: Request, exc: DistributionError):
        code = str(exc)
        status = 409 if code == "stale_manifest" else 422
        return JSONResponse(status_code=status, content={"detail": code})

    @app.get("/health")
    def health():
        return {"status": "ok", "schema_version": 1}

    @app.get("/v1/manifest")
    def manifest():
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
    def direct(request: DirectRequest):
        return service.direct(request.manifest_id, request.path, request.sha256)

    @app.get("/v1/proxy/{token}")
    def proxy(token: str, request: Request):
        range_header = request.headers.get("range")
        if not valid_range_header(range_header):
            raise HTTPException(status_code=416, detail="invalid_range")
        bound = service.consume_proxy(token)
        remote = "/Neri_Data/Model/" + bound.path
        link = service.store.resolve_link(remote)
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": (
                f'attachment; filename="{bound.path.rsplit("/", 1)[-1]}"'
            ),
        }
        status_code = 206 if range_header else 200
        return StreamingResponse(
            service.store.iter_bytes(link, range_header=range_header),
            media_type="application/octet-stream",
            status_code=status_code,
            headers=headers,
        )

    return app
