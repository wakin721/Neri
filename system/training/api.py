"""Local-only explicit consent API. Ordinary settings cannot grant consent."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictBool


class PrivacyChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agreement_version: str
    training_enabled: StrictBool


def privacy_router(queue_provider):
    router = APIRouter()

    def ensure_native_client(request):
        # Flutter desktop requests have no Origin; reject browser-created consent changes.
        if request.headers.get("origin"):
            raise HTTPException(status_code=403, detail="请在 Neri 软件内修改隐私设置。")

    @router.get("/api/privacy")
    def status():
        return queue_provider().status()

    @router.put("/api/privacy")
    def choose(choice: PrivacyChoice, request: Request):
        ensure_native_client(request)
        try:
            return queue_provider().set_consent(choice.agreement_version, choice.training_enabled)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None

    @router.delete("/api/privacy/queue")
    def clear(request: Request):
        ensure_native_client(request)
        return queue_provider().clear_pending()

    return router
