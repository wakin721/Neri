"""Local FastAPI router for model synchronization status and manual runs."""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from fastapi import APIRouter

from .manager import ModelSyncManager


def model_sync_status_payload(manager: ModelSyncManager) -> dict:
    return asdict(manager.status())


def start_model_sync_payload(manager: ModelSyncManager) -> dict:
    return asdict(manager.start())


def create_model_sync_router(
    manager_provider: Callable[[], ModelSyncManager],
) -> APIRouter:
    router = APIRouter()

    @router.get('/api/model-sync/status')
    def model_sync_status():
        return model_sync_status_payload(manager_provider())

    @router.post('/api/model-sync/run', status_code=202)
    def run_model_sync():
        return start_model_sync_payload(manager_provider())

    return router
