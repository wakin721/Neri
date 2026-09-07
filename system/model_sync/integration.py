"""Wire model synchronization into the existing Neri FastAPI application."""
from __future__ import annotations

from fastapi import FastAPI

from . import get_model_sync_manager
from .api import create_model_sync_router
from .bootstrap import install_model_sync


def wire_model_sync(app: FastAPI) -> FastAPI:
    app.include_router(create_model_sync_router(get_model_sync_manager))
    install_model_sync(app, manager_provider=get_model_sync_manager)
    return app
