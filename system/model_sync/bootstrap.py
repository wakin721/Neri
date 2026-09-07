"""Install non-blocking model synchronization around an existing FastAPI lifespan."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI

from .layout import migrate_legacy_layout

logger = logging.getLogger(__name__)


def install_model_sync(
    app: FastAPI,
    *,
    manager_provider: Callable,
    migrate: Callable = migrate_legacy_layout,
) -> None:
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            migrate()
        except Exception as error:
            logger.warning('Model layout migration unavailable (%s)', type(error).__name__)

        async with original_lifespan(application):
            try:
                manager_provider().start()
            except Exception as error:
                logger.warning('Model synchronization unavailable (%s)', type(error).__name__)
            yield

    app.router.lifespan_context = lifespan
