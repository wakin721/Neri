"""Model synchronization subsystem."""
from __future__ import annotations

import threading

from .client import ModelDistributionClient
from .layout import get_model_layout
from .manager import ModelSyncManager

_manager: ModelSyncManager | None = None
_manager_lock = threading.Lock()


def get_model_sync_manager() -> ModelSyncManager:
    global _manager
    if _manager is not None:
        return _manager
    with _manager_lock:
        if _manager is None:
            _manager = ModelSyncManager(get_model_layout(), ModelDistributionClient())
        return _manager


__all__ = ["get_model_sync_manager"]
