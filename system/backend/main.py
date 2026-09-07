"""FastAPI entrypoint for the Flutter-based Neri client."""

from .main_core import *  # noqa: F401,F403 - preserve the historical module surface
from .main_core import app
from system.model_sync.integration import wire_model_sync

wire_model_sync(app)
