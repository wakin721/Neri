"""FastAPI entrypoint for the Flutter-based Neri client."""

from . import model_services as _model_services
from . import services as _services

# Patch only the four model-catalog hooks before main_core imports them.  The
# rest of system.backend.services remains the historical module so tests and
# runtime patches still affect the globals used by its functions.
for _name in (
    "model_directory",
    "classification_model_directory",
    "list_available_models",
    "list_available_classification_models",
):
    setattr(_services, _name, getattr(_model_services, _name))

from .main_core import *  # noqa: F401,F403 - preserve the historical module surface
from .main_core import app
from system.model_sync.integration import wire_model_sync

wire_model_sync(app)
