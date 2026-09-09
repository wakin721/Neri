"""FastAPI entrypoint for the Flutter-based Neri client."""

from . import model_services as _model_services
from . import services as _services

# Patch canonical model hooks and DINOv3-aware class loading before main_core
# imports the historical service surface.
for _name in (
    "model_directory",
    "classification_model_directory",
    "list_available_models",
    "list_available_classification_models",
    "list_model_classes",
):
    setattr(_services, _name, getattr(_model_services, _name))

from .main_core import *  # noqa: F401,F403 - preserve the historical module surface
from .main_core import app
from .dinov3_environment import dinov3_environment_router
from system.model_sync.integration import wire_model_sync

app.include_router(dinov3_environment_router())
wire_model_sync(app)
