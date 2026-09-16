"""FastAPI entrypoint for the Flutter-based Neri client."""

from . import model_services as _model_services
from . import services as _services
from .runtime_patches import install_runtime_patches

for _name in (
    "model_directory",
    "classification_model_directory",
    "list_available_models",
    "list_available_classification_models",
    "list_model_classes",
):
    setattr(_services, _name, getattr(_model_services, _name))

install_runtime_patches(_services)

from .main_core import *  # noqa: F401,F403
from .main_core import app
from .dinov2_environment import dinov2_environment_router
from .dinov2_feedback_api import dinov2_feedback_router
from system.model_sync.integration import wire_model_sync

app.include_router(dinov2_environment_router())
app.include_router(dinov2_feedback_router())
wire_model_sync(app)
