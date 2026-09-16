"""Active backend core during the DINOv2 clean-break migration."""
from .main_core_legacy import *  # noqa: F401,F403
from .main_core_legacy import app

# The frozen core creates the historical DINOv3 Registry routes at import time.
# Remove them from the active app, then install the DINOv2-only replacement.
app.router.routes[:] = [
    route
    for route in app.router.routes
    if not str(getattr(route, "path", "")).startswith("/api/dinov3/registry")
]

from system.dinov2.api import dinov2_registry_router

app.include_router(dinov2_registry_router())
