"""FastAPI main app after the DINOv2 clean break."""
from .main_core_legacy import *  # noqa: F401,F403
from .main_core_legacy import app  # noqa: F401
