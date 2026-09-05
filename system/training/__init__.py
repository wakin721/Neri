"""Optional model improvement services; no cloud credentials live in the client."""
from __future__ import annotations

import os
import threading
from pathlib import Path

_instance = None
_lock = threading.Lock()


def get_queue():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                from .queue import TrainingQueue
                custom = os.environ.get("NERI_TRAINING_STATE_DIR")
                if custom:
                    directory = Path(custom)
                elif os.name == "nt":
                    directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "Neri" / "training"
                else:
                    directory = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "neri" / "training"
                _instance = TrainingQueue(directory)
    return _instance
