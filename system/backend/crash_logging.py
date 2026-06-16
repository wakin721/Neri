"""Crash logging helpers for the local backend process."""

from __future__ import annotations

from datetime import datetime, timezone
import faulthandler
import logging
from pathlib import Path
import sys
import threading
import traceback
from types import TracebackType


def configure_backend_crash_logging() -> Path:
    """Configure best-effort logs for uncaught backend failures."""

    log_path = _backend_runtime_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _install_exception_hooks(log_path)
        _enable_fault_handler()
        logging.getLogger(__name__).info("Backend crash logging enabled: %s", log_path)
    except Exception:  # noqa: BLE001 - diagnostics must never prevent startup
        logging.getLogger(__name__).exception("Failed to configure backend crash logging")
    return log_path


def _backend_runtime_log_path() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "crash.log"


def _install_exception_hooks(log_path: Path) -> None:
    def handle_exception(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        _write_exception(log_path, "Uncaught exception", exc_type, exc_value, exc_traceback)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        thread_name = args.thread.name if args.thread else "unknown"
        _write_exception(
            log_path,
            f"Uncaught thread exception in {thread_name}",
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
        )
        threading.__excepthook__(args)

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception


def _enable_fault_handler() -> None:
    try:
        faulthandler.enable(all_threads=True)
    except Exception:  # noqa: BLE001 - crash logging must never break startup
        logging.getLogger(__name__).exception("Failed to enable faulthandler")


def _write_exception(
    log_path: Path,
    title: str,
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n============================================================\n")
            log_file.write(f"[{datetime.now(timezone.utc).isoformat()}] {title}\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=log_file)
    except Exception:
        pass
