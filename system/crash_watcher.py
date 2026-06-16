"""Out-of-process crash watcher for the Flutter frontend.

The Flutter process starts this helper and passes its PID. The helper waits
outside the main process, so it can still show a system dialog after the main
window has crashed or exited with a non-zero code.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0x00000000
SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259

MB_OK = 0x00000000
MB_ICONERROR = 0x00000010
MB_SETFOREGROUND = 0x00010000
MB_TOPMOST = 0x00040000


def main() -> int:
    args = _parse_args()
    _write_diagnostic(args.diagnostic_log, f"Python watcher started. MainPid={args.main_pid}")

    exit_code = _wait_for_process(args.main_pid, args.diagnostic_log)
    session_data = _session_data(args.session)
    if _session_status(session_data) == "normal":
        _write_diagnostic(args.diagnostic_log, "Session marked normal. Exiting without popup.")
        return 0
    if exit_code == 0:
        _write_diagnostic(args.diagnostic_log, "Exit code is 0. Exiting without popup.")
        return 0

    log_start_offset = _crash_log_start_offset(session_data)
    reason = _latest_reason(args.frontend_log, log_start_offset) or _latest_reason(
        args.backend_log,
        log_start_offset,
    )
    if not reason:
        exit_code_text = "unknown" if exit_code is None else str(exit_code)
        reason = f"主进程异常退出，退出码：{exit_code_text}。"

        _append_fallback_crash_log(args.frontend_log, reason, exit_code)

    _write_startup_report(
        args.startup_report,
        reason,
        args.frontend_log,
        args.backend_log,
        args.diagnostic_log,
    )
    restart_message, restart_started = _restart_main_process(
        args.restart_executable,
        args.restart_arg,
        args.restart_cwd,
        args.diagnostic_log,
    )
    if restart_started:
        _write_diagnostic(
            args.diagnostic_log,
            "Restart started; startup crash report will be shown by the main process.",
        )
        return 0

    message = "\n".join(
        [
            "Neri 已异常退出。",
            restart_message,
            "",
            f"原因：{reason}",
            "",
            f"崩溃日志：{args.frontend_log}",
        ]
    )
    _show_message_box(args.title + " 崩溃提示", message, args.diagnostic_log)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a process and show a crash dialog.")
    parser.add_argument("--main-pid", required=True, type=int)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--frontend-log", required=True, type=Path)
    parser.add_argument("--backend-log", required=True, type=Path)
    parser.add_argument("--diagnostic-log", required=True, type=Path)
    parser.add_argument("--restart-executable", type=Path)
    parser.add_argument("--restart-cwd", type=Path)
    parser.add_argument("--restart-arg", action="append", default=[])
    parser.add_argument("--startup-report", type=Path)
    parser.add_argument("--title", default="Neri")
    return parser.parse_args()


def _wait_for_process(pid: int, diagnostic_log: Path) -> int | None:
    if sys.platform != "win32":
        _write_diagnostic(diagnostic_log, "Non-Windows platform; no native process wait available.")
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int

    handle = kernel32.OpenProcess(
        SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
        False,
        pid,
    )
    if not handle:
        error = ctypes.get_last_error()
        _write_diagnostic(diagnostic_log, f"OpenProcess failed for pid={pid}. error={error}")
        return None

    try:
        _write_diagnostic(diagnostic_log, "Attached to main process.")
        wait_result = kernel32.WaitForSingleObject(handle, INFINITE)
        _write_diagnostic(diagnostic_log, f"WaitForSingleObject returned {wait_result}.")
        if wait_result != WAIT_OBJECT_0:
            return None

        exit_code = ctypes.c_uint32(STILL_ACTIVE)
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            error = ctypes.get_last_error()
            _write_diagnostic(diagnostic_log, f"GetExitCodeProcess failed. error={error}")
            return None

        value = int(exit_code.value)
        _write_diagnostic(diagnostic_log, f"Main process exited. ExitCode={value}")
        return value
    finally:
        kernel32.CloseHandle(handle)


def _session_data(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _session_status(data: dict) -> str:
    return str(data.get("status", ""))


def _crash_log_start_offset(data: dict) -> int:
    try:
        return max(0, int(data.get("crash_log_start_offset", 0)))
    except Exception:
        return 0


def _latest_reason(path: Path, start_offset: int = 0) -> str | None:
    text = _tail_text(path, start_offset=start_offset)
    if not text.strip():
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("Message:"):
            return line[len("Message:") :].strip()
    for line in reversed(lines):
        if _looks_like_reason(line):
            return line
    return None


def _tail_text(path: Path, max_bytes: int = 80_000, start_offset: int = 0) -> str:
    try:
        with path.open("rb") as file:
            file.seek(0, 2)
            size = file.tell()
            if start_offset >= size:
                return ""
            file.seek(max(start_offset, size - max_bytes))
            return file.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _looks_like_reason(line: str) -> bool:
    lower = line.lower()
    return any(
        marker in lower
        for marker in (
            "error",
            "exception",
            "traceback",
            "failed",
            "crash",
            "错误",
            "异常",
            "失败",
            "崩溃",
        )
    )


def _append_fallback_crash_log(path: Path, reason: str, exit_code: int | None) -> None:
    exit_code_text = "unknown" if exit_code is None else str(exit_code)
    exception_code = _windows_exception_code(exit_code)
    details = [
        "Origin: Windows process watchdog",
        f"Message: {reason}",
        f"Exit code: {exit_code_text}",
    ]
    if exception_code:
        details.append(f"Windows exception code: {exception_code}")
    details.extend(
        [
            "",
            (
                "No Dart/Flutter crash details were written before the process "
                "terminated, so the watchdog generated this fallback crash record."
            ),
        ]
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().isoformat()
        with path.open("a", encoding="utf-8") as file:
            file.write("============================================================\n")
            file.write(f"[Frontend process crash] {timestamp}\n")
            file.write("\n".join(details).rstrip())
            file.write("\n\n")
    except Exception:
        pass


def _windows_exception_code(exit_code: int | None) -> str | None:
    if exit_code is None:
        return None
    try:
        return f"0x{exit_code & 0xFFFFFFFF:08X}"
    except Exception:
        return None


def _write_startup_report(
    path: Path | None,
    reason: str,
    frontend_log: Path,
    backend_log: Path,
    diagnostic_log: Path,
) -> None:
    if path is None:
        _write_diagnostic(diagnostic_log, "No startup crash report path was supplied.")
        return

    payload = {
        "title": "程序崩溃提示",
        "message": "Neri 上次运行异常退出，崩溃日志已保存。",
        "details": f"原因：{reason}",
        "logPath": str(frontend_log),
        "createdAt": datetime.now().astimezone().isoformat(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        _write_diagnostic(diagnostic_log, f"Wrote startup crash report: {path}")
    except Exception as error:
        _write_diagnostic(diagnostic_log, f"Failed to write startup crash report: {error!r}")


def _restart_main_process(
    executable: Path | None,
    restart_args: list[str],
    cwd: Path | None,
    diagnostic_log: Path,
) -> tuple[str, bool]:
    if executable is None:
        _write_diagnostic(diagnostic_log, "No restart executable was supplied.")
        return "未配置自动重启路径。", False
    if not executable.exists():
        _write_diagnostic(diagnostic_log, f"Restart executable not found: {executable}")
        return "未能自动重启程序。", False

    working_directory = cwd if cwd is not None else executable.parent
    try:
        flags = 0
        if sys.platform == "win32":
            flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            [str(executable), *restart_args],
            cwd=str(working_directory),
            close_fds=True,
            creationflags=flags,
        )
        _write_diagnostic(
            diagnostic_log,
            f"Restarted main process. pid={process.pid}; executable={executable}",
        )
        return "程序正在自动重启。", True
    except Exception as error:
        _write_diagnostic(diagnostic_log, f"Failed to restart main process: {error!r}")
        return "未能自动重启程序。", False


def _show_message_box(title: str, message: str, diagnostic_log: Path) -> None:
    if sys.platform != "win32":
        _write_diagnostic(diagnostic_log, "MessageBoxW is only available on Windows.")
        print(f"{title}\n{message}", file=sys.stderr)
        return

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.MessageBoxW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        ]
        user32.MessageBoxW.restype = ctypes.c_int
        _write_diagnostic(diagnostic_log, "Attempting crash popup with MessageBoxW.")
        result = user32.MessageBoxW(
            None,
            message,
            title,
            MB_OK | MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST,
        )
        _write_diagnostic(diagnostic_log, f"MessageBoxW returned {result}.")
    except Exception as error:
        _write_diagnostic(diagnostic_log, f"MessageBoxW failed: {error!r}")


def _write_diagnostic(path: Path, message: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().isoformat()
        with path.open("a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
