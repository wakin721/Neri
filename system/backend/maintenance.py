"""Environment maintenance helpers for the local FastAPI backend."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = "721"
BACKEND_APP = "system.backend.main:app"
ACTIVE_STATES = {"starting", "waiting_for_backend", "running", "restarting"}
ULTRALYTICS_PACKAGE_SPEC = "ultralytics>=8.4.13"
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYTORCH_XPU_INDEX_URL = "https://download.pytorch.org/whl/xpu"
PIP_SOURCES = {
    "official": ("官方源", "https://pypi.org/simple"),
    "aliyun": ("阿里源", "https://mirrors.aliyun.com/pypi/simple"),
    "tsinghua": ("清华源", "https://pypi.tuna.tsinghua.edu.cn/simple"),
}

_PACKAGE_SPEC_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:\[[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*\])?"
    r"(?:(?:==|~=|!=|<=|>=|<|>)[A-Za-z0-9_.!*+\-]+)?$"
)


def project_root() -> Path:
    """Return the repository/application root."""

    return Path(__file__).resolve().parents[2]


def toolkit_python() -> Path:
    """Return the embedded Python executable required for maintenance."""

    return project_root() / "toolkit" / "python.exe"


def maintenance_status_path() -> Path:
    return project_root() / "temp" / "backend_maintenance_status.json"


def maintenance_log_path() -> Path:
    return project_root() / "temp" / "backend_maintenance.log"


def read_maintenance_status() -> dict[str, Any]:
    path = maintenance_status_path()
    if not path.exists():
        return {"state": "idle", "message": ""}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"state": "idle", "message": ""}


def start_pytorch_install(env_choice: str, package_source: str = "official") -> dict[str, Any]:
    env_choice = (env_choice or "自动检测").strip()
    source_key, _, _ = _resolve_package_source(package_source)
    message = "PyTorch 安装已开始，Python 后端会在安装完成后自动重启。"
    return _start_maintenance(
        operation="install_pytorch",
        extra_args=["--env-choice", env_choice, "--package-source", source_key],
        message=message,
    )


def start_package_reinstall(package_spec: str, package_source: str = "official") -> dict[str, Any]:
    package_spec = _validate_package_spec(package_spec)
    source_key, _, _ = _resolve_package_source(package_source)
    message = f"{package_spec} 重新安装已开始，Python 后端会在安装完成后自动重启。"
    return _start_maintenance(
        operation="reinstall_package",
        extra_args=["--package", package_spec, "--package-source", source_key],
        message=message,
    )


def start_yolo_dependencies_install(
    env_choice: str,
    package_source: str = "official",
) -> dict[str, Any]:
    env_choice = (env_choice or "自动检测").strip()
    source_key, _, _ = _resolve_package_source(package_source)
    message = "YOLO 依赖安装已开始，将先安装 PyTorch，再安装 ultralytics，完成后自动重启 Python 后端。"
    return _start_maintenance(
        operation="install_yolo_dependencies",
        extra_args=["--env-choice", env_choice, "--package-source", source_key],
        message=message,
    )


def schedule_backend_shutdown(delay: float = 1.0) -> None:
    """Exit the current backend after the HTTP response has had time to flush."""

    def _exit() -> None:
        os._exit(0)

    timer = threading.Timer(delay, _exit)
    timer.daemon = False
    timer.start()


def _start_maintenance(operation: str, extra_args: list[str], message: str) -> dict[str, Any]:
    python_exe = toolkit_python()
    if not python_exe.exists():
        raise FileNotFoundError(f"未找到嵌入式 Python: {python_exe}")

    current_status = read_maintenance_status()
    if current_status.get("state") in ACTIVE_STATES:
        raise RuntimeError("已有环境维护任务正在执行，请等待其完成。")

    status = {
        "operation": operation,
        "state": "starting",
        "message": message,
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "log_path": str(maintenance_log_path()),
    }
    _write_status(status, replace=True)

    command = [
        str(python_exe),
        str(project_root() / "system" / "backend" / "maintenance.py"),
        "--operation",
        operation,
        "--backend-pid",
        str(os.getpid()),
        *extra_args,
    ]
    subprocess.Popen(
        command,
        cwd=str(project_root()),
        creationflags=_creation_flags(),
        close_fds=True,
    )
    return status


def _validate_package_spec(package_spec: str) -> str:
    cleaned = package_spec.strip()
    if not cleaned:
        raise ValueError("请输入要重新安装的 Python 包名。")
    if not _PACKAGE_SPEC_PATTERN.fullmatch(cleaned):
        raise ValueError("只支持单个包规格，例如 numpy 或 numpy==1.26.4。")
    if cleaned.startswith("-"):
        raise ValueError("包名不能是 pip 参数。")
    return cleaned


def _resolve_package_source(package_source: str) -> tuple[str, str, str]:
    cleaned = (package_source or "official").strip().lower()
    alias_map = {
        "official": "official",
        "pypi": "official",
        "官方源": "official",
        "aliyun": "aliyun",
        "ali": "aliyun",
        "阿里源": "aliyun",
        "清华源": "tsinghua",
        "tsinghua": "tsinghua",
        "tuna": "tsinghua",
    }
    key = alias_map.get(cleaned, "official")
    label, url = PIP_SOURCES[key]
    return key, label, url


def _pip_source_args(package_source: str) -> list[str]:
    _, _, source_url = _resolve_package_source(package_source)
    return ["--index-url", source_url]


def _run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation",
        required=True,
        choices=["install_pytorch", "reinstall_package", "install_yolo_dependencies"],
    )
    parser.add_argument("--backend-pid", type=int, required=True)
    parser.add_argument("--env-choice", default="自动检测")
    parser.add_argument("--package", default="")
    parser.add_argument("--package-source", default="official")
    args = parser.parse_args(argv)

    try:
        _write_status(
            {
                "operation": args.operation,
                "state": "waiting_for_backend",
                "message": "正在等待旧 Python 后端退出...",
            }
        )
        _wait_for_process_exit(args.backend_pid)
        _wait_for_port_release()

        if args.operation == "install_pytorch":
            _run_pytorch_install(args.env_choice, args.package_source)
        elif args.operation == "install_yolo_dependencies":
            _run_yolo_dependencies_install(args.env_choice, args.package_source)
        else:
            _run_package_reinstall(args.package, args.package_source)

        _write_status(
            {
                "state": "restarting",
                "message": "安装完成，正在重启 Python 后端...",
                "return_code": 0,
            }
        )
        _start_backend()
        _write_status(
            {
                "state": "completed",
                "message": "安装完成，Python 后端已重启。",
                "return_code": 0,
                "error": None,
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - maintenance must recover the backend
        _write_status(
            {
                "state": "failed",
                "message": f"环境维护失败，正在尝试恢复 Python 后端: {exc}",
                "return_code": 1,
                "error": str(exc),
            }
        )
        try:
            _start_backend()
        except Exception as restart_exc:  # noqa: BLE001
            _write_status(
                {
                    "message": f"环境维护失败，且 Python 后端恢复失败: {restart_exc}",
                    "error": f"{exc}; restart failed: {restart_exc}",
                }
            )
        return 1


def _run_pytorch_install(env_choice: str, package_source: str = "official") -> None:
    python_exe = toolkit_python()
    actual_env, index_url = _resolve_pytorch_index(env_choice)
    _, source_label, source_url = _resolve_package_source(package_source)
    install_command = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        index_url,
    ]
    if source_url != PIP_SOURCES["official"][1]:
        install_command.extend(["--extra-index-url", source_url])
    commands = [
        [str(python_exe), "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
        install_command,
    ]
    _write_status(
        {
            "state": "running",
            "message": f"正在安装 PyTorch ({actual_env}，{source_label})...",
            "command": " && ".join(_format_command(command) for command in commands),
        }
    )
    _run_commands(commands)


def _run_package_reinstall(package_spec: str, package_source: str = "official") -> None:
    package_spec = _validate_package_spec(package_spec)
    python_exe = toolkit_python()
    _, source_label, _ = _resolve_package_source(package_source)
    commands = [
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            package_spec,
            *_pip_source_args(package_source),
        ],
    ]
    _write_status(
        {
            "state": "running",
            "message": f"正在从{source_label}重新安装 {package_spec}...",
            "command": _format_command(commands[0]),
        }
    )
    _run_commands(commands)


def _run_yolo_dependencies_install(env_choice: str, package_source: str = "official") -> None:
    _run_pytorch_install(env_choice, package_source)
    _run_ultralytics_install(package_source)


def _run_ultralytics_install(package_source: str = "official") -> None:
    python_exe = toolkit_python()
    _, source_label, _ = _resolve_package_source(package_source)
    command = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        "--upgrade",
        ULTRALYTICS_PACKAGE_SPEC,
        *_pip_source_args(package_source),
    ]
    _write_status(
        {
            "state": "running",
            "message": f"正在从{source_label}安装 ultralytics...",
            "command": _format_command(command),
        }
    )
    _run_commands([command])


def _run_commands(commands: list[list[str]]) -> None:
    log_path = maintenance_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[{_utc_now()}] maintenance start\n")
        for command in commands:
            log.write(f"\n> {_format_command(command)}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=str(project_root()),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"命令执行失败 ({completed.returncode}): {_format_command(command)}")
        log.write(f"\n[{_utc_now()}] maintenance completed\n")


def _resolve_pytorch_index(env_choice: str) -> tuple[str, str]:
    choice = (env_choice or "自动检测").strip()
    if choice == "自动检测":
        return _auto_detect_pytorch_index()
    if "XPU" in choice.upper():
        return "Intel XPU", PYTORCH_XPU_INDEX_URL
    if "CPU" in choice.upper():
        return "CPU Only", PYTORCH_CPU_INDEX_URL

    match = re.search(r"CUDA\s*(\d+)\.(\d+)", choice, flags=re.IGNORECASE)
    if match:
        major, minor = match.groups()
        return f"CUDA {major}.{minor}", f"https://download.pytorch.org/whl/cu{major}{minor}"

    return "CPU Only", PYTORCH_CPU_INDEX_URL


def _auto_detect_pytorch_index() -> tuple[str, str]:
    cuda_choice = _detect_nvidia_cuda_pytorch_index()
    if cuda_choice is not None:
        return cuda_choice

    intel_choice = _detect_intel_xpu_pytorch_index()
    if intel_choice is not None:
        return intel_choice

    return "CPU Only (未检测到支持的 GPU)", PYTORCH_CPU_INDEX_URL


def _detect_nvidia_cuda_pytorch_index() -> tuple[str, str] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            encoding="gbk",
            errors="ignore",
            timeout=10,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if not match:
        return "CUDA 12.4 (自动检测默认)", "https://download.pytorch.org/whl/cu124"

    major, minor = (int(value) for value in match.groups())
    if major >= 13:
        return f"CUDA 13.0 (检测到 CUDA {major}.{minor})", "https://download.pytorch.org/whl/cu130"
    if major >= 12 and minor >= 8:
        return f"CUDA 12.8 (检测到 CUDA {major}.{minor})", "https://download.pytorch.org/whl/cu128"
    if major >= 12 and minor >= 6:
        return f"CUDA 12.6 (检测到 CUDA {major}.{minor})", "https://download.pytorch.org/whl/cu126"
    if major >= 12 and minor >= 4:
        return f"CUDA 12.4 (检测到 CUDA {major}.{minor})", "https://download.pytorch.org/whl/cu124"
    if major >= 12 and minor >= 1:
        return f"CUDA 12.1 (检测到 CUDA {major}.{minor})", "https://download.pytorch.org/whl/cu121"
    if major >= 11 and minor >= 8:
        return f"CUDA 11.8 (检测到 CUDA {major}.{minor})", "https://download.pytorch.org/whl/cu118"
    return f"CPU Only (CUDA {major}.{minor} 过旧)", PYTORCH_CPU_INDEX_URL


def _detect_intel_xpu_pytorch_index() -> tuple[str, str] | None:
    gpu_names = _windows_video_controller_names()
    intel_gpu_names = [
        name for name in gpu_names if "intel" in _normalize_hardware_name(name)
    ]
    if not intel_gpu_names:
        return None

    cpu_names = _windows_processor_names()
    for gpu_name in intel_gpu_names:
        if _is_supported_intel_xpu_gpu(gpu_name, cpu_names):
            return f"Intel XPU (检测到 {gpu_name})", PYTORCH_XPU_INDEX_URL

    return f"CPU Only (Intel GPU 暂不支持 XPU: {intel_gpu_names[0]})", PYTORCH_CPU_INDEX_URL


def _is_supported_intel_xpu_gpu(gpu_name: str, cpu_names: Sequence[str]) -> bool:
    gpu_text = _normalize_hardware_name(gpu_name)
    if "intel" not in gpu_text or "arc" not in gpu_text:
        return False

    if re.search(r"\barc\b.*\b[ab]\s*series\b", gpu_text):
        return True
    if re.search(r"\barc\b.*\b[ab]\d{2,4}m?\b", gpu_text):
        return True
    if re.search(r"\barc\b.*\b1[34]0[vt]\b", gpu_text):
        return True

    return any(_is_supported_core_ultra_arc_cpu(name) for name in cpu_names)


def _is_supported_core_ultra_arc_cpu(cpu_name: str) -> bool:
    cpu_text = _normalize_hardware_name(cpu_name)
    if "core" not in cpu_text or "ultra" not in cpu_text:
        return False
    if any(
        codename in cpu_text
        for codename in ("meteor lake", "arrow lake", "lunar lake", "panther lake")
    ):
        return True

    for model, suffix in re.findall(r"\b([123]\d{2})([a-z]{1,2})\b", cpu_text):
        generation = int(model[0])
        suffix = suffix.lower()
        if generation == 1 and suffix.startswith("h"):
            return True
        if generation == 2 and (suffix.startswith("h") or suffix.startswith("v")):
            return True
        if generation == 3:
            return True
    return False


def _windows_video_controller_names() -> list[str]:
    names = _powershell_cim_property_values("Win32_VideoController", "Name")
    if names:
        return names
    return _wmic_property_values(["wmic", "path", "win32_VideoController", "get", "Name"])


def _windows_processor_names() -> list[str]:
    names = _powershell_cim_property_values("Win32_Processor", "Name")
    if names:
        return names
    return _wmic_property_values(["wmic", "cpu", "get", "Name"])


def _powershell_cim_property_values(cim_class: str, property_name: str) -> list[str]:
    if os.name != "nt":
        return []
    command = f"Get-CimInstance {cim_class} | Select-Object -ExpandProperty {property_name}"
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return _clean_command_lines(result.stdout)


def _wmic_property_values(command: Sequence[str]) -> list[str]:
    if os.name != "nt":
        return []
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return [line for line in _clean_command_lines(result.stdout) if line.lower() != "name"]


def _clean_command_lines(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.strip()]


def _normalize_hardware_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[\(\)\[\],._+\-/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _start_backend() -> None:
    python_exe = toolkit_python()
    command = [
        str(python_exe),
        "-m",
        "uvicorn",
        BACKEND_APP,
        "--app-dir",
        ".",
        "--host",
        BACKEND_HOST,
        "--port",
        BACKEND_PORT,
    ]
    subprocess.Popen(
        command,
        cwd=str(project_root()),
        creationflags=_creation_flags(),
        close_fds=True,
    )


def _wait_for_process_exit(pid: int, timeout_seconds: float = 30) -> None:
    if os.name == "nt":
        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if handle:
            try:
                ctypes.windll.kernel32.WaitForSingleObject(handle, int(timeout_seconds * 1000))
                return
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.25)


def _wait_for_port_release(timeout_seconds: float = 30) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((BACKEND_HOST, int(BACKEND_PORT))) != 0:
                return
        time.sleep(0.25)


def _write_status(updates: dict[str, Any], *, replace: bool = False) -> None:
    path = maintenance_status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        status: dict[str, Any] = {}
    else:
        status = read_maintenance_status()
    status.update(updates)
    status["updated_at"] = _utc_now()
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(_run_cli())
