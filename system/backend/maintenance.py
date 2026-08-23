"""Environment maintenance helpers for the local FastAPI backend."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from system.config import XPU_ENABLED

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = "721"
BACKEND_APP = "system.backend.main:app"
ACTIVE_STATES = {"starting", "waiting_for_backend", "running", "restarting"}
ULTRALYTICS_PACKAGE_SPEC = "ultralytics>=8.4.13"
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYTORCH_XPU_INDEX_URL = "https://download.pytorch.org/whl/xpu"
PYTORCH_CUDA_132_INDEX_URL = "https://download.pytorch.org/whl/cu132"
PYTORCH_CUDA_130_INDEX_URL = "https://download.pytorch.org/whl/cu130"
INTEL_GRAPHICS_DRIVER_PAGE_URL = (
    "https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html"
)
INTEL_GRAPHICS_DRIVER_DOWNLOAD_URL = (
    "https://downloadmirror.intel.com/919751/gfx_win_101.8801.exe"
)
INTEL_GRAPHICS_DRIVER_SHA256 = (
    "118F1C82871210A52FA2C2488A29E67423A1ED4E263DFCF94C3F64D05B3E0926"
)
INTEL_GRAPHICS_DRIVER_FILENAME = "gfx_win_101.8801.exe"
PIP_SOURCES = {
    "official": ("官方源", "https://pypi.org/simple"),
    "aliyun": ("阿里源", "https://mirrors.aliyun.com/pypi/simple"),
    "tsinghua": ("清华源", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    "nju": ("南京大学源", "https://mirror.nju.edu.cn/pypi/web/simple"),
}
PUBLIC_IP_COUNTRY_URL = "https://api.country.is/"
PUBLIC_IP_COUNTRY_TIMEOUT_SECONDS = 1.5
PACKAGE_SOURCE_CACHE_SECONDS = 30 * 60
HARDWARE_PROBE_TIMEOUT_SECONDS = 8
NVIDIA_SMI_TIMEOUT_SECONDS = 3
HARDWARE_STATUS_CACHE_SECONDS = 60
PIP_NETWORK_TIMEOUT_SECONDS = 60
PIP_NETWORK_RETRIES = 8
WINDOWS_LEGACY_DIRECTORY_PATH_LIMIT = 247

_PYTORCH_LONG_PATH_PROBE = Path(
    "toolkit",
    "Lib",
    "site-packages",
    "torch-2.13.0+cpu.dist-info",
    "licenses",
    "third_party",
    "kineto",
    "libkineto",
    "third_party",
    "dynolog",
    "third_party",
    "prometheus-cpp",
    "3rdparty",
    "civetweb",
    "src",
    "third_party",
    "duktape-1.5.2",
)
_PYTORCH_PACKAGE_ARTIFACTS = {
    "torch": ("torch", "functorch", "torchgen"),
    "torchvision": ("torchvision", "torchvision.libs"),
    "torchaudio": ("torchaudio", "torchaudio.lib", "torchaudio.libs"),
}

_package_source_cache: tuple[float, str] | None = None
_package_source_cache_lock = threading.Lock()
_intel_driver_status_cache: tuple[float, dict[str, Any]] | None = None
_intel_driver_status_cache_lock = threading.Lock()

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


def logs_dir() -> Path:
    """Return the project software log directory."""

    return project_root() / "logs"


def maintenance_status_path() -> Path:
    return logs_dir() / "backend_maintenance_status.json"


def maintenance_log_path() -> Path:
    return logs_dir() / "backend_maintenance.log"


def maintenance_pip_cache_dir() -> Path:
    """Return the app-owned pip cache used only during environment maintenance."""

    return project_root() / "temp" / "maintenance_pip_cache"


def toolkit_site_packages_dir() -> Path:
    """Return the embedded Python environment's site-packages directory."""

    return toolkit_python().parent / "Lib" / "site-packages"


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


def intel_graphics_driver_status() -> dict[str, Any]:
    """Return best-effort Intel display driver status for XPU installs."""

    global _intel_driver_status_cache

    now = time.monotonic()
    with _intel_driver_status_cache_lock:
        if (
            _intel_driver_status_cache is not None
            and now - _intel_driver_status_cache[0] < HARDWARE_STATUS_CACHE_SECONDS
        ):
            return dict(_intel_driver_status_cache[1])

        status = _probe_intel_graphics_driver_status()
        _intel_driver_status_cache = (time.monotonic(), dict(status))
        return status


def _reset_intel_driver_status_cache() -> None:
    global _intel_driver_status_cache
    with _intel_driver_status_cache_lock:
        _intel_driver_status_cache = None


def _probe_intel_graphics_driver_status() -> dict[str, Any]:
    status = _empty_intel_driver_status()
    if os.name != "nt":
        status.update(
            {
                "checked": True,
                "message": "Intel XPU 驱动自动检测仅支持 Windows。",
            }
        )
        return status

    try:
        payload = _query_windows_display_devices()
    except Exception as exc:  # noqa: BLE001 - best-effort hardware probing
        status.update(
            {
                "checked": False,
                "error": str(exc),
                "message": f"检测 Intel 显卡驱动失败: {exc}",
            }
        )
        return status

    devices = _as_list(payload.get("devices"))
    drivers = _as_list(payload.get("drivers"))
    raw_processor_names = payload.get("processors")
    if not isinstance(raw_processor_names, list):
        raw_processor_names = [raw_processor_names] if raw_processor_names else []
    processor_names = [
        str(value).strip()
        for value in raw_processor_names
        if str(value).strip()
    ]
    intel_devices = [device for device in devices if _is_intel_display_device(device)]
    intel_drivers = [driver for driver in drivers if _is_intel_display_driver(driver)]
    installed_driver = next(
        (driver for driver in intel_drivers if _is_official_intel_display_driver(driver)),
        None,
    )
    display_item = installed_driver or (intel_drivers[0] if intel_drivers else None)
    status.update(
        {
            "checked": True,
            "gpu_present": bool(intel_devices or intel_drivers),
            "driver_installed": installed_driver is not None,
            "driver_name": _first_text(display_item, "DeviceName", "Name"),
            "driver_version": _first_text(display_item, "DriverVersion"),
            "driver_provider": _first_text(display_item, "DriverProviderName", "Manufacturer"),
            "device_name": _first_text(
                intel_devices[0] if intel_devices else display_item,
                "Name",
                "DeviceName",
            ),
            "processor_names": processor_names,
            "message": (
                "已检测到 Intel 显卡驱动。"
                if installed_driver is not None
                else "未检测到可用的 Intel 官方显卡驱动。"
            ),
        }
    )
    return status


def resolve_pytorch_install_plan(env_choice: str) -> dict[str, Any]:
    """Resolve a PyTorch target and any Intel driver preflight requirement."""

    requested = (env_choice or "自动检测").strip()
    actual_env, index_url = _resolve_pytorch_index(requested)
    is_xpu = index_url.rstrip("/").endswith("/xpu")
    driver_status = intel_graphics_driver_status() if is_xpu else _empty_intel_driver_status()
    return {
        "env_choice": requested,
        "actual_env": actual_env,
        "index_url": index_url,
        "is_xpu": is_xpu,
        "intel_driver": driver_status,
        "needs_intel_driver": bool(
            is_xpu
            and driver_status.get("gpu_present")
            and not driver_status.get("driver_installed")
        ),
        "intel_driver_page_url": INTEL_GRAPHICS_DRIVER_PAGE_URL,
        "intel_driver_download_url": INTEL_GRAPHICS_DRIVER_DOWNLOAD_URL,
    }


def download_intel_graphics_driver() -> Path:
    """Download Intel's official graphics driver installer and verify it."""

    download_url = INTEL_GRAPHICS_DRIVER_DOWNLOAD_URL
    _validate_intel_download_url(download_url)
    target_dir = project_root() / "temp" / "intel_driver"
    target_dir.mkdir(parents=True, exist_ok=True)
    installer_path = target_dir / INTEL_GRAPHICS_DRIVER_FILENAME

    expected_sha256 = INTEL_GRAPHICS_DRIVER_SHA256.lower()
    if installer_path.exists() and _sha256_file(installer_path) == expected_sha256:
        return installer_path
    if installer_path.exists():
        installer_path.unlink()

    temp_path = installer_path.with_suffix(installer_path.suffix + ".download")
    if temp_path.exists():
        temp_path.unlink()

    request = urllib.request.Request(
        download_url,
        headers={"User-Agent": "Neri Intel XPU dependency installer"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310 - URL is validated above.
        _validate_intel_download_url(response.geturl())
        with temp_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

    actual_sha256 = _sha256_file(temp_path)
    if actual_sha256 != expected_sha256:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Intel 显卡驱动下载校验失败，已取消运行安装程序。"
        )

    temp_path.replace(installer_path)
    return installer_path


def install_intel_graphics_driver() -> Path:
    """Download and run Intel's graphics driver installer."""

    installer_path = download_intel_graphics_driver()
    completed = subprocess.run(
        [str(installer_path)],
        cwd=str(installer_path.parent),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Intel 显卡驱动安装程序退出码为 {completed.returncode}。")
    return installer_path


def start_pytorch_install(
    env_choice: str,
    package_source: str = "auto",
    install_intel_driver: bool = False,
) -> dict[str, Any]:
    env_choice = (env_choice or "自动检测").strip()
    source_key, _, _ = resolve_package_source(package_source)
    message = (
        "Intel 显卡驱动与 PyTorch 安装已开始，Python 后端会在安装完成后自动重启。"
        if install_intel_driver
        else "PyTorch 安装已开始，Python 后端会在安装完成后自动重启。"
    )
    extra_args = ["--env-choice", env_choice, "--package-source", source_key]
    if install_intel_driver:
        extra_args.append("--install-intel-driver")
    return _start_maintenance(
        operation="install_pytorch",
        extra_args=extra_args,
        message=message,
    )


def start_package_reinstall(package_spec: str, package_source: str = "auto") -> dict[str, Any]:
    package_spec = _validate_package_spec(package_spec)
    source_key, _, _ = resolve_package_source(package_source)
    message = f"{package_spec} 重新安装已开始，Python 后端会在安装完成后自动重启。"
    return _start_maintenance(
        operation="reinstall_package",
        extra_args=["--package", package_spec, "--package-source", source_key],
        message=message,
    )


def start_yolo_dependencies_install(
    env_choice: str,
    package_source: str = "auto",
    install_intel_driver: bool = False,
) -> dict[str, Any]:
    env_choice = (env_choice or "自动检测").strip()
    source_key, _, _ = resolve_package_source(package_source)
    message = (
        "YOLO 依赖安装已开始，将先安装 Intel 显卡驱动和 PyTorch，再安装 ultralytics，完成后自动重启 Python 后端。"
        if install_intel_driver
        else "YOLO 依赖安装已开始，将先安装 PyTorch，再安装 ultralytics，完成后自动重启 Python 后端。"
    )
    extra_args = ["--env-choice", env_choice, "--package-source", source_key]
    if install_intel_driver:
        extra_args.append("--install-intel-driver")
    return _start_maintenance(
        operation="install_yolo_dependencies",
        extra_args=extra_args,
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
        "progress": 0,
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "log_path": str(maintenance_log_path()),
        "status_path": str(maintenance_status_path()),
    }
    _write_status(status, replace=True)

    command = [
        str(python_exe),
        "-m",
        "system.backend.maintenance",
        "--operation",
        operation,
        "--backend-pid",
        str(os.getpid()),
        *extra_args,
    ]
    process = subprocess.Popen(
        command,
        cwd=str(project_root()),
        creationflags=_creation_flags(),
        close_fds=True,
    )
    status["maintenance_pid"] = process.pid
    _write_status({"maintenance_pid": process.pid})
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


def _public_ip_country_code() -> str | None:
    """Return the public-IP country code, or None when detection is unavailable."""

    request = urllib.request.Request(
        PUBLIC_IP_COUNTRY_URL,
        headers={"User-Agent": "Neri package source selector"},
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed HTTPS endpoint.
            request,
            timeout=PUBLIC_IP_COUNTRY_TIMEOUT_SECONDS,
        ) as response:
            payload = json.loads(response.read(4096).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None
    country = payload.get("country")
    if not isinstance(country, str):
        return None
    normalized = country.strip().upper()
    return normalized if len(normalized) == 2 else None


def _automatic_package_source() -> str:
    """Use Aliyun in mainland China and official PyPI everywhere else."""

    return "aliyun" if is_mainland_china() else "official"


def is_mainland_china() -> bool:
    """Return whether the shared automatic source detector selects mainland China."""

    global _package_source_cache

    now = time.monotonic()
    with _package_source_cache_lock:
        if (
            _package_source_cache is not None
            and now - _package_source_cache[0] < PACKAGE_SOURCE_CACHE_SECONDS
        ):
            return _package_source_cache[1] == "aliyun"

        # A mainland-China timezone is a reliable, zero-network-cost hint for
        # the common case. The public-IP lookup remains the authority elsewhere.
        mainland_china = (
            _local_timezone_indicates_mainland_china()
            or _public_ip_country_code() == "CN"
        )
        source = "aliyun" if mainland_china else "official"
        _package_source_cache = (now, source)
        return mainland_china


def _local_timezone_indicates_mainland_china() -> bool:
    timezone_names = {
        str(value).strip().lower().replace("\\", "/")
        for value in (*time.tzname, os.environ.get("TZ", ""))
        if value
    }
    return bool(
        timezone_names
        & {
            "asia/shanghai",
            "china standard time",
            "中国标准时间",
            "prc",
        }
    )


def _reset_package_source_cache() -> None:
    """Clear automatic source state for tests and explicit runtime refreshes."""

    global _package_source_cache
    with _package_source_cache_lock:
        _package_source_cache = None


def resolve_package_source(package_source: str = "auto") -> tuple[str, str, str]:
    """Resolve an automatic or explicitly selected Python package source."""

    cleaned = (package_source or "auto").strip().lower()
    alias_map = {
        "auto": "auto",
        "automatic": "auto",
        "自动": "auto",
        "自动检测": "auto",
        "official": "official",
        "pypi": "official",
        "官方源": "official",
        "aliyun": "aliyun",
        "ali": "aliyun",
        "阿里源": "aliyun",
        "清华源": "tsinghua",
        "tsinghua": "tsinghua",
        "tuna": "tsinghua",
        "nju": "nju",
        "nanjing": "nju",
        "南京大学源": "nju",
        "南大源": "nju",
    }
    key = alias_map.get(cleaned, "official")
    if key == "auto":
        key = _automatic_package_source()
    label, url = PIP_SOURCES[key]
    return key, label, url


def _pip_source_args(package_source: str) -> list[str]:
    _, _, source_url = resolve_package_source(package_source)
    return ["-i", source_url]


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
    parser.add_argument("--package-source", default="auto")
    parser.add_argument("--install-intel-driver", action="store_true")
    args = parser.parse_args(argv)

    try:
        _write_status(
            {
                "operation": args.operation,
                "state": "waiting_for_backend",
                "message": "正在等待旧 Python 后端退出...",
                "progress": 5,
            }
        )
        _wait_for_process_exit(args.backend_pid)
        _wait_for_port_release()

        if args.operation == "install_pytorch":
            _run_pytorch_install(
                args.env_choice,
                args.package_source,
                install_intel_driver=args.install_intel_driver,
            )
        elif args.operation == "install_yolo_dependencies":
            _run_yolo_dependencies_install(
                args.env_choice,
                args.package_source,
                install_intel_driver=args.install_intel_driver,
            )
        else:
            _run_package_reinstall(args.package, args.package_source)

        _write_status(
            {
                "state": "running",
                "message": "安装完成，正在清理依赖下载缓存...",
                "progress": 92,
            }
        )
        cache_cleanup_error = _clear_maintenance_pip_cache()
        _write_status(
            {
                "state": "restarting",
                "message": "安装完成，正在重启 Python 后端...",
                "progress": 95,
                "return_code": 0,
            }
        )
        _start_backend()
        _write_status(
            {
                "state": "completed",
                "message": (
                    "安装完成，下载缓存已清理，Python 后端已重启。"
                    if cache_cleanup_error is None
                    else "安装完成，Python 后端已重启，但下载缓存清理失败。"
                ),
                "progress": 100,
                "return_code": 0,
                "error": None,
                "cache_cleanup_error": cache_cleanup_error,
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


def _run_pytorch_install(
    env_choice: str,
    package_source: str = "auto",
    *,
    install_intel_driver: bool = False,
    progress_start: int = 10,
    progress_end: int = 90,
) -> None:
    python_exe = toolkit_python()
    _ensure_pytorch_install_path_supported()
    _repair_broken_pytorch_installations()
    plan = resolve_pytorch_install_plan(env_choice)
    actual_env = str(plan["actual_env"])
    index_url = str(plan["index_url"])
    _, source_label, source_url = resolve_package_source(package_source)
    command_progress_start = progress_start
    if plan.get("needs_intel_driver"):
        if not install_intel_driver:
            raise RuntimeError(
                "安装 Intel XPU 版本 PyTorch 前需要先安装 Intel 显卡驱动。"
            )
        driver_progress_end = progress_start + max(
            1,
            round((progress_end - progress_start) * 0.3),
        )
        _install_intel_graphics_driver_for_xpu(
            progress_start=progress_start,
            progress_end=driver_progress_end,
        )
        command_progress_start = driver_progress_end

    install_command = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        "--progress-bar",
        "raw",
        "--timeout",
        str(PIP_NETWORK_TIMEOUT_SECONDS),
        "--retries",
        str(PIP_NETWORK_RETRIES),
        "--cache-dir",
        str(maintenance_pip_cache_dir()),
        "torch",
        "torchvision",
        "--index-url",
        index_url,
        "--extra-index-url",
        source_url,
    ]
    commands = [
        [str(python_exe), "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
        install_command,
    ]
    _write_status(
        {
            "state": "running",
            "message": f"正在安装 PyTorch ({actual_env}，{source_label})...",
            "progress": command_progress_start,
            "command": " && ".join(_format_command(command) for command in commands),
        }
    )
    _run_commands(
        commands,
        progress_start=command_progress_start,
        progress_end=progress_end,
        progress_weights=[0.05, 0.95],
        progress_message=f"正在安装 PyTorch ({actual_env}，{source_label})...",
        command_messages=[
            "正在卸载旧版 PyTorch、torchvision 和 torchaudio...",
            f"正在安装 PyTorch ({actual_env}，{source_label})...",
        ],
    )


def _ensure_pytorch_install_path_supported() -> None:
    """Reject legacy Windows paths that cannot hold PyTorch's package metadata."""

    if os.name != "nt" or _windows_long_paths_enabled():
        return
    projected_path = project_root() / _PYTORCH_LONG_PATH_PROBE
    if len(str(projected_path)) <= WINDOWS_LEGACY_DIRECTORY_PATH_LIMIT:
        return
    raise RuntimeError(
        "当前 Neri 安装路径过长，Windows 无法写入 PyTorch 文件。"
        "请将整个 Neri 文件夹移动或重新解压到 C:\\Neri、D:\\Neri 等短路径后重试；"
        "移动后无需重新下载安装包。"
    )


def _windows_long_paths_enabled() -> bool:
    """Return whether Windows has opted in to Win32 long-path support."""

    if os.name != "nt":
        return True
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        return int(value) == 1
    except (OSError, TypeError, ValueError):
        return False


def _repair_broken_pytorch_installations() -> list[Path]:
    """Remove only PyTorch distributions whose pip RECORD file is missing."""

    site_packages = toolkit_site_packages_dir()
    if not site_packages.is_dir():
        return []
    resolved_site_packages = site_packages.resolve()
    resolved_toolkit = toolkit_python().parent.resolve()
    try:
        resolved_site_packages.relative_to(resolved_toolkit)
    except ValueError as exc:
        raise RuntimeError(
            f"拒绝清理 Neri 内嵌 Python 之外的依赖目录: {site_packages}"
        ) from exc

    removed: list[Path] = []
    for package_name, artifact_names in _PYTORCH_PACKAGE_ARTIFACTS.items():
        metadata_dirs = [
            path
            for path in site_packages.glob(f"{package_name}-*.dist-info")
            if path.is_dir()
        ]
        artifacts = [
            site_packages / artifact_name
            for artifact_name in artifact_names
            if (site_packages / artifact_name).exists()
        ]
        artifacts.extend(metadata_dirs)
        if not artifacts:
            continue
        if metadata_dirs and all(
            (metadata_dir / "RECORD").is_file() for metadata_dir in metadata_dirs
        ):
            continue

        for artifact in dict.fromkeys(artifacts):
            if artifact.is_symlink() or artifact.is_file():
                artifact.unlink()
            elif artifact.exists():
                shutil.rmtree(artifact)
            removed.append(artifact)

    if removed:
        log_path = maintenance_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            names = ", ".join(path.name for path in removed)
            log.write(
                f"[{_utc_now()}] removed incomplete PyTorch artifacts: {names}\n"
            )
    return removed


def _run_package_reinstall(package_spec: str, package_source: str = "auto") -> None:
    package_spec = _validate_package_spec(package_spec)
    python_exe = toolkit_python()
    _, source_label, _ = resolve_package_source(package_source)
    commands = [
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--progress-bar",
            "raw",
            "--timeout",
            str(PIP_NETWORK_TIMEOUT_SECONDS),
            "--retries",
            str(PIP_NETWORK_RETRIES),
            "--cache-dir",
            str(maintenance_pip_cache_dir()),
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
            "progress": 10,
            "command": _format_command(commands[0]),
        }
    )
    _run_commands(
        commands,
        progress_start=10,
        progress_end=90,
        progress_message=f"正在从{source_label}重新安装 {package_spec}...",
    )


def _run_yolo_dependencies_install(
    env_choice: str,
    package_source: str = "auto",
    *,
    install_intel_driver: bool = False,
) -> None:
    _run_pytorch_install(
        env_choice,
        package_source,
        install_intel_driver=install_intel_driver,
        progress_start=10,
        progress_end=70,
    )
    _run_ultralytics_install(package_source, progress_start=70, progress_end=90)


def _run_ultralytics_install(
    package_source: str = "auto",
    *,
    progress_start: int = 70,
    progress_end: int = 90,
) -> None:
    python_exe = toolkit_python()
    _, source_label, _ = resolve_package_source(package_source)
    command = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        "--progress-bar",
        "raw",
        "--timeout",
        str(PIP_NETWORK_TIMEOUT_SECONDS),
        "--retries",
        str(PIP_NETWORK_RETRIES),
        "--cache-dir",
        str(maintenance_pip_cache_dir()),
        "--upgrade",
        ULTRALYTICS_PACKAGE_SPEC,
        *_pip_source_args(package_source),
    ]
    _write_status(
        {
            "state": "running",
            "message": f"正在从{source_label}安装 ultralytics...",
            "progress": progress_start,
            "command": _format_command(command),
        }
    )
    _run_commands(
        [command],
        progress_start=progress_start,
        progress_end=progress_end,
        progress_message=f"正在从{source_label}安装 ultralytics...",
    )


def _run_commands(
    commands: list[list[str]],
    *,
    progress_start: int,
    progress_end: int,
    progress_weights: Sequence[float] | None = None,
    progress_message: str = "正在安装依赖...",
    command_messages: Sequence[str] | None = None,
) -> None:
    if not commands:
        return
    weights = list(progress_weights or [1.0] * len(commands))
    if len(weights) != len(commands) or any(weight < 0 for weight in weights):
        raise ValueError("维护命令的进度权重无效。")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("维护命令的进度权重总和必须大于 0。")
    messages = list(command_messages or [progress_message] * len(commands))
    if len(messages) != len(commands):
        raise ValueError("维护命令与阶段提示的数量不一致。")

    log_path = maintenance_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[{_utc_now()}] maintenance start\n")
        completed_weight = 0.0
        for index, command in enumerate(commands):
            command_message = messages[index]
            command_progress_start = progress_start + round(
                (progress_end - progress_start) * completed_weight / total_weight
            )
            command_progress_end = progress_start + round(
                (progress_end - progress_start)
                * (completed_weight + weights[index])
                / total_weight
            )
            log.write(f"\n> {_format_command(command)}\n")
            log.flush()
            _write_status(
                {"progress": command_progress_start, "message": command_message}
            )
            process = subprocess.Popen(
                command,
                cwd=str(project_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            tracker = _PipProgressTracker(
                command,
                progress_start=command_progress_start,
                progress_end=command_progress_end,
                base_message=command_message,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    tracker.consume(line)
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"命令执行失败 ({return_code}): {_format_command(command)}")
            completed_weight += weights[index]
            _write_status({"progress": command_progress_end, "message": command_message})
        log.write(f"\n[{_utc_now()}] maintenance completed\n")


class _PipProgressTracker:
    _RAW_PROGRESS_PATTERN = re.compile(r"^Progress\s+(\d+)\s+of\s+(\d+)\s*$", re.IGNORECASE)

    def __init__(
        self,
        command: Sequence[str],
        *,
        progress_start: int,
        progress_end: int,
        base_message: str,
    ) -> None:
        lowered_args = [str(part).lower() for part in command]
        self._enabled = "pip" in lowered_args and "install" in lowered_args
        self._pytorch_install = "torch" in lowered_args and "torchvision" in lowered_args
        self._progress_start = progress_start
        self._progress_end = progress_end
        self._base_message = base_message
        self._last_progress = progress_start
        self._last_status_write = 0.0
        self._download_label = "依赖包"
        self._download_range = (0.08, 0.82)
        self._download_count = 0

    def consume(self, line: str) -> None:
        if not self._enabled:
            return
        text = line.strip()
        if not text:
            return
        lowered = text.lower()

        if lowered.startswith("collecting "):
            self._update(0.02, self._base_message)
            return
        if lowered.startswith("downloading "):
            self._start_download(lowered)
            return
        if lowered.startswith("installing collected packages"):
            self._update(0.94, "正在写入已下载的依赖文件...")
            return
        if lowered.startswith("successfully installed"):
            self._update(1.0, self._base_message, force=True)
            return

        match = self._RAW_PROGRESS_PATTERN.match(text)
        if match is None:
            return
        current, total = (int(value) for value in match.groups())
        if total <= 0:
            return
        fraction = min(1.0, max(0.0, current / total))
        range_start, range_end = self._download_range
        local_progress = range_start + (range_end - range_start) * fraction
        message = (
            f"正在下载 {self._download_label}"
            f"（{_format_transfer_size(current)} / {_format_transfer_size(total)}）..."
        )
        self._update(local_progress, message)

    def _start_download(self, lowered_line: str) -> None:
        is_metadata = ".metadata" in lowered_line
        if is_metadata:
            self._download_label = "依赖信息"
            self._download_range = (0.02, 0.06)
            return

        self._download_label = _download_label(lowered_line)
        if self._pytorch_install:
            if self._download_label == "PyTorch":
                self._download_range = (0.08, 0.84)
            elif self._download_label == "torchvision":
                self._download_range = (0.84, 0.91)
            else:
                self._download_range = (0.06, 0.08)
            return

        if self._download_count == 0:
            self._download_range = (0.08, 0.82)
        else:
            self._download_range = (0.82, 0.90)
        self._download_count += 1

    def _update(self, local_progress: float, message: str, *, force: bool = False) -> None:
        absolute_progress = self._progress_start + round(
            (self._progress_end - self._progress_start)
            * min(1.0, max(0.0, local_progress))
        )
        absolute_progress = max(self._last_progress, absolute_progress)
        now = time.monotonic()
        if (
            not force
            and absolute_progress == self._last_progress
            and now - self._last_status_write < 1.5
        ):
            return
        self._last_progress = absolute_progress
        self._last_status_write = now
        _write_status({"progress": absolute_progress, "message": message})


def _download_label(lowered_line: str) -> str:
    if "torchvision-" in lowered_line:
        return "torchvision"
    if re.search(r"(?:^|[/\\])torch-", lowered_line) or " torch-" in lowered_line:
        return "PyTorch"
    if "ultralytics-" in lowered_line:
        return "ultralytics"
    return "依赖包"


def _format_transfer_size(size_bytes: int) -> str:
    value = float(max(0, size_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            precision = 0 if unit == "B" else 1
            return f"{value:.{precision}f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def _clear_maintenance_pip_cache() -> str | None:
    """Delete only Neri's maintenance cache and leave the user's global pip cache alone."""

    cache_dir = maintenance_pip_cache_dir().resolve()
    expected_parent = (project_root() / "temp").resolve()
    if cache_dir.parent != expected_parent:
        return f"拒绝清理非预期缓存目录: {cache_dir}"
    if not cache_dir.exists():
        return None
    try:
        shutil.rmtree(cache_dir)
    except OSError as exc:
        message = str(exc)
        log_path = maintenance_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"[{_utc_now()}] pip cache cleanup failed: {message}\n")
        return message
    return None


def _empty_intel_driver_status() -> dict[str, Any]:
    return {
        "checked": False,
        "gpu_present": False,
        "driver_installed": False,
        "driver_name": None,
        "driver_version": None,
        "driver_provider": None,
        "device_name": None,
        "processor_names": [],
        "message": "",
        "error": None,
    }


def _query_windows_display_devices() -> dict[str, Any]:
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$devices = Get-CimInstance Win32_VideoController |
    Select-Object Name, @{Name='Manufacturer'; Expression={$_.AdapterCompatibility}}, PNPDeviceID, Status
$hasIntelDisplay = @(
    $devices | Where-Object {
        "$($_.Name) $($_.Manufacturer) $($_.PNPDeviceID)" -match '(?i)Intel|VEN_8086'
    }
).Count -gt 0
$drivers = @()
$processors = @()
if ($hasIntelDisplay) {
    $drivers = Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceClass = 'DISPLAY'" |
        Select-Object DeviceName, Manufacturer, DriverProviderName, DriverVersion, DeviceID, InfName
    $processors = Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name
}
[pscustomobject]@{ devices = @($devices); drivers = @($drivers); processors = @($processors) } |
    ConvertTo-Json -Depth 5 -Compress
"""
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=HARDWARE_PROBE_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        try:
            return _query_windows_display_devices_with_pnputil()
        except Exception as fallback_exc:  # noqa: BLE001 - report both probes
            message = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"{message or 'PowerShell 显卡驱动检测失败'}; pnputil 兜底检测失败: {fallback_exc}"
            ) from fallback_exc
    output = completed.stdout.strip()
    if not output:
        return {"devices": [], "drivers": []}
    data = json.loads(output)
    if not isinstance(data, dict):
        return {"devices": [], "drivers": []}
    return data


def _query_windows_display_devices_with_pnputil() -> dict[str, Any]:
    completed = subprocess.run(
        ["pnputil", "/enum-devices", "/class", "Display", "/drivers"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=HARDWARE_PROBE_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(message or "pnputil 显卡驱动检测失败")
    return _parse_pnputil_display_output(completed.stdout)


def _parse_pnputil_display_output(output: str) -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    drivers: list[dict[str, Any]] = []
    blocks = re.split(r"(?m)^Instance ID:\s*", output)
    for block in blocks[1:]:
        lines = block.splitlines()
        if not lines:
            continue
        instance_id = lines[0].strip()
        device = {
            "Name": _search_pnputil_value(block, "Device Description"),
            "Manufacturer": _search_pnputil_value(block, "Manufacturer Name"),
            "PNPDeviceID": instance_id,
            "Status": _search_pnputil_value(block, "Status"),
        }
        devices.append(device)
        for driver_block in re.split(r"(?m)^\s+Driver Name:\s*", block)[1:]:
            driver_lines = driver_block.splitlines()
            if not driver_lines:
                continue
            drivers.append(
                {
                    "DeviceName": device["Name"],
                    "Manufacturer": device["Manufacturer"],
                    "DriverProviderName": _search_pnputil_value(driver_block, "Provider Name"),
                    "DriverVersion": _search_pnputil_value(driver_block, "Driver Version"),
                    "DeviceID": " ".join(
                        part
                        for part in [
                            instance_id,
                            _search_pnputil_value(driver_block, "Matching Device ID"),
                            _search_pnputil_value(driver_block, "Driver Status"),
                        ]
                        if part
                    ),
                    "DriverStatus": _search_pnputil_value(driver_block, "Driver Status"),
                    "InfName": driver_lines[0].strip(),
                }
            )
    return {"devices": devices, "drivers": drivers}


def _search_pnputil_value(block: str, label: str) -> str | None:
    match = re.search(rf"(?im)^\s*{re.escape(label)}:\s*(.+?)\s*$", block)
    return match.group(1).strip() if match else None


def _as_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _is_intel_display_device(item: dict[str, Any]) -> bool:
    text = _joined_item_text(
        item,
        "Name",
        "Manufacturer",
        "PNPDeviceID",
        "DeviceName",
        "DeviceID",
    ).upper()
    return "VEN_8086" in text or "INTEL" in text


def _is_intel_display_driver(item: dict[str, Any]) -> bool:
    text = _joined_item_text(
        item,
        "DeviceName",
        "Manufacturer",
        "DriverProviderName",
        "DeviceID",
        "InfName",
    ).upper()
    return "VEN_8086" in text or "INTEL" in text


def _is_official_intel_display_driver(item: dict[str, Any]) -> bool:
    provider = (_first_text(item, "DriverProviderName", "Manufacturer") or "").upper()
    status = (_first_text(item, "DriverStatus") or "").upper()
    driver_text = _joined_item_text(
        item,
        "DeviceName",
        "Manufacturer",
        "DriverProviderName",
        "DeviceID",
    ).upper()
    if "MICROSOFT" in provider:
        return False
    if status and "INSTALLED" not in status:
        return False
    return bool(_first_text(item, "DriverVersion")) and (
        "INTEL" in provider or "INTEL" in driver_text or "VEN_8086" in driver_text
    )


def _joined_item_text(item: dict[str, Any] | None, *keys: str) -> str:
    if not item:
        return ""
    return " ".join(str(item.get(key) or "") for key in keys)


def _first_text(item: dict[str, Any] | None, *keys: str) -> str | None:
    if not item:
        return None
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _validate_intel_download_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host != "downloadmirror.intel.com":
        raise ValueError(f"不是受信任的 Intel 驱动下载地址: {url}")


def _install_intel_graphics_driver_for_xpu(
    *,
    progress_start: int,
    progress_end: int,
) -> None:
    log_path = maintenance_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _write_status(
        {
            "state": "running",
            "message": "正在从 Intel 官网下载显卡驱动安装程序...",
            "progress": progress_start,
            "command": INTEL_GRAPHICS_DRIVER_DOWNLOAD_URL,
        }
    )
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[{_utc_now()}] Intel graphics driver download start\n")
        log.write(f"> {INTEL_GRAPHICS_DRIVER_DOWNLOAD_URL}\n")
        log.flush()
        installer_path = download_intel_graphics_driver()
        log.write(f"Downloaded Intel driver installer: {installer_path}\n")
        log.flush()

        _write_status(
            {
                "state": "running",
                "message": "正在运行 Intel 显卡驱动安装程序，请按安装程序提示完成操作...",
                "progress": progress_start
                + round((progress_end - progress_start) * 0.5),
                "command": _format_command([str(installer_path)]),
            }
        )
        log.write(f"> {_format_command([str(installer_path)])}\n")
        log.flush()
        completed = subprocess.run(
            [str(installer_path)],
            cwd=str(installer_path.parent),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Intel 显卡驱动安装程序退出码为 {completed.returncode}。")
        log.write(f"[{_utc_now()}] Intel graphics driver installer completed\n")
        _write_status({"progress": progress_end})

    _reset_intel_driver_status_cache()
    refreshed = intel_graphics_driver_status()
    if refreshed.get("gpu_present") and not refreshed.get("driver_installed"):
        _write_status(
            {
                "state": "running",
                "message": "Intel 驱动安装程序已退出；如安装程序提示重启，请重启后再使用 XPU 加速。",
            }
        )


def _resolve_pytorch_index(env_choice: str) -> tuple[str, str]:
    choice = (env_choice or "自动检测").strip()
    if choice == "自动检测":
        return _auto_detect_pytorch_index()
    if "XPU" in choice.upper():
        if not XPU_ENABLED:
            raise RuntimeError("Intel XPU 安装当前已暂时关闭，请选择 CUDA 或 CPU Only。")
        return "Intel XPU", PYTORCH_XPU_INDEX_URL
    if "CPU" in choice.upper():
        return "CPU Only", PYTORCH_CPU_INDEX_URL

    match = re.search(r"CUDA\s*(\d+)\.(\d+)", choice, flags=re.IGNORECASE)
    if match:
        major, minor = (int(value) for value in match.groups())
        if major > 13 or (major == 13 and minor >= 2):
            return "CUDA 13.2", PYTORCH_CUDA_132_INDEX_URL
        if major == 13:
            return "CUDA 13.0", PYTORCH_CUDA_130_INDEX_URL
        return f"CUDA {major}.{minor}", f"https://download.pytorch.org/whl/cu{major}{minor}"

    return "CPU Only", PYTORCH_CPU_INDEX_URL


def _auto_detect_pytorch_index() -> tuple[str, str]:
    cuda_choice = _detect_nvidia_cuda_pytorch_index()
    if cuda_choice is not None:
        return cuda_choice

    if XPU_ENABLED:
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
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    cuda_version = _parse_nvidia_cuda_version(result.stdout)
    if cuda_version is None:
        raise RuntimeError(
            "检测到 NVIDIA GPU，但无法从 nvidia-smi 识别 CUDA 驱动版本；"
            "请在环境维护中手动选择 CUDA 版本。"
        )

    major, minor = cuda_version
    if (major, minor) >= (13, 2):
        return f"CUDA 13.2 (检测到 CUDA {major}.{minor})", PYTORCH_CUDA_132_INDEX_URL
    if (major, minor) >= (13, 0):
        return f"CUDA 13.0 (检测到 CUDA {major}.{minor})", PYTORCH_CUDA_130_INDEX_URL
    if (major, minor) >= (12, 8):
        return f"CUDA 12.8 (检测到 CUDA {major}.{minor})", "https://download.pytorch.org/whl/cu128"
    if (major, minor) >= (12, 6):
        return f"CUDA 12.6 (检测到 CUDA {major}.{minor})", "https://download.pytorch.org/whl/cu126"
    if (major, minor) >= (12, 4):
        return f"CUDA 12.4 (检测到 CUDA {major}.{minor})", "https://download.pytorch.org/whl/cu124"
    if (major, minor) >= (12, 1):
        return f"CUDA 12.1 (检测到 CUDA {major}.{minor})", "https://download.pytorch.org/whl/cu121"
    if (major, minor) >= (11, 8):
        return f"CUDA 11.8 (检测到 CUDA {major}.{minor})", "https://download.pytorch.org/whl/cu118"
    return f"CPU Only (CUDA {major}.{minor} 过旧)", PYTORCH_CPU_INDEX_URL


def _parse_nvidia_cuda_version(output: str) -> tuple[int, int] | None:
    match = re.search(
        r"\bCUDA(?:\s+UMD)?\s+Version:\s*(\d+)\.(\d+)\b",
        output,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    major, minor = (int(value) for value in match.groups())
    return major, minor


def _detect_intel_xpu_pytorch_index() -> tuple[str, str] | None:
    status = intel_graphics_driver_status()
    if not status.get("gpu_present"):
        return None

    gpu_name = str(status.get("device_name") or status.get("driver_name") or "Intel GPU")
    cpu_names = [
        str(value)
        for value in status.get("processor_names", [])
        if str(value).strip()
    ]
    if _is_supported_intel_xpu_gpu(gpu_name, cpu_names):
        return f"Intel XPU (检测到 {gpu_name})", PYTORCH_XPU_INDEX_URL

    return f"CPU Only (Intel GPU 暂不支持 XPU: {gpu_name})", PYTORCH_CPU_INDEX_URL


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
            timeout=HARDWARE_PROBE_TIMEOUT_SECONDS,
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
            timeout=HARDWARE_PROBE_TIMEOUT_SECONDS,
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
