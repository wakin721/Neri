import os
import re
import subprocess
from pathlib import Path

BASE_VERSION = "3.0.5-beta2"
DEFAULT_BUILD_CODE = "19e9af"
DEFAULT_BUILD_NUMBER = 416

_BUILD_CODE_PATTERN = re.compile(r"^[0-9a-fA-F]{6}$")
_BUILD_NUMBER_PATTERN = re.compile(r"^\d{1,5}$")
_VERSION_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)+")


def _normalize_build_code(value: str | None) -> str | None:
    if not value:
        return None
    build_code = value.strip().lower()
    return build_code if _BUILD_CODE_PATTERN.fullmatch(build_code) else None


def _normalize_build_number(value: str | int | None) -> int | None:
    if value is None:
        return None
    build_number = str(value).strip()
    if not _BUILD_NUMBER_PATTERN.fullmatch(build_number):
        return None
    parsed = int(build_number)
    return parsed if 1 <= parsed <= 65535 else None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _git_build_code() -> str | None:
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["git", "rev-parse", "--short=6", "HEAD"],
            cwd=_repo_root(),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _normalize_build_code(result.stdout)


def _git_build_number() -> int | None:
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=_repo_root(),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _normalize_build_number(result.stdout)


def get_build_code() -> str:
    return (
        _normalize_build_code(os.environ.get("NERI_BUILD_CODE"))
        or _git_build_code()
        or DEFAULT_BUILD_CODE
    )


def get_build_number() -> int:
    return (
        _normalize_build_number(os.environ.get("NERI_BUILD_NUMBER"))
        or _normalize_build_number(os.environ.get("GITHUB_RUN_NUMBER"))
        or _git_build_number()
        or DEFAULT_BUILD_NUMBER
    )


def format_app_version(build_code: str | None = None) -> str:
    return f"{BASE_VERSION}({build_code or get_build_code()})"


def format_pubspec_version(build_number: str | int | None = None) -> str:
    normalized = _normalize_build_number(build_number)
    return f"{BASE_VERSION}+{normalized if normalized is not None else get_build_number()}"


BUILD_CODE = get_build_code()
BUILD_NUMBER = get_build_number()
APP_VERSION = format_app_version(BUILD_CODE)
APP_TITLE = f"Neri v{APP_VERSION}"

_display_match = _VERSION_NUMBER_PATTERN.search(BASE_VERSION)
APP_DISPLAY_VERSION = f"Neri {_display_match.group(0)}" if _display_match else "Neri"
