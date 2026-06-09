import argparse
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from system.version import BASE_VERSION, DEFAULT_BUILD_CODE  # noqa: E402

BUILD_CODE_PATTERN = re.compile(r"^[0-9a-fA-F]{6}$")


def normalize_build_code(value: str | None) -> str | None:
    if not value:
        return None
    build_code = value.strip().lower()
    return build_code if BUILD_CODE_PATTERN.fullmatch(build_code) else None


def git_build_code() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=6", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return normalize_build_code(result.stdout)


def choose_build_code(args: argparse.Namespace) -> str:
    explicit = normalize_build_code(args.code)
    if args.code and explicit is None:
        raise SystemExit("--code must be exactly 6 hexadecimal characters")
    return (
        explicit
        or normalize_build_code(os.environ.get("NERI_BUILD_CODE"))
        or git_build_code()
        or (secrets.token_hex(3) if args.random else DEFAULT_BUILD_CODE)
    )


def app_version(build_code: str) -> str:
    return f"{BASE_VERSION}({build_code})"


def pubspec_version(build_code: str) -> str:
    return f"{BASE_VERSION}+{build_code}"


def replace_once(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Expected one match in {path}, found {count}")
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize Neri build code across backend and Flutter metadata."
    )
    parser.add_argument("--code", help="Explicit 6-character hexadecimal build code.")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate a random build code when Git and NERI_BUILD_CODE are unavailable.",
    )
    args = parser.parse_args()

    build_code = choose_build_code(args)
    full_version = app_version(build_code)

    replace_once(
        ROOT / "system" / "version.py",
        r'^DEFAULT_BUILD_CODE = "[0-9a-fA-F]{6}"$',
        f'DEFAULT_BUILD_CODE = "{build_code}"',
    )
    replace_once(
        ROOT / "frontend" / "pubspec.yaml",
        r"^version:\s*.+$",
        f"version: {pubspec_version(build_code)}",
    )
    replace_once(
        ROOT / "frontend" / "lib" / "src" / "screens" / "settings_screen.dart",
        r"defaultValue:\s*'[^']+'",
        f"defaultValue: '{full_version}'",
    )

    print(f"Build code: {build_code}")
    print(f"App version: {full_version}")
    print(f"Flutter define: --dart-define=NERI_FRONTEND_VERSION={full_version}")


if __name__ == "__main__":
    main()
