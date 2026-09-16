from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

REPLACEMENTS = (
    ("DinoV3", "DinoV2"),
    ("dinoV3", "dinoV2"),
    ("DINOv3", "DINOv2"),
    ("dinov3", "dinov2"),
)


def swap(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def rewrite(path: Path, transform=swap) -> None:
    if not path.is_file():
        return
    old = path.read_text(encoding="utf-8")
    new = transform(old)
    if new != old:
        path.write_text(new, encoding="utf-8")


def strip_flutter_bridges() -> None:
    api = ROOT / "frontend/lib/src/api_client.dart"
    text = api.read_text(encoding="utf-8")
    text = re.sub(
        r"\n  // Temporary source-compatibility bridge\..*?\n  void _ensureSuccess",
        "\n  void _ensureSuccess",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\n@Deprecated\('Use DinoV2ComponentStatus\.[^\n]*\n"
        r"typedef DinoV3ComponentStatus = DinoV2ComponentStatus;\n?",
        "\n",
        text,
    )
    api.write_text(text, encoding="utf-8")

    settings = ROOT / "frontend/lib/src/models/settings.dart"
    text = settings.read_text(encoding="utf-8")
    text = re.sub(
        r"\n  @Deprecated\('Use isDinoV2\.[^\n]*\n  bool get isDinoV3 => isDinoV2;\n",
        "\n",
        text,
    )
    settings.write_text(text, encoding="utf-8")


def migrate_frontend() -> None:
    strip_flutter_bridges()
    lib_root = ROOT / "frontend/lib"

    for path in sorted(lib_root.rglob("*.dart")):
        if "dinov3" not in path.name.lower():
            rewrite(path)

    old_files = sorted(
        (p for p in lib_root.rglob("*.dart") if "dinov3" in p.name.lower()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for src in old_files:
        dst = src.with_name(src.name.replace("dinov3", "dinov2"))
        if dst.exists():
            src.unlink()
            continue
        dst.write_text(swap(src.read_text(encoding="utf-8")), encoding="utf-8")
        src.unlink()

    test_root = ROOT / "frontend/test"
    for path in sorted(test_root.glob("*.dart")):
        if "dinov3" not in path.name.lower():
            rewrite(path)
    for src in sorted(test_root.glob("*dinov3*.dart")):
        dst = src.with_name(src.name.replace("dinov3", "dinov2"))
        if dst.exists():
            src.unlink()
            continue
        dst.write_text(swap(src.read_text(encoding="utf-8")), encoding="utf-8")
        src.unlink()


def migrate_backend_hooks() -> None:
    rewrite(ROOT / "system/backend/services_legacy.py")
    rewrite(ROOT / "system/image_processor_legacy.py")

    facade = ROOT / "system/backend/services.py"
    text = facade.read_text(encoding="utf-8")
    text = text.replace(
        "            # Temporary cleanup alias for the frozen generic job loop only.\n"
        "            # This does not enable DINOv3 inference and is deleted in Task 12.\n"
        "            detector.dinov3_runtime = runtime\n",
        "",
    )
    text = text.replace(
        "# These assignments redirect global lookups inside the frozen generic service\n"
        "# implementation. No DINOv3 checkpoint/encoder/runtime is loaded through them.\n"
        "_legacy._validate_dinov3_job_options = _validate_dinov2_job_options\n"
        "_legacy._dinov3_manifest_payload = _dinov2_manifest_payload\n"
        "_legacy._persist_dinov3_observations = _persist_dinov2_observations\n",
        "# These assignments redirect DINO hooks inside the generic service implementation.\n"
        "_legacy._validate_dinov2_job_options = _validate_dinov2_job_options\n"
        "_legacy._dinov2_manifest_payload = _dinov2_manifest_payload\n"
        "_legacy._persist_dinov2_observations = _persist_dinov2_observations\n",
    )
    facade.write_text(text, encoding="utf-8")


def clean_dinov2_base_names() -> None:
    for relative in (
        "system/dinov2/feedback_base.py",
        "system/dinov2/registry_base.py",
        "system/dinov2/feedback_index.py",
        "system/dinov2/prototype_bank.py",
        "system/dinov2/simple_shot.py",
    ):
        rewrite(ROOT / relative)


def main() -> None:
    migrate_backend_hooks()
    clean_dinov2_base_names()
    migrate_frontend()


if __name__ == "__main__":
    main()
