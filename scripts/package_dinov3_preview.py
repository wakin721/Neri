"""Package an existing calibrated deployment, preserving all inference assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from system.dinov3.runtime import read_manifest, resolve_manifest_asset
from system.dinov3.experiment import EXPERIMENT_RECIPE, PREVIEW_DISPLAY_NAME


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    payload = read_manifest(manifest)
    if payload.get("inference_recipe") != EXPERIMENT_RECIPE:
        raise ValueError("Preview release requires the validated full experiment recipe")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    renamed = dict(payload, display_name=PREVIEW_DISPLAY_NAME)
    sources = {}
    for key, name, digest_field in (
        ("checkpoint", "model.neri-model", "checkpoint_sha256"),
        ("encoder_weights", "encoder.pth", "encoder_sha256"),
        ("calibration", "calibration.npz", "calibration_sha256"),
    ):
        source = resolve_manifest_asset(manifest, payload[key])
        if sha(source) != payload[digest_field]:
            raise ValueError(f"Invalid source digest: {key}")
        renamed[key] = f"assets/{name}"
        sources[renamed[key]] = source
    source_code = resolve_manifest_asset(manifest, payload["encoder_code"])
    renamed["encoder_code"] = "assets/encoder-code"
    for source in sorted(source_code.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts or source.suffix in {".pyc", ".pyo"}:
            continue
        source.resolve().relative_to(source_code.resolve())
        sources[f"assets/encoder-code/{source.relative_to(source_code).as_posix()}"] = source
    required = ("assets/encoder-code/LICENSE.md", "assets/encoder-code/dinov3/hub/backbones.py")
    if any(path not in sources for path in required):
        raise ValueError("Official encoder code and license must be included")
    # Read the serialized classifier metadata without re-saving its parameters.
    checkpoint = resolve_manifest_asset(manifest, payload["checkpoint"])
    with zipfile.ZipFile(checkpoint) as archive:
        head = json.loads(archive.read("manifest.json"))
    assert head["inference_recipe"] == EXPERIMENT_RECIPE and head["is_calibrated"] is True
    prefix = PREVIEW_DISPLAY_NAME + "/"
    manifest_name = PREVIEW_DISPLAY_NAME + ".neri.json"
    manifest_bytes = (json.dumps(renamed, ensure_ascii=False, indent=2)+"\n").encode("utf-8")
    inventory = {name: dict(size=path.stat().st_size, sha256=sha(path)) for name, path in sources.items()}
    inventory[manifest_name] = dict(size=len(manifest_bytes), sha256=hashlib.sha256(manifest_bytes).hexdigest())
    receipt = dict(name=PREVIEW_DISPLAY_NAME, inference_recipe=EXPERIMENT_RECIPE,
                   known_species=len(head["classes"]), files=inventory,
                   checkpoint_sha256=payload["checkpoint_sha256"],
                   encoder_sha256=payload["encoder_sha256"],
                   calibration_sha256=payload["calibration_sha256"])
    temporary = output.with_suffix(output.suffix+".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr(prefix+manifest_name, manifest_bytes)
            for name, source in sources.items():
                archive.write(source, prefix+name)
            archive.writestr(prefix+"package_inventory.json", json.dumps(receipt, ensure_ascii=False, indent=2))
        with zipfile.ZipFile(temporary) as archive:
            for name, expected in inventory.items():
                with archive.open(prefix+name) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected["sha256"] or archive.getinfo(prefix+name).file_size != expected["size"]:
                    raise ValueError(f"Package verification failed: {name}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha(output)
    output.with_suffix(output.suffix+".sha256").write_text(f"{digest}  {output.name}\n", "ascii")
    report = {**receipt, "archive": str(output), "archive_size": output.stat().st_size,
              "archive_sha256": digest, "passed": True}
    output.with_suffix(".package-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.manifest, args.output)
