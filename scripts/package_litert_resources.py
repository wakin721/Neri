"""Copy portable resources with a LiteRT-only model allowlist; verify the result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

MODEL_EXTENSIONS = {".tflite", ".litert"}
EXCLUDED_EXTENSIONS = {
    ".pt", ".pth", ".onnx", ".engine", ".safetensors", ".neri-model",
    ".ckpt", ".pb", ".h5", ".hdf5", ".bin", ".npz", ".npy",
    ".zip", ".tar", ".tgz", ".gz", ".7z",
}
MODEL_ROOTS = {"model", "model_cls", "dinov2", "dinov3"}
SIDECARS = (".labels.txt", ".labels.json", ".LICENSE.md", ".SOURCE.md")
CANONICAL_DIRECTORIES = ("detect/bundled", "detect/user", "detect/sync", "cls/bundled", "cls/user", "cls/sync")
DETECTOR_DIRECTORIES = ("detect/bundled", "detect/user", "detect/sync")
BUILTIN_DETECTOR = 'model/detect/bundled/MDV6-yolov10-c.tflite'
INVENTORY = "packaged-models.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def is_excluded(path):
    return path.suffix.lower() in EXCLUDED_EXTENSIONS or path.name.lower().endswith(".neri.json")


def copy_resources(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate resource trees")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Build resources into an empty destination; existing models are never deleted")
    destination.mkdir(parents=True, exist_ok=True)
    for relative in CANONICAL_DIRECTORIES:
        (destination / "model" / relative).mkdir(parents=True, exist_ok=True)

    def copy_file(path, relative):
        path.resolve().relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    def copy_non_models(directory):
        for path in sorted(directory.iterdir()):
            if path.is_symlink():
                raise ValueError(f"Resource symlinks are not allowed: {path}")
            if path.is_dir():
                if path.name.lower() not in MODEL_ROOTS | {'bundled-models'} and path.name != "__pycache__":
                    copy_non_models(path)
            elif not is_excluded(path) and path.name != INVENTORY:
                copy_file(path, path.relative_to(source))

    copy_non_models(source)
    model_root = source / "model"
    for relative in DETECTOR_DIRECTORIES:
        directory = model_root / relative
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in MODEL_EXTENSIONS:
                continue
            if path.is_symlink():
                raise ValueError(f"Model symlinks are not allowed: {path}")
            copy_file(path, path.relative_to(source))
            for suffix in SIDECARS:
                sidecar = path.with_name(path.stem + suffix)
                if sidecar.is_file():
                    copy_file(sidecar, sidecar.relative_to(source))
    tracker = model_root / "tracker.yaml"
    if tracker.is_file():
        copy_file(tracker, Path("model/tracker.yaml"))
    models = [path for path in sorted(destination.rglob("*"))
              if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS]
    if not models:
        raise ValueError("The lite package must contain at least one LiteRT model")
    inventory = dict(policy="litert-only", models=[dict(
        path=path.relative_to(destination).as_posix(), size=path.stat().st_size, sha256=sha(path),
    ) for path in models])
    (destination / INVENTORY).write_text(json.dumps(inventory, indent=2)+"\n", "utf-8")
    return verify_resources(destination)


def verify_resources(resources, *, require_bundled_detector=False):
    resources = Path(resources).resolve()
    inventory = json.loads((resources / INVENTORY).read_text("utf-8"))
    if inventory.get("policy") != "litert-only" or not inventory.get("models"):
        raise ValueError("Missing LiteRT-only model inventory")
    if inventory.get("classifier_assets"):
        raise ValueError("Bundled classifier assets are not allowed in lite package")
    expected = {}
    for item in inventory["models"]:
        path = resources / item["path"]
        path.resolve().relative_to(resources)
        if tuple(part.lower() for part in Path(item["path"]).parts[:2]) == ("model", "cls"):
            raise ValueError(f"Classifier models are not allowed in lite package: {item['path']}")
        if path.suffix.lower() not in MODEL_EXTENSIONS or not path.is_file():
            raise ValueError(f"Invalid packaged model: {path}")
        if sha(path) != item["sha256"] or path.stat().st_size != item["size"]:
            raise ValueError(f"Packaged model checksum mismatch: {path}")
        if item["path"] in expected:
            raise ValueError("Duplicate model inventory entry")
        expected[item["path"]] = item
    actual = set()
    for path in resources.rglob("*"):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"Resource symlinks are not allowed: {path}")
        relative = path.relative_to(resources)
        parts_lower = tuple(part.lower() for part in relative.parts)
        if parts_lower and parts_lower[0] == "bundled-models":
            raise ValueError(f"Bundled model shard directory is not allowed in package: {relative}")
        if is_excluded(path):
            raise ValueError(f"Non-LiteRT model/archive in package: {relative}")
        if len(parts_lower) >= 2 and parts_lower[:2] == ("model", "cls"):
            raise ValueError(f"Classifier assets are not allowed in lite package: {relative}")
        if path.suffix.lower() in MODEL_EXTENSIONS:
            actual.add(relative.as_posix())
        elif relative.parts[0].lower() in MODEL_ROOTS:
            if relative.as_posix() == "model/tracker.yaml":
                continue
            if not any(path.name == model.stem + suffix
                       for suffix in SIDECARS
                       for model in path.parent.iterdir()
                       if model.is_file() and model.suffix.lower() in MODEL_EXTENSIONS):
                raise ValueError(f"Unexpected model asset in package: {relative}")
    if actual != set(expected):
        raise ValueError("Packaged model inventory differs from actual model files")
    if require_bundled_detector and BUILTIN_DETECTOR not in expected:
        raise ValueError('Missing bundled LiteRT detector: ' + BUILTIN_DETECTOR)
    return dict(passed=True, policy="litert-only", models=list(expected.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify", type=Path, help="Read-only verification of an existing res directory")
    parser.add_argument('--require-bundled-detector', action='store_true')
    args = parser.parse_args()
    if args.verify:
        report = verify_resources(args.verify, require_bundled_detector=args.require_bundled_detector)
    elif args.source and args.destination:
        report = copy_resources(args.source, args.destination)
        if args.require_bundled_detector:
            report = verify_resources(args.destination, require_bundled_detector=args.require_bundled_detector)
    else:
        parser.error("Use --source/--destination or --verify")
    print(json.dumps(report, ensure_ascii=False, indent=2))
