#!/usr/bin/env python3
"""Calibrate Multi-dual rejection and write a derived Neri model manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.dinov3.calibration import calibrate_multi_dual, cl2n_transform, prototype_signals


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to read the DINOv3 checkpoint") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("head_type") != "multi_prototype":
        raise ValueError("Expected a Multi-prototype checkpoint payload")
    for field in ("classes", "feature_center", "prototypes"):
        if field not in payload:
            raise ValueError(f"Checkpoint is missing {field}")
    return payload


def _tensor_numpy(value, *, name: str) -> np.ndarray:
    if not hasattr(value, "detach"):
        raise ValueError(f"Checkpoint {name} must be a tensor")
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _feature_cache_path(source_run: Path) -> Path:
    for name in ("image_features.npz", "event_features.npz"):
        candidate = source_run / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No image_features.npz or event_features.npz found in {source_run}"
    )


def _row_id(row: dict[str, Any]) -> str | None:
    for key in ("image_id", "event_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return None


def _cache_ids(cache) -> np.ndarray | None:
    for key in ("image_ids", "event_ids", "ids"):
        if key in cache.files:
            return np.asarray(cache[key]).astype(str)
    return None


def _validate_source_protocol(
    source_run: Path,
    checkpoint_payload: dict[str, Any],
    *,
    feature_dim: int,
    classes: tuple[str, ...],
) -> dict[str, Any]:
    protocol_path = source_run / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Missing source protocol required for safe calibration: {protocol_path}"
        )
    protocol = _read_json(protocol_path)
    if not isinstance(protocol, dict):
        raise ValueError("source-run protocol.json must contain an object")

    checkpoint_fingerprint = str(
        checkpoint_payload.get("data_fingerprint", "")
    ).strip()
    source_fingerprint = str(
        protocol.get("data_fingerprint")
        or protocol.get("source_data_fingerprint")
        or ""
    ).strip()
    if not checkpoint_fingerprint:
        raise ValueError("Checkpoint data_fingerprint is required for safe calibration")
    if not source_fingerprint:
        raise ValueError("source-run protocol is missing data_fingerprint")
    if source_fingerprint != checkpoint_fingerprint:
        raise ValueError(
            "source-run data_fingerprint does not match checkpoint data_fingerprint"
        )

    source_dim = protocol.get("feature_dim")
    if source_dim is not None and int(source_dim) != int(feature_dim):
        raise ValueError("source-run feature_dim does not match checkpoint")

    checkpoint_encoder = str(
        checkpoint_payload.get("encoder_sha256", "")
    ).strip().lower()
    source_encoder = str(
        protocol.get("encoder_sha256")
        or protocol.get("source_encoder_sha256")
        or ""
    ).strip().lower()
    if checkpoint_encoder and source_encoder and checkpoint_encoder != source_encoder:
        raise ValueError("source-run encoder_sha256 does not match checkpoint")

    source_classes = protocol.get("classes") or protocol.get("source_known_classes")
    if isinstance(source_classes, (list, tuple)):
        available = {str(name) for name in source_classes}
        missing = [name for name in classes if name not in available]
        if missing:
            raise ValueError(
                "source-run protocol is missing checkpoint classes: "
                + ", ".join(missing)
            )
    return protocol


def load_calibration_source(source_run: Path, *, feature_dim: int):
    split_path = source_run / "splits.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing source split file: {split_path}")
    rows = _read_json(split_path)
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("splits.json must contain a non-empty list of objects")

    cache_path = _feature_cache_path(source_run)
    with np.load(cache_path, allow_pickle=False) as cache:
        if "features" not in cache.files:
            raise ValueError(f"{cache_path.name} is missing features")
        raw = np.asarray(cache["features"], dtype=np.float32)
        cache_ids = _cache_ids(cache)

    if raw.ndim != 2 or raw.shape[1] != int(feature_dim):
        raise ValueError(
            f"Feature cache shape {raw.shape} does not have feature_dim {feature_dim}"
        )

    row_ids = [_row_id(row) for row in rows]
    if cache_ids is not None and all(value is not None for value in row_ids):
        if cache_ids.ndim != 1 or len(cache_ids) != len(raw):
            raise ValueError("Feature-cache id count does not match feature rows")
        typed_row_ids = [str(value) for value in row_ids]
        if len(set(typed_row_ids)) != len(typed_row_ids):
            raise ValueError("splits.json contains duplicate feature-cache ids")
        typed_cache_ids = [str(value) for value in cache_ids.tolist()]
        if len(set(typed_cache_ids)) != len(typed_cache_ids):
            raise ValueError("Feature cache contains duplicate ids")

        rows_by_id = dict(zip(typed_row_ids, rows, strict=True))
        missing = [value for value in typed_cache_ids if value not in rows_by_id]
        if missing:
            raise ValueError(
                "Feature-cache ids are missing from splits.json: "
                + ", ".join(missing[:5])
            )
        rows = [rows_by_id[value] for value in typed_cache_ids]
    elif raw.shape[0] != len(rows):
        raise ValueError(
            f"Feature cache shape {raw.shape} does not match splits ({len(rows)}, {feature_dim})"
        )

    if not np.isfinite(raw).all() or not np.allclose(
        np.linalg.norm(raw, axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("Expected finite L2-normalized frozen DINOv3 source features")

    truth = np.asarray([str(row.get("species", "")) for row in rows])
    split = np.asarray([str(row.get("split", "")) for row in rows])
    if np.any(truth == "") or np.any(split == ""):
        raise ValueError("Every split row must contain species and split")
    return raw, truth, split, cache_path


def _default_output_manifest(manifest: Path) -> Path:
    name = manifest.name
    if name.endswith(".neri.json"):
        name = name[: -len(".neri.json")] + ".multi-dual.neri.json"
    else:
        name = manifest.stem + ".multi-dual.neri.json"
    return manifest.with_name(name)


def _default_calibration_path(output_manifest: Path) -> Path:
    name = output_manifest.name
    if name.endswith(".neri.json"):
        name = name[: -len(".neri.json")] + ".calibration.json"
    else:
        name = output_manifest.stem + ".calibration.json"
    return output_manifest.with_name(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path)
    parser.add_argument("--out-calibration", type=Path)
    parser.add_argument("--max-known-frr", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    source_run = args.source_run.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not source_run.is_dir():
        raise FileNotFoundError(source_run)

    output_manifest = (
        args.out_manifest.expanduser().resolve()
        if args.out_manifest is not None
        else _default_output_manifest(manifest_path)
    )
    if output_manifest.parent != manifest_path.parent:
        raise ValueError("Derived manifest must stay beside the source manifest/checkpoint")
    output_calibration = (
        args.out_calibration.expanduser().resolve()
        if args.out_calibration is not None
        else _default_calibration_path(output_manifest)
    )
    if not args.force:
        for output in (output_manifest, output_calibration):
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    payload = _load_checkpoint_payload(checkpoint_path)
    classes = tuple(str(name) for name in payload["classes"])
    feature_center = _tensor_numpy(payload["feature_center"], name="feature_center")
    prototypes = _tensor_numpy(payload["prototypes"], name="prototypes")
    feature_dim = int(payload.get("feature_dim", prototypes.shape[1]))
    if feature_center.shape != (feature_dim,) or prototypes.ndim != 2 or prototypes.shape[1] != feature_dim:
        raise ValueError("Checkpoint feature_center/prototype dimensions are inconsistent")

    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("backend") != "dinov3":
        raise ValueError("Expected a DINOv3 .neri.json manifest")
    checkpoint_name = manifest.get("checkpoint")
    if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
        raise ValueError("Manifest checkpoint is missing")
    if (manifest_path.parent / checkpoint_name).resolve() != checkpoint_path:
        raise ValueError("Manifest checkpoint does not match --checkpoint")

    source_protocol = _validate_source_protocol(
        source_run,
        payload,
        feature_dim=feature_dim,
        classes=classes,
    )
    raw, truth, split, cache_path = load_calibration_source(
        source_run,
        feature_dim=feature_dim,
    )
    transformed = cl2n_transform(raw, feature_center)
    signals = prototype_signals(transformed, prototypes)

    known_val = np.isin(split, ("val", "validation")) & np.isin(truth, classes)
    proxy = split == "proxy_unknown_val"
    if not np.any(known_val):
        raise ValueError("No known validation images are available for calibration")
    if not np.any(proxy):
        raise ValueError("No proxy_unknown_val images are available for calibration")

    calibration = calibrate_multi_dual(
        signals["winner_cosine"][known_val],
        signals["winner_squared_distance"][known_val],
        signals["winner_cosine"][proxy],
        signals["winner_squared_distance"][proxy],
        truth[proxy],
        max_known_frr=args.max_known_frr,
        grid_size=args.grid_size,
    )

    source_fingerprint = str(
        source_protocol.get("data_fingerprint")
        or source_protocol.get("source_data_fingerprint")
    )
    derived = dict(manifest)
    derived["rejection"] = {
        "mode": "multi_dual",
        "cosine_threshold": calibration["cosine_threshold"],
        "squared_distance_threshold": calibration["squared_distance_threshold"],
    }
    derived["rejection_calibration"] = {
        "method": "known validation + proxy_unknown_val joint grid search",
        "objective": "maximize mean(known acceptance, proxy species-macro rejection) subject to known FRR cap",
        "feature_transform": "subtract stored feature_center, then L2 normalize (CL2N)",
        "known_validation_images": int(np.sum(known_val)),
        "proxy_unknown_images": int(np.sum(proxy)),
        "proxy_unknown_species": int(len(set(truth[proxy].tolist()))),
        "known_false_rejection_rate": calibration["known_false_rejection_rate"],
        "proxy_macro_rejection_rate": calibration["proxy_macro_rejection_rate"],
        "proxy_pooled_rejection_rate": calibration["proxy_pooled_rejection_rate"],
        "objective_value": calibration["objective"],
        "max_known_frr": calibration["max_known_frr"],
        "grid_size": calibration["grid_size"],
        "source_feature_cache": cache_path.name,
        "source_data_fingerprint": source_fingerprint,
        "test_isolation": "test and unknown_test rows are not used for threshold calibration",
    }

    output_manifest.write_text(
        json.dumps(derived, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_calibration.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(output_manifest),
                "calibration": str(output_calibration),
                "rejection": derived["rejection"],
                "known_validation_images": int(np.sum(known_val)),
                "proxy_unknown_images": int(np.sum(proxy)),
                "known_validation_frr": calibration["known_false_rejection_rate"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
