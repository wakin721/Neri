"""Fit, update and calibrate a deployable DINOv3 Ridge + Within model offline.

Input NPZ: features[N,768], labels[N], cameras[N], sequences[N],
encoder_sha256 (scalar); optionally contents, sites, adapted_features for C,
and origin / transform for an already frozen initial geometry.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from system.dinov3.deployment import publish_model
from system.dinov3.encoder import PREPROCESSING, sha256
from system.dinov3.model import DinoV3RidgeWithinModel
from system.dinov3.experiment import INFERENCE_RECIPES, PRODUCTION_RECIPE
from system.dinov3.runtime import read_manifest, resolve_manifest_asset


def read_features(path, encoder_sha256):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    if str(data.pop("encoder_sha256", "")) != encoder_sha256:
        raise ValueError(f"{path}: feature encoder SHA-256 is missing or mismatched")
    if not {"features", "labels", "cameras", "sequences"} <= data.keys():
        raise ValueError(f"{path}: missing feature rows or camera/sequence labels")
    if data["features"].ndim != 2 or data["features"].shape[1] != 768:
        raise ValueError(f"{path}: expected 768-dimensional CLS features")
    return data


def recalibrate(model, data):
    adapted = data.get("adapted_features")
    if adapted is None and "datetimes" in data:
        from system.dinov3.context import FastCausalHistoryIndex, adapt_with_history
        count = len(data["features"])
        sites = data.get("sites", data["cameras"])
        contents = data.get("contents", np.asarray([f"cal:{i}" for i in range(count)]))
        if any(np.shape(values) != (count,) for values in (data["datetimes"], sites, contents)):
            raise ValueError("C datetimes, sites and contents must align with features")
        records = [dict(camera_component=str(sites[i]), seq_id=str(data["sequences"][i]),
                        datetime=str(data["datetimes"][i]), content_hash=str(contents[i])) for i in range(count)]
        adapted = adapt_with_history(data["features"], FastCausalHistoryIndex(records).build())
    model.calibrate(data["features"], data["labels"], data["cameras"], data["sequences"],
                    sites=data.get("sites"), contents=data.get("contents"),
                    adapted_features=adapted)


def train(args):
    if args.command == "fit":
        digest = sha256(args.encoder_weights)
        bank = read_features(args.bank, digest)
        cal = read_features(args.calibration, digest)
        model = DinoV3RidgeWithinModel(inference_recipe=getattr(args, "inference_recipe", PRODUCTION_RECIPE))
        model.fit(bank["features"], bank["labels"], bank["cameras"], bank["sequences"],
                  origin=bank.get("origin"), transform=bank.get("transform"),
                  metadata=dict(encoder_sha256=digest, preprocessing=PREPROCESSING,
                                source="DINOv3 + Ridge + Within; 2026-11-07", gradient_updates=0))
        recalibrate(model, cal)
        return publish_model(model, args.manifest, encoder_weights=args.encoder_weights,
                             encoder_code=args.encoder_code, calibration_data=cal,
                             display_name=getattr(args, "display_name", "Neri1.1"))
    payload = read_manifest(args.manifest)
    model = DinoV3RidgeWithinModel.load(resolve_manifest_asset(args.manifest, payload["checkpoint"]))
    digest = payload["encoder_sha256"]
    if sha256(resolve_manifest_asset(args.manifest, payload["checkpoint"])) != payload["checkpoint_sha256"]:
        raise ValueError("Existing model checkpoint SHA-256 mismatch")
    if args.command == "update":
        supports = read_features(args.supports, digest)
        model.partial_fit(supports["features"], supports["labels"], supports["cameras"], supports["sequences"])
    cal = read_features(args.calibration, digest)
    recalibrate(model, cal)
    return publish_model(model, args.manifest, existing_manifest=payload,
                         display_name=payload.get("display_name", "Neri1.1"), calibration_data=cal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Train from trusted supports and labeled C only")
    fit.add_argument("--bank", type=Path, required=True)
    fit.add_argument("--encoder-weights", type=Path, required=True)
    fit.add_argument("--encoder-code", type=Path, required=True)
    fit.add_argument("--inference-recipe", choices=INFERENCE_RECIPES, default=PRODUCTION_RECIPE)
    fit.add_argument("--display-name", default="Neri1.1")
    update = commands.add_parser("update", help="Supplement old/new species, then recalibrate both paths")
    update.add_argument("--supports", type=Path, required=True)
    commands.add_parser("calibrate", help="Recalibrate C without changing the frozen geometry/supports")
    for command in commands.choices.values():
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        result = train(args)
    print(result)


if __name__ == "__main__":
    main()
