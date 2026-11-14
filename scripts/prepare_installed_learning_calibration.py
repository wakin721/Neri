"""Prepare a verified calibration attachment without changing an installed model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from system.dinov3.encoder import sha256
from system.dinov3.model import DinoV3RidgeWithinModel
from system.dinov3.runtime import read_manifest, resolve_manifest_asset


def prepare(installation, training_run, output):
    installation, training_run, output = map(lambda p: Path(p).resolve(), (installation, training_run, output))
    settings_path = installation / "temp/settings.json"
    settings = json.loads(settings_path.read_text("utf-8"))
    manifest = Path(settings["selected_classification_model"]).resolve()
    manifest.relative_to(installation)
    payload = read_manifest(manifest)
    checkpoint = resolve_manifest_asset(manifest, payload["checkpoint"])
    if sha256(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError("Installed checkpoint hash differs")
    model = DinoV3RidgeWithinModel.load(checkpoint)
    verification = json.loads((training_run / "feature_verification.json").read_text("utf-8"))
    if verification.get("passed") is not True:
        raise ValueError("Source feature verification failed")
    provenance = model.metadata
    for key in ("encoder_sha256", "features_sha256"):
        if verification[key] != provenance[key]:
            raise ValueError(f"Training provenance differs: {key}")
    if verification["manifest_sha256"] != provenance["reviewed_manifest_sha256"]:
        raise ValueError("Reviewed training manifest differs")
    for filename, expected in (("features.npy", provenance["features_sha256"]),
                               ("reviewed.jsonl", provenance["reviewed_manifest_sha256"]),
                               ("partition.json", provenance["partition_sha256"])):
        if sha256(training_run / filename) != expected:
            raise ValueError(f"Training artifact hash differs: {filename}")
    partition = json.loads((training_run / "partition.json").read_text("utf-8"))
    indices = np.asarray(partition["indices"]["calibration"], dtype=np.int64)
    feature_cache = np.load(training_run / "features.npy", allow_pickle=False, mmap_mode="r")
    with np.load(training_run / "calibration_replay.npz", allow_pickle=False) as archive:
        fields = {"features", "labels", "cameras", "sequences", "sites", "contents", "adapted_features", "datetimes"}
        data = {key: archive[key].copy() for key in fields if key in archive}
        np.testing.assert_array_equal(data["features"], feature_cache[indices])
        prediction = model.predict(data["features"], data["cameras"], data["sequences"],
                                   contents=data.get("contents"), sites=data.get("sites"),
                                   adapted_features=data.get("adapted_features"))
        np.testing.assert_array_equal(prediction["predictions"], archive["predictions"])
        np.testing.assert_array_equal(prediction["accepted"], archive["accepted"])
        np.testing.assert_allclose(prediction["known_score"], archive["known_score"], rtol=0, atol=2e-6)
        error = float(np.max(abs(prediction["known_score"] - archive["known_score"])))
    data["encoder_sha256"] = np.asarray(payload["encoder_sha256"])
    output.mkdir(parents=True, exist_ok=True)
    calibration = output / "calibration.npz"
    np.savez_compressed(calibration, **data)
    digest = sha256(calibration)
    filename = manifest.name.removesuffix(".neri.json") + f"-learning-calibration-{digest[:12]}.npz"
    payload.update(calibration=filename, calibration_sha256=digest)
    prepared = output / manifest.name
    prepared.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    relative = manifest.relative_to(installation)
    extra = [dict(Source=str(calibration), Rel=str(relative.parent / filename)),
             dict(Source=str(prepared), Rel=str(relative), ExpectedTargetSha256=sha256(manifest))]
    (output / "extra-files.json").write_text(json.dumps(extra, ensure_ascii=False, indent=2), "utf-8")
    report = dict(installation=str(installation), selected_model=str(manifest), classes=len(model.classes),
                  calibration_rows=len(data["features"]), predictions_match=True, acceptance_match=True,
                  max_gate_error=error, checkpoint_sha256=payload["checkpoint_sha256"],
                  settings_sha256=sha256(settings_path), calibration_sha256=digest,
                  extra_files=str(output / "extra-files.json"))
    (output / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        prepare(args.installation, args.training_run, args.output)
