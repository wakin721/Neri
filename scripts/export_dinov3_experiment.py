"""Export the 2026-11-07 initial joint model using Bank and calibration C only.

Reads verified frozen embeddings and the original frozen geometry. No evaluation
features or labels participate in fitting or calibration. Additional supports
are admitted separately with train_dinov3_ridge_within.py update.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from system.dinov3.deployment import publish_model
from system.dinov3.encoder import PREPROCESSING, sha256
from system.dinov3.model import DinoV3RidgeWithinModel


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def export(source, manifest):
    source = Path(source).resolve()
    folder = source / "runs/dinov3_context_comparison_20261107"
    feature_dir = folder / "dinov3_features"
    verification = json.loads((feature_dir / "verification.json").read_text("utf-8"))
    if verification.get("status") != "passed":
        raise ValueError("Source feature verification has not passed")
    for name, key in (("features_subset.npy", "features_sha256"), ("indices.npy", "indices_sha256")):
        if sha256(feature_dir / name) != verification[key]:
            raise ValueError(f"Source cache hash mismatch: {name}")
    indices = np.load(feature_dir / "indices.npy", allow_pickle=False)
    features = np.load(feature_dir / "features_subset.npy", allow_pickle=False, mmap_mode="r")
    position = {int(index): i for i, index in enumerate(indices)}
    def take(rows):
        return features[[position[int(i)] for i in rows]]
    ledger = source / "runs/review_quality_macro_20261105"
    with np.load(ledger / "partition.npz", allow_pickle=False) as parts:
        bank, calibration = parts["bank"].copy(), parts["val"].copy()
        unknown = parts["validation_unknown"].copy()
    if set(bank) & set(calibration):
        raise ValueError("Bank and C must be disjoint")
    manifest_source = source / "runs/iwildcam_initial_human_tests_20260930/features/dinov2/manifest.json"
    rows = json.loads(manifest_source.read_text("utf-8"))["rows"]
    def meta(ii, keys):
        return [np.asarray([str(rows[int(i)][k]) for i in ii]) for k in keys]
    weights = source / "models/dinov3-vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    extraction = json.loads((feature_dir / "protocol.json").read_text("utf-8"))
    if extraction.get("weight_sha256") != sha256(weights):
        raise ValueError("Frozen source features were extracted with different encoder weights")
    provenance = dict(encoder_sha256=sha256(weights), preprocessing=PREPROCESSING, gradient_updates=0,
                      source="2026-11-07 Bank/C joint model; no E fitting", source_features_sha256=verification["features_sha256"],
                      partition_sha256=sha256(ledger / "partition.npz"),
                      geometry_sha256=sha256(folder / "dinov3/geometry.npz"),
                      source_manifest_sha256=sha256(manifest_source))
    with np.load(folder / "dinov3/geometry.npz", allow_pickle=False) as geometry:
        origin = geometry["within_origin"].copy()
        transform = np.linalg.cholesky(geometry["within_precision"])
    model = DinoV3RidgeWithinModel().fit(take(bank), *meta(bank, ("species", "location", "seq_id")),
                                        origin=origin, transform=transform, metadata=provenance)
    with np.load(ledger / "numeric/history_validation.npz", allow_pickle=False) as archive:
        history = archive["history"].copy()
    # Replay the frozen same-pool history indices with exactly the reference
    # NumPy attention math; no evaluation pool is loaded into this adapter.
    cal_set = set(map(int, calibration))
    if any(int(i) not in cal_set for i in np.unique(history[history >= 0])):
        raise ValueError("C history must stay inside the C pool")
    raw = take(calibration)
    adapted = raw.copy()
    for begin in range(0, len(raw), 128):
        hh = history[begin:begin + 128]
        valid = hh >= 0
        active = valid.any(1)
        if not active.any():
            continue
        q = raw[begin:begin + len(hh)][active]
        h = take(np.where(hh[active] >= 0, hh[active], calibration[0]).ravel()).reshape((-1, hh.shape[1], 768)).copy()
        mask = valid[active]
        h[~mask] = 0
        logits = np.einsum("bd,bmd->bm", unit(q), unit(h)) / .1
        logits[~mask] = -np.inf
        attention = np.exp(logits - logits.max(1, keepdims=True))
        attention /= attention.sum(1, keepdims=True)
        pooled = np.einsum("bm,bmd->bd", attention, h)
        adapted[begin + np.flatnonzero(active)] = unit(.9 * q + .1 * pooled)
    labels, cameras, sequences, sites, contents = meta(calibration, ("species", "location", "seq_id", "camera_component", "jpeg_md5"))
    model.calibrate(raw, labels, cameras, sequences, sites=sites, contents=contents,
                    adapted_features=adapted, unknown_mask=np.isin(calibration, unknown))
    result = model.predict(raw, cameras, sequences, sites=sites, contents=contents, adapted_features=adapted)
    parity = {}
    audited = source / "runs/old_species_accuracy_20261107_r2"
    if (audited / "independent_audit.json").is_file():
        audit = json.loads((audited / "independent_audit.json").read_text("utf8"))
        frozen = json.loads((audited / "decisions_frozen.json").read_text("utf8"))
        baseline = audited / "predictions/baseline.npz"
        if not audit.get("passed") or sha256(baseline) != frozen["prediction_files"]["baseline.npz"]:
            raise ValueError("Audited joint-reference baseline hash mismatch")
        # Only C arrays are accessed. E remains outside all export decisions.
        with np.load(baseline, allow_pickle=False) as reference:
            expected_scores = reference["score_c__top75_radial08"]
            expected_classes = model.classes[reference["pred_c"]]
        np.testing.assert_allclose(result["known_score"], expected_scores, atol=2e-5, rtol=0)
        np.testing.assert_array_equal(result["predictions"], expected_classes)
        expected_threshold = frozen["baseline"]["top75_radial08"]["threshold"]
        np.testing.assert_array_equal(result["accepted"], expected_scores >= expected_threshold)
        parity = dict(calibration_rows_compared=len(calibration), classification_matches=True, acceptance_matches=True,
                      gate_max_abs_error=float(np.max(abs(result["known_score"] - expected_scores))),
                      threshold_abs_error=abs(float(result["threshold"]) - expected_threshold))
    path = publish_model(model, manifest, encoder_weights=weights, encoder_code=source / "models/dinov3-code",
                         calibration_data=dict(features=raw, labels=labels, cameras=cameras, sequences=sequences,
                                               sites=sites, contents=contents, adapted_features=adapted))
    report = dict(manifest=str(path), classes=len(model.classes), supports=len(bank), calibration=len(calibration),
                  auxiliary_unknown=int(np.isin(calibration, unknown).sum()), threshold=float(result["threshold"]), parity=parity)
    Path(manifest).with_suffix(".export-report.json").write_text(json.dumps(report, indent=2) + "\n", "utf8")
    print(json.dumps(report, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Neri_plus experiment root")
    parser.add_argument("--manifest", type=Path, required=True, help="Output *.neri.json inside res/model/cls/user")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        export(args.source, args.manifest)


if __name__ == "__main__":
    main()
