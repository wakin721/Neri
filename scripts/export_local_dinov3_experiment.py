"""Package the frozen local comparison's complete experimental B/16 recipe."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.compare_local_dinov3_protocols import rows_read, query, write
from system.dinov3.classifier import DinoV3Classifier
from system.dinov3.deployment import publish_model
from system.dinov3.encoder import PREPROCESSING, sha256
from system.dinov3.experiment import EXPERIMENT_RECIPE, PREVIEW_DISPLAY_NAME
from system.dinov3.model import DinoV3RidgeWithinModel
from system.dinov3.runtime import read_manifest


def export(source, comparison, manifest):
    split = json.loads((comparison/"partition.json").read_text("utf8"))
    if sha256(comparison/"current_reviewed.jsonl") != split["manifest_sha256"]:
        raise ValueError("Frozen photo-use manifest changed")
    audit = json.loads((comparison/"input_audit.json").read_text("utf8"))
    if not audit["all_current_images_content_hashed"] or audit["changed_content"] or audit["new_files"]:
        raise ValueError("Full local photo content audit is required")
    base = DinoV3RidgeWithinModel.load(comparison/"camera_split/frozen_dual.neri-model")
    if base.metadata["partition_sha256"] != sha256(comparison/"partition.json"):
        raise ValueError("Checkpoint and frozen photo-use split differ")
    cache = Path(audit.get("features_file", REPO/"runs/dinov3_reviewed_20261107/features.npy"))
    if sha256(cache) != audit["features_sha256"]:
        raise ValueError("Feature cache SHA-256 mismatch")
    rows = rows_read(comparison/"current_reviewed.jsonl")
    raw = np.load(cache,allow_pickle=False,mmap_mode="r")[[r["feature_index"] for r in rows]]
    assignments = {int(k):v for k,v in split["assignments"].items()}
    choose = lambda names: np.asarray(sorted(i for i,p in assignments.items() if p in names),dtype=np.int64)
    model = deepcopy(base)
    model.inference_recipe = EXPERIMENT_RECIPE
    model.tau = .5
    model.metadata.update(preprocessing=PREPROCESSING, gradient_updates=0,
                          source="Frozen local B/C/E comparison; experimental Within/Ridge blend and event gate",
                          not_for_production=False, experimental=True)
    cq = query(rows,raw,choose({"calibration","calibration_unknown"}))
    # The reference uses crop-content hashes, with whole events rather than target tracks.
    model.calibrate(cq["raw"],cq["labels"],cq["cameras"],cq["sequences"],
                    contents=cq["crop_hashes"],adapted_features=cq["adapted"])
    reference_path = comparison/"camera_split/experiment_full.npz"
    frozen = json.loads((comparison/"camera_split/decisions_frozen.json").read_text("utf8"))
    if sha256(reference_path) != frozen["files"][reference_path.name]:
        raise ValueError("Frozen experiment reference changed")
    errors = {}
    calibration_predictions = None
    for prefix,q in (("c",cq),("e",query(rows,raw,choose({"evaluation","evaluation_unknown"})))):
        actual = DinoV3Classifier(model).classify_features(q["raw"],camera_ids=q["cameras"],
                    sequence_ids=q["sequences"],content_ids=q["crop_hashes"],adapted_features=q["adapted"])
        with np.load(reference_path,allow_pickle=False) as ref:
            np.testing.assert_array_equal(q["indices"],ref[prefix+"_indices"])
            np.testing.assert_array_equal([p.best_known_species for p in actual],ref[prefix+"_pred"])
            np.testing.assert_array_equal([p.accepted for p in actual],ref[prefix+"_accepted"])
            error = float(np.max(abs(np.asarray([p.known_score for p in actual])-ref[prefix+"_score"])))
            np.testing.assert_allclose([p.known_score for p in actual],ref[prefix+"_score"],atol=1e-10,rtol=0)
        errors[prefix] = dict(images=len(actual),classification_matches=True,acceptance_matches=True,max_gate_error=error)
        if prefix == "c":
            calibration_predictions = actual
    calibration = dict(features=cq["raw"],labels=cq["labels"],cameras=cq["cameras"],
                       sequences=cq["sequences"],contents=cq["crop_hashes"],source_contents=cq["contents"],adapted_features=cq["adapted"])
    path = publish_model(model,manifest,encoder_weights=source/"models/dinov3-vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                         encoder_code=source/"models/dinov3-code",display_name=PREVIEW_DISPLAY_NAME,
                         calibration_data=calibration,existing_manifest=read_manifest(manifest) if manifest.exists() else None)
    verify = comparison/"experiment_deployment"
    verify.mkdir(exist_ok=True)
    np.savez_compressed(verify/"calibration_replay.npz",**calibration,
                        predictions=np.asarray([p.best_known_species for p in calibration_predictions]),
                        accepted=np.asarray([p.accepted for p in calibration_predictions]),
                        known_score=np.asarray([p.known_score for p in calibration_predictions]),
                        encoder_sha256=np.asarray(model.metadata["encoder_sha256"]))
    result = dict(passed=True,manifest=str(path),inference_recipe=EXPERIMENT_RECIPE,
                  classes=len(model.classes),supports=int(model.ridge_head.class_counts_.sum()),
                  threshold=model.threshold,gate_mode="sequence",parity=errors)
    write(verify/"export_report.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("F:/files/python/Neri_plus"))
    parser.add_argument("--comparison",type=Path,default=REPO/"runs/local_protocol_comparison_20261114")
    parser.add_argument("--manifest",type=Path,default=REPO/"res/model/cls/user/dinov3-experiment-local.neri.json")
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        export(args.source,args.comparison,args.manifest)


if __name__=="__main__":
    main()
