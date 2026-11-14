"""Replay frozen decisions while checking independent frame display evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime_root.resolve()))
    import numpy as np
    from threadpoolctl import threadpool_limits
    import system
    from system.dinov3.runtime import load_dinov3_model, read_manifest, resolve_manifest_asset
    from system.dinov3.encoder import sha256
    from system.dinov3.context import FastCausalHistoryIndex, adapt_with_history
    if not Path(system.__file__).resolve().is_relative_to(args.runtime_root.resolve()):
        raise ValueError("Imported a different runtime")
    payload = read_manifest(args.manifest)
    checkpoint = resolve_manifest_asset(args.manifest, payload["checkpoint"])
    calibration = resolve_manifest_asset(args.manifest, payload["calibration"])
    assert sha256(calibration) == payload["calibration_sha256"]
    reference = args.comparison/"camera_split/experiment_full.npz"
    frozen = json.loads((args.comparison/"camera_split/decisions_frozen.json").read_text("utf8"))
    assert sha256(reference) == frozen["files"][reference.name]
    results = {}
    with threadpool_limits(limits=4):
        runtime = load_dinov3_model(args.manifest, encoder_factory=lambda *a, **kw: None)
        def verify(prefix, data):
            predictions = runtime.classifier.classify_features(data["features"],camera_ids=data["cameras"],
                sequence_ids=data["sequences"],content_ids=data["contents"],adapted_features=data["adapted_features"])
            with np.load(reference,allow_pickle=False) as ref:
                np.testing.assert_array_equal([p.best_known_species for p in predictions],ref[prefix+"_pred"])
                np.testing.assert_array_equal([p.accepted for p in predictions],ref[prefix+"_accepted"])
                error = float(np.max(abs(np.asarray([p.known_score for p in predictions])-ref[prefix+"_score"])))
                np.testing.assert_allclose([p.known_score for p in predictions],ref[prefix+"_score"],atol=1e-10,rtol=0)
            groups = defaultdict(list)
            for i,p in enumerate(predictions):
                assert p.confidence_kind == "sigmoid_frame_class_margin"
                assert np.isfinite(p.confidence) and 0 <= p.confidence <= 1
                expected = 1/(1+np.exp(-np.clip(p.frame_classification_margin,-700,700)))
                np.testing.assert_allclose(p.confidence,expected,atol=1e-12,rtol=0)
                candidate = p.as_candidate()
                assert candidate["conf"] == candidate["frame_classification_confidence"] == p.confidence
                assert candidate["sequence_confidence"] == p.sequence_confidence
                groups[(str(data["cameras"][i]),str(data["sequences"][i]))].append(p)
            varied = 0
            for group in groups.values():
                np.testing.assert_allclose([p.known_score for p in group],group[0].known_score,atol=1e-12,rtol=0)
                if len(group)>1 and np.ptp([p.confidence for p in group])>1e-9:
                    varied += 1
            return dict(rows=len(predictions),classification_matches=True,acceptance_matches=True,
                        max_gate_error=error,events_with_distinct_frame_evidence=varied,
                        accepted_frames_below_half_evidence=sum(p.accepted and p.confidence<.5 for p in predictions))
        with np.load(calibration,allow_pickle=False) as c:
            results["calibration"] = verify("c",{k:c[k] for k in ("features","cameras","sequences","contents","adapted_features")})
        rows = [json.loads(line) for line in (args.comparison/"current_reviewed.jsonl").read_text("utf8").splitlines()]
        split = json.loads((args.comparison/"partition.json").read_text("utf8"))
        audit = json.loads((args.comparison/"input_audit.json").read_text("utf8"))
        cache = Path(audit.get("features_file",args.comparison.parent/"dinov3_reviewed_20261107/features.npy"))
        assert sha256(cache) == audit["features_sha256"]
        features = np.load(cache,allow_pickle=False,mmap_mode="r")
        indices = np.asarray(sorted(int(k) for k,p in split["assignments"].items() if p in {"evaluation","evaluation_unknown"}))
        with np.load(reference,allow_pickle=False) as ref:
            np.testing.assert_array_equal(indices,ref["e_indices"])
        selected = [rows[int(i)] for i in indices]
        x = features[[r["feature_index"] for r in selected]]
        records = [dict(camera_component=r["camera"],seq_id=r["sequence"],datetime=r["datetime"],
                        content_hash=r["source_sha256"] or r["sha256"]) for r in selected]
        results["evaluation"] = verify("e",dict(features=x,cameras=[r["camera"] for r in selected],
            sequences=[r["sequence"] for r in selected],contents=[r["sha256"] for r in selected],
            adapted_features=adapt_with_history(x,FastCausalHistoryIndex(records).build())))
        assert sum(r["events_with_distinct_frame_evidence"] for r in results.values())>0
        runtime.close()
    output = dict(passed=True,runtime_root=str(args.runtime_root.resolve()),manifest=str(args.manifest.resolve()),
                  checkpoint_sha256=sha256(checkpoint),parity=results,
                  confidence_kind="sigmoid_frame_class_margin",calibrated_probability=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n","utf8")
    print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
