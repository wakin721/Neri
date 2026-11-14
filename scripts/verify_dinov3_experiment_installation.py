"""Replay local experiment decisions through the installed software and encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--comparison",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    installation=args.installation.resolve()
    sys.path.insert(0,str(installation))
    import numpy as np
    from threadpoolctl import threadpool_limits
    import system
    from system.dinov3.runtime import load_dinov3_model,read_manifest,resolve_manifest_asset
    from system.dinov3.encoder import sha256
    from system.dinov3.context import FastCausalHistoryIndex,adapt_with_history
    from system.model_sync.layout import get_model_layout
    from system.model_sync.catalog import discover_models
    from system.backend.model_services import list_model_classes
    from system.backend.dinov3_learning import learning_status
    if not Path(system.__file__).resolve().is_relative_to(installation):
        raise ValueError("Imported workspace backend instead of the installed one")
    manifest=args.manifest.resolve()
    if not manifest.is_relative_to(installation):
        raise ValueError("Model must belong to the installation")
    payload=read_manifest(manifest)
    if payload.get("inference_recipe")!="experiment_full":
        raise ValueError("Expected the full experimental recipe")
    reference=args.comparison/"camera_split/experiment_full.npz"
    frozen=json.loads((args.comparison/"camera_split/decisions_frozen.json").read_text("utf8"))
    assert sha256(reference)==frozen["files"][reference.name]
    with threadpool_limits(limits=4):
        runtime=load_dinov3_model(manifest)
        calibration=resolve_manifest_asset(manifest,payload["calibration"])
        assert sha256(calibration)==payload["calibration_sha256"]
        with np.load(calibration,allow_pickle=False) as c:
            actual=runtime.classifier.classify_features(c["features"],camera_ids=c["cameras"],
                   sequence_ids=c["sequences"],content_ids=c["contents"],adapted_features=c["adapted_features"])
        parity={}
        def verify(prefix,predictions,indices=None):
            with np.load(reference,allow_pickle=False) as ref:
                if indices is not None:
                    np.testing.assert_array_equal(indices,ref[prefix+"_indices"])
                np.testing.assert_array_equal([p.best_known_species for p in predictions],ref[prefix+"_pred"])
                np.testing.assert_array_equal([p.accepted for p in predictions],ref[prefix+"_accepted"])
                error=float(np.max(abs(np.asarray([p.known_score for p in predictions])-ref[prefix+"_score"])))
                np.testing.assert_allclose([p.known_score for p in predictions],ref[prefix+"_score"],atol=1e-10,rtol=0)
            assert all(p.gate_mode=="sequence" and p.box_known_score is None for p in predictions)
            return dict(rows=len(predictions),classification_matches=True,acceptance_matches=True,max_gate_error=error)
        parity["calibration"]=verify("c",actual)
        rows=[json.loads(line) for line in (args.comparison/"current_reviewed.jsonl").read_text("utf8").splitlines()]
        split=json.loads((args.comparison/"partition.json").read_text("utf8"))
        audit=json.loads((args.comparison/"input_audit.json").read_text("utf8"))
        cache=Path(audit.get("features_file",args.comparison.parent/"dinov3_reviewed_20261107/features.npy"))
        assert sha256(cache)==audit["features_sha256"]
        features=np.load(cache,allow_pickle=False,mmap_mode="r")
        indices=np.asarray(sorted(int(k) for k,p in split["assignments"].items() if p in {"evaluation","evaluation_unknown"}))
        selected=[rows[int(i)] for i in indices]
        x=features[[r["feature_index"] for r in selected]]
        records=[dict(camera_component=r["camera"],seq_id=r["sequence"],datetime=r["datetime"],
                      content_hash=r["source_sha256"] or r["sha256"]) for r in selected]
        adapted=adapt_with_history(x,FastCausalHistoryIndex(records).build())
        actual=runtime.classifier.classify_features(x,camera_ids=[r["camera"] for r in selected],
               sequence_ids=[r["sequence"] for r in selected],content_ids=[r["sha256"] for r in selected],adapted_features=adapted)
        parity["evaluation"]=verify("e",actual,indices)
        bank=sorted(int(k) for k,p in split["assignments"].items() if p=="bank")[:32]
        images=[rows[i] for i in bank]
        encoded=runtime.classifier.encoder.encode([r["file"] for r in images])
        expected=features[[r["feature_index"] for r in images]]
        np.testing.assert_allclose(encoded,expected,atol=1e-6,rtol=1e-5)
        catalog=discover_models(get_model_layout(installation/"res"),"cls")
        assert any(Path(item.path).resolve()==manifest and item.backend=="dinov3" for item in catalog)
        status=learning_status(manifest)
        assert status["calibration_available"] and status["rejection"]["gate_mode"]=="sequence"
        result=dict(passed=True,installation=str(installation),manifest=str(manifest),display_name=payload["display_name"],
                    backend_module=str(Path(system.__file__).resolve()),classes=len(list_model_classes(str(manifest))),
                    inference_recipe=runtime.model.inference_recipe,parity=parity,image_replay_count=len(images),
                    image_feature_max_error=float(np.max(abs(encoded-expected))),torch_device=runtime.classifier.encoder.device,
                    visible_in_model_catalog=True,automatic_learning_available=True)
        runtime.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n","utf8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
