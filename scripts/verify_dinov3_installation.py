"""Verify the installed joint runtime against frozen training outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    installation = args.installation.resolve()
    sys.path.insert(0, str(installation))
    import system
    from system.dinov3.runtime import load_dinov3_model
    from system.backend.model_services import list_available_classification_models, list_model_classes, _to_model_info
    if not Path(system.__file__).resolve().is_relative_to(installation):
        raise ValueError("Verification imported the workspace backend instead of the installed one")
    settings = json.loads((installation / "temp/settings.json").read_text("utf8"))
    manifest = Path(settings["selected_classification_model"]).resolve()
    if not manifest.is_relative_to(installation):
        raise ValueError("Selected model must belong to the target installation")
    if settings["video_mode"] != "skip":
        raise ValueError("The installed photo sequence model must skip videos")
    with threadpool_limits(limits=4):
        runtime = load_dinov3_model(manifest)
        with np.load(args.training_run / "calibration_replay.npz", allow_pickle=False) as replay:
            actual = runtime.classifier.classify_features(replay["features"], camera_ids=replay["cameras"],
                sequence_ids=replay["sequences"], content_ids=replay["contents"], adapted_features=replay["adapted_features"])
            np.testing.assert_array_equal([p.best_known_species for p in actual], replay["predictions"])
            np.testing.assert_array_equal([p.accepted for p in actual], replay["accepted"])
            error = float(np.max(abs(np.asarray([p.known_score for p in actual]) - replay["known_score"])))
            np.testing.assert_allclose([p.known_score for p in actual], replay["known_score"], atol=2e-6, rtol=0)
        rows = [json.loads(line) for line in (args.training_run / "reviewed.jsonl").read_text("utf8").splitlines()]
        feature_cache = np.load(args.training_run / "features.npy", allow_pickle=False, mmap_mode="r")
        image_features = runtime.classifier.encoder.encode([r["file"] for r in rows[:32]])
        np.testing.assert_allclose(image_features, feature_cache[:32], atol=1e-6, rtol=1e-5)
        image_error = float(np.max(abs(image_features - feature_cache[:32])))
        classes = list_model_classes(str(manifest))
        catalog = list_available_classification_models()
        if not any(Path(item.path).resolve() == manifest and item.backend == "dinov3" for item in catalog):
            from system.model_sync.layout import get_model_layout
            from system.model_sync.catalog import discover_models
            catalog = [_to_model_info(item) for item in discover_models(get_model_layout(installation / "res"), "cls")]
        if not any(Path(item.path).resolve() == manifest and item.backend == "dinov3" for item in catalog):
            raise ValueError("Installed model is not visible in the software catalog")
        result = dict(passed=True, installation=str(installation), selected_model=str(manifest),
            backend_module=str(Path(system.__file__).resolve()), classes=len(classes),
            calibration_replay_rows=len(actual), classification_matches=True, acceptance_matches=True,
            max_gate_error=error, image_replay_count=32, max_feature_error=image_error,
            torch_device=runtime.classifier.encoder.device, video_mode=settings["video_mode"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        runtime.close()


if __name__ == "__main__":
    main()
