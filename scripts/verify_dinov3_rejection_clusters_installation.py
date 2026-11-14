"""Verify deployed files and run the installed rejection pipeline on synthetic crops."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.installation.resolve()
    report_dir = args.report_directory.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads((report_dir / "plan.json").read_text(encoding="utf-8-sig"))
    assert plan["status"] == "completed"
    for operation in plan["files"]:
        assert digest(operation["Target"]) == operation["After"], operation["Relative"]
    protected = json.loads((report_dir / "protected_before.json").read_text("utf-8"))
    for path, expected in protected.items():
        assert digest(path) == expected, f"Protected model/settings changed: {path}"
    # Keep all verification media, checkpoints and settings separate from user data.
    os.chdir(report_dir)
    sys.path.insert(0, str(root))
    import numpy as np
    import sklearn
    from system.backend import services
    from system.backend.models import DetectionItem, JobState, JobSummary
    from system.backend.job_state_store import JobStateStore
    from system.dinov3 import processor, rejection_clustering, runtime
    for module in (services, processor, rejection_clustering, runtime):
        assert Path(module.__file__).resolve().is_relative_to(root), module.__file__

    settings = json.loads((root / "temp/settings.json").read_text(encoding="utf-8-sig"))
    manifest = Path(settings["selected_classification_model"])
    payload = runtime.read_manifest(manifest)
    loaded = runtime.load_dinov3_model(manifest, encoder_factory=lambda *a, **kw: None)
    fingerprint = f"{payload['encoder_sha256']}:{payload['checkpoint_sha256']}"
    assert loaded.classifier.model_fingerprint == fingerprint
    loaded.close()
    del loaded

    class Prediction:
        def __init__(self, feature):
            self.accepted = int(np.argmax(feature[:3])) == 2
            self.species = self.best_known_species = "Known"
            self.known_score = 2.0 if self.accepted else -2.0
            self.threshold = 0.0
            self.confidence = .8 if self.accepted else .2
            self.ridge_score = .7
            self.within_score = .6
            self.embedding = feature
            self.candidates = ()

        def as_candidate(self, **kwargs):
            return dict(name="Known", source="dinov3_ridge_within", accepted=self.accepted,
                        conf=self.confidence, known_score=self.known_score, threshold=self.threshold)

    classifier = SimpleNamespace(
        backend="dinov3", model_fingerprint=fingerprint,
        model=SimpleNamespace(inference_recipe="experiment_full"),
        classify_features=lambda features, **kwargs: [Prediction(f) for f in features],
    )
    collector = processor.DinoV3CropCollector()
    detector = SimpleNamespace(dinov3_classifier=classifier, _dinov3_collector=collector)
    rng = np.random.default_rng(20261114)
    items = []
    for index in range(13):
        path = report_dir / "media" / ("north" if index % 2 else "south") / f"{index:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"Synthetic cached crop {index}".encode())
        feature = rng.normal(0, .002, 768).astype(np.float32)
        feature[min(index // 6, 2)] += 1
        feature /= np.linalg.norm(feature)
        collector.record_crop(processor.DinoV3CropRecord(str(path), 0, (0, 0, 20, 20), .9, feature))
        items.append(DetectionItem(filename=path.name, path=str(path), file_type="jpg",
                                   date_taken=f"2026-11-14T{index:02d}:00:00"))
    updated = processor.finalize_photo_job(
        detector, items, report_dir / "media", job_id="installed-smoke",
        save_batch_fn=services._save_detection_data_batch, apply_data_fn=services._apply_detection_data,
    )
    clusters = {}
    accepted = 0
    run_ids = set()
    for item in updated:
        assert item.detection_data["sequence_completed"] is True
        path = Path(item.path)
        stored = services._load_detection_data_for_path(path, [path.parent], recursive=False)
        assert stored["检测框"] == item.detection_data["检测框"]
        box = item.detection_boxes[0]
        metadata = box.get("rejection_cluster")
        if metadata is None:
            assert item.detection_data["检测框"][0]["accepted"] is True
            accepted += 1
        else:
            assert box["species"] == "Unknown" and metadata["status"] == "clustered"
            clusters.setdefault(metadata["cluster_id"], set()).add(path.parent)
            run_ids.add(metadata["run_id"])
            assert metadata["event_count"] == 6
    assert accepted == 1 and len(clusters) == 2 and len(run_ids) == 1
    assert all(len(folders) == 2 for folders in clusters.values())
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    job = JobSummary(id="smoke", state=JobState.COMPLETED, input_dir=str(report_dir),
                     results=updated, created_at=now, updated_at=now)
    state_path = report_dir / "verification_job.json"
    JobStateStore(state_path).save({"smoke": job}, {})
    restored = JobSummary.model_validate(JobStateStore(state_path).load()["jobs"]["smoke"])
    assert restored.results == updated

    # Import/routing readiness only: avoid resuming automatic learning workers.
    print("Installed clustering and database roundtrips passed; checking API readiness.", flush=True)
    faulthandler.dump_traceback_later(30)
    from fastapi.testclient import TestClient
    from system.backend.main import app
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    client.close()
    faulthandler.cancel_dump_traceback_later()
    for path, expected in protected.items():
        assert digest(path) == expected, f"Protected model/settings changed during verification: {path}"
    report = dict(
        passed=True, installation=str(root), backup=plan["backup"],
        installed_files_verified=len(plan["files"]), protected_files_verified=len(protected),
        sklearn_version=sklearn.__version__, selected_model=str(manifest),
        model_fingerprint_matches=True, synthetic_photos=13, rejected_photos=12,
        clusters=2, cross_folder_grouping=True, source_database_roundtrip=True,
        sqlite_job_roundtrip=True, backend_health=response.json(),
        verification_scope="installed runtime with synthetic cached features; encoder not rerun",
    )
    (report_dir / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
