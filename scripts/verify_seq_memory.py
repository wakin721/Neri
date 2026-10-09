"""Check sequence isolation, deployment compatibility and calibration diagnostics."""
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

PROJECT = Path(r"E:\Files\Neri")
SOURCE = Path(r"F:\files\python\Neri_plus")
OUT = PROJECT / "runs/seq_memory_reviewed_20261009"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(PROJECT))

from pipeline.sequence_rules import validate_partition
from training.training_free_memory_head import MemoryHead, camera_selection_split
from system.dinov2.memory_checkpoint import load_memory_checkpoint
from system.dinov2.memory_classifier import MemoryDinoV2Classifier


def main():
    snapshot = json.loads((OUT / "features/snapshot.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "deployment/manifest.json").read_text(encoding="utf-8"))
    rows = snapshot["rows"]
    with np.load(OUT / "features/features.npz", allow_pickle=False) as z:
        raw = z["features"].copy()
        np.testing.assert_array_equal(z["ids"], [r["image_id"] for r in rows])
    y = np.array([r["species"] for r in rows])
    cams = np.array([r["camera"] for r in rows])
    bank, cal = camera_selection_split(y, cams, seed=manifest["seed"])
    validate_partition(rows, {str(r["id"]): "bank" if b else "calibration" for r, b in zip(rows, bank)})
    assert not set(cams[bank]) & set(cams[cal])
    assert int(bank.sum()) == manifest["bank_images"] and int(cal.sum()) == manifest["calibration_images"]
    head_path = OUT / "deployment/memory_head.npz"
    head = MemoryHead.load(head_path)
    closed, opened, scores = head.predict(raw[cal])
    correct = closed == y[cal]
    per_class = []
    for cls in head.classes:
        keep = y[cal] == cls
        per_class.append(dict(species=str(cls), bank_images=int(np.sum(bank & (y == cls))),
                              calibration_images=int(keep.sum()),
                              closed_accuracy=float(correct[keep].mean()) if keep.any() else None,
                              false_rejection_rate=float((opened[keep] == "Unknown").mean()) if keep.any() else None))
    checkpoint = load_memory_checkpoint(head_path)
    runtime = MemoryDinoV2Classifier(checkpoint)
    # Cover both bank/calibration, all classes, and the rejection boundary.
    sample = np.unique(np.concatenate((np.linspace(0, len(rows)-1, 128, dtype=int),
        np.array([np.flatnonzero(y == c)[0] for c in head.classes]),
        np.flatnonzero(cal)[np.argsort(np.abs(scores-head.threshold))[:32]])))
    expected_closed, expected_open, expected_scores = head.predict(raw[sample])
    actual = runtime.classify_features(raw[sample])
    np.testing.assert_array_equal(expected_closed, [p.best_known_species for p in actual])
    np.testing.assert_array_equal(expected_open, [p.species for p in actual])
    np.testing.assert_allclose(expected_scores, [p.known_score for p in actual], atol=2e-6, rtol=0)
    report = dict(status="passed", classes=len(head.classes), sequence_samples=len(rows),
        bank_images=int(bank.sum()), calibration_images=int(cal.sum()),
        calibration_covered_classes=len(set(y[cal])), calibration_missing_species=manifest["calibration_missing_species"],
        calibration_closed_accuracy=float(correct.mean()),
        calibration_balanced_closed_accuracy=float(np.mean([r["closed_accuracy"] for r in per_class if r["closed_accuracy"] is not None])),
        calibration_correct_and_accepted=float((correct & (opened != "Unknown")).mean()),
        calibration_false_rejection_rate=float((opened == "Unknown").mean()), threshold=head.threshold,
        config=asdict(head.config), gradient_updates=0,
        sequence_partition_isolation=True, camera_partition_isolation=True,
        runtime_parity_samples=len(sample), runtime_max_score_difference=float(np.max(np.abs(expected_scores-np.array([p.known_score for p in actual])))),
        head_sha256=hashlib.sha256(head_path.read_bytes()).hexdigest(),
        limitation="Calibration diagnostics only; no independent test or unknown-species evaluation. Sparse initial classes retained; single-acquisition classes do not meet new-class admission requirements.")
    (OUT / "deployment/evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "deployment/per_class_calibration.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    assignments = [dict(id=r["id"], species=r["species"], camera=r["camera"], acquisition_id=r["acquisition_id"],
                        split="bank" if b else "calibration") for r, b in zip(rows, bank)]
    (OUT / "partition.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in assignments), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
