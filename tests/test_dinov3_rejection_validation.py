import json

import numpy as np
import pytest

from system.dinov3.rejection_validation import select_rejection_controls, evaluate_rejection_controls


def row(identity, label, camera, sequence):
    return dict(id=identity, species=label, camera=camera, sequence=sequence,
                sha256=identity, image_id=identity, source_sha256="", file=f"{identity}.jpg")


def fixture_rows():
    rows = [row(f"A-{camera}-{seq}", "A", camera, f"s{seq}") for camera in ("c1", "c2", "c3") for seq in range(3)]
    rows += [row(f"B-{i}", "B", "c4", "single-event") for i in range(3)]
    rows += [row("U-1", "U", "c5", "u1"), row("U-2", "U", "c5", "u2")]
    rows += [row("bank-A", "A", "bank-camera", "bank"), row("evaluation-B", "B", "eval-camera", "eval")]
    return rows


def test_selection_is_reproducible_balanced_and_uses_only_heldout_independent_photos():
    rows = fixture_rows()
    cal = np.arange(14)
    selection = select_rejection_controls(rows, cal, ["A", "B"], per_species=3)
    assert selection == select_rejection_controls(rows, cal[::-1], ["B", "A"], per_species=3)
    assert selection["known_counts_per_species"] == {"A": 3, "B": 1}
    assert selection["missing_known_species"] == []
    assert selection["auxiliary_unknown_indices"] == [12, 13]
    a = [i for i in selection["known_indices"] if rows[i]["species"] == "A"]
    assert len({rows[i]["camera"] for i in a}) == 3
    assert set(selection["known_indices"]) <= set(cal)
    assert {rows[i]["species"] for i in selection["known_indices"]} == {"A", "B"}
    assert not selection["selected_using_scores"]


def test_known_controls_measure_false_rejection_without_becoming_unknown_targets():
    rows = fixture_rows()
    cal = np.arange(14)
    selection = select_rejection_controls(rows, cal, ["A", "B"], per_species=3)
    scores = np.ones(len(cal))
    scores[selection["known_indices"][0]] = -1
    scores[selection["known_indices"][-1]] = -1
    scores[12:] = -1
    predictions = np.asarray([r["species"] for r in rows[:14]])
    report, samples = evaluate_rejection_controls(rows, cal, selection, predictions, scores >= 0, scores, 0.)
    assert report["known_photos"] == 4 and report["known_rejected_photos"] == 2
    assert report["known_macro_false_rejection"] == pytest.approx(2 / 3)
    assert report["known_micro_false_rejection"] == .5
    assert report["auxiliary_unknown_macro_rejection"] == 1.
    assert report["threshold_changed"] is False
    assert report["full_calibration_known_photos"] == 12
    assert all(r["expected_accept"] for r in samples if r["species"] in {"A", "B"})
    with pytest.raises(ValueError, match="threshold"):
        evaluate_rejection_controls(rows, cal, selection, predictions, np.ones(len(cal), dtype=bool), scores, 0.)
    invalid = dict(selection, known_indices=[14])
    with pytest.raises(ValueError, match="held-out"):
        evaluate_rejection_controls(rows, cal, invalid, predictions, scores >= 0, scores, 0.)


def test_default_takes_up_to_fifty_independent_events_and_reports_sparse_species():
    rows = [row(f"A-{i}", "A", "camera", f"event-{i}") for i in range(80)]
    rows += [row("B-only", "B", "sparse-camera", "sparse-event")]
    selection = select_rejection_controls(rows, np.arange(len(rows)), ["A", "B"])
    assert selection["requested_known_photos_per_species"] == 50
    assert selection["known_counts_per_species"] == {"A": 50, "B": 1}
    report, _ = evaluate_rejection_controls(rows, np.arange(len(rows)), selection,
        np.asarray([r["species"] for r in rows]), np.ones(len(rows), dtype=bool), np.ones(len(rows)), 0.)
    assert report["known_species_below_requested_count"] == ["B"]
    assert report["full_calibration_known_photos"] == len(rows)


def test_training_freezes_rejection_controls_before_model_fit(monkeypatch, tmp_path):
    from scripts.train_reviewed_dinov3 import fit
    from system.dinov3.model import DinoV3RidgeWithinModel
    from system.dinov3.encoder import sha256
    rows = [row(f"{label}-{camera}-{event}", label, f"c{camera}", f"s{event}")
            for label in ("A", "B") for camera in range(4) for event in range(4)]
    rows += [row("unknown", "U", "cu", "su")]
    manifest = tmp_path / "reviewed.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf8")
    np.save(tmp_path / "features.npy", np.ones((len(rows), 4), dtype=np.float32))
    audit = dict(known_classes=["A", "B"], unknown_classes=["U"], manifest_sha256=sha256(manifest))
    (tmp_path / "input_audit.json").write_text(json.dumps(audit), "utf8")
    (tmp_path / "feature_verification.json").write_text(json.dumps(dict(passed=True,
        features_sha256=sha256(tmp_path / "features.npy"), manifest_sha256=sha256(manifest), encoder_sha256="a" * 64)), "utf8")
    def inspect_fit(self, *args, **kwargs):
        selection = json.loads((tmp_path / "rejection_validation/selection.json").read_text("utf8"))
        partition = json.loads((tmp_path / "partition.json").read_text("utf8"))["indices"]
        selected = set(selection["known_indices"])
        assert selected <= set(partition["calibration"])
        assert not selected & (set(partition["bank"]) | set(partition["evaluation"]))
        raise RuntimeError("selection verified before fit")
    monkeypatch.setattr(DinoV3RidgeWithinModel, "fit", inspect_fit)
    with pytest.raises(RuntimeError, match="verified before fit"):
        fit(tmp_path / "source", tmp_path)
