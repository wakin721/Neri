"""Species-balanced, held-out Known controls for checking rejection errors."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np


def select_rejection_controls(rows, calibration_indices, known_classes, *, per_species=50, seed=20261107):
    """Select at most one photo per event; spread each species across cameras.

    Selection reads identities and labels only, never features or model scores.
    Known controls remain Known and are expected to be accepted by the gate.
    """
    if isinstance(per_species, bool) or not isinstance(per_species, int) or per_species < 1:
        raise ValueError("per_species must be a positive integer")
    indices = np.asarray(calibration_indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
            or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(rows))):
        raise ValueError("calibration_indices must be unique valid row indices")
    classes = sorted(set(map(str, known_classes)))
    known = set(classes)
    def rank(*parts):
        return hashlib.sha256(json.dumps([seed, *map(str, parts)], ensure_ascii=False).encode("utf8")).hexdigest()
    candidates = defaultdict(list)
    unknown = []
    for i in map(int, indices):
        label = str(rows[i]["species"])
        if label in known:
            candidates[label].append(i)
        else:
            unknown.append(i)
    selected = []
    coverage = {}
    for label in classes:
        ordered = sorted(candidates[label], key=lambda i: (rank(label, rows[i]["id"], rows[i]["sha256"]), str(rows[i]["id"])))
        cameras = defaultdict(list)
        seen_events, seen_content, seen_images = set(), set(), set()
        for i in ordered:
            row = rows[i]
            event = (str(row["camera"]), str(row["sequence"]))
            content = str(row["sha256"])
            image = str(row.get("source_sha256") or row.get("image_id") or row["id"])
            if event in seen_events or content in seen_content or image in seen_images:
                continue
            seen_events.add(event)
            seen_content.add(content)
            seen_images.add(image)
            cameras[event[0]].append(i)
        camera_order = sorted(cameras, key=lambda camera: rank(label, camera))
        chosen = []
        while len(chosen) < per_species and any(cameras.values()):
            for camera in camera_order:
                if cameras[camera] and len(chosen) < per_species:
                    chosen.append(cameras[camera].pop(0))
        coverage[label] = len(chosen)
        selected.extend(chosen)
    return dict(schema_version=1, seed=seed, requested_known_photos_per_species=per_species,
                known_classes=classes, known_indices=selected, auxiliary_unknown_indices=sorted(unknown),
                known_counts_per_species=coverage, missing_known_species=[c for c in classes if not coverage[c]],
                selected_using_scores=False, source_pool="calibration_only",
                known_role="expected_accept", auxiliary_unknown_role="expected_reject",
                selection="camera round-robin; at most one image per event/content/source image within each species")


def evaluate_rejection_controls(rows, calibration_indices, selection, predictions, accepted, scores, threshold):
    """Audit sampled rows using scores from the complete calibration sequences."""
    indices = list(map(int, calibration_indices))
    if len(set(indices)) != len(indices):
        raise ValueError("Calibration indices must be unique")
    predictions = np.asarray(predictions)
    accepted = np.asarray(accepted, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if any(a.shape != (len(indices),) for a in (predictions, accepted, scores)):
        raise ValueError("Calibration predictions must align with calibration indices")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Finite calibration scores and threshold required")
    if not np.array_equal(accepted, scores >= threshold):
        raise ValueError("Acceptance flags must match the frozen rejection threshold")
    positions = {i: pos for pos, i in enumerate(indices)}
    classes = set(selection["known_classes"])
    controls = [(int(i), True) for i in selection["known_indices"]]
    controls += [(int(i), False) for i in selection["auxiliary_unknown_indices"]]
    if len({i for i, _ in controls}) != len(controls):
        raise ValueError("Rejection controls must be unique")
    records = []
    for i, is_known in controls:
        if i not in positions:
            raise ValueError("Rejection controls must remain in the held-out calibration pool")
        row, pos = rows[i], positions[i]
        if (str(row["species"]) in classes) != is_known:
            raise ValueError("Known controls must not be relabeled as auxiliary Unknown")
        records.append(dict(index=i, id=str(row["id"]), file=str(row.get("file", "")), species=str(row["species"]),
            camera=str(row["camera"]), sequence=str(row["sequence"]), sha256=str(row["sha256"]),
            role="known_expected_accept" if is_known else "unknown_expected_reject", expected_accept=is_known,
            predicted_species=str(predictions[pos]), accepted=bool(accepted[pos]),
            known_score=float(scores[pos]), threshold=float(threshold),
            rejection_error=bool(accepted[pos]) != is_known))
    per_species = []
    for label in sorted({r["species"] for r in records}):
        group = [r for r in records if r["species"] == label]
        is_known = label in classes
        rejected = sum(not r["accepted"] for r in group)
        full_positions = [pos for pos, i in enumerate(indices) if str(rows[i]["species"]) == label]
        full_events = {(str(rows[indices[pos]]["camera"]), str(rows[indices[pos]]["sequence"])) for pos in full_positions}
        per_species.append(dict(species=label, expected_accept=is_known, photos=len(group), rejected=rejected,
            accepted=len(group)-rejected, rejection_rate=rejected/len(group),
            false_rejection_rate=rejected/len(group) if is_known else None,
            correct_classifications=sum(r["predicted_species"] == label for r in group) if is_known else None,
            full_calibration_photos=len(full_positions), full_calibration_independent_events=len(full_events),
            full_calibration_rejection_rate=float(np.mean(~accepted[full_positions]))))
    known_species = [r for r in per_species if r["expected_accept"]]
    unknown_species = [r for r in per_species if not r["expected_accept"]]
    known_records = [r for r in records if r["expected_accept"]]
    report = dict(schema_version=1, requested_known_photos_per_species=selection["requested_known_photos_per_species"],
        seed=selection["seed"], known_photos=len(known_records), known_species=len(known_species),
        known_rejected_photos=sum(not r["accepted"] for r in known_records),
        known_macro_false_rejection=float(np.mean([r["rejection_rate"] for r in known_species])) if known_species else None,
        known_micro_false_rejection=float(np.mean([not r["accepted"] for r in known_records])) if known_records else None,
        full_calibration_known_photos=sum(r["full_calibration_photos"] for r in known_species),
        full_calibration_known_macro_false_rejection=float(np.mean([r["full_calibration_rejection_rate"] for r in known_species])) if known_species else None,
        auxiliary_unknown_photos=len(records)-len(known_records), auxiliary_unknown_species=len(unknown_species),
        auxiliary_unknown_macro_rejection=float(np.mean([r["rejection_rate"] for r in unknown_species])) if unknown_species else None,
        missing_known_species=selection["missing_known_species"], threshold=float(threshold), threshold_changed=False,
        known_species_below_requested_count=[r["species"] for r in known_species if r["photos"] < selection["requested_known_photos_per_species"]],
        score_context="complete calibration sequences and causal history before sampling; cached frozen model decisions",
        scope="calibration diagnostic controls; no threshold selection or independent test claim", per_species=per_species)
    return report, records
