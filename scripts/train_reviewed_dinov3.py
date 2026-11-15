"""Audit reviewed crops, extract frozen DINOv3 features, and train the joint model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from system.dinov3.encoder import DinoV3Encoder, PREPROCESSING, sha256
from system.dino.preprocess import preprocess_image
from system.dinov3.experiment import PRODUCTION_DISPLAY_NAME

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf8")
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=True)
    reviewed = source / "data/by_species/00已校验"
    manifest = out / "inputs.jsonl"
    if manifest.exists():
        audit = json.loads((out / "input_audit.json").read_text("utf8"))
        if sha256(manifest) != audit["input_freeze_sha256"]:
            raise ValueError("Frozen reviewed manifest changed")
        return
    photos = sorted(p for p in reviewed.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    by_id = {p.stem: p for p in photos}
    if len(by_id) != len(photos):
        raise ValueError("Reviewed crop IDs must be unique")
    counts = Counter(p.relative_to(reviewed).parts[0] for p in photos)
    print(f"Reviewed input: {len(photos)} photos, {len(counts)} species", flush=True)
    metadata = {}
    catalog = source / "data/sequence_recovered_20261006/all_valid_crops.jsonl"
    with catalog.open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            if str(row["id"]) in by_id:
                metadata[str(row["id"])] = row
    rows = []
    for i, photo in enumerate(photos, 1):
        original = metadata.get(photo.stem, {})
        relative = photo.relative_to(reviewed)
        camera = str(original.get("camera") or relative.parent.as_posix().split("/", 1)[-1])
        sequence = str(original.get("acquisition_id") or f"missing:{photo.stem.split('_')[0]}")
        image_id = str(original.get("image_id") or photo.stem.split("_")[0])
        # Current reviewed labels override all historical automatic labels.
        stat = photo.stat()
        rows.append(dict(id=photo.stem, file=str(photo), species=relative.parts[0],
            camera=camera, sequence=sequence, datetime=original.get("timestamp") or "",
            size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
            source_sha256=original.get("source_sha256") or "",
            image_id=image_id, metadata_recovered=bool(original)))
        if i % 10000 == 0:
            print(f"Freeze metadata {i}/{len(photos)}", flush=True)
    with manifest.open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = dict(reviewed_root=str(reviewed), reviewed_images=len(rows), species=len(counts),
        reviewed_class_counts=dict(sorted(counts.items())), known_classes=sorted(k for k, n in counts.items() if n >= 10),
        unknown_classes=sorted(k for k, n in counts.items() if n < 10),
        metadata_recovered=sum(r["metadata_recovered"] for r in rows),
        missing_timestamps=sum(not r["datetime"] for r in rows), input_freeze_sha256=sha256(manifest),
        label_source="current reviewed folder; user confirmed 45 Known / 42 auxiliary Unknown policy",
        partition_policy="fixed seed 20261107; camera preferred; acquisition/source-image/content components never cross pools")
    write_json(out / "input_audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k not in {"reviewed_class_counts", "known_classes", "unknown_classes"}}, ensure_ascii=False), flush=True)


def extract(source, out, device=None):
    import torch
    rows = read_rows(out / "inputs.jsonl")
    weights = source / "models/dinov3-vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    protocol = dict(input_freeze_sha256=sha256(out / "inputs.jsonl"), encoder_sha256=sha256(weights),
                    preprocessing=PREPROCESSING, dimension=768, batch_size=32, gradient_updates=0,
                    precision="float32; no autocast; matmul TF32 disabled; cudnn TF32 enabled")
    protocol_path = out / "extraction_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text("utf8")) != protocol:
        raise ValueError("Extraction protocol changed")
    write_json(protocol_path, protocol)
    feature_path = out / "features.npy"
    done_path = out / "feature_verification.json"
    if done_path.exists():
        verified = json.loads(done_path.read_text("utf8"))
        if sha256(feature_path) != verified["features_sha256"]:
            raise ValueError("Frozen feature cache hash mismatch")
        return
    progress = out / "extraction_progress.json"
    completed = json.loads(progress.read_text("utf8"))["completed"] if progress.exists() else 0
    torch.set_num_threads(1)
    torch.manual_seed(20261107)
    encoder = DinoV3Encoder(weights, source / "models/dinov3-code", protocol["encoder_sha256"], device=device)
    features = np.lib.format.open_memmap(feature_path, mode="r+" if feature_path.exists() else "w+",
                                        dtype=np.float32, shape=(len(rows), 768))
    digest_path = out / "image_sha256.npy"
    digests = np.lib.format.open_memmap(digest_path, mode="r+" if digest_path.exists() else "w+",
                                       dtype="S64", shape=(len(rows),))
    def decode(row):
        path = Path(row["file"])
        stat = path.stat()
        if stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError(f"Reviewed image changed after freeze: {row['file']}")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        after_stat = path.stat()
        if after_stat.st_size != stat.st_size or after_stat.st_mtime_ns != stat.st_mtime_ns:
            raise ValueError(f"Reviewed image changed during decode: {row['file']}")
        with Image.open(io.BytesIO(raw)) as image:
            tensor = preprocess_image(image)
        return tensor, digest
    if completed >= 32:
        anchor = list(map(decode, rows[:32]))
        replay = encoder.encode_preprocessed([value[0] for value in anchor])
        np.testing.assert_allclose(replay, features[:32], atol=1e-6, rtol=1e-5)
        print("Resume anchor replay passed", flush=True)
    started, last, start_count = time.monotonic(), time.monotonic(), completed
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = [pool.submit(decode, r) for r in rows[completed:completed + 32]]
        while completed < len(rows):
            decoded = [f.result() for f in pending]
            batch = [value[0] for value in decoded]
            following = completed + len(batch)
            pending = [pool.submit(decode, r) for r in rows[following:following + 32]]
            values = encoder.encode_preprocessed(batch)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite extracted feature")
            features[completed:following] = values
            digests[completed:following] = [value[1] for value in decoded]
            completed = following
            now = time.monotonic()
            if now - last >= 10 or completed == len(rows):
                features.flush()
                digests.flush()
                rate = (completed - start_count) / max(now - started, .001)
                status = dict(completed=completed, total=len(rows), images_per_second=rate,
                              elapsed_seconds=now - started, estimated_remaining_seconds=(len(rows)-completed)/max(rate,.001))
                write_json(progress, status)
                print(json.dumps(status), flush=True)
                last = now
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)
    errors = []
    for begin in sorted({0, (len(rows) // 2 // 32) * 32, max(0, ((len(rows)-1)//32)*32)}):
        decoded = list(map(decode, rows[begin:begin+32]))
        repeated = encoder.encode_preprocessed([value[0] for value in decoded])
        np.testing.assert_array_equal([value[1] for value in decoded], digests[begin:begin+len(decoded)].astype(str))
        cached = features[begin:begin+len(repeated)]
        errors.append(float(np.max(abs(repeated - cached))))
        np.testing.assert_allclose(repeated, cached, atol=1e-6, rtol=1e-5)
    manifest = out / "reviewed.jsonl"
    with manifest.open("w", encoding="utf8") as stream:
        for row, digest in zip(rows, digests):
            stream.write(json.dumps(dict(row, sha256=digest.decode("ascii")), ensure_ascii=False) + "\n")
    audit = json.loads((out / "input_audit.json").read_text("utf8"))
    audit["manifest_sha256"] = sha256(manifest)
    write_json(out / "input_audit.json", audit)
    write_json(done_path, dict(passed=True, images=len(rows), features_sha256=sha256(feature_path),
                              manifest_sha256=audit["manifest_sha256"],
                              batch_replay_max_errors=errors, **protocol))
    print("FEATURE EXTRACTION VERIFIED", flush=True)


def make_partition(rows, known_classes, seed=20261107):
    """Freeze camera-preferred pools with strict event/content/image isolation."""
    known_classes = set(known_classes)
    by_hash = defaultdict(list)
    for i, row in enumerate(rows):
        by_hash[row["sha256"]].append(i)
    retained, excluded, conflicts = [], [], []
    for digest, indices in by_hash.items():
        if len({rows[i]["species"] for i in indices}) != 1:
            conflicts.extend(indices)
        else:
            retained.append(indices[0])
            excluded.extend(indices[1:])
    retained.sort()
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for i, row in enumerate(rows):
        identities = [("sequence", row["camera"], row["sequence"]), ("image_id", row["image_id"]),
                      ("crop_sha256", row["sha256"])]
        if row.get("source_sha256"):
            identities.append(("source_sha256", row["source_sha256"]))
        for key in identities:
            if key in owner:
                parent[find(i)] = find(owner[key])
            else:
                owner[key] = i
    components = defaultdict(list)
    for i in retained:
        components[find(i)].append(i)
    groups = {k: set(rows[i]["species"] for i in ii) for k, ii in components.items()}
    protected = {k for k, labels in groups.items() if labels - known_classes}
    def rank(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    assignments = {}
    for key, ii in components.items():
        camera = min(rows[i]["camera"] for i in ii)
        value = int(rank(camera)[:8], 16) % 100
        assignments[key] = "calibration" if key in protected or value < 20 else "evaluation" if value < 30 else "bank"
    def counts(pool):
        return Counter(c for k, labels in groups.items() if assignments[k] == pool for c in labels & known_classes)
    # Every requested Known species retains a trusted Bank component. Protected
    # auxiliary-unknown components are never admitted to fitting.
    for label in sorted(known_classes):
        if not any(label in groups[k] and assignments[k] == "bank" for k in groups):
            candidates = [k for k in groups if label in groups[k] and k not in protected]
            if not candidates:
                raise ValueError(f"No leakage-free training event for requested Known class: {label}")
            assignments[min(candidates, key=lambda k: rank(str(k)))] = "bank"
    # Repair coverage only when a class has enough independent components.
    # Moving one event may share a camera; every such exception is reported.
    for pool in ("calibration", "evaluation"):
        for label in sorted(known_classes):
            if counts(pool)[label] or sum(label in values for values in groups.values()) < 3:
                continue
            bank_counts = counts("bank")
            candidates = [k for k in groups if assignments[k] == "bank" and label in groups[k]
                          and all(bank_counts[c] > 1 for c in groups[k] & known_classes)]
            if candidates:
                assignments[min(candidates, key=lambda k: rank(str(k)))] = pool
    pools = {name: [] for name in ("bank", "calibration", "evaluation")}
    for key, ii in components.items():
        pools[assignments[key]].extend(ii)
    pools = {k: np.asarray(sorted(ii), dtype=np.int64) for k, ii in pools.items()}
    if set(rows[i]["species"] for i in pools["bank"]) != known_classes:
        raise ValueError("Final Bank class coverage differs from requested Known classes")
    owners = {}
    for pool, ii in pools.items():
        for i in ii:
            row = rows[int(i)]
            keys = [("sha256", row["sha256"]), ("image", row["image_id"]),
                    ("sequence", row["camera"], row["sequence"])]
            if row.get("source_sha256"):
                keys.append(("source_sha256", row["source_sha256"]))
            for key in keys:
                if key in owners and owners[key] != pool:
                    raise ValueError(f"Partition leakage in {key[0]}")
                owners[key] = pool
    camera_sets = {k: {rows[int(i)]["camera"] for i in ii} for k, ii in pools.items()}
    exceptions = {}
    for a, b in (("bank", "calibration"), ("bank", "evaluation"), ("calibration", "evaluation")):
        shared = camera_sets[a] & camera_sets[b]
        exceptions[f"{a}__{b}"] = sorted(shared)
    info = dict(seed=seed, duplicate_crops_excluded=len(excluded), conflicting_label_crops_excluded=len(conflicts),
                conflict_ids=[rows[i]["id"] for i in conflicts], retained_images=len(retained),
                acquisition_components=len(components), camera_overlaps=exceptions,
                strict_acquisition_source_image_content_isolation=True,
                full_camera_isolation=not any(exceptions.values()),
                known_species_missing_calibration=sorted(known_classes-set(rows[int(i)]["species"] for i in pools["calibration"])),
                known_species_missing_evaluation=sorted(known_classes-set(rows[int(i)]["species"] for i in pools["evaluation"])))
    return pools, info


def freeze_rejection_controls(out, rows, pools, known_classes, per_species=50, seed=20261107, validation_out=None):
    from system.dinov3.rejection_validation import select_rejection_controls
    cal = set(map(int, pools["calibration"]))
    if cal & (set(map(int, pools["bank"])) | set(map(int, pools["evaluation"]))):
        raise ValueError("Rejection controls require calibration isolated from Bank/E")
    selection = select_rejection_controls(rows, pools["calibration"], known_classes, per_species=per_species, seed=seed)
    selection["partition_sha256"] = sha256(out / "partition.json")
    selection["reviewed_manifest_sha256"] = sha256(out / "reviewed.jsonl")
    folder = validation_out or out / "rejection_validation"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "selection.json"
    if path.exists() and json.loads(path.read_text("utf8")) != selection:
        raise ValueError("Frozen rejection controls changed; use a separate --validation-out")
    write_json(path, selection)
    return selection, folder


def check_rejection(out, per_species=50, seed=20261107, validation_out=None):
    """Backfill diagnostics from frozen C decisions, without fitting or changing gates."""
    from system.dinov3.model import DinoV3RidgeWithinModel
    from system.dinov3.rejection_validation import evaluate_rejection_controls
    from system.dinov3.runtime import read_manifest, resolve_manifest_asset
    manifest_path = out / "deployment/dinov3-reviewed.neri.json"
    manifest = read_manifest(manifest_path)
    checkpoint = resolve_manifest_asset(manifest_path, manifest["checkpoint"])
    if sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Model checkpoint hash mismatch")
    model = DinoV3RidgeWithinModel.load(checkpoint)
    rows = read_rows(out / "reviewed.jsonl")
    partition = json.loads((out / "partition.json").read_text("utf8"))
    if (sha256(out / "partition.json") != model.provenance["partition_sha256"]
            or sha256(out / "reviewed.jsonl") != model.provenance["reviewed_manifest_sha256"]):
        raise ValueError("Frozen model partition or reviewed data changed")
    pools = {k: np.asarray(v, dtype=np.int64) for k, v in partition["indices"].items()}
    selection, folder = freeze_rejection_controls(out, rows, pools, model.classes, per_species, seed, validation_out)
    with np.load(out / "calibration_replay.npz", allow_pickle=False) as replay:
        truth = np.asarray([rows[int(i)]["species"] for i in pools["calibration"]])
        np.testing.assert_array_equal(replay["labels"], truth)
        report, records = evaluate_rejection_controls(rows, pools["calibration"], selection,
            replay["predictions"], replay["accepted"], replay["known_score"], model.threshold)
        positions = {int(i): p for p, i in enumerate(pools["calibration"])}
        selected = np.asarray([positions[i] for i in selection["known_indices"]], dtype=np.int64)
        np.savez_compressed(folder / "known_acceptance_controls.npz",
            features=replay["features"][selected], adapted_features=replay["adapted_features"][selected],
            labels=replay["labels"][selected], cameras=replay["cameras"][selected],
            sequences=replay["sequences"][selected], indices=selection["known_indices"],
            expected_accept=np.ones(len(selected), dtype=bool), accepted=replay["accepted"][selected],
            known_score=replay["known_score"][selected], threshold=model.threshold,
            score_context="scores computed on complete calibration sequences, then sampled")
    report["model_checkpoint_sha256"] = manifest["checkpoint_sha256"]
    report["partition_sha256"] = selection["partition_sha256"]
    write_json(folder / "report.json", report)
    with (folder / "pool.jsonl").open("w", encoding="utf8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    with (folder / "samples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (folder / "per_species.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["per_species"][0]))
        writer.writeheader()
        writer.writerows(report["per_species"])
    print(json.dumps({k: v for k, v in report.items() if k != "per_species"}, ensure_ascii=False, indent=2), flush=True)
    return report


def fit(source, out, known_probes_per_species=50, probe_seed=20261107, validation_out=None):
    from system.dinov3.classifier import DinoV3Classifier
    from system.dinov3.context import FastCausalHistoryIndex, adapt_with_history
    from system.dinov3.deployment import publish_model
    from system.dinov3.model import DinoV3RidgeWithinModel
    rows = read_rows(out / "reviewed.jsonl")
    audit = json.loads((out / "input_audit.json").read_text("utf8"))
    verified = json.loads((out / "feature_verification.json").read_text("utf8"))
    if not verified["passed"] or sha256(out / "features.npy") != verified["features_sha256"]:
        raise ValueError("Training requires verified frozen features")
    if sha256(out / "reviewed.jsonl") != audit["manifest_sha256"] or audit["manifest_sha256"] != verified["manifest_sha256"]:
        raise ValueError("Reviewed labels/content manifest changed after feature extraction")
    raw = np.load(out / "features.npy", allow_pickle=False, mmap_mode="r")
    pools, isolation = make_partition(rows, audit["known_classes"])
    frozen = out / "partition.json"
    protocol = dict(**isolation, indices={k: ii.tolist() for k, ii in pools.items()},
                    manifest_sha256=audit["manifest_sha256"], known_classes=audit["known_classes"],
                    auxiliary_unknown_classes=audit["unknown_classes"], model_family="ridge_within_top75_radial08")
    if frozen.exists() and json.loads(frozen.read_text("utf8")) != protocol:
        raise ValueError("Frozen training partition changed")
    write_json(frozen, protocol)
    _, control_folder = freeze_rejection_controls(out, rows, pools, audit["known_classes"], known_probes_per_species, probe_seed, validation_out)
    def metadata(ii, key):
        return np.asarray([str(rows[int(i)][key]) for i in ii])
    bank, cal, evaluation = (pools[k] for k in ("bank", "calibration", "evaluation"))
    provenance = dict(encoder_sha256=verified["encoder_sha256"], preprocessing=PREPROCESSING, gradient_updates=0,
        reviewed_manifest_sha256=audit["manifest_sha256"], features_sha256=verified["features_sha256"],
        partition_sha256=sha256(frozen), training_source=str(source / "data/by_species/00已校验"),
        rejection_controls_selection_sha256=sha256(control_folder / "selection.json"),
        class_policy="reviewed crop count >=10 Known; <10 auxiliary Unknown; fixed before extraction")
    print(f"Fit joint model: Bank={len(bank)} C={len(cal)} E={len(evaluation)}", flush=True)
    model = DinoV3RidgeWithinModel(metadata=provenance).fit(raw[bank], metadata(bank,"species"),
        metadata(bank,"camera"), metadata(bank,"sequence"))
    def _crop_content_id(row):
        frame_key = str(row["source_sha256"] or row.get("image_id") or row["sha256"])
        row_id = str(row["id"])
        box_suffix = row_id.rsplit("_", 1)[1] if "_" in row_id and row_id.rsplit("_", 1)[1].isdigit() else "0"
        return f"{frame_key}:{int(box_suffix)}"
    def query(ii):
        features = raw[ii]
        records = [dict(camera_component=rows[int(i)]["camera"], seq_id=rows[int(i)]["sequence"],
                        datetime=rows[int(i)]["datetime"], content_hash=rows[int(i)]["source_sha256"] or rows[int(i)]["sha256"])
                   for i in ii]
        adapted = adapt_with_history(features, FastCausalHistoryIndex(records).build())
        crop_contents = np.asarray([_crop_content_id(rows[int(i)]) for i in ii])
        return features, adapted, metadata(ii,"camera"), metadata(ii,"sequence"), crop_contents
    c_raw, c_adapted, c_camera, c_seq, c_content = query(cal)
    calibration = model.calibrate(c_raw, metadata(cal,"species"), c_camera, c_seq, contents=c_content, adapted_features=c_adapted)
    deployment = out / "deployment"
    deployment.mkdir(exist_ok=True)
    frozen_model = out / "frozen.neri-model"
    model.save(frozen_model)
    restored = DinoV3RidgeWithinModel.load(frozen_model)
    before = model.predict(c_raw, c_camera, c_seq, contents=c_content, adapted_features=c_adapted)
    after = restored.predict(c_raw, c_camera, c_seq, contents=c_content, adapted_features=c_adapted)
    np.testing.assert_array_equal(before["predictions"], after["predictions"])
    np.testing.assert_array_equal(before["accepted"], after["accepted"])
    np.testing.assert_allclose(before["known_score"], after["known_score"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(before["box_known_score"], after["box_known_score"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(before["sequence_known_score"], after["sequence_known_score"], atol=1e-10, rtol=0)
    software = DinoV3Classifier(restored).classify_features(c_raw, camera_ids=c_camera, sequence_ids=c_seq,
                                                         content_ids=c_content, adapted_features=c_adapted)
    np.testing.assert_array_equal(after["predictions"], [p.best_known_species for p in software])
    np.testing.assert_array_equal(after["accepted"], [p.accepted for p in software])
    c_truth = metadata(cal,"species")
    c_unknown = ~np.isin(c_truth, model.classes)
    def macro(values, labels):
        return float(np.mean([values[labels == c].mean() for c in np.unique(labels)])) if len(labels) else None
    report = dict(passed=True, known_classes=len(model.classes), auxiliary_unknown_classes=len(audit["unknown_classes"]),
        reviewed_images=len(rows), bank_images=len(bank), calibration_images=len(cal), evaluation_images=len(evaluation),
        calibration_aux_unknown_images=int(c_unknown.sum()), threshold=model.threshold,
        box_threshold=model.box_threshold, sequence_threshold=model.threshold,
        calibration_unknown_macro_rejection=macro(~after["accepted"][c_unknown], c_truth[c_unknown]),
        calibration_known_macro_false_rejection=macro(~after["accepted"][~c_unknown], c_truth[~c_unknown]),
        calibration_known_macro_closed_accuracy=macro(after["predictions"][~c_unknown] == c_truth[~c_unknown], c_truth[~c_unknown]),
        runtime_calibration_parity=True, save_load_parity=True, isolation=isolation,
        limitation="E contains held-out Known acquisition components only; unknown rejection is C calibration, not an independent shifted-site test")
    report["auxiliary_unknown_species_missing_calibration"] = sorted(set(audit["unknown_classes"]) - set(c_truth[c_unknown]))
    report["calibration_auxiliary_unknown_species"] = len(np.unique(c_truth[c_unknown]))
    if len(evaluation):
        e_raw, e_adapted, e_camera, e_seq, e_content = query(evaluation)
        # Parameters and gate cutoff are already saved and frozen before E scoring.
        predicted = restored.predict(e_raw, e_camera, e_seq, contents=e_content, adapted_features=e_adapted)
        truth = metadata(evaluation,"species")
        correct = predicted["predictions"] == truth
        report.update(evaluation_known_macro_closed_accuracy=macro(correct, truth),
            evaluation_known_macro_correct_acceptance=macro(correct & predicted["accepted"], truth),
            evaluation_known_macro_false_rejection=macro(~predicted["accepted"], truth),
            evaluation_known_species=len(np.unique(truth)))
        np.savez_compressed(out / "evaluation_predictions.npz", indices=evaluation, labels=truth,
            predictions=predicted["predictions"], accepted=predicted["accepted"], known_score=predicted["known_score"],
            box_known_score=predicted["box_known_score"], box_threshold=predicted["box_threshold"],
            box_accepted=predicted["box_accepted"], sequence_known_score=predicted["sequence_known_score"],
            sequence_threshold=predicted["sequence_threshold"], sequence_accepted=predicted["sequence_accepted"])
    np.savez_compressed(out / "calibration_replay.npz", features=c_raw, adapted_features=c_adapted,
                        encoder_sha256=np.asarray(verified["encoder_sha256"]),
                        cameras=c_camera, sequences=c_seq, contents=c_content, labels=c_truth,
                        predictions=after["predictions"], accepted=after["accepted"], known_score=after["known_score"],
                        box_known_score=after["box_known_score"], box_threshold=after["box_threshold"],
                        box_accepted=after["box_accepted"], sequence_known_score=after["sequence_known_score"],
                        sequence_threshold=after["sequence_threshold"], sequence_accepted=after["sequence_accepted"])
    manifest_path = deployment / "dinov3-reviewed.neri.json"
    existing = None
    if manifest_path.exists():
        from system.dinov3.runtime import read_manifest
        existing = read_manifest(manifest_path)
    manifest = publish_model(model, manifest_path,
        encoder_weights=source/"models/dinov3-vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        encoder_code=source/"models/dinov3-code", existing_manifest=existing,
        display_name=PRODUCTION_DISPLAY_NAME, calibration_data=dict(features=c_raw, adapted_features=c_adapted,
            cameras=c_camera, sequences=c_seq, contents=c_content, labels=c_truth))
    report["model_manifest"] = str(manifest)
    write_json(out / "training_report.json", report)
    validation = check_rejection(out, known_probes_per_species, probe_seed, validation_out)
    report["rejection_validation"] = {k: v for k, v in validation.items() if k != "per_species"}
    write_json(out / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"TRAINING VERIFIED: {manifest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "extract", "fit", "check-rejection", "all"), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--known-probes-per-species", type=int, default=50)
    parser.add_argument("--probe-seed", type=int, default=20261107)
    parser.add_argument("--validation-out", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage in {"prepare", "all"}:
            prepare(args.source, args.out)
        if args.stage in {"extract", "all"}:
            extract(args.source, args.out, args.device)
        if args.stage in {"fit", "all"}:
            fit(args.source, args.out, args.known_probes_per_species, args.probe_seed, args.validation_out)
        if args.stage == "check-rejection":
            check_rejection(args.out, args.known_probes_per_species, args.probe_seed, args.validation_out)


if __name__ == "__main__":
    main()
