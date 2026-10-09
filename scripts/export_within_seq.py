"""Export a Within-Seq Val90 head from an audited low-count species partition.

The source directory supplies protocol.json, partition.jsonl, features.npz and
extraction_report.json. Class roles use reviewed crop counts before sequence
selection. Calibration matches the desktop's independent single-frame queries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from system.dinov2.checkpoint import load_checkpoint
from system.dinov2.memory_classifier import MemoryDinoV2Classifier
from system.dinov2.within_seq import WithinSeqModel, calibration_threshold


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), 'utf8')


def export(source, output):
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source / 'protocol.json').read_text('utf8'))
    counts = protocol['reviewed_class_counts']
    known_classes = {c for c, n in counts.items() if n >= 10}
    unknown_classes = set(counts) - known_classes
    if known_classes != set(protocol['known_classes']) or unknown_classes != set(protocol['unknown_classes']):
        raise ValueError('Class roles must follow the reviewed-image <10 rule')
    rows = [json.loads(line) for line in (source / 'partition.jsonl').read_text('utf8').splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Partition crop IDs must be unique')
    with np.load(source / 'features.npz', allow_pickle=False) as archive:
        raw = archive['features'].copy()
        np.testing.assert_array_equal(archive['ids'], [r['id'] for r in rows])
    labels = np.array([r['species'] for r in rows])
    cameras = np.array([r['camera'] for r in rows])
    sequences = np.array([r['acquisition_id'] for r in rows])
    bank = np.array([r['role'] == 'bank' for r in rows])
    known = np.array([r['role'] == 'calibration_known' for r in rows])
    unknown = np.array([r['role'] == 'calibration_unknown' for r in rows])
    if not np.all(bank | known | unknown) or set(labels[bank]) != known_classes or set(labels[known]) - known_classes or set(labels[unknown]) != unknown_classes:
        raise ValueError('Invalid partition roles or species coverage')
    for species in unknown_classes:
        if np.sum(labels[unknown] == species) != counts[species]:
            raise ValueError('Every reviewed low-count image must be reserved as Unknown')
    # No held-out crop, original image, content duplicate or temporal group may
    # influence the bank, the class centers, or the whitening transform.
    for key in ('id', 'sha256', 'source_sha256', 'source_image_id', 'burst_id', 'acquisition_id'):
        bank_keys = {str(r[key]) for r, b in zip(rows, bank) if b and r.get(key)}
        cal_keys = {str(r[key]) for r, b in zip(rows, bank) if not b and r.get(key)}
        if bank_keys & cal_keys:
            raise ValueError(f'Bank/calibration overlap in {key}')
    shared_cameras = set(cameras[bank]) & set(cameras[~bank])
    shared_bank_species = set(labels[bank & np.isin(cameras, list(shared_cameras))])
    if shared_bank_species - set(protocol['camera_exception_species']):
        raise ValueError('Undeclared bank/calibration camera overlap')
    model = WithinSeqModel(raw[bank], labels[bank], cameras[bank], sequences[bank])
    extraction = json.loads((source / 'extraction_report.json').read_text('utf8'))
    provenance = dict(encoder_sha256=extraction['encoder_sha256'], preprocessing=extraction['preprocessing'],
        feature_dim=768, gradient_updates=0, rebuilding_from_known_only=True,
        class_role_protocol_sha256=sha(source / 'protocol.json'),
        source_partition_sha256=sha(source / 'partition.jsonl'), source_features_sha256=sha(source / 'features.npz'))
    metadata = dict(version=1, rejection_mode='within_seq', threshold=0.,
        config=dict(neighbors=3, centroid_weight=.5, camera_pooling=True, margin_weight=1.),
        calibration=dict(images=int(unknown.sum()), sequences=int(unknown.sum()), target_unknown_recall=.9),
        provenance=provenance)
    path = output / 'memory_head.npz'

    def save_head():
        np.savez_compressed(path, metadata=json.dumps(metadata, ensure_ascii=False),
            center=model.classification_center.astype(np.float32), features=model.memory.astype(np.float32),
            centroids=model.memory_centroids.astype(np.float32), labels=labels[bank], cameras=cameras[bank],
            classes=model.classes, raw_features=raw[bank], sequences=sequences[bank])

    save_head()
    classifier = MemoryDinoV2Classifier(load_checkpoint(path))
    indices = np.flatnonzero(known | unknown)
    predictions = classifier.classify_features(raw[indices])
    scores = np.array([p.known_score for p in predictions])
    unknown_mask = unknown[indices]
    target_labels = labels[indices]
    unknown_scores = scores[unknown_mask]
    exact_threshold = calibration_threshold(unknown_scores)
    # Include boundary ties under strict < rejection, with a representable
    # float32 guard to preserve decisions across runtime BLAS batch sizes.
    threshold = max(exact_threshold, float(np.nextafter(np.float32(exact_threshold), np.float32(np.inf))))
    rejected = scores < threshold
    closed = np.array([p.best_known_species for p in predictions])
    metadata['threshold'] = threshold
    metadata['calibration'].update(point='Val90', unknown_species=len(unknown_classes),
        empirical_aux_unknown_recall=float(rejected[unknown_mask].mean()),
        empirical_known_frr=float(rejected[~unknown_mask].mean()), known_images=int(known.sum()),
        finite_sample_rank=int(np.ceil(.9 * len(unknown_scores))),
        query_mode='independent_single_frame', original_unknown_acquisitions=len(set(zip(cameras[unknown], sequences[unknown]))),
        independent_test=False, exact_val90_threshold=exact_threshold,
        floating_guard='next float32 above the Val90 boundary')
    save_head()
    deployed = MemoryDinoV2Classifier(load_checkpoint(path))
    replay = []
    for start in range(0, len(indices), 73):
        replay.extend(deployed.classify_features(raw[indices[start:start + 73]]))
    replay_scores = np.array([p.known_score for p in replay])
    decisions = np.array([not p.accepted for p in replay])
    np.testing.assert_array_equal(rejected, decisions)
    np.testing.assert_array_equal(closed, [p.best_known_species for p in replay])
    np.testing.assert_allclose(scores, replay_scores, atol=2e-6, rtol=0)
    if decisions[unknown_mask].mean() < .9:
        raise ValueError('Deployed runtime does not achieve the Val90 calibration target')
    # Sequence aggregation is a separate diagnostic, not a new threshold fit.
    grouped = deployed.classify_features(raw[unknown], camera_ids=cameras[unknown], sequence_ids=sequences[unknown])
    report = dict(method='frozen_memory_within_seq', display_name='DINOv2 Within-Seq', default_rejection='Val90',
        classes=len(model.classes), bank_images=int(bank.sum()), calibration_known_images=int(known.sum()),
        calibration_unknown_images=int(unknown.sum()), unknown_classes=len(unknown_classes), threshold=threshold,
        auxiliary_unknown_recall=float(decisions[unknown_mask].mean()), unknown_rejected=int(decisions[unknown_mask].sum()),
        known_calibration_frr=float(decisions[~unknown_mask].mean()),
        known_closed_accuracy=float(np.mean(closed[~unknown_mask] == target_labels[~unknown_mask])),
        known_correct_and_accepted=float(np.mean((closed == target_labels)[~unknown_mask] & ~decisions[~unknown_mask])),
        unknown_macro_recall=float(np.mean([decisions[target_labels == c].mean() for c in sorted(unknown_classes)])),
        sequence_unknown_recall=float(np.mean([not p.accepted for p in grouped])),
        calibration=metadata['calibration'], provenance=provenance, head_sha256=sha(path),
        class_role_policy=dict(min_known_reviewed_images=10, count_unit='verified crop images before sequence selection',
            reviewed_class_counts=counts, known_classes=sorted(known_classes), unknown_classes=sorted(unknown_classes)),
        partition_isolation=dict(original_image_content_burst_acquisition=True,
            full_camera_isolation=not bool(shared_cameras), camera_exception_species=protocol['camera_exception_species']),
        known_calibration_missing_classes=sorted(known_classes - set(target_labels[~unknown_mask])),
        runtime_replay='passed', runtime_replay_samples=len(indices),
        runtime_max_score_difference=float(np.max(np.abs(scores - replay_scores))),
        scope='Internal calibration only; no independent test. Sparse species previously belonged to the old head; all transforms are rebuilt from Known-only bank.',
        user_selected_policy='Prioritize unknown rejection with Val90; no Known FRR constraint',
        query_without_sequence='Each crop is an independent single-frame sequence; no pooling of unrelated animals.',
        feedback='New model fingerprint; historical Memory stores remain isolated and preserved.')
    save_json(output / 'manifest.json', report)
    per_class = [dict(species=c, role='Unknown' if c in unknown_classes else 'Known', images=int(np.sum(target_labels == c)),
        rejection_rate=float(decisions[target_labels == c].mean())) for c in sorted(set(target_labels))]
    with (output / 'per_class_calibration.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    print(json.dumps({k: report[k] for k in ('classes', 'bank_images', 'calibration_unknown_images', 'threshold',
        'auxiliary_unknown_recall', 'known_calibration_frr', 'sequence_unknown_recall', 'runtime_replay')}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        export(args.source_dir, args.output_dir)
