"""Reproducible reviewed sequence-balanced DINOv2 Memory build and verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
OUT = SOURCE = REVIEWED = CATALOG = None


def configure(args):
    global OUT, SOURCE, REVIEWED, CATALOG
    global select_verified_bank, validate_partition, MemoryHead, camera_selection_split
    global load_memory_checkpoint, MemoryDinoV2Classifier
    OUT = args.output_dir.resolve()
    if OUT.exists():
        raise FileExistsError(f"Use a new output directory: {OUT}")
    SOURCE = args.source_root.resolve()
    REVIEWED = SOURCE / 'data/by_species/00已校验'
    CATALOG = SOURCE / 'data/sequence_recovered_20261006'
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(PROJECT), str(SOURCE)]
    from pipeline.sequence_rules import select_verified_bank, validate_partition
    from training.training_free_memory_head import MemoryHead, camera_selection_split
    from system.dinov2.memory_checkpoint import load_memory_checkpoint
    from system.dinov2.memory_classifier import MemoryDinoV2Classifier


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def rows(path):
    return [json.loads(s) for s in Path(path).read_text('utf8').splitlines() if s.strip()]


def command(*args):
    subprocess.run([sys.executable, '-B', '-X', 'utf8', '-u', *map(str, args)], cwd=SOURCE, check=True)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'input_audit.json').exists():
        return
    verified = rows(CATALOG / 'verified_crops.jsonl')
    files = [p for p in REVIEWED.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    by_id = {p.stem: p for p in files}
    assert len(by_id) == len(files) and set(by_id) == {r['id'] for r in verified}
    print(f'Audit {len(files)} reviewed crops', flush=True)
    hashes = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (identity, digest) in enumerate(zip(by_id, pool.map(sha, by_id.values())), 1):
            hashes[identity] = digest
            if i % 1000 == 0:
                print(f'Hash audit {i}/{len(files)}', flush=True)
    for r in verified:
        p = by_id[r['id']]
        assert r['sha256'] == hashes[r['id']] and r['species'] == p.relative_to(REVIEWED).parts[0]
        r['file'] = r['review_file'] = str(p)
    bank = select_verified_bank(verified)
    assert {r['id'] for r in bank} == {r['id'] for r in rows(CATALOG / 'classification_bank.jsonl')}
    manifest = OUT / 'classification_bank.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in bank), encoding='utf8')
    counts = Counter(r['species'] for r in bank)
    audit = dict(status='passed', reviewed_root=str(REVIEWED), reviewed_images=len(files), verified_hashes=len(hashes),
        classes=len(counts), sequence_bank_images=len(bank), sequence_catalog=str(CATALOG),
        sequence_manifest_sha256=sha(manifest), class_counts=dict(sorted(counts.items())),
        single_acquisition_species=sorted(c for c, n in counts.items() if n < 2),
        initial_bank_policy='All verified folder species retained as trusted initial classes; single-acquisition species do not meet incremental new-class admission requirements.')
    save(OUT / 'input_audit.json', audit)
    print('Input audit passed', flush=True)


def verify():
    snapshot = read(OUT / 'features/snapshot.json')
    manifest = read(OUT / 'deployment/manifest.json')
    data = snapshot['rows']
    with np.load(OUT / 'features/features.npz', allow_pickle=False) as z:
        raw = z['features'].copy()
        np.testing.assert_array_equal(z['ids'], [r['image_id'] for r in data])
    y = np.array([r['species'] for r in data])
    cams = np.array([r['camera'] for r in data])
    bank, cal = camera_selection_split(y, cams, seed=manifest['seed'])
    validate_partition(data, {str(r['id']): 'bank' if b else 'calibration' for r, b in zip(data, bank)})
    assert not set(cams[bank]) & set(cams[cal])
    path = OUT / 'deployment/memory_head.npz'
    head = MemoryHead.load(path)
    closed, opened, scores = head.predict(raw[cal])
    runtime = MemoryDinoV2Classifier(load_memory_checkpoint(path))
    runtime_cal = runtime.classify_features(raw[cal])
    runtime_scores = np.array([p.known_score for p in runtime_cal])
    np.testing.assert_array_equal(closed, [p.best_known_species for p in runtime_cal])
    np.testing.assert_allclose(scores, runtime_scores, atol=2e-6, rtol=0)
    guard = manifest.get('numerical_stabilization')
    if not np.array_equal(opened, [p.species for p in runtime_cal]):
        accepted = scores >= head.threshold
        lower = float(max(scores[~accepted].max(), runtime_scores[~accepted].max()))
        upper = float(min(scores[accepted].min(), runtime_scores[accepted].min()))
        assert lower < upper
        original = head.threshold
        head.threshold = (lower + upper) / 2
        assert head.threshold <= original
        guard = dict(original_rank_threshold=original, final_threshold=head.threshold,
            rejected_score_upper_bound=lower, accepted_score_lower_bound=upper,
            method='midpoint preserving every original calibration acceptance decision across research and Neri runtime')
        head.calibration['numerical_stabilization'] = guard
        shutil.copy2(path, path.with_name('memory_head.rank_cutoff.npz'))
        temporary = path.with_name('memory_head.stabilized.npz')
        head.save(temporary)
        temporary.replace(path)
        manifest.update(threshold=head.threshold, calibration=head.calibration, head_sha256=sha(path), numerical_stabilization=guard)
        save(OUT / 'deployment/manifest.json', manifest)
        runtime = MemoryDinoV2Classifier(load_memory_checkpoint(path))
        new_closed, new_opened, scores = head.predict(raw[cal])
        np.testing.assert_array_equal(opened, new_opened)
        runtime_cal = runtime.classify_features(raw[cal])
        np.testing.assert_array_equal(new_opened, [p.species for p in runtime_cal])
        closed, opened = new_closed, new_opened
        print(json.dumps(dict(numerical_stabilization=guard), ensure_ascii=False), flush=True)
    sample = np.unique(np.concatenate((np.linspace(0, len(data)-1, 128, dtype=int),
        np.array([np.flatnonzero(y == c)[0] for c in head.classes]),
        np.flatnonzero(cal)[np.argsort(np.abs(scores-head.threshold))[:32]])))
    ec, eo, es = head.predict(raw[sample])
    actual = runtime.classify_features(raw[sample])
    np.testing.assert_array_equal(ec, [p.best_known_species for p in actual])
    np.testing.assert_array_equal(eo, [p.species for p in actual])
    np.testing.assert_allclose(es, [p.known_score for p in actual], atol=2e-6, rtol=0)
    correct = closed == y[cal]
    per_class = []
    for cls in head.classes:
        keep = y[cal] == cls
        per_class.append(dict(species=str(cls), bank_images=int(np.sum(bank & (y == cls))), calibration_images=int(keep.sum()),
            closed_accuracy=float(correct[keep].mean()) if keep.any() else None,
            false_rejection_rate=float((opened[keep] == 'Unknown').mean()) if keep.any() else None))
    report = dict(status='passed', classes=len(head.classes), sequence_samples=len(data), bank_images=int(bank.sum()),
        calibration_images=int(cal.sum()), calibration_covered_classes=len(set(y[cal])),
        calibration_missing_species=manifest['calibration_missing_species'],
        calibration_closed_accuracy=float(correct.mean()),
        calibration_balanced_closed_accuracy=float(np.mean([r['closed_accuracy'] for r in per_class if r['closed_accuracy'] is not None])),
        calibration_correct_and_accepted=float((correct & (opened != 'Unknown')).mean()),
        calibration_false_rejection_rate=float((opened == 'Unknown').mean()), threshold=head.threshold,
        config=asdict(head.config), gradient_updates=0, sequence_partition_isolation=True, camera_partition_isolation=True,
        runtime_parity_samples=len(sample), runtime_calibration_parity_samples=int(cal.sum()),
        runtime_max_score_difference=float(np.max(np.abs(es-np.array([p.known_score for p in actual])))),
        numerical_stabilization=guard, head_sha256=sha(path),
        limitation='Calibration diagnostics only; no independent test or unknown-species evaluation. Sparse initial classes retained; single-acquisition classes do not meet new-class admission requirements.')
    save(OUT / 'deployment/evaluation.json', report)
    with (OUT / 'deployment/per_class_calibration.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class[0])); writer.writeheader(); writer.writerows(per_class)
    assignments = [dict(id=r['id'], species=r['species'], camera=r['camera'], acquisition_id=r['acquisition_id'],
                        split='bank' if b else 'calibration') for r, b in zip(data, bank)]
    (OUT / 'partition.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in assignments), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    prepare()
    if not (OUT / 'features/features.npz').exists():
        command('-m', 'training.extract_incremental_memory_features', '--data', REVIEWED,
            '--manifest', OUT / 'classification_bank.jsonl', '--previous', SOURCE / 'runs/memory_head_verified_20260927/features',
            '--encoder', SOURCE / 'models/dinov2-base', '--out', OUT / 'features', '--batch-size', 16)
    if not (OUT / 'deployment/memory_head.npz').exists():
        command('-m', 'training.export_verified_standard_memory', '--features', OUT / 'features',
            '--config-head', SOURCE / 'runs/training_free_head_20260921/deployment/memory_head.npz',
            '--out', OUT / 'deployment', '--target-frr', .04, '--seed', 20260923)
    with threadpool_limits(limits=4):
        verify()
    print(f'COMPLETE {OUT}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='Neri_plus checkout with the reviewed dataset, training modules and encoder')
    parser.add_argument('--output-dir', type=Path, required=True, help='New experiment directory; existing directories are refused')
    configure(parser.parse_args())
    main()
