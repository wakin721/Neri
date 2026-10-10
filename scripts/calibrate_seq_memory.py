"""Rebuild the reviewed sequence-balanced head with <10-image species as Unknown.

Use original reviewed-image counts for class roles, fixed prior scoring config,
and Val90 image-micro rejection on all 123 held-out unknown crops. Never reuse
their exemplars, centroids or contribution to the previous feature center.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
SEED = 20260923
OUT = SOURCE = PRIOR = REVIEWED = CATALOG = None


def configure(args):
    global OUT, SOURCE, PRIOR, REVIEWED, CATALOG
    global select_verified_bank, validate_partition, MemoryHead, camera_selection_split
    global load_memory_checkpoint, MemoryDinoV2Classifier
    OUT = args.output_dir.resolve()
    if OUT.exists():
        raise FileExistsError(f"Use a new output directory: {OUT}")
    SOURCE = args.source_root.resolve()
    PRIOR = args.prior_dir.resolve()
    REVIEWED = SOURCE / 'data/by_species/00已校验'
    CATALOG = SOURCE / 'data/sequence_recovered_20261006/verified_crops.jsonl'
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(PROJECT), str(SOURCE)]
    from pipeline.sequence_rules import select_verified_bank, validate_partition
    from training.training_free_memory_head import MemoryHead, camera_selection_split
    from system.dinov2.memory_checkpoint import load_memory_checkpoint
    from system.dinov2.memory_classifier import MemoryDinoV2Classifier
    OUT.mkdir(parents=True)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), 'utf8')


def jsonl(path, values):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values), 'utf8')


def read(path):
    return json.loads(Path(path).read_text('utf8'))


def prepare():
    if (OUT / 'protocol.json').exists():
        return
    verified = [json.loads(s) for s in CATALOG.read_text('utf8').splitlines() if s.strip()]
    files = {p.stem: p for p in REVIEWED.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    assert len(files) == len(verified) == 10564
    assert set(files) == {r['id'] for r in verified}
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(zip(files, pool.map(sha, files.values())))
    for row in verified:
        file = files[row['id']]
        assert hashes[row['id']] == row['sha256']
        assert file.relative_to(REVIEWED).parts[0] == row['species']
        row['file'] = row['review_file'] = str(file)
    counts = Counter(r['species'] for r in verified)
    unknown_classes = {c for c,n in counts.items() if n < 10}
    known_classes = set(counts) - unknown_classes
    unknown = [r for r in verified if r['species'] in unknown_classes]
    assert len(known_classes) == 42 and len(unknown_classes) == 43 and len(unknown) == 123
    keys = ('id','image_id','sha256','source_sha256','burst_id','acquisition_id')
    forbidden = {key:{str(r[key]) for r in unknown if r.get(key)} for key in keys}
    unknown_cameras = {r['camera'] for r in unknown}
    selected = select_verified_bank([r for r in verified if r['species'] in known_classes])
    # Entire shared original/temporal groups are held out, regardless of crop label.
    free = [r for r in selected if not any(r.get(key) and str(r[key]) in forbidden[key] for key in keys)]
    independent_classes = {r['species'] for r in free if r['camera'] not in unknown_cameras}
    exceptions = sorted(known_classes - independent_classes)
    assert exceptions == ['赤麻鸭']
    # Only this class can train on its sole shared camera; all other known species
    # from the 61 unknown cameras remain in calibration. No calibration scores
    # participate in this metadata-only partition decision.
    eligible = [r for r in free if r['camera'] not in unknown_cameras or r['species'] in exceptions]
    y = np.array([r['species'] for r in eligible])
    cameras = np.array([r['camera'] for r in eligible])
    bank_mask, _ = camera_selection_split(y, cameras, seed=SEED)
    bank_ids = {r['id'] for r,b in zip(eligible,bank_mask) if b}
    bank = [r for r in selected if r['id'] in bank_ids]
    known_cal = [r for r in selected if r['id'] not in bank_ids]
    assert set(r['species'] for r in bank) == known_classes
    all_rows = bank + known_cal + unknown
    assignments = {r['id']:'bank' if r['id'] in bank_ids else 'calibration' for r in all_rows}
    validate_partition(all_rows, assignments)
    overlap_cameras = sorted({r['camera'] for r in bank} & {r['camera'] for r in known_cal+unknown})
    assert len(overlap_cameras) == 1
    assert {r['species'] for r in bank if r['camera'] in overlap_cameras} == {'赤麻鸭'}
    for row in all_rows:
        row['role'] = ('bank' if row['id'] in bank_ids else
                       'calibration_unknown' if row['species'] in unknown_classes else 'calibration_known')
        row['source_image_id'] = row['image_id']
        row['image_id'] = row['id']
    jsonl(OUT / 'partition.jsonl', all_rows)
    roles = [dict(species=c, reviewed_images=counts[c], role='Unknown' if c in unknown_classes else 'Known',
                  bank_sequences=sum(r['species']==c for r in bank),
                  calibration_images=sum(r['species']==c for r in known_cal+unknown)) for c in sorted(counts)]
    with (OUT / 'class_roles.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(roles[0])); writer.writeheader(); writer.writerows(roles)
    protocol = dict(version=1, seed=SEED, reviewed_root=str(REVIEWED), reviewed_images=len(verified),
        count_unit='original verified crop images before sequence selection; strictly less than10 is Unknown',
        known_classes=sorted(known_classes), unknown_classes=sorted(unknown_classes),
        reviewed_class_counts=dict(sorted(counts.items())), bank_images=len(bank),
        calibration_known_images=len(known_cal), calibration_unknown_images=len(unknown),
        calibration_unknown_acquisitions=len({r['acquisition_id'] for r in unknown}),
        partition='Reserve all Unknown images and shared original/temporal groups; reserve all unknown cameras except the sole-camera class 赤麻鸭. Additional Known calibration cameras use the fixed seed.',
        camera_exception_species=exceptions, overlapping_camera_ids=overlap_cameras,
        image_content_original_burst_acquisition_isolation=True, full_camera_isolation=False,
        scoring='Existing Seq Memory config and runtime knownness (best class score plus margin). Individual verified-crop calibration; original-image micro weights, not species macro or adaptive Within-Seq scoring.',
        threshold_rule='Val90: ceil(0.90*n) ascending unknown scores; strict score<threshold rejection with boundary ties included.',
        known_frr_constraint=None, selection_by_scores=False,
        unknown_previously_in_old_head=True, rebuild='Fit center, exemplars and centroids afresh from42 Known classes only; previous85-class head never scores the new calibration.',
        source_catalog_sha256=sha(CATALOG), old_model_sha256=sha(PRIOR / 'deployment/memory_head.npz'),
        old_calibration_kept_as_backup=True, limitation='Calibration diagnostics only; no independent test.123 images/117 acquisition proxies,43 sparse species; shared-camera exception explicitly retained.')
    save(OUT / 'protocol.json', protocol)
    print(json.dumps({k:protocol[k] for k in ['bank_images','calibration_known_images','calibration_unknown_images','camera_exception_species']},ensure_ascii=False), flush=True)


def extract():
    if (OUT / 'features.npz').exists():
        return
    rows = [json.loads(s) for s in (OUT / 'partition.jsonl').read_text('utf8').splitlines()]
    previous = read(PRIOR / 'features/snapshot.json')
    with np.load(PRIOR / 'features/features.npz',allow_pickle=False) as z:
        old = z['features'].copy()
        np.testing.assert_array_equal(z['ids'],[r['image_id'] for r in previous['rows']])
    by_hash = {r['sha256']:i for i,r in enumerate(previous['rows'])}
    features = np.empty((len(rows),768),dtype=np.float32)
    missing = []
    for i,row in enumerate(rows):
        if row['sha256'] in by_hash:
            features[i] = old[by_hash[row['sha256']]]
        else:
            missing.append(i)
    print(f'Feature reuse {len(rows)-len(missing)}; extract {len(missing)}',flush=True)
    encoder = SOURCE / 'models/dinov2-base'
    assert sha(encoder / 'model.safetensors') == previous['encoder_sha256']
    assert previous['preprocessing'] == 'letterbox224_imagenet'
    device='cache-only'
    if missing:
        import torch
        from transformers import AutoModel
        from training.dinov2_common import CropDataset
        torch.set_num_threads(4)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=AutoModel.from_pretrained(encoder,local_files_only=True).to(device).eval().requires_grad_(False)
        loader=torch.utils.data.DataLoader(CropDataset([rows[i] for i in missing]),batch_size=16,shuffle=False,num_workers=0)
        written=0
        with torch.inference_mode():
            for batch in loader:
                x=model(pixel_values=batch.to(device)).last_hidden_state[:,0]
                x=torch.nn.functional.normalize(x.float(),dim=1).cpu().numpy()
                features[missing[written:written+len(x)]]=x
                written+=len(x)
        assert written==len(missing)
    assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-4)
    np.savez_compressed(OUT / 'features.npz',features=features,ids=np.array([r['id'] for r in rows]))
    save(OUT / 'extraction_report.json',dict(images=len(rows),reused=len(rows)-len(missing),extracted=len(missing),device=device,encoder_sha256=previous['encoder_sha256'],preprocessing=previous['preprocessing']))


def fit():
    destination=OUT / 'deployment'
    destination.mkdir(exist_ok=True)
    protocol=read(OUT / 'protocol.json')
    rows=[json.loads(s) for s in (OUT / 'partition.jsonl').read_text('utf8').splitlines()]
    with np.load(OUT / 'features.npz',allow_pickle=False) as z:
        features=z['features'].copy()
        np.testing.assert_array_equal(z['ids'],[r['id'] for r in rows])
    labels=np.array([r['species'] for r in rows])
    camera=np.array([r['camera'] for r in rows])
    bank=np.array([r['role']=='bank' for r in rows])
    known=np.array([r['role']=='calibration_known' for r in rows])
    unknown=np.array([r['role']=='calibration_unknown' for r in rows])
    old=MemoryHead.load(PRIOR / 'deployment/memory_head.npz')
    head=MemoryHead(features[bank],labels[bank],camera[bank],old.config)
    assert set(head.classes)==set(protocol['known_classes'])
    assert not set(head.labels) & set(protocol['unknown_classes'])
    head.provenance=dict(old.provenance, class_role_protocol_sha256=sha(OUT / 'protocol.json'),
                         source_partition_sha256=sha(OUT / 'partition.jsonl'), calibration_point='Val90',
                         rebuilding_from_known_only=True)
    head.threshold=old.threshold # Temporary score-only checkpoint; never installed.
    head.calibration=dict(images=int(known.sum()+unknown.sum()),purpose='temporary score extraction')
    provisional=destination / 'score_only_head.npz'
    head.save(provisional)
    runtime=MemoryDinoV2Classifier(load_memory_checkpoint(provisional))
    cal_indices=np.flatnonzero(known|unknown)
    # Fixed blocks reduce memory and mirror the software's canonical scoring.
    predictions=[]
    with threadpool_limits(4):
        for start in range(0,len(cal_indices),128):
            predictions.extend(runtime.classify_features(features[cal_indices[start:start+128]]))
    scores=np.array([p.known_score for p in predictions],dtype=np.float64)
    targets=labels[cal_indices]
    unknown_mask=unknown[cal_indices]
    us=scores[unknown_mask]
    rank=int(np.ceil(.90*len(us)))
    boundary=float(np.sort(us)[rank-1])
    strict_boundary_threshold=float(np.nextafter(boundary,np.inf))
    # A float32-compatible guard above the boundary avoids scalar downcast and
    # batched BLAS roundoff at the exact cutoff; it may reject additional ties.
    head.threshold=float(np.nextafter(np.float32(boundary),np.float32(np.inf),dtype=np.float32))
    assert head.threshold>boundary
    rejected=scores<head.threshold
    assert float(rejected[unknown_mask].mean())>=.90
    closed=np.array([p.best_known_species for p in predictions])
    correct=closed==targets
    head.calibration=dict(images=len(cal_indices),point='Val90',target_unknown_micro_recall=.90,
        unknown_images=len(us),unknown_species=len(protocol['unknown_classes']),
        known_images=int(known.sum()), unknown_rejection_recall=float(rejected[unknown_mask].mean()),
        empirical_known_frr=float(rejected[~unknown_mask].mean()),
        empirical_frr=float(rejected[~unknown_mask].mean()),
        finite_sample_rank=rank,order_statistic_boundary=boundary, strict_float64_threshold=strict_boundary_threshold,
        threshold=head.threshold, floating_guard='next float32 above boundary; verify deployed runtime accepts identical decisions',
        count_unit='verified crop images; image micro, all low-count species excluded from training',
        camera_isolation=False,camera_exception_species=protocol['camera_exception_species'],
        acquisition_original_image_content_isolation=True,independent_test=False)
    final=destination / 'memory_head.npz'
    head.save(final)
    deployed=MemoryDinoV2Classifier(load_memory_checkpoint(final))
    replay=[]
    with threadpool_limits(4):
        # Deliberately change batch boundaries to exercise numerical stability.
        for start in range(0,len(cal_indices),73):
            replay.extend(deployed.classify_features(features[cal_indices[start:start+73]]))
    replay_scores=np.array([p.known_score for p in replay])
    decisions=np.array([not p.accepted for p in replay])
    np.testing.assert_array_equal(rejected,decisions)
    np.testing.assert_array_equal(closed,[p.best_known_species for p in replay])
    np.testing.assert_allclose(scores,replay_scores,rtol=0,atol=2e-6)
    threshold=head.threshold
    report=dict(status='passed',known_classes=len(head.classes),unknown_classes=len(protocol['unknown_classes']),
        bank_images=int(bank.sum()), calibration_known_images=int(known.sum()),calibration_unknown_images=len(us),
        threshold=threshold, unknown_rejected=int(decisions[unknown_mask].sum()),
        unknown_micro_recall=float(decisions[unknown_mask].mean()),
        unknown_macro_recall=float(np.mean([decisions[targets==c].mean() for c in protocol['unknown_classes']])),
        known_false_rejection_rate=float(decisions[~unknown_mask].mean()),
        known_closed_accuracy=float(correct[~unknown_mask].mean()),
        known_correct_and_accepted=float((correct & ~decisions)[~unknown_mask].mean()),
        config=asdict(head.config),gradient_updates=0,runtime_parity_samples=len(cal_indices),
        runtime_max_score_difference=float(np.max(np.abs(scores-replay_scores))),
        known_calibration_covered_classes=len(set(targets[~unknown_mask])),
        camera_exception_species=protocol['camera_exception_species'],
        classifier_sha256=sha(final),limitation=protocol['limitation'])
    save(destination / 'evaluation.json',report)
    manifest=dict(classes=head.classes.tolist(),config=asdict(head.config),threshold=threshold,
        provenance=head.provenance,calibration=head.calibration,head_sha256=sha(final),
        role_protocol=protocol,evaluation=report)
    save(destination / 'manifest.json',manifest)
    scored=[dict(id=rows[i]['id'],species=rows[i]['species'],role=rows[i]['role'],
                 best_known_species=str(closed[j]),knownness=float(replay_scores[j]),rejected=bool(decisions[j]))
            for j,i in enumerate(cal_indices)]
    jsonl(destination / 'calibration_scores.jsonl',scored)
    per_class=[dict(species=c,role='Unknown' if c in protocol['unknown_classes'] else 'Known',
        images=int((targets==c).sum()),rejection_rate=float(decisions[targets==c].mean())) for c in sorted(set(targets))]
    with (destination / 'per_class_calibration.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_class[0]));writer.writeheader();writer.writerows(per_class)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='Neri_plus checkout with the frozen historical reviewed dataset')
    parser.add_argument('--prior-dir', type=Path, required=True, help='Prior reviewed-sequence run with deployment/ and features/')
    parser.add_argument('--output-dir', type=Path, required=True, help='New calibration directory')
    configure(parser.parse_args())
    prepare()
    extract()
    fit()
