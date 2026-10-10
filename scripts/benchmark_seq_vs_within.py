"""Paired, frozen-feature Seq Memory / Within-Seq evaluation.

See --help for source datasets and a new output directory.
Writes only experiment outputs. No installed model is changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
OUT = SOURCE = LOCAL_RUN = IWILDCAM_RUN = CCT20_FEATURES = None


def configure_source(source_root):
    global SOURCE, MemoryConfig, MemoryHead
    SOURCE = source_root.resolve()
    sys.path[:0] = [str(PROJECT), str(SOURCE)]
    from training.training_free_memory_head import MemoryConfig, MemoryHead


sys.path.insert(0, str(PROJECT))
from system.dinov2.checkpoint import load_checkpoint
from system.dinov2.memory_classifier import MemoryDinoV2Classifier
from system.dinov2.within_seq import WithinSeqModel, calibration_threshold
from sklearn.metrics import roc_auc_score

SEEDS = [20261009, 20261010, 20261011]
MAIN = ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'opossum', 'rabbit',
        'raccoon', 'rodent', 'skunk', 'squirrel']
RESULTS, SWEEP, PER_CLASS, SPLITS = [], [], [], []
CHECKS = []


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_data(name, metadata, feature_path, *, local=False):
    if local:
        rows = [json.loads(s) for s in metadata.read_text('utf8').splitlines() if s]
        source_meta = json.loads((metadata.parent/'extraction_report.json').read_text('utf8'))
    else:
        source_meta = json.loads(metadata.read_text('utf8'))
        rows = source_meta['rows']
    assert source_meta['encoder_sha256'] == 'd73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841'
    with np.load(feature_path, allow_pickle=False) as z:
        features = z['features'].copy()
        ids = z['ids'].astype(str)
    np.testing.assert_array_equal(ids, [r.get('id', r['image_id']) if local else r['image_id'] for r in rows])
    assert len(set(ids)) == len(ids)
    assert np.isfinite(features).all()
    assert np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4)
    return dict(name=name, rows=rows, raw=features, ids=ids,
                labels=np.array([r['species'] for r in rows]),
                cameras=np.array([str(r.get('camera', r.get('location'))) for r in rows]),
                sequences=np.array([str(r.get('acquisition_id', r.get('seq_id'))) for r in rows]),
                roles=np.array([r.get('official_split', r['role'] if 'role' in r else '') for r in rows]),
                metadata_path=str(metadata), metadata_sha256=sha(metadata),
                features_path=str(feature_path), features_sha256=sha(feature_path),
                encoder_sha256=source_meta['encoder_sha256'])


def groups(data, indices, *, species=False):
    found = defaultdict(list)
    for i in indices:
        key = (str(data['cameras'][i]), str(data['sequences'][i]))
        if species:
            key = (str(data['labels'][i]), *key)
        found[key].append(int(i))
    return [np.array(v, dtype=int) for v in found.values()]


def capped(data, indices, cap, seed, *, representatives=False):
    """Metadata-only sampling, whole sequences, fixed per-species image ceiling."""
    rng = np.random.default_rng(seed)
    selected = []
    for c in sorted(set(data['labels'][indices])):
        local = indices[data['labels'][indices] == c]
        gg = groups(data, local, species=True)
        total = 0
        for j in rng.permutation(len(gg)):
            g = gg[j]
            if representatives:
                g = np.array([min(g, key=lambda i: data['ids'][i])])
            if total and total + len(g) > cap:
                continue
            selected.extend(g.tolist())
            total += len(g)
            if total >= cap:
                break
    return np.array(sorted(selected), dtype=int)


def remove_leakage(data, pools):
    """Remove every selected group spanning bank/cal/test; class labels never fix IDs."""
    bad = set()
    audits = {}
    # Recovered temporal groups may contain different species. Drop entire
    # query groups before comparing either method; never use truth to split a
    # query group into convenient same-species sequences. Keep the fixed bank.
    source_groups = groups(data, np.arange(len(data['raw'])))
    mixed_rows = set(int(i) for g in source_groups if len(set(data['labels'][g])) > 1 for i in g)
    mixed_query_rows = {int(i) for p, ii in pools.items() if p != 'bank' for i in ii if i in mixed_rows}
    audits['mixed_query_images'] = len(mixed_query_rows)
    bad.update(mixed_query_rows)
    for key in ['image_id', 'sha256', 'source_sha256', 'source_image_id', 'burst_id', 'acquisition_id']:
        values = defaultdict(list)
        for pool, ii in pools.items():
            for i in ii:
                value = data['rows'][i].get(key)
                if value:
                    values[str(value)].append((pool, int(i)))
        conflicts = [v for v in values.values() if len({p for p, _ in v}) > 1]
        audits[key] = len(conflicts)
        bad.update(i for v in conflicts for _, i in v)
    values = defaultdict(list)
    for pool, ii in pools.items():
        for i in ii:
            values[(data['cameras'][i], data['sequences'][i])].append((pool, int(i)))
    conflicts = [v for v in values.values() if len({p for p, _ in v}) > 1]
    audits['camera_sequence'] = len(conflicts)
    bad.update(i for v in conflicts for _, i in v)
    cleaned = {p: np.array([i for i in ii if i not in bad], dtype=int) for p, ii in pools.items()}
    assert all(len(ii) for ii in cleaned.values())
    return cleaned, dict(conflicting_groups=audits, excluded_images=len(bad))


def mean_by_sequence(data, indices, scores):
    result = scores.copy()
    local = defaultdict(list)
    for j, i in enumerate(indices):
        local[(data['cameras'][i], data['sequences'][i])].append(j)
    for jj in local.values():
        result[jj] = scores[jj].mean()
    return result


def fit_score(data, bank, query, tag):
    dest = OUT / tag
    dest.mkdir(exist_ok=True)
    cache = dest / 'scores.npz'
    fingerprint = hashlib.sha256((data['features_sha256'] + json.dumps(bank.tolist()) + json.dumps(query.tolist())
                                  + sha(PROJECT/'system/dinov2/within_seq.py')
                                  + sha(PROJECT/'system/dinov2/memory_classifier.py')).encode()).hexdigest()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if str(z['fingerprint']) == fingerprint:
                print(f'{tag}: reused verified score cache', flush=True)
                return {k: z[k].copy() for k in ['seq_pred', 'seq_score', 'within_pred', 'within_score']}
    raw, y, cams, seq = [data[k] for k in ['raw', 'labels', 'cameras', 'sequences']]
    print(f'{tag}: fitting {len(bank)} bank representatives; scoring {len(query)} images', flush=True)
    start = time.perf_counter()
    memory = MemoryHead(raw[bank], y[bank], cams[bank], MemoryConfig(1, .5, True, 1.))
    memory.threshold = 0.
    provenance = dict(preprocessing='letterbox224_imagenet', feature_dim=768, gradient_updates=0,
                      encoder_sha256=data['encoder_sha256'], source_features_sha256=data['features_sha256'])
    memory.provenance = provenance
    memory.calibration = dict(images=len(query), purpose='temporary score extraction; never deployed')
    memory.save(dest/'seq_memory.npz')
    model = WithinSeqModel(raw[bank], y[bank], cams[bank], seq[bank])
    meta = dict(version=1, rejection_mode='within_seq', threshold=0.,
                config=dict(neighbors=3, centroid_weight=.5, camera_pooling=True, margin_weight=1.),
                calibration=dict(images=len(query), sequences=len(query), target_unknown_recall=.9,
                                 purpose='temporary score extraction; never deployed'), provenance=provenance)
    np.savez_compressed(dest/'within_seq.npz', metadata=json.dumps(meta),
                        center=model.classification_center.astype(np.float32),
                        features=model.memory.astype(np.float32), centroids=model.memory_centroids.astype(np.float32),
                        labels=y[bank], cameras=cams[bank], classes=model.classes, raw_features=raw[bank], sequences=seq[bank])
    del model
    outputs = {}
    elapsed = {}
    for method, filename in [('seq', 'seq_memory.npz'), ('within', 'within_seq.npz')]:
        classifier = MemoryDinoV2Classifier(load_checkpoint(dest/filename))
        pred, scores = [], []
        method_start = time.perf_counter()
        for offset in range(0, len(query), 512):
            ii = query[offset:offset+512]
            predictions = classifier.classify_features(raw[ii])
            pred.extend(p.best_known_species for p in predictions)
            scores.extend(p.known_score for p in predictions)
        elapsed[method] = time.perf_counter() - method_start
        outputs[f'{method}_pred'] = np.array(pred)
        outputs[f'{method}_score'] = np.array(scores)
        sample = query[:min(257, len(query))]
        replay = classifier.classify_features(raw[sample])
        np.testing.assert_array_equal(outputs[f'{method}_pred'][:len(sample)], [p.best_known_species for p in replay])
        np.testing.assert_allclose(outputs[f'{method}_score'][:len(sample)], [p.known_score for p in replay], atol=2e-6, rtol=0)
        if method == 'seq':
            research_pred, _, research_scores = memory.predict(raw[sample])
            np.testing.assert_array_equal(research_pred, [p.best_known_species for p in replay])
            np.testing.assert_allclose(research_scores, [p.known_score for p in replay], atol=2e-6, rtol=0)
        else:
            pooled = classifier.classify_features(raw[sample], camera_ids=cams[sample], sequence_ids=seq[sample])
            expected = mean_by_sequence(data, sample, np.array([p.known_score for p in replay]))
            np.testing.assert_allclose(expected, [p.known_score for p in pooled], atol=2e-6, rtol=0)
        del classifier
    CHECKS[:] = [r for r in CHECKS if r['tag'] != tag]
    CHECKS.append(dict(tag=tag, bank_images=len(bank), query_images=len(query),
                       runtime_replay_samples=min(257, len(query)), runtime_parity='passed',
                       within_sequence_parity='passed', score_seconds=elapsed,
                       total_seconds=time.perf_counter()-start))
    np.savez_compressed(cache, fingerprint=fingerprint, query_indices=query, bank_indices=bank, **outputs)
    print(f'{tag}: scores and runtime replay passed in {time.perf_counter()-start:.1f}s', flush=True)
    save_json(OUT/'runtime_checks.json', CHECKS)
    return outputs


def metrics(labels, pred, score, threshold, classes, novel):
    known = np.isin(labels, list(classes))
    unknown = ~known
    rejected = score < threshold
    correct = pred == labels
    kk, uu = int(known.sum()), int(unknown.sum())
    assert kk and uu
    unseen = np.isin(labels, list(novel))
    accepted = known & ~rejected
    def macro(mask, values):
        return float(np.mean([values[mask & (labels == c)].mean() for c in sorted(set(labels[mask]))]))
    return dict(known_images=kk, unknown_images=uu, novel_unknown_images=int(unseen.sum()),
                known_species=len(set(labels[known])), unknown_species=len(set(labels[unknown])),
                known_frr=float(rejected[known].mean()), unknown_recall=float(rejected[unknown].mean()),
                novel_unknown_recall=float(rejected[unseen].mean()) if unseen.any() else None,
                known_closed_accuracy=float(correct[known].mean()),
                known_correct_accepted=float((correct & ~rejected)[known].mean()),
                accepted_known_accuracy=float(correct[accepted].mean()) if accepted.any() else None,
                known_macro_frr=macro(known, rejected), unknown_macro_recall=macro(unknown, rejected),
                known_macro_correct_accepted=macro(known, correct & ~rejected),
                auroc_unknown=float(roc_auc_score(unknown, -score)))


def camera_bootstrap_delta(data, ii, a, b, mask, seed):
    cameras = np.unique(data['cameras'][ii][mask])
    rows = [np.flatnonzero(mask & (data['cameras'][ii] == c)) for c in cameras]
    count = np.array([len(r) for r in rows])
    difference = np.array([np.sum(b[r].astype(float)-a[r].astype(float)) for r in rows])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(rows), size=(2000, len(rows)))
    values = difference[sampled].sum(1) / count[sampled].sum(1)
    low, high = np.quantile(values, [.025, .975])
    return float(low), float(high), len(cameras)


def evaluate(data, pools, classes, aux, novel, seed, split_name, scored, query, *, internal=False):
    position = {int(i): j for j, i in enumerate(query)}
    variants = [('Seq Memory', 'seq', False), ('Within-Seq single', 'within', False),
                ('Seq Memory + sequence mean (control)', 'seq', True), ('Within-Seq sequence', 'within', True)]
    if internal:
        variants = variants[:2]
    all_decisions = {}
    for variant, method, pooled in variants:
        def extract(indices):
            jj = np.array([position[int(i)] for i in indices])
            scores = scored[method+'_score'][jj]
            if pooled:
                scores = mean_by_sequence(data, indices, scores)
            return scored[method+'_pred'][jj], scores
        cp, cs = extract(pools['cal'])
        cm = np.isin(data['labels'][pools['cal']], list(aux))
        assert cm.any()
        target_thresholds = {target: calibration_threshold(cs[cm], target) for target in [.80, .85, .90, .95]}
        threshold = target_thresholds[.90]
        for domain, indices in pools.items():
            if domain in ['bank', 'cal']:
                continue
            pred, score = extract(indices)
            labels = data['labels'][indices]
            m = metrics(labels, pred, score, threshold, classes, novel)
            base = dict(dataset=data['name'], split=split_name, seed=seed, domain=domain,
                        method=variant, evaluation_kind='internal_calibration' if internal else 'held_out',
                        bank_images=len(pools['bank']), bank_classes=len(classes),
                        calibration_unknown_images=int(cm.sum()), calibration_unknown_species=len(set(data['labels'][pools['cal']][cm])),
                        threshold=threshold, calibration_unknown_recall=float(np.mean(cs[cm] < threshold)))
            ck = np.isin(data['labels'][pools['cal']], list(classes))
            base['calibration_known_frr'] = float(np.mean(cs[ck] < threshold)) if ck.any() else None
            row = base | m
            RESULTS.append(row)
            all_decisions[variant, domain] = score < threshold
            for target, cutoff in target_thresholds.items():
                SWEEP.append(base | dict(target=target, threshold=cutoff) | metrics(labels, pred, score, cutoff, classes, novel))
            for c in sorted(set(labels)):
                mask = labels == c
                PER_CLASS.append(dict(dataset=data['name'], split=split_name, seed=seed, domain=domain, method=variant,
                                      species=c, role='Known' if c in classes else 'Novel Unknown' if c in novel else 'Aux-class Unknown',
                                      images=int(mask.sum()), rejection_rate=float(np.mean(score[mask] < threshold)),
                                      correct_and_accepted=float(np.mean((pred[mask] == c) & (score[mask] >= threshold))) if c in classes else None))
            np.savez_compressed(OUT/split_name/(domain+'_'+method+('_sequence' if pooled else '_single')+'.npz'),
                                ids=data['ids'][indices], truth=labels, predicted=pred, knownness=score,
                                rejected=score<threshold, cameras=data['cameras'][indices], sequences=data['sequences'][indices], threshold=threshold)
    for domain, ii in pools.items():
        if domain in ['cal', 'bank']:
            continue
        known = np.isin(data['labels'][ii], list(classes))
        a = all_decisions['Seq Memory', domain]
        b = all_decisions['Within-Seq single', domain]
        low, high, ncam = camera_bootstrap_delta(data, ii, a, b, known, seed)
        for row in RESULTS:
            if row['split'] == split_name and row['domain'] == domain and row['method'] == 'Within-Seq single':
                row.update(paired_known_frr_delta=float(b[known].mean()-a[known].mean()),
                           paired_known_frr_delta_ci95_low=low, paired_known_frr_delta_ci95_high=high,
                           bootstrap_known_cameras=ncam)
    meta = dict(name=split_name, dataset=data['name'], seed=seed, classes=sorted(classes),
                auxiliary_unknown_classes=sorted(aux), novel_unknown_classes=sorted(novel),
                pool_counts={p: len(ii) for p, ii in pools.items()},
                pool_sequence_counts={p: len(groups(data, ii)) for p, ii in pools.items()},
                multi_image_sequences={p: sum(len(g)>1 for g in groups(data, ii)) for p, ii in pools.items()},
                bank_cal_camera_overlap=len(set(data['cameras'][pools['bank']]) & set(data['cameras'][pools['cal']])),
                cal_test_camera_overlap={p: len(set(data['cameras'][pools['cal']]) & set(data['cameras'][ii]))
                                         for p, ii in pools.items() if p not in ['bank', 'cal']},
                source_metadata_sha256=data['metadata_sha256'], source_features_sha256=data['features_sha256'])
    SPLITS.append(meta)
    save_json(OUT/split_name/'partition.json', meta)
    csv_write(OUT/split_name/'partition_ids.csv', [dict(id=data['ids'][i], species=data['labels'][i], camera=data['cameras'][i],
              sequence=data['sequences'][i], pool=p) for p, ii in pools.items() for i in ii])
    csv_write(OUT/'results.csv', RESULTS)
    csv_write(OUT/'threshold_sweep.csv', SWEEP)
    csv_write(OUT/'per_species.csv', PER_CLASS)
    save_json(OUT/'partitions.json', SPLITS)
    print(split_name, [(r['domain'], r['method'], round(r['known_frr']*100, 2), round(r['unknown_recall']*100, 2))
                      for r in RESULTS if r['split']==split_name], flush=True)


def camera_partition(data, candidate, seed):
    cams = np.unique(data['cameras'][candidate])
    rng = np.random.default_rng(seed)
    ordered = rng.permutation(cams)
    cal_cameras = set(ordered[:max(1, int(.4*len(cams)))])
    return np.isin(data['cameras'], list(cal_cameras))


def run_local():
    root = LOCAL_RUN
    data = load_data('Local reviewed', root/'partition.jsonl', root/'features.npz', local=True)
    bank = np.flatnonzero(data['roles']=='bank')
    query = np.flatnonzero(data['roles']!='bank')
    classes = set(data['labels'][bank])
    unknowns = sorted(set(data['labels'][query])-classes)
    scored = fit_score(data, bank, query, 'local_bank')
    aux = set(unknowns)
    # Historical common-pool reference, explicitly not an independent test.
    name = 'local_internal_reference'; (OUT/name).mkdir(exist_ok=True)
    evaluate(data, dict(bank=bank, cal=query, calibration_reference=query), classes, aux, set(),
             SEEDS[0], name, scored, query, internal=True)
    for seed in SEEDS:
        ordered = np.random.default_rng(seed).permutation(unknowns)
        aux = set(ordered[:22]); novel = set(ordered[22:])
        calcam = camera_partition(data, query, seed)
        cal = query[calcam[query] & np.isin(data['labels'][query], list(classes|aux))]
        test = query[~calcam[query]]
        pools, audit = remove_leakage(data, dict(bank=bank, cal=cal, test=test))
        assert np.array_equal(pools['bank'], bank)
        name = f'local_r{SEEDS.index(seed)+1}'; (OUT/name).mkdir(exist_ok=True)
        save_json(OUT/name/'leakage_audit.json', audit)
        evaluate(data, pools, classes, aux, novel, seed, name, scored, query)


def run_iwildcam():
    root = IWILDCAM_RUN
    metadata = json.loads((root/'manifest.json').read_text('utf8'))
    data = load_data('iWildCam', root/'manifest.json', root/'features.npz')
    classes = set(metadata['known_classes'])
    unknowns = sorted(metadata['unknown_classes'])
    bank = np.flatnonzero(data['roles']=='bank')
    query = np.flatnonzero(data['roles']!='bank')
    scored = fit_score(data, bank, query, 'iwildcam_bank')
    for seed in SEEDS:
        ordered = np.random.default_rng(seed).permutation(unknowns)
        aux, novel = set(ordered[:5]), set(ordered[5:])
        held = np.flatnonzero(data['roles']=='test')
        calcam = camera_partition(data, held, seed)
        cal = np.r_[np.flatnonzero(data['roles']=='calibration'), held[calcam[held] & np.isin(data['labels'][held], list(aux))]]
        test = held[~calcam[held]]
        pools, audit = remove_leakage(data, dict(bank=bank, cal=cal, test=test))
        assert np.array_equal(pools['bank'], bank)
        for a, b in [('bank', 'cal'), ('bank', 'test'), ('cal', 'test')]:
            assert not set(data['cameras'][pools[a]]) & set(data['cameras'][pools[b]])
        name = f'iwildcam_r{SEEDS.index(seed)+1}'; (OUT/name).mkdir(exist_ok=True)
        save_json(OUT/name/'leakage_audit.json', audit)
        evaluate(data, pools, classes, aux, novel, seed, name, scored, query)


def run_cct20():
    root = CCT20_FEATURES
    data = load_data('CCT20', root/'snapshot.json', root/'features.npz')
    for seed in SEEDS:
        ordered = np.random.default_rng(seed).permutation(sorted(MAIN))
        classes, aux, novel = set(ordered[:7]), set(ordered[7:9]), set(ordered[9:])
        roles, y = data['roles'], data['labels']
        bank = capped(data, np.flatnonzero((roles=='train') & np.isin(y, list(classes))), 500, seed, representatives=True)
        cal = capped(data, np.flatnonzero(np.isin(roles, ['cis_val', 'trans_val']) & np.isin(y, list(classes|aux))), 1000, seed)
        pools = dict(bank=bank, cal=cal)
        for domain in ['cis_test', 'trans_test']:
            pools[domain] = capped(data, np.flatnonzero((roles==domain) & np.isin(y, MAIN)), 1000, seed)
        pools, audit = remove_leakage(data, pools)
        assert set(data['labels'][pools['bank']]) == classes
        assert not set(data['cameras'][pools['bank']]) & set(data['cameras'][pools['trans_test']])
        name = f'cct20_r{SEEDS.index(seed)+1}'; (OUT/name).mkdir(exist_ok=True)
        save_json(OUT/name/'leakage_audit.json', audit)
        query = np.unique(np.concatenate([ii for p, ii in pools.items() if p!='bank']))
        scored = fit_score(data, pools['bank'], query, name)
        evaluate(data, pools, classes, aux, novel, seed, name, scored, query)


def summarize():
    aggregate = []
    grouped = defaultdict(list)
    for r in RESULTS:
        grouped[r['dataset'], r['domain'], r['method']].append(r)
    metrics_list = ['known_frr', 'unknown_recall', 'novel_unknown_recall', 'known_closed_accuracy',
                    'known_correct_accepted', 'auroc_unknown', 'known_macro_frr', 'unknown_macro_recall']
    for (dataset, domain, method), rr in grouped.items():
        row = dict(dataset=dataset, domain=domain, method=method, repetitions=len(rr),
                   known_images_min=min(r['known_images'] for r in rr), known_images_max=max(r['known_images'] for r in rr),
                   unknown_images_min=min(r['unknown_images'] for r in rr), unknown_images_max=max(r['unknown_images'] for r in rr))
        for key in metrics_list:
            vv = [r[key] for r in rr if r[key] is not None]
            row[key+'_mean'] = float(np.mean(vv)) if vv else None
            row[key+'_std'] = float(np.std(vv, ddof=1)) if len(vv)>1 else 0. if vv else None
        aggregate.append(row)
    csv_write(OUT/'summary.csv', aggregate)
    save_json(OUT/'summary.json', aggregate)
    plot(aggregate)
    report(aggregate)


def plot(aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    domains = [('Local reviewed', 'test'), ('CCT20', 'cis_test'), ('CCT20', 'trans_test'), ('iWildCam', 'test')]
    variants = ['Seq Memory', 'Within-Seq single', 'Seq Memory + sequence mean (control)', 'Within-Seq sequence']
    labels = ['Seq Memory', 'Within-Seq single', 'Seq + mean control', 'Within-Seq sequence']
    colors = ['#376798', '#d37f3a', '#7397bc', '#e4ae7e']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    for ax, metric, title in zip(axes, ['unknown_recall', 'known_frr', 'known_correct_accepted'],
                               ['Unknown recall (%)', 'Known false rejection (%)', 'Known correct & accepted (%)']):
        for j, (method, label, color) in enumerate(zip(variants, labels, colors)):
            rows = [next(r for r in aggregate if (r['dataset'], r['domain'], r['method'])==(dataset, domain, method)) for dataset, domain in domains]
            values = [r[metric+'_mean']*100 for r in rows]
            err = [r[metric+'_std']*100 for r in rows]
            ax.bar(np.arange(4)+(j-1.5)*.19, values, width=.18, yerr=err, capsize=2, label=label, color=color)
        ax.set_xticks(np.arange(4), ['Local', 'CCT20 cis', 'CCT20 trans', 'iWildCam'], rotation=20)
        ax.set_ylim(0, 100); ax.set_title(title); ax.set_ylabel('%'); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Paired frozen-feature comparison: Val90 calibration, 3 predeclared splits\nError bars: split SD; thresholds fitted separately; held-out recall may be below 90%', fontsize=12)
    fig.savefig(OUT/'comparison.png', dpi=170)
    plt.close(fig)


def report(aggregate):
    def pct(value):
        return '—' if value is None else f'{value*100:.2f}%'
    def table(rows, fields, headings):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headings)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(r.get(k, '')))+'</td>' for k in fields)+'</tr>' for r in rows)+'</tbody></table>'
    def get(dataset, domain, method):
        return next(r for r in aggregate if (r['dataset'],r['domain'],r['method']) == (dataset,domain,method))
    takeaways = []
    for dataset,domain in [('Local reviewed','test'),('CCT20','cis_test'),('CCT20','trans_test'),('iWildCam','test')]:
        a,b = get(dataset,domain,'Seq Memory'), get(dataset,domain,'Within-Seq single')
        takeaways.append(f"{dataset}/{domain}：Within-Seq 单帧相对 Seq Memory，未知召回改变 {(b['unknown_recall_mean']-a['unknown_recall_mean'])*100:+.2f} 个百分点，Known 误拒改变 {(b['known_frr_mean']-a['known_frr_mean'])*100:+.2f} 个百分点，正确且接受改变 {(b['known_correct_accepted_mean']-a['known_correct_accepted_mean'])*100:+.2f} 个百分点。")
    findings = ''.join('<p>'+html.escape(t)+'</p>' for t in takeaways)
    view = []
    for r in aggregate:
        view.append(dict(dataset=r['dataset']+'/'+r['domain'], method=r['method'], reps=r['repetitions'],
                         known=f"{r['known_images_min']}–{r['known_images_max']}", unknown=f"{r['unknown_images_min']}–{r['unknown_images_max']}",
                         recall=pct(r['unknown_recall_mean']), novel=pct(r['novel_unknown_recall_mean']),
                         frr=pct(r['known_frr_mean']), closed=pct(r['known_closed_accuracy_mean']),
                         accepted=pct(r['known_correct_accepted_mean']), auroc=f"{r['auroc_unknown_mean']:.4f}"))
    rows = table(view, ['dataset','method','reps','known','unknown','recall','novel','frr','closed','accepted','auroc'],
                 ['数据/评估池','方法','重复','Known张数','Unknown张数','未知召回','未参与校准物种召回','已知误拒','闭集准确率','正确且接受','未知AUROC'])
    split_view = [dict(split=r['name'], known=len(r['classes']), aux=len(r['auxiliary_unknown_classes']),
                       novel=len(r['novel_unknown_classes']), pools=json.dumps(r['pool_counts']),
                       multi=json.dumps(r['multi_image_sequences']), overlap=json.dumps(r['cal_test_camera_overlap'])) for r in SPLITS]
    detail = table(split_view, ['split','known','aux','novel','pools','multi','overlap'],
                   ['划分','Known类','辅助未知类','新未知类','各池图片数','各池多图序列数','校准/测试共用相机数'])
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Seq Memory 与 Within-Seq 多组数据对比</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#243341}}h1{{font-size:26px}}h2{{font-size:21px}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}}th,td{{padding:7px 8px;text-align:left;border-bottom:1px solid #dce2e7}}th{{background:#f0f3f6}}img{{width:100%;height:auto}}.note{{background:#f3f6f8;padding:16px}}a{{color:#245f98}}</style>
<h1>Seq Memory 与 Within-Seq：多组冻结特征对照</h1>
<p>同一 DINOv2-B/14 编码器，同一建库图片；每种方法独立按辅助未知集校准 Val90。比较本地已校验、CCT20 同域/跨相机、iWildCam；预先确定 3 个种子 {SEEDS}，不按测试成绩选模型。主要结果是三次划分的等权均值，完整逐次结果与阈值见 CSV。</p>
<p>结果依赖数据域。本地和 CCT20 上，Within-Seq 单帧提高未知拒识，同时增加 Known 误拒；iWildCam 上，未知召回接近，Within-Seq 的 Known 误拒较低。CCT20 的真实多帧序列平均改善两种方法的 AUROC 和多数工作点；本地/iWildCam 的序列证据有限，不外推连拍收益。</p>
{findings}
<img src="comparison.png" alt="四个数据池的未知召回、已知误拒和正确接受率对比">
{rows}
<h2>公平性与算法范围</h2>
<p>Seq Memory 使用当前旧头的 k=1、类别质心权重 0.5、相机池化、margin=1。Within-Seq 分类固定 k=3，拒识由 Known 专用层次等权类内白化、自适应 k 和分数 2s₁−s₂ 决定。二者分类 k 不同属于当前实现本身，因此闭集准确率差异不能归因于拒识分支。</p>
<p>单帧条件按每张图片独立评分。Within-Seq sequence 使用 (camera,sequence) 的图像 knownness 均值；Seq Memory + sequence mean 是额外消融控制，并非原 Seq Memory 默认接口。四种条件分别在相同辅助图片上重新校准 Val90，不共用数值阈值。序列评分仅影响拒识，分类保持逐图输出。长序列按图像微平均保留权重。</p>
<h2>数据与划分</h2>
<p>本地参考池：原 42 Known / 43 稀疏 Unknown 划分，3311 个建库序列代表；4281 Known 与 123 Unknown 是历史校准池。reference 行仅对单帧评分复核历史结果，不是独立测试。另用元数据划分其相机，40% 校准、60% 测试；每轮 22 个辅助未知物种、21 个不参与校准的新未知物种。42 类和初始变换固定，赤麻鸭的历史共享相机例外保留。原 Known 池已按序列选代表，不能凭它推断连拍收益。</p>
<p>CCT20：每轮 11 个主要动物类随机分为 7 Known、2 辅助未知、2 新未知；Known 的官方 train 建库，cis_val+trans_val 校准，官方 cis_test / trans_test 测试。校准不含新未知物种。每类建库最多 500 个序列代表、每类每测试域最多 1000 张，按完整序列元数据采样，若单一序列自身超过上限则保留该完整序列。不同轮次改变类别角色和元数据采样；同域允许共享相机，跨域测试相机与建库相机严格分离。</p>
<p>iWildCam：使用既有 30 Known / 10 Unknown、每序列一张的裁剪缓存。固定 6427 张初始 Known 库；每轮 5 个辅助、5 个新未知物种。原 test 的 40% 相机仅允许辅助未知用于校准，其余相机用于测试；Known 校准使用原 calibration 相机。bank/cal/test 相机完全隔离。该缓存没有多帧序列，因此两个序列条件应与各自单帧条件一致。</p>
{detail}
<h2>验证与限制</h2>
<p>所有得分由项目真实 MemoryDinoV2Classifier 产生。每个模型用 257 个样本改变分块大小核对得分与分类；Seq Memory 与研究评分器核对；Within-Seq 显式序列 API 与跨批次平均核对。bank/cal/test 按图片 ID、内容哈希、原图、burst、采集组和相机/序列联合键检查，冲突整体排除。源数据内含多个真实类别的查询序列整体排除，四种条件使用同一过滤后数据池；不使用真值把序列拆成同物种子组。固定已审核建库保留原有数据。逐划分审计在 leakage_audit.json。</p>
<p class="note">Val90 的 90% 是辅助未知校准目标，不是独立测试保证。本地稀疏物种辅助池很小，结果不稳定；本地相机/采集序列是恢复的代理标签。新划分测试未用于此次阈值或变换拟合，但数据曾参与项目历史实验，属于回顾性留出验证。CCT20 的鸟/啮齿类是数据集类别，不全是物种。不同数据集预处理和领域不同，跨数据集绝对数值不作因果比较；两算法在同一数据池内配对比较。本轮未执行人工审核、增量新类注册或聚类闭环。误差条是划分标准差，不是置信区间；按相机配对 bootstrap 的 95% 区间另在 results.csv。</p>
<p>附件：<a href="results.csv">逐次结果</a> · <a href="summary.csv">均值与标准差</a> · <a href="threshold_sweep.csv">Val80/85/90/95</a> · <a href="per_species.csv">逐物种</a> · <a href="protocol.json">协议与源文件哈希</a> · <a href="runtime_checks.json">运行时验证</a> · <a href="verification.json">独立重算验证</a> · <code>scripts/benchmark_seq_vs_within.py</code> · <code>scripts/verify_seq_vs_within_results.py</code>。各划分目录保留样本 ID、预测和得分，模型文件阈值为 0 的评分头只供实验，不部署。</p></html>'''
    (OUT/'REPORT.html').write_text(text, encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='Neri_plus checkout providing training modules')
    parser.add_argument('--local-run', type=Path, required=True, help='Local Val90 partition and feature directory')
    parser.add_argument('--iwildcam-run', type=Path, required=True, help='iWildCam manifest and feature directory')
    parser.add_argument('--cct20-features', type=Path, required=True, help='CCT20 snapshot and feature directory')
    parser.add_argument('--output-dir', type=Path, required=True, help='New experiment directory')
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if OUT.exists():
        raise FileExistsError(f"Use a new output directory: {OUT}")
    configure_source(args.source_root)
    LOCAL_RUN, IWILDCAM_RUN, CCT20_FEATURES = [p.resolve() for p in (args.local_run, args.iwildcam_run, args.cct20_features)]
    OUT.mkdir(parents=True)
    protocol = dict(seeds=SEEDS, historical_protocol_date='2026-10-09', target=.90,
                    scoring='real Neri runtime; frozen encoder; exact paired bank and calibration pools',
                    methods=['Seq Memory k1', 'Within-Seq k3 classifier + adaptive whitened rejection'],
                    sensitivity_targets=[.80,.85,.90,.95],
                    source_hashes={str(p.relative_to(PROJECT)): sha(p) for p in [PROJECT/'system/dinov2/within_seq.py',
                        PROJECT/'system/dinov2/memory_classifier.py', PROJECT/'system/dinov2/memory_checkpoint.py', Path(__file__)]},
                    no_installed_model_changes=True, selection_uses_scores=False,
                    uncertainty='split mean/SD; paired camera bootstrap 2000 draws, 95% percentile CI',
                    local='existing Known bank; heldout-pool cameras 40/60; auxiliary/novel species 22/21',
                    cct20='official train/val/test; class roles 7/2/2; bank 500 reps/class; query 1000 frames/class/domain, whole sequences',
                    iwildcam='existing location split; auxiliary/novel 5/5; test cameras repartitioned 40/60 for cal/test')
    save_json(OUT/'protocol.json', protocol)
    with threadpool_limits(limits=4):
        run_local()
        run_iwildcam()
        run_cct20()
        summarize()
    print('Completed:', OUT/'REPORT.html', flush=True)
