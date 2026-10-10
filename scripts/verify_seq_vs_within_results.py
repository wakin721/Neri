"""Independently recompute the fixed three-dataset experiment from saved predictions."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--local-run', type=Path, required=True)
    parser.add_argument('--iwildcam-run', type=Path, required=True)
    parser.add_argument('--cct20-features', type=Path, required=True)
    parser.add_argument('--report-file', type=Path, help='Optional new report file; otherwise print only')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.report_file and args.report_file.exists():
        raise FileExistsError(args.report_file)
    import csv
    import json
    from collections import defaultdict
    import numpy as np

    ROOT = args.run_dir.resolve()
    local, iwildcam, cct20 = [p.resolve() for p in (args.local_run, args.iwildcam_run, args.cct20_features)]
    data_paths = {
        'Local reviewed': (local/'partition.jsonl', local/'features.npz'),
        'iWildCam': (iwildcam/'manifest.json', iwildcam/'features.npz'),
        'CCT20': (cct20/'snapshot.json', cct20/'features.npz'),
    }
    datasets = {}
    for name, (meta, path) in data_paths.items():
        records = [json.loads(s) for s in meta.read_text('utf8').splitlines()] if meta.suffix=='.jsonl' else json.loads(meta.read_text('utf8'))['rows']
        with np.load(path, allow_pickle=False) as z:
            ids = z['ids'].astype(str)
        assert len(ids) == len(records)
        assert list(ids) == [r.get('id', r['image_id']) if meta.suffix=='.jsonl' else r['image_id'] for r in records]
        datasets[name] = records, {identity: i for i, identity in enumerate(ids)}

    rows = list(csv.DictReader((ROOT/'results.csv').open(encoding='utf-8-sig')))
    assert len(rows) == 50
    shared = {}
    checks = 0
    for row in rows:
        split = row['split']
        part = json.loads((ROOT/split/'partition.json').read_text('utf8'))
        classes, aux, novel = [set(part[k]) for k in ['classes','auxiliary_unknown_classes','novel_unknown_classes']]
        assert not classes & aux and not classes & novel and not aux & novel
        records, by_id = datasets[row['dataset']]
        ledger = list(csv.DictReader((ROOT/split/'partition_ids.csv').open(encoding='utf-8-sig')))
        pools = {p: [r for r in ledger if r['pool']==p] for p in part['pool_counts']}
        assert set(r['species'] for r in pools['bank']) == classes
        assert set(r['species'] for r in pools['cal']) <= classes | aux
        assert set(r['id'] for r in pools['bank']).isdisjoint(r['id'] for r in pools['cal'])
        if split != 'local_internal_reference':
            assert set(r['id'] for r in pools['cal']).isdisjoint(r['id'] for r in pools[row['domain']])
            for pool in ['cal',row['domain']]:
                gg = defaultdict(set)
                for r in pools[pool]:
                    gg[r['camera'],r['sequence']].add(r['species'])
                assert all(len(labels)==1 for labels in gg.values())
        method = 'seq' if row['method'].startswith('Seq Memory') else 'within'
        pooled = row['method'] in ['Within-Seq sequence','Seq Memory + sequence mean (control)']
        suffix = 'sequence' if pooled else 'single'
        prediction_path = ROOT/split/(row['domain']+'_'+method+'_'+suffix+'.npz')
        with np.load(prediction_path, allow_pickle=False) as z:
            truth, pred, score, ids = [z[k].copy() for k in ['truth','predicted','knownness','ids']]
            rejected = z['rejected'].copy()
            threshold = float(z['threshold'])
        assert threshold == float(row['threshold'])
        np.testing.assert_array_equal(score<threshold, rejected)
        np.testing.assert_array_equal(ids,[r['id'] for r in pools[row['domain']]])
        np.testing.assert_array_equal(truth,[r['species'] for r in pools[row['domain']]])
        key = split, row['domain']
        if key in shared:
            np.testing.assert_array_equal(shared[key], ids)
        else:
            shared[key] = ids
        known = np.isin(truth,list(classes)); unknown = ~known; unseen = np.isin(truth,list(novel))
        assert int(known.sum())==int(row['known_images'])
        assert int(unknown.sum())==int(row['unknown_images'])
        assert int(unseen.sum())==int(row['novel_unknown_images'])
        values = {
            'known_frr': rejected[known].mean(),
            'unknown_recall': rejected[unknown].mean(),
            'known_closed_accuracy': (pred[known]==truth[known]).mean(),
            'known_correct_accepted': ((pred[known]==truth[known]) & ~rejected[known]).mean(),
        }
        if unseen.any():
            values['novel_unknown_recall'] = rejected[unseen].mean()
        for key,value in values.items():
            assert abs(float(row[key])-value)<1e-12
            checks += 1
        cache_name = 'local_bank' if row['dataset']=='Local reviewed' else 'iwildcam_bank' if row['dataset']=='iWildCam' else split
        with np.load(ROOT/cache_name/'scores.npz',allow_pickle=False) as z:
            scores = z[method+'_score'].copy()
            positions = {int(i):j for j,i in enumerate(z['query_indices'])}
        cal = pools['cal']
        cs = np.array([scores[positions[by_id[r['id']]]] for r in cal])
        if pooled:
            gg = defaultdict(list)
            for i,r in enumerate(cal):
                gg[r['camera'],r['sequence']].append(i)
            for g in gg.values():
                cs[g] = cs[g].mean()
        mask = np.array([r['species'] in aux for r in cal])
        assert int(mask.sum())==int(row['calibration_unknown_images'])
        boundary = np.sort(cs[mask])[int(np.ceil(.9*mask.sum()))-1]
        assert threshold == np.nextafter(boundary,np.inf)
        assert (cs[mask]<threshold).mean() >= .9
        assert abs((cs[mask]<threshold).mean()-float(row['calibration_unknown_recall']))<1e-12
        checks += 3

    # iWildCam has one frame per remaining pure-species sequence: aggregation must
    # make exactly the same decisions as single-frame scoring.
    for split in ['iwildcam_r1','iwildcam_r2','iwildcam_r3']:
        for method in ['seq','within']:
            with np.load(ROOT/split/f'test_{method}_single.npz') as a, np.load(ROOT/split/f'test_{method}_sequence.npz') as b:
                np.testing.assert_array_equal(a['knownness'],b['knownness'])
                np.testing.assert_array_equal(a['rejected'],b['rejected'])

    result = dict(status='passed', result_rows=len(rows), held_out_cohorts=len(shared)-1,
                  recomputed_metrics_and_calibration_checks=checks,
                  common_evaluation_ids='passed', auxiliary_thresholds='passed',
                  pure_query_sequence_labels='passed', iwildcam_single_sequence_equivalence='passed')
    if args.report_file:
        args.report_file.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
