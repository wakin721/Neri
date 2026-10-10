"""Independently recompute the fixed 90-round unknown-species rotation experiment."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--metadata-file', type=Path, help='Override the source snapshot path recorded in protocol.json')
    parser.add_argument('--report-file', type=Path, help='Optional new report file; otherwise print only')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.report_file and args.report_file.exists():
        raise FileExistsError(args.report_file)
    import csv
    import json
    from collections import Counter,defaultdict
    import numpy as np

    ROOT=args.run_dir.resolve()
    protocol=json.loads((ROOT/'protocol.json').read_text('utf8'))
    metadata = args.metadata_file or Path(protocol['metadata_source'])
    source=json.loads(metadata.read_text('utf8'))['rows']
    source_by_id={r['image_id']:r for r in source}
    results=list(csv.DictReader((ROOT/'results.csv').open(encoding='utf-8-sig')))
    schedule=list(csv.DictReader((ROOT/'role_schedule.csv').open(encoding='utf-8-sig')))
    assert len(results)==270
    assert len(schedule)==2700
    for c in protocol['included_species']:
        assert Counter(r['role'] for r in schedule if r['species']==c)=={'Known':60,'Auxiliary Unknown':15,'Novel Unknown':15}
    methods={'Seq Memory':'seq','Memory k3 control':'k3','Within-Seq single':'within'}
    prediction_ids={};within_pred={};control_pred={};checks=0
    for row in results:
        tag=row['experiment'];meta=json.loads((ROOT/tag/'partition.json').read_text('utf8'))
        known,aux,novel=[set(meta[k]) for k in ['known_classes','auxiliary_unknown_classes','novel_unknown_classes']]
        assert len(known)==20 and len(aux)==5 and len(novel)==5
        assert not known&aux and not known&novel and not aux&novel
        for r in schedule:
            if int(r['camera_partition'])==meta['camera_partition'] and int(r['rotation'])==meta['rotation']:
                assert r['species'] in {'Known':known,'Auxiliary Unknown':aux,'Novel Unknown':novel}[r['role']]
        ledger=list(csv.DictReader((ROOT/tag/'partition_ids.csv').open(encoding='utf-8-sig')))
        pools={p:[r for r in ledger if r['pool']==p] for p in ['bank','cal','test']}
        assert set(r['species'] for r in pools['bank'])==known
        assert set(r['species'] for r in pools['cal'])==known|aux
        assert set(r['species'] for r in pools['test'])==known|aux|novel
        owners={}
        for p,rr in pools.items():
            for r in rr:
                s=source_by_id[r['id']]
                assert s['species']==r['species'] and s['camera']==r['camera'] and s['acquisition_id']==r['sequence']
                for key in ['image_id','sha256','source_sha256','source_image_id','burst_id','acquisition_id','camera']:
                    value=s.get(key)
                    if value:
                        identity=(key,str(value))
                        assert identity not in owners or owners[identity]==p
                        owners[identity]=p
        with np.load(ROOT/tag/(methods[row['method']]+'_predictions.npz'),allow_pickle=False) as z:
            truth,pred,score,rejected,cs,cy,ids=[z[k].copy() for k in ['truth','predicted','knownness','rejected','calibration_score','calibration_truth','test_ids']]
            cutoff=float(z['threshold'])
            np.testing.assert_array_equal(z['calibration_ids'],[r['id'] for r in pools['cal']])
        assert cutoff==float(row['threshold'])
        np.testing.assert_array_equal(ids,[r['id'] for r in pools['test']])
        np.testing.assert_array_equal(truth,[r['species'] for r in pools['test']])
        np.testing.assert_array_equal(cy,[r['species'] for r in pools['cal']])
        np.testing.assert_array_equal(rejected,score<cutoff)
        if tag in prediction_ids:np.testing.assert_array_equal(ids,prediction_ids[tag])
        else:prediction_ids[tag]=ids
        if row['method']=='Within-Seq single':within_pred[tag]=pred
        if row['method']=='Memory k3 control':control_pred[tag]=pred
        am=np.isin(cy,list(aux));km=np.isin(truth,list(known));um=~km;nm=np.isin(truth,list(novel))
        assert cutoff==np.nextafter(np.sort(cs[am])[int(np.ceil(.9*am.sum()))-1],np.inf)
        assert np.mean(cs[am]<cutoff)>=.9
        assert int(am.sum())==int(row['calibration_unknown_images'])
        metrics={'known_frr':rejected[km].mean(),'unknown_recall':rejected[um].mean(),
                 'novel_unknown_recall':rejected[nm].mean(),'known_correct_accepted':((pred[km]==truth[km])&~rejected[km]).mean(),
                 'known_closed_accuracy':(pred[km]==truth[km]).mean(),
                 'known_macro_frr':np.mean([rejected[truth==c].mean() for c in known])}
        oracle=np.nextafter(np.sort(score[nm])[int(np.ceil(.9*nm.sum()))-1],np.inf)
        metrics['oracle_novel90_known_frr']=np.mean(score[km]<oracle)
        metrics['oracle_novel_recall']=np.mean(score[nm]<oracle)
        for key,value in metrics.items():
            assert abs(float(row[key])-value)<1e-12,(tag,row['method'],key)
            checks+=1
        # Known-only transforms are independently checked against the head labels.
        with np.load(ROOT/tag/'within_seq.npz',allow_pickle=False) as head:
            assert set(head['labels'])==known and len(head['labels'])==len(pools['bank'])
        with np.load(ROOT/tag/'seq_memory.npz',allow_pickle=False) as head:
            assert set(head['labels'])==known and len(head['labels'])==len(pools['bank'])
    for tag in within_pred:
        np.testing.assert_array_equal(within_pred[tag],control_pred[tag])
    assert len(within_pred)==90
    summary=json.loads((ROOT/'summary.json').read_text('utf8'))[0]
    pairs=list(csv.DictReader((ROOT/'paired_comparison.csv').open(encoding='utf-8-sig')))
    assert len(pairs)==90
    delta=np.array([float(p['frr_delta']) for p in pairs])
    assert int((delta>1e-12).sum())==summary['seq_lower_frr']
    assert int((delta< -1e-12).sum())==summary['within_lower_frr']
    assert abs(delta.mean()-summary['mean_frr_delta'])<1e-12
    sweeps=list(csv.DictReader((ROOT/'threshold_sweep.csv').open(encoding='utf-8-sig')))
    assert len(sweeps)==1080
    for row in sweeps:
        tag=row['experiment'];meta=json.loads((ROOT/tag/'partition.json').read_text('utf8'))
        known,aux,novel=[set(meta[k]) for k in ['known_classes','auxiliary_unknown_classes','novel_unknown_classes']]
        with np.load(ROOT/tag/(methods[row['method']]+'_predictions.npz'),allow_pickle=False) as z:
            truth,score,cs,cy=[z[k].copy() for k in ['truth','knownness','calibration_score','calibration_truth']]
        target=float(row['target']);am=np.isin(cy,list(aux));km=np.isin(truth,list(known));nm=np.isin(truth,list(novel))
        cutoff=np.nextafter(np.sort(cs[am])[int(np.ceil(target*am.sum()))-1],np.inf)
        assert cutoff==float(row['threshold'])
        assert abs(float(row['known_frr'])-np.mean(score[km]<cutoff))<1e-12
        assert abs(float(row['novel_unknown_recall'])-np.mean(score[nm]<cutoff))<1e-12
        oracle=np.nextafter(np.sort(score[nm])[int(np.ceil(target*nm.sum()))-1],np.inf)
        assert abs(float(row['oracle_novel_known_frr'])-np.mean(score[km]<oracle))<1e-12
        checks+=4
    result=dict(status='passed',rounds=90,method_results=270,recomputed_metric_checks=checks,
                balanced_unknown_species_roles=True,unknown_excluded_from_known_heads=True,
                camera_original_hash_sequence_isolation=True,identical_paired_query_ids=True,
                same_classifier_control_predictions=True,Val90_threshold_recomputed=True,
                test_ROC_diagnostic_recomputed=True,paired_summary_recomputed=True,
                sensitivity_rows_recomputed=len(sweeps))
    if args.report_file:
        args.report_file.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
