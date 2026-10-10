"""Replay the fixed Val90 calibration through a selected Neri runtime; never install a model."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--app-root', type=Path, required=True, help='Neri runtime directory; use its Python interpreter')
    parser.add_argument('--installed', action='store_true', help='Validate app-root/res/model/DINOv2/memory_head.npz instead of the run checkpoint')
    parser.add_argument('--report-file', type=Path, help='Optional new report file; otherwise print only')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.report_file and args.report_file.exists():
        raise FileExistsError(args.report_file)
    import json
    import numpy as np

    OUT=args.run_dir.resolve()
    APP=args.app_root.resolve()
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(APP))
    from system.dinov2.checkpoint import load_checkpoint
    from system.dinov2.memory_classifier import MemoryDinoV2Classifier

    stage=APP/'res/model/DINOv2/memory_head.npz' if args.installed else OUT/'deployment/memory_head.npz'
    head=load_checkpoint(stage)
    protocol=json.loads((OUT/'protocol.json').read_text('utf8'))
    rows=[json.loads(s) for s in (OUT/'partition.jsonl').read_text('utf8').splitlines()]
    expected={r['id']:r for r in [json.loads(s) for s in (OUT/'deployment/calibration_scores.jsonl').read_text('utf8').splitlines()]}
    assert set(head.classes)==set(protocol['known_classes'])
    assert not set(head.labels)&set(protocol['unknown_classes'])
    with np.load(OUT/'features.npz',allow_pickle=False) as z:
        features=z['features'].copy()
        np.testing.assert_array_equal(z['ids'],[r['id'] for r in rows])
    indices=[i for i,r in enumerate(rows) if r['role']!='bank']
    runtime=MemoryDinoV2Classifier(head)
    results=[]
    for start in range(0,len(indices),64):
        ii=indices[start:start+64]
        for i,pred in zip(ii,runtime.classify_features(features[ii])):
            row=rows[i]; prior=expected[row['id']]
            assert pred.best_known_species==prior['best_known_species']
            assert (not pred.accepted)==prior['rejected']
            assert abs(pred.known_score-prior['knownness'])<2e-6
            results.append(dict(id=row['id'],role=row['role'],rejected=not pred.accepted,
                                delta=abs(pred.known_score-prior['knownness'])))
    unknown=[r for r in results if r['role']=='calibration_unknown']
    known=[r for r in results if r['role']=='calibration_known']
    assert len(unknown)==123 and sum(r['rejected'] for r in unknown)==111
    assert sum(r['rejected'] for r in unknown)/len(unknown)>=.90
    report=dict(status='passed',checkpoint=str(stage),app_runtime=str(APP/'system/dinov2/memory_classifier.py'),
        python=sys.executable,known_classes=len(head.classes),exemplars=len(head.features),
        threshold=head.threshold,calibration_samples=len(results),unknown_images=len(unknown),
        unknown_rejected=sum(r['rejected'] for r in unknown),
        unknown_micro_recall=sum(r['rejected'] for r in unknown)/len(unknown),
        known_false_rejection_rate=sum(r['rejected'] for r in known)/len(known),
        max_score_difference=max(r['delta'] for r in results),fingerprint=head.fingerprint,
        known_calibration_missing_classes=sorted(set(head.classes)-{r['species'] for r in rows if r['role']=='calibration_known'}))
    if args.report_file:
        args.report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
