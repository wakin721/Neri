"""Summarize the four predeclared calibration targets without altering the original experiment."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New summary directory')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    import csv
    import json
    import numpy as np

    ROOT=args.run_dir.resolve()
    OUT=args.output_dir.resolve()
    OUT.mkdir(parents=True)
    rows=list(csv.DictReader((ROOT/'threshold_sweep.csv').open(encoding='utf-8-sig')))
    assert len(rows)==1080
    summaries=[]
    for target in [.8,.85,.9,.95]:
        selected=[r for r in rows if float(r['target'])==target]
        pairs=[]
        for tag in sorted({r['experiment'] for r in selected}):
            rr={r['method']:r for r in selected if r['experiment']==tag}
            a,b,k=rr['Seq Memory'],rr['Within-Seq single'],rr['Memory k3 control']
            pairs.append((float(b['known_frr'])-float(a['known_frr']),
                          float(b['known_frr'])-float(k['known_frr']),
                          float(b['oracle_novel_known_frr'])-float(a['oracle_novel_known_frr'])))
        assert len(pairs)==90
        values=np.array(pairs)
        summaries.append(dict(calibration_target=target,rounds=90,
            seq_lower_known_frr=int(np.sum(values[:,0]>1e-12)),within_lower_known_frr=int(np.sum(values[:,0]< -1e-12)),
            mean_known_frr_delta=float(values[:,0].mean()),memory_k3_lower_frr=int(np.sum(values[:,1]>1e-12)),
            seq_lower_oracle_novel_frr=int(np.sum(values[:,2]>1e-12)),mean_oracle_frr_delta=float(values[:,2].mean()),
            seq_mean_known_frr=float(np.mean([float(r['known_frr']) for r in selected if r['method']=='Seq Memory'])),
            within_mean_known_frr=float(np.mean([float(r['known_frr']) for r in selected if r['method']=='Within-Seq single'])),
            seq_mean_novel_recall=float(np.mean([float(r['novel_unknown_recall']) for r in selected if r['method']=='Seq Memory'])),
            within_mean_novel_recall=float(np.mean([float(r['novel_unknown_recall']) for r in selected if r['method']=='Within-Seq single']))))
    with (OUT/'sensitivity_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    (OUT/'sensitivity_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf8')
    section='<h2 id="sensitivity">辅助阈值敏感性（预先指定的工作点）</h2><table><tr><th>辅助未知目标</th><th>Seq误拒较低/90</th><th>Within误拒较低/90</th><th>平均Within−Seq误拒差</th><th>同等测试新未知召回时Seq误拒较低/90</th></tr>'
    for r in summaries:
        section+=f"<tr><td>{r['calibration_target']:.0%}</td><td>{r['seq_lower_known_frr']}</td><td>{r['within_lower_known_frr']}</td><td>{r['mean_known_frr_delta']*100:+.2f} 个百分点</td><td>{r['seq_lower_oracle_novel_frr']}</td></tr>"
    section+='</table><p>各目标分别在辅助未知图片上取顺序统计阈值，不使用测试指标选择目标。等测试召回列仍是回顾性ROC诊断。<a href="sensitivity_summary.csv">完整敏感性汇总</a>。</p>'
    (OUT/'sensitivity.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>阈值敏感性</title>'+section+'</html>', encoding='utf8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
