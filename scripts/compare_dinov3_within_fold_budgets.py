"""Within each species-exclusion model, vary proxy photo count without pooling.

Reuse verified reduced-B/whitening models and label-free scores. Budget 1000 is
an upper bound; report actual independent acquisitions without duplication.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.compare_local_dinov3_protocols import csvwrite,freeze,read,rows_read,write
from scripts.compare_dinov3_joint_calibration import load_npz
from scripts.simulate_dinov3_unknown import acquisition_units,proxy_order,population_rows,aggregate_populations,summary_rows
from system.dinov3.encoder import sha256
from system.dinov3.balanced_unknown_calibration import fit_unknown_threshold
from system.dinov3.joint_calibration import fit_joint_threshold


def fit_selected_negatives(data,positions):
    positions=np.asarray(positions,dtype=np.int64)
    known=np.asarray(data['known'],dtype=bool)
    if not len(positions) or known[positions].any() or len(set(positions))!=len(positions):
        raise ValueError('Proxy samples must be distinct actual held-out Unknown rows')
    if set(data['labels'][positions])!=set(data['labels'][~known]):
        raise ValueError('Every budget must retain the same held-species coverage')
    scores=np.r_[data['known_score'][known],data['known_score'][positions]]
    labels=np.r_[data['labels'][known],data['labels'][positions]]
    roles=np.r_[np.ones(known.sum(),dtype=bool),np.zeros(len(positions),dtype=bool)]
    return dict(macro90=dict(fit_unknown_threshold(data['known_score'][positions],data['labels'][positions],target=.9),feasible=True),
                joint_equal=fit_joint_threshold(scores,labels,roles),
                joint_cost2=fit_joint_threshold(scores,labels,roles,unknown_cost=2/3))


def run(args):
    if (args.out/'verification.json').exists():
        return
    source=read(args.source/'plan.json')
    if not read(args.source/'independent_verification.json')['passed']:
        raise ValueError('Species exclusion and whitening must already be verified')
    rows=rows_read(args.comparison/'current_reviewed.jsonl')
    if sha256(args.comparison/'current_reviewed.jsonl')!=source['reviewed_manifest_sha256']:
        raise ValueError('Reviewed metadata changed')
    if sha256(source['installed_manifest'])!=source['installed_manifest_sha256']:
        raise ValueError('Installed model changed')
    plan=[]
    for fold,held in enumerate(source['held_folds']):
        folder=args.source/f'fold_{fold}'
        frozen=read(folder/'fitted.json')
        if any(sha256(folder/name)!=digest for name,digest in frozen['files'].items()):
            raise ValueError('Verified shadow files changed')
        data=load_npz(folder/'simulation_C.npz')
        held_indices=[int(i) for i in data['indices'] if rows[int(i)]['species'] in held]
        units=acquisition_units(rows,held_indices)
        for seed in source['seeds']:
            order=proxy_order(rows,units,seed)
            for budget in source['budgets']:
                selected=order[:budget]
                plan.append(dict(fold=fold,seed=seed,budget=budget,actual_images=len(selected),
                                 held_species=held,available_units=len(units),selected=selected))
    freeze(args.out/'plan.json',dict(source=str(args.source),source_plan_sha256=sha256(args.source/'plan.json'),
          scope='separate threshold fit inside each fixed species-exclusion model; no cross-model calibration pooling',
          policies=['macro90','joint_equal','joint_cost2'],samples=plan,application_changed=False))
    thresholds=[]
    for p in plan:
        data=load_npz(args.source/f"fold_{p['fold']}/simulation_C.npz")
        positions=np.searchsorted(data['indices'],[x['index'] for x in p['selected']])
        np.testing.assert_array_equal(data['indices'][positions],[x['index'] for x in p['selected']])
        for policy,fit in fit_selected_negatives(data,positions).items():
            thresholds.append(dict(fold=p['fold'],seed=p['seed'],budget=p['budget'],actual_images=p['actual_images'],
                                   policy=policy,threshold=fit['threshold'],fit=fit))
    freeze(args.out/'thresholds_before_E.json',dict(plan_sha256=sha256(args.out/'plan.json'),thresholds=thresholds))
    original=set(source['known_classes'])
    rates=defaultdict(list)
    per_fold=[]
    detailed=[]
    for p in thresholds:
        data=load_npz(args.source/f"fold_{p['fold']}/E_scores.npz")
        np.testing.assert_array_equal(data['indices'],source['evaluation'])
        accepted=data['known_score']>=p['threshold']
        by_species=population_rows(data['labels'],data['predictions'],accepted,set(data['classes']),original)
        key=(p['seed'],p['budget'],p['policy'])
        rates[key].extend(by_species)
        aggregate=aggregate_populations(by_species)
        per_fold.append(dict(fold=p['fold'],seed=p['seed'],budget=p['budget'],actual_images=p['actual_images'],
                             policy=p['policy'],threshold=p['threshold'],**aggregate))
        detailed.extend(dict(fold=p['fold'],seed=p['seed'],budget=p['budget'],policy=p['policy'],**r) for r in by_species)
    combined=[dict(seed=seed,budget=budget,policy=policy,**aggregate_populations(values))
              for (seed,budget,policy),values in rates.items()]
    summarized=summary_rows(combined,('budget','policy'),('known_rejection_macro','known_correct_acceptance_macro',
                            'real_unknown_rejection_macro','proxy_unknown_rejection_macro'))
    csvwrite(args.out/'per_fold_metrics.csv',per_fold)
    csvwrite(args.out/'per_species.csv',detailed)
    csvwrite(args.out/'summary.csv',summarized)
    write(args.out/'metrics.json',combined)
    # Verify nesting, unchanged species coverage, and independent acquired events.
    for fold in range(5):
        for seed in source['seeds']:
            previous=set()
            for p in [s for s in plan if s['fold']==fold and s['seed']==seed]:
                current={s['index'] for s in p['selected']}
                assert previous<=current
                assert len({s['unit'] for s in p['selected']})==len(current)
                assert {rows[i]['species'] for i in current}==set(p['held_species'])
                previous=current
    assert sha256(source['installed_manifest'])==source['installed_manifest_sha256']
    write(args.out/'verification.json',dict(passed=True,fixed_models_and_species_within_each_fold=True,
          known_C_fixed_within_each_fold=True,nested_independent_event_sampling=True,thresholds_fit_on_C_only=True,
          same_real_unknown_E_21_species_28_photos=True,application_model_unchanged=True,threshold_conditions=len(thresholds)))
    names=dict(macro90='未知Macro90',joint_equal='等成本联合',joint_cost2='未知错误成本×2')
    lines=['# 每轮固定留出物种：未知照片数量对照','',
       '复用已核验的5个整物种排除模型。每轮分别拟合门限，不跨轮汇总C分数；白化、Ridge、Within均已按剩余B重新拟合。', '',
       '固定每轮6—10个被留出物种，物种覆盖、正例和模型不随预算改变。每个独立事件只取一张，1000为上限，实际各轮仅361—362个可用事件。所有阈值先用C确定，再读取同一固定E的已保存分数。','',
       '|预算|每轮实际照片|策略|已知误拒|正确接纳|真实未知拒识|模拟未知E拒识|','|---|---|---|---:|---:|---:|---:|']
    for r in summarized:
        actual=[x['actual_images'] for x in plan if x['budget']==r['budget']]
        count=str(min(actual)) if min(actual)==max(actual) else f'{min(actual)}—{max(actual)}'
        lines.append(f"|{r['budget']}|{count}|{names[r['policy']]}|{r['known_rejection_macro']*100:.2f}%|{r['known_correct_acceptance_macro']*100:.2f}%|{r['real_unknown_rejection_macro']*100:.2f}%|{r['proxy_unknown_rejection_macro']*100:.2f}%|")
    lines+=['','按物种在适用影子模型中先平均，再作物种宏平均，最后对三个种子取均值。这里是35—39类留类模型的结果，不是完整45类软件模型的误拒率；真实未知E始终为21类28张。', '',
            '模拟未知E是本轮被移出的原已知物种，另行报告。所有E文件和各模型内部类别集合在预算之间不变；不把跨模型反复评分当成更多独立测试照片。', '',
            '未凑出每轮1000张独立未知照片，因此最后一档只代表实际361—362张，不能声称完成每轮1000张的效果验证。相机仍为目录代理，历史E已曝光，结果是探索性诊断。当前软件门限未改动。']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n','utf8')
    print('SUMMARY',summarized)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=REPO/'runs/simulated_unknown_20261114')
    p.add_argument('--comparison',type=Path,default=REPO/'runs/local_protocol_comparison_20261114')
    p.add_argument('--out',type=Path,default=REPO/'runs/within_fold_unknown_20261114')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    run(args)


if __name__=='__main__':main()
