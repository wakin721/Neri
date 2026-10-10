"""Balanced unknown-species rotations on local human-reviewed frozen features.

30 class-role rotations x 3 independently selected camera partitions. Every
species is Known 60 times, auxiliary Unknown 15 times and novel Unknown 15 times.
See --help for dataset and output directory arguments.
"""
import argparse
from collections import defaultdict, Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
OUT = FEATURE_DIR = None
import benchmark_seq_vs_within as base
SEEDS = [20261009,20261010,20261011]
RESULTS, SWEEPS, SPECIES = [], [], []


def save(path, value):
    base.save_json(path, value)


def camera_splits(data):
    y,cams = data['labels'],data['cameras']
    pure = np.ones(len(y),bool)
    for group in base.groups(data,np.arange(len(y))):
        if len(set(y[group]))>1:
            pure[group] = False
    candidates = [str(c) for c in np.unique(y) if np.sum((y==c)&pure)>=20 and len(set(cams[(y==c)&pure]))>=6]
    unique_cams, codes = np.unique(cams,return_inverse=True)
    counts = np.zeros((len(candidates),len(unique_cams)),int)
    pure_counts = counts.copy()
    for j,c in enumerate(candidates):
        np.add.at(counts[j],codes[y==c],1)
        np.add.at(pure_counts[j],codes[(y==c)&pure],1)
    found=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed); best=None
        for trial in range(3000):
            perm = rng.permutation(len(unique_cams))
            cut1,cut2 = int(.5*len(perm)),int(.75*len(perm))
            b,cl,t = perm[:cut1],perm[cut1:cut2],perm[cut2:]
            eligible = (counts[:,b].sum(1)>=5)&(pure_counts[:,cl].sum(1)>=3)&(pure_counts[:,t].sum(1)>=3)
            imbalance = float(np.abs(pure_counts[:,cl].sum(1)/pure_counts.sum(1)-.25).sum()
                              +np.abs(pure_counts[:,t].sum(1)/pure_counts.sum(1)-.25).sum())
            merit = (int(eligible.sum()),-imbalance)
            if best is None or merit>best[0]:
                best=(merit,eligible.copy(),b.copy(),cl.copy(),t.copy(),trial)
        eligible = {c for c,flag in zip(candidates,best[1]) if flag}
        pools={name:np.flatnonzero(np.isin(cams,unique_cams[ii])) for name,ii in zip(['bank','cal','test'],best[2:5])}
        pools,audit = base.remove_leakage(data,pools)
        eligible={c for c in eligible if all(np.sum(y[ii]==c)>=minimum for ii,minimum in zip(pools.values(),[5,3,3]))}
        found.append(dict(seed=seed,trial=best[5],eligible=eligible,pools=pools,audit=audit))
    common=set.intersection(*(d['eligible'] for d in found))
    assert len(common)==30, f'Unexpected metadata eligibility: {len(common)}'
    for d in found:
        d['pools']={p:ii[np.isin(y[ii],list(common))] for p,ii in d['pools'].items()}
    return sorted(common),found,candidates


def k3_control(data,bank,query,tag,scored):
    """Use precisely the Within-Seq classifier arrays, with Memory rejection."""
    folder=OUT/tag
    cache=folder/'memory_k3_scores.npz'
    identity=base.sha(folder/'within_seq.npz')
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:
            if str(z['within_head_sha256'])==identity:
                np.testing.assert_array_equal(z['predicted'],scored['within_pred'])
                return z['knownness'].copy()
    with np.load(folder/'within_seq.npz',allow_pickle=False) as z:
        meta=json.loads(str(z['metadata'].item()));meta.pop('rejection_mode')
        arrays={k:z[k].copy() for k in ['center','features','centroids','labels','cameras','classes']}
    np.savez_compressed(folder/'memory_k3_control.npz',metadata=json.dumps(meta),**arrays)
    runtime=base.MemoryDinoV2Classifier(base.load_checkpoint(folder/'memory_k3_control.npz'))
    predictions=[]
    for start in range(0,len(query),512):
        predictions.extend(runtime.classify_features(data['raw'][query[start:start+512]]))
    pred=np.array([p.best_known_species for p in predictions])
    score=np.array([p.known_score for p in predictions])
    np.testing.assert_array_equal(pred,scored['within_pred'])
    np.savez_compressed(cache,predicted=pred,knownness=score,query_indices=query,within_head_sha256=identity)
    return score


def empirical_roc_frr(known_scores,unknown_scores,target):
    """Retrospective test-ROC diagnostic, never a deployable calibration."""
    cutoff=base.calibration_threshold(unknown_scores,target)
    return float(np.mean(known_scores<cutoff)),float(np.mean(unknown_scores<cutoff))


def run_rotation(data,pools,common,known,aux,novel,camera_id,rotation):
    tag=f'camera{camera_id}_roles{rotation:02d}'
    bank=pools['bank'][np.isin(data['labels'][pools['bank']],list(known))]
    cal=pools['cal'][np.isin(data['labels'][pools['cal']],list(known|aux))]
    test=pools['test']
    query=np.unique(np.r_[cal,test])
    assert not set(data['labels'][bank]) & (aux|novel)
    assert set(data['labels'][bank])==known
    assert not set(data['labels'][cal]) & novel
    scored=base.fit_score(data,bank,query,tag)
    k3=k3_control(data,bank,query,tag,scored)
    position={int(i):j for j,i in enumerate(query)}
    ci=np.array([position[int(i)] for i in cal]);ti=np.array([position[int(i)] for i in test])
    cy,ty=data['labels'][cal],data['labels'][test]
    aux_mask=np.isin(cy,list(aux));known_mask=np.isin(ty,list(known));novel_mask=np.isin(ty,list(novel))
    variants=[('Seq Memory','seq',scored['seq_score'],scored['seq_pred']),
              ('Memory k3 control','k3',k3,scored['within_pred']),
              ('Within-Seq single','within',scored['within_score'],scored['within_pred'])]
    manifest=dict(camera_partition=camera_id,rotation=rotation,known_classes=sorted(known),
                  auxiliary_unknown_classes=sorted(aux),novel_unknown_classes=sorted(novel),
                  counts=dict(bank=len(bank),calibration=len(cal),test=len(test)),
                  camera_isolation=True,unknown_excluded_from_all_transforms=True)
    save(OUT/tag/'partition.json',manifest)
    base.csv_write(OUT/tag/'partition_ids.csv',[dict(id=data['ids'][i],species=data['labels'][i],
                  camera=data['cameras'][i],sequence=data['sequences'][i],pool=p)
                  for p,ii in [('bank',bank),('cal',cal),('test',test)] for i in ii])
    decisions={}
    for method,key,score,pred in variants:
        cs=score[ci];ts=score[ti];tp=pred[ti]
        threshold=base.calibration_threshold(cs[aux_mask],.9)
        values=base.metrics(ty,tp,ts,threshold,known,novel)
        oracle,oracle_recall=empirical_roc_frr(ts[known_mask],ts[novel_mask],.9)
        row=dict(experiment=tag,camera_partition=camera_id,rotation=rotation,method=method,
                 bank_images=len(bank),known_classes=20,auxiliary_unknown_classes=5,novel_unknown_classes=5,
                 calibration_unknown_images=int(aux_mask.sum()),threshold=threshold,
                 calibration_unknown_recall=float(np.mean(cs[aux_mask]<threshold)),
                 auxiliary_species='|'.join(sorted(aux)),novel_species='|'.join(sorted(novel)),
                 oracle_novel90_known_frr=oracle,oracle_novel_recall=oracle_recall)|values
        RESULTS.append(row)
        decisions[method]=ts<threshold
        for target in [.80,.85,.90,.95]:
            cutoff=base.calibration_threshold(cs[aux_mask],target)
            oracle,_=empirical_roc_frr(ts[known_mask],ts[novel_mask],target)
            SWEEPS.append(dict(experiment=tag,camera_partition=camera_id,rotation=rotation,method=method,
                               target=target,threshold=cutoff,oracle_novel_known_frr=oracle)
                          |base.metrics(ty,tp,ts,cutoff,known,novel))
        for c in common:
            mask=ty==c
            SPECIES.append(dict(experiment=tag,camera_partition=camera_id,rotation=rotation,method=method,
                                species=c,role='Known' if c in known else 'Auxiliary Unknown' if c in aux else 'Novel Unknown',
                                test_images=int(mask.sum()),rejection_rate=float(np.mean(ts[mask]<threshold))))
        np.savez_compressed(OUT/tag/(key+'_predictions.npz'),test_ids=data['ids'][test],truth=ty,
                            predicted=tp,knownness=ts,rejected=ts<threshold,threshold=threshold,
                            calibration_ids=data['ids'][cal],calibration_truth=cy,calibration_score=cs,
                            cameras=data['cameras'][test])
    low,high,ncam=base.camera_bootstrap_delta(data,test,decisions['Seq Memory'],decisions['Within-Seq single'],known_mask,20261009+rotation+100*camera_id)
    for r in RESULTS[-3:]:
        if r['method']=='Within-Seq single':
            r.update(paired_frr_delta_ci95_low=low,paired_frr_delta_ci95_high=high,known_test_cameras=ncam)
    base.csv_write(OUT/'results.csv',RESULTS)
    base.csv_write(OUT/'threshold_sweep.csv',SWEEPS)
    base.csv_write(OUT/'per_species.csv',SPECIES)
    return values


def analysis():
    pairs=[]
    for camera in [1,2,3]:
        for rotation in range(30):
            rr={r['method']:r for r in RESULTS if r['camera_partition']==camera and r['rotation']==rotation}
            a,b,k=rr['Seq Memory'],rr['Within-Seq single'],rr['Memory k3 control']
            pairs.append(dict(experiment=a['experiment'],camera_partition=camera,rotation=rotation,
                              auxiliary_species=a['auxiliary_species'],novel_species=a['novel_species'],
                              seq_frr=a['known_frr'],within_frr=b['known_frr'],memory_k3_frr=k['known_frr'],
                              seq_novel_recall=a['novel_unknown_recall'],within_novel_recall=b['novel_unknown_recall'],
                              frr_delta=b['known_frr']-a['known_frr'],macro_frr_delta=b['known_macro_frr']-a['known_macro_frr'],
                              frr_delta_same_classifier=b['known_frr']-k['known_frr'],
                              novel_recall_delta=b['novel_unknown_recall']-a['novel_unknown_recall'],
                              seq_auroc=a['auroc_unknown'],within_auroc=b['auroc_unknown'],
                              auroc_delta=b['auroc_unknown']-a['auroc_unknown'],
                              seq_oracle_novel90_frr=a['oracle_novel90_known_frr'],within_oracle_novel90_frr=b['oracle_novel90_known_frr'],
                              oracle_novel90_frr_delta=b['oracle_novel90_known_frr']-a['oracle_novel90_known_frr'],
                              ci95_low=b['paired_frr_delta_ci95_low'],ci95_high=b['paired_frr_delta_ci95_high']))
    base.csv_write(OUT/'paired_comparison.csv',pairs)
    summary=[]
    for scope,subset in [('all',pairs)]+[(f'camera{i}',[p for p in pairs if p['camera_partition']==i]) for i in [1,2,3]]:
        d=np.array([p['frr_delta'] for p in subset]);oracle=np.array([p['oracle_novel90_frr_delta'] for p in subset])
        summary.append(dict(scope=scope,rounds=len(subset),seq_lower_frr=int(np.sum(d>1e-12)),within_lower_frr=int(np.sum(d< -1e-12)),
                            ties=int(np.sum(np.abs(d)<=1e-12)),mean_frr_delta=float(d.mean()),median_frr_delta=float(np.median(d)),
                            p10_frr_delta=float(np.quantile(d,.1)),p90_frr_delta=float(np.quantile(d,.9)),
                            min_frr_delta=float(d.min()),max_frr_delta=float(d.max()),
                            seq_lower_macro_frr=sum(p['macro_frr_delta']>1e-12 for p in subset),
                            memory_k3_lower_frr=sum(p['frr_delta_same_classifier']>1e-12 for p in subset),
                            seq_lower_oracle_novel90_frr=int(np.sum(oracle>1e-12)),
                            mean_oracle_novel90_frr_delta=float(oracle.mean()),
                            mean_novel_recall_delta=float(np.mean([p['novel_recall_delta'] for p in subset])),
                            mean_auroc_delta=float(np.mean([p['auroc_delta'] for p in subset]))))
    save(OUT/'summary.json',summary);base.csv_write(OUT/'summary.csv',summary)
    base.csv_write(OUT/'counterexamples.csv',[p for p in pairs if p['frr_delta']<0])
    method_summary=[]
    for method in ['Seq Memory','Memory k3 control','Within-Seq single']:
        rr=[r for r in RESULTS if r['method']==method]
        method_summary.append(dict(method=method,**{k:float(np.mean([r[k] for r in rr])) for k in
             ['known_frr','known_macro_frr','unknown_recall','novel_unknown_recall','known_correct_accepted','auroc_unknown','oracle_novel90_known_frr']}))
    base.csv_write(OUT/'method_summary.csv',method_summary)
    roles=[]
    for role in ['auxiliary_species','novel_species']:
        species=sorted({c for p in pairs for c in p[role].split('|')})
        for c in species:
            pp=[p for p in pairs if c in p[role].split('|')]
            roles.append(dict(role=role,species=c,rounds=len(pp),seq_lower_frr=sum(p['frr_delta']>0 for p in pp),
                              mean_frr_delta=float(np.mean([p['frr_delta'] for p in pp]))))
    base.csv_write(OUT/'species_role_associations.csv',roles)
    plot(pairs)
    report(summary,method_summary,pairs)
    print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)


def plot(pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    colors=['#376798','#d37f3a','#56967b']
    for i,color in zip([1,2,3],colors):
        pp=[p for p in pairs if p['camera_partition']==i]
        axes[0].scatter([p['seq_frr']*100 for p in pp],[p['within_frr']*100 for p in pp],s=28,alpha=.7,color=color,label=f'Camera split {i}')
        axes[1].plot(range(30),[p['frr_delta']*100 for p in pp],marker='.',linewidth=1,color=color,label=f'Camera split {i}')
        axes[2].scatter([p['novel_recall_delta']*100 for p in pp],[p['frr_delta']*100 for p in pp],s=28,alpha=.7,color=color)
    values=[p[k]*100 for p in pairs for k in ['seq_frr','within_frr']]
    lim=max(values)*1.05;axes[0].plot([0,lim],[0,lim],color='gray',linewidth=1,linestyle='--')
    axes[0].set(xlabel='Seq Memory known FRR (%)',ylabel='Within-Seq known FRR (%)',title='Above diagonal: Seq Memory rejects fewer Known',xlim=(0,lim),ylim=(0,lim))
    axes[1].axhline(0,color='gray',linewidth=1);axes[1].set(xlabel='Balanced class-role rotation',ylabel='Within minus Seq known FRR (pp)',title='Direction across all 90 rounds')
    axes[2].axhline(0,color='gray',linewidth=1);axes[2].axvline(0,color='gray',linewidth=1);axes[2].set(xlabel='Within minus Seq novel recall (pp)',ylabel='Within minus Seq known FRR (pp)',title='Rejection benefit and Known cost')
    for ax in axes:ax.grid(alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle('Local reviewed species rotations: 20 Known / 5 auxiliary Unknown / 5 novel Unknown\nSeparate Val90 calibration; camera-disjoint bank/calibration/test; 30 roles x 3 camera splits',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)


def report(summary,methods,pairs):
    s=summary[0]
    def pct(v):return f'{v*100:.2f}%'
    def table(rows,keys):
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
    method_table=table([{'方法':r['method'],'已知微误拒':pct(r['known_frr']),'已知宏误拒':pct(r['known_macro_frr']),
                         '新未知召回':pct(r['novel_unknown_recall']),'正确且接受':pct(r['known_correct_accepted']),
                         'AUROC':f"{r['auroc_unknown']:.4f}",'测试ROC新未知90%时已知误拒':pct(r['oracle_novel90_known_frr'])} for r in methods],
                       ['方法','已知微误拒','已知宏误拒','新未知召回','正确且接受','AUROC','测试ROC新未知90%时已知误拒'])
    summary_table=table([{'相机划分':r['scope'],'轮数':r['rounds'],'Seq微误拒较低':r['seq_lower_frr'],
                          'Within微误拒较低':r['within_lower_frr'],'Seq宏误拒较低':r['seq_lower_macro_frr'],
                          'Memory k3误拒较低':r['memory_k3_lower_frr'],
                          '同等测试新未知90%时Seq误拒较低':r['seq_lower_oracle_novel90_frr']} for r in summary],
                        ['相机划分','轮数','Seq微误拒较低','Within微误拒较低','Seq宏误拒较低','Memory k3误拒较低','同等测试新未知90%时Seq误拒较低'])
    reverse=sorted([p for p in pairs if p['frr_delta']<0],key=lambda p:p['frr_delta'])
    reverse_table=table([{'实验':p['experiment'],'辅助未知':p['auxiliary_species'],'测试新未知':p['novel_species'],
                          'Seq误拒':pct(p['seq_frr']),'Within误拒':pct(p['within_frr']),
                          '误拒差(百分点)':f"{p['frr_delta']*100:+.2f}"} for p in reverse[:15]],
                        ['实验','辅助未知','测试新未知','Seq误拒','Within误拒','误拒差(百分点)']) if reverse else '<p>本次没有反向轮次。</p>'
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>本地未知物种轮换</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:0 24px;color:#243341}}table{{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}}td,th{{padding:8px;border-bottom:1px solid #dce2e7;text-align:left}}th{{background:#f0f3f6}}img{{width:100%}}h1{{font-size:26px}}h2{{font-size:21px}}.note{{padding:16px;background:#f3f6f8}}</style>
<h1>Seq Memory 的低误拒率能否跨未知物种保持？</h1>
<p>90 轮本地已校验数据对照中，Seq Memory 的已知微平均误拒率低于 Within-Seq 的轮数为 <b>{s['seq_lower_frr']}/90</b>，Within-Seq 较低为 {s['within_lower_frr']}/90。Within−Seq 的误拒差平均 {s['mean_frr_delta']*100:+.2f} 个百分点，中位数 {s['median_frr_delta']*100:+.2f}，第10–90百分位为 {s['p10_frr_delta']*100:+.2f} 至 {s['p90_frr_delta']*100:+.2f}。</p>
<img src="comparison.png" alt="90轮误拒率对照与拒识召回权衡">
{method_table}{summary_table}
<h2>协议</h2>
<p>使用全部已校验序列代表特征缓存：85类、7709个代表特征，编码器与预处理冻结。仅根据数据元信息筛选有至少20个纯物种查询序列代表、至少6台相机的候选；分别用种子 {SEEDS} 搜索3000次全局相机 50%/25%/25% 划分，优化类别覆盖和样本比例，不读取预测分数。取三个划分中均有至少5个建库、3个校准、3个测试代表的30类交集。单相机和稀疏物种不进入这次轮换，不把它们固定充当未知。</p>
<p>每个相机划分用一个预先固定的类别排列做30次循环位移：前20类Known、随后5类辅助Unknown、最后5类新Unknown。所有30类每个相机划分恰好Known20次、辅助未知5次、新未知5次；合计Known60次、辅助15次、新未知15次。未知物种不进入示例库、分类中心、质心或白化估计；每轮从当轮Known独立重建两种头。两个方法共享bank/cal/test样本；各自只在辅助未知校准Val90。新未知校准相机的图像不用于阈值或测试。</p>
<p>bank/cal/test相机完全分离。不同真值物种混合的查询时间代理序列整体移除，避免按标签拆组；同时检查图片、原图、哈希、burst、acquisition隔离。只评估单帧已校验序列代表，不把本结果当作多帧聚合收益。固定库仍保留审核通过的多物种采集组中各自物种裁剪。</p>
<p>原Seq Memory为k=1、类别质心权重0.5、margin=1。Within-Seq分类k=3，拒识使用层次类内白化、自适应邻域。Memory k3 control逐项复制Within-Seq的分类中心、示例和质心，用原Memory拒识分数；其每张闭集预测必须与Within-Seq完全一致。这一对照用于检查差异是否来自拒识分支。</p>
<h2>同等未知召回的诊断</h2>
<p>辅助集Val90并不使两模型的测试未知召回相等。为区分单纯阈值松紧与分数分离度，另外从新未知测试分数的ROC取达到90%召回的最小严格阈值并核对Known误拒。该“oracle”值使用测试真值，仅是回顾性判别能力诊断，不是可部署阈值；不用于选择主实验参数。此条件下Seq的误拒较低为 {s['seq_lower_oracle_novel90_frr']}/90 轮，平均Within−Seq误拒差 {s['mean_oracle_novel90_frr_delta']*100:+.2f} 个百分点。另保存Val80/85/90/95辅助阈值敏感性。</p>
<h2>反向情况（误拒率 Within-Seq 更低）</h2>{reverse_table}
<p>完整反例见counterexamples.csv；species_role_associations.csv是物种参与角色时的条件均值，其他物种同时变化，不能解释成单个物种的因果作用。</p>
<p class="note">这些是有限本地数据的回顾性稳健性检查，90轮共享图像和物种，不是90个独立样本，胜率不作独立二项检验。按相机的配对bootstrap区间只反映单轮采样差异。时间分组是恢复的采集代理。结论覆盖所选30类和当前实现；不能推广为所有物种、现场新域或论文原模型必然成立。已知宏平均误拒与微平均同时报告，避免大类主导结论。初始编码器无梯度更新。</p>
<p><a href="results.csv">逐轮结果</a> · <a href="paired_comparison.csv">配对差值</a> · <a href="counterexamples.csv">全部反例</a> · <a href="role_schedule.csv">物种角色轮换</a> · <a href="protocol.json">协议</a> · <a href="verification.json">独立验证</a> · <code>scripts/benchmark_unknown_rotations.py</code></p></html>'''
    (OUT/'REPORT.html').write_text(text,encoding='utf8')


def main():
    root=FEATURE_DIR
    data=base.load_data('Local reviewed rotation',root/'snapshot.json',root/'features.npz')
    common,partitions,candidates=camera_splits(data)
    schedule=[]
    for camera,d in enumerate(partitions,1):
        order=np.random.default_rng(d['seed']+100000).permutation(common).tolist()
        for rotation in range(30):
            rotated=order[rotation:]+order[:rotation]
            for j,c in enumerate(rotated):
                schedule.append(dict(camera_partition=camera,rotation=rotation,species=c,
                                     role='Known' if j<20 else 'Auxiliary Unknown' if j<25 else 'Novel Unknown'))
    base.csv_write(OUT/'role_schedule.csv',schedule)
    for c in common:
        assert Counter(r['role'] for r in schedule if r['species']==c)==dict(Known=60,**{'Auxiliary Unknown':15,'Novel Unknown':15})
    save(OUT/'protocol.json',dict(seeds=SEEDS,rotations_per_camera=30,total_rounds=90,known_classes_per_round=20,
        auxiliary_unknown_classes_per_round=5,novel_unknown_classes_per_round=5,balanced_role_counts_per_species=dict(known=60,auxiliary=15,novel=15),
        metadata_candidates=candidates,included_species=common,source_features_sha256=data['features_sha256'],
        source_metadata_sha256=data['metadata_sha256'],feature_source=data['features_path'],metadata_source=data['metadata_path'],
        encoder_sha256=data['encoder_sha256'],script_sha256=base.sha(Path(__file__)),
        partitions=[dict(seed=d['seed'],metadata_trial=d['trial'],eligible_species=sorted(d['eligible']),
                         pool_counts={p:len(ii) for p,ii in d['pools'].items()},leakage_audit=d['audit']) for d in partitions],
        target=.9,selection_uses_scores=False,primary='single-frame actual Neri runtime, separately calibrated Val90',
        controls='same exact Within-Seq classification arrays, Memory k3 rejection; retrospective test ROC novel90',
        no_installed_model_changes=True,overlapping_rotations_not_independent=True))
    print('Frozen protocol:',common,flush=True)
    start=time.perf_counter()
    for camera,d in enumerate(partitions,1):
        for rotation in range(30):
            rr=[r for r in schedule if r['camera_partition']==camera and r['rotation']==rotation]
            roles={role:{r['species'] for r in rr if r['role']==role} for role in ['Known','Auxiliary Unknown','Novel Unknown']}
            run_rotation(data,d['pools'],common,roles['Known'],roles['Auxiliary Unknown'],roles['Novel Unknown'],camera,rotation)
            if (rotation+1)%5==0:
                print(f'PROGRESS {30*(camera-1)+rotation+1}/90; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    analysis()
    print('Completed:',OUT/'REPORT.html',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='Neri_plus checkout providing training modules')
    parser.add_argument('--feature-dir', type=Path, required=True, help='Reviewed-run features/ with snapshot.json and features.npz')
    parser.add_argument('--output-dir', type=Path, required=True, help='New experiment directory')
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if OUT.exists():
        raise FileExistsError(f"Use a new output directory: {OUT}")
    FEATURE_DIR = args.feature_dir.resolve()
    base.configure_source(args.source_root)
    base.OUT = OUT
    OUT.mkdir(parents=True)
    with threadpool_limits(limits=4):
        main()
