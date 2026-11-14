"""C-only joint calibration with camera/component OOF checks and a frozen E.

The deployed model and its automatic-learning policy are not modified. All
candidate policies are declared before C fitting and saved before reading E.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.compare_local_dinov3_protocols import camera_components, csvwrite, freeze, metrics, read, rows_read, write
from scripts.expand_dinov3_calibration import paired_interval, signature
from system.dinov3.balanced_unknown_calibration import fit_unknown_threshold
from system.dinov3.encoder import sha256
from system.dinov3.experiment import pooled_evidence
from system.dinov3.joint_calibration import fit_joint_threshold, threshold_frontier
from system.dinov3.model import DinoV3RidgeWithinModel

SEED = 20261114
CASES = {"C27": "baseline", "C47": "expanded_same_model"}
POLICIES = {
    "joint_equal_cost": dict(unknown_cost=.5),
    "joint_unknown_cost2": dict(unknown_cost=2/3),
    "joint_unknown90": dict(unknown_cost=.5, min_unknown_rejection=.90),
    "joint_known80_unknown90": dict(unknown_cost=.5, min_unknown_rejection=.90, min_known_acceptance=.80),
}


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def normalization(pooled, radial, known):
    def stats(values):
        median = float(np.median(values[known]))
        return dict(median=median, scale=max(float(1.4826*np.median(abs(values[known]-median))), 1e-6))
    return dict(margin=stats(pooled), radial=stats(radial))


def gate_scores(pooled, radial, stats):
    m, r = stats["margin"], stats["radial"]
    return (pooled-m["median"])/m["scale"]-.08*(radial-r["median"])/r["scale"]


def make_group_folds(groups, known, max_folds=5):
    """Metadata-only balancing; whole alias-linked camera groups stay together."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    u_groups = [g for g in unique if (~known[groups == g]).any()]
    nfold = min(max_folds, len(u_groups))
    if nfold < 2:
        raise ValueError("At least two independent Unknown camera groups are required")
    rank = lambda g: hashlib.sha256(f"{SEED}:{g}".encode()).hexdigest()
    masses = {g: (int((groups == g).sum()), int((~known[groups == g]).sum())) for g in unique}
    sizes, unknowns = np.zeros(nfold, dtype=int), np.zeros(nfold, dtype=int)
    assigned = {}
    for g in sorted(u_groups, key=lambda g: (-masses[g][1], rank(g))):
        fold = min(range(nfold), key=lambda f: (unknowns[f], sizes[f], f))
        assigned[g] = fold
        sizes[fold] += masses[g][0]
        unknowns[fold] += masses[g][1]
    for g in sorted(set(unique)-set(u_groups), key=lambda g: (-masses[g][0], rank(g))):
        fold = min(range(nfold), key=lambda f: (sizes[f], f))
        assigned[g] = fold
        sizes[fold] += masses[g][0]
    folds = np.asarray([assigned[g] for g in groups], dtype=np.int64)
    for fold in range(nfold):
        validation = folds == fold
        if not known[validation].any() or not (~known[validation]).any():
            raise ValueError("A camera validation fold lacks Known or Unknown; do not fabricate an OOF result")
    return folds


def fit_policies(scores, labels, known):
    reference = fit_unknown_threshold(scores[~known], labels[~known], target=.9, weighting="species")
    selected = {"macro90": dict(reference, feasible=True, policy="unknown_only_macro90")}
    selected.update({name: fit_joint_threshold(scores, labels, known, **config) for name, config in POLICIES.items()})
    return selected


def cross_validate(data, pooled, radial, groups):
    known = np.isin(data["labels"], data["classes"])
    folds = make_group_folds(groups, known)
    oof = {name: np.zeros(len(known), dtype=bool) for name in ("macro90", *POLICIES)}
    covered = {name: np.zeros(len(known), dtype=bool) for name in oof}
    fold_records = []
    for fold in np.unique(folds):
        validation = folds == fold
        train = ~validation
        # Median/MAD also exclude the held-out fold, including its role labels.
        stats = normalization(pooled[train], radial[train], known[train])
        train_scores = gate_scores(pooled[train], radial[train], stats)
        validation_scores = gate_scores(pooled[validation], radial[validation], stats)
        for name, fit in fit_policies(train_scores, data["labels"][train], known[train]).items():
            record = dict(fold=int(fold), policy=name, feasible=fit["feasible"], threshold=fit["threshold"],
                          train_images=int(train.sum()), validation_images=int(validation.sum()))
            if fit["feasible"]:
                accepted = validation_scores >= fit["threshold"]
                oof[name][validation] = accepted
                covered[name][validation] = True
                record.update(metrics(data["labels"][validation], data["predictions"][validation], accepted, data["classes"]))
            fold_records.append(record)
    summary = []
    for name in oof:
        row = dict(policy=name, folds=int(folds.max()+1), complete_coverage=bool(covered[name].all()),
                   covered_photos=int(covered[name].sum()), total_photos=len(known))
        if row["complete_coverage"]:
            row.update(metrics(data["labels"], data["predictions"], oof[name], data["classes"]))
        summary.append(row)
    return folds, oof, covered, summary, fold_records


def run(args):
    if (args.out/"thresholds_frozen_before_E.json").exists():
        if not (args.out/"verification.json").exists() or not read(args.out/"verification.json")["passed"]:
            raise FileExistsError("Frozen selection already exists; use a new --out to retry")
        print("REUSE completed frozen joint comparison", flush=True)
        return
    source_plan = read(args.expansion/"plan.json")
    if sha256(source_plan["baseline_manifest"]) != source_plan["baseline_manifest_sha256"]:
        raise ValueError("Installed experimental model changed; refresh the baseline comparison first")
    if not read(args.expansion/"verification.json")["passed"]:
        raise ValueError("Source expansion comparison is not verified")
    sources = {case: args.expansion/folder for case, folder in CASES.items()}
    freeze(args.out/"protocol.json", dict(seed=SEED, policies=POLICIES,
           populations="Known and Unknown each have their own species-equal weights; image totals are not priors",
           primary="joint_equal_cost", tie_break="least strict cutoff among minimum-loss candidates",
           threshold_selection="C only; all policies reported; no E best-policy selection",
           cross_validation="5 camera-component folds; train-fold Known median/MAD only; C history is label-free and within camera",
           sources={case: dict(folder=str(folder), C_predictions_sha256=sha256(folder/"predictions_C.npz"),
                               C_inputs_sha256=sha256(folder/"calibration_inputs.npz"),
                               checkpoint_sha256=sha256(folder/"model.neri-model")) for case, folder in sources.items()},
           baseline_manifest_sha256=source_plan["baseline_manifest_sha256"],
           B_C_camera_sharing=dict(C27=0, C47=len(source_plan["audits"]["expanded_same_model"]["B_C_shared_cameras"])) ,
           production_policy_changed=False))
    rows = rows_read(args.comparison/"current_reviewed.jsonl")
    if sha256(args.comparison/"current_reviewed.jsonl") != source_plan["reviewed_manifest_sha256"]:
        raise ValueError("Reviewed metadata differs from the source experiment")
    components, _ = camera_components(rows)
    row_group = {i: g for g, indices in enumerate(components) for i in indices}
    model = DinoV3RidgeWithinModel.load(sources["C27"]/"model.neri-model")
    model_signature = signature(model)
    selection, cv_summary, cv_folds, C_data, C_metrics = {}, [], [], {}, []
    for case, folder in sources.items():
        print("FIT C-only joint policies and camera CV", case, flush=True)
        data = load_npz(folder/"predictions_C.npz")
        inputs = load_npz(folder/"calibration_inputs.npz")
        np.testing.assert_array_equal(data["labels"], inputs["labels"])
        np.testing.assert_array_equal(data["cameras"], inputs["cameras"])
        other = DinoV3RidgeWithinModel.load(folder/"model.neri-model")
        if signature(other) != model_signature:
            raise ValueError("C27/C47 classification parameters differ")
        within = model._compute_within_scores(inputs["features"], adapted_features=inputs["adapted_features"])
        norm = np.linalg.norm((inputs["adapted_features"].astype(np.float64)-model.origin_)@model.transform_, axis=1)
        _, pooled, radial = pooled_evidence(within, norm, inputs["cameras"], inputs["sequences"])
        known = np.isin(data["labels"], data["classes"])
        stats = normalization(pooled, radial, known)
        reconstructed = gate_scores(pooled, radial, stats)
        np.testing.assert_allclose(reconstructed, data["known_score"], atol=1e-10, rtol=0)
        fits = fit_policies(data["known_score"], data["labels"], known)
        np.testing.assert_allclose(fits["macro90"]["threshold"], data["threshold"], atol=1e-10, rtol=0)
        C_data[case] = data
        groups = np.asarray([row_group[int(i)] for i in data["indices"]])
        folds, oof, covered, summary, fold_details = cross_validate(data, pooled, radial, groups)
        cv_summary.extend(dict(case=case, **r) for r in summary)
        cv_folds.extend(dict(case=case, **r) for r in fold_details)
        case_out = args.out/case
        case_out.mkdir(exist_ok=True)
        np.savez_compressed(case_out/"C_components_and_folds.npz", indices=data["indices"], labels=data["labels"],
                            cameras=data["cameras"], groups=groups, folds=folds, pooled_margin=pooled, radial=radial,
                            **{name+"_oof_accepted": flags for name, flags in oof.items()},
                            **{name+"_oof_covered": flags for name, flags in covered.items()})
        frontier = threshold_frontier(data["known_score"], data["labels"], known)
        np.savez_compressed(case_out/"C_threshold_frontier.npz", **frontier)
        for name, fit in fits.items():
            if fit["feasible"]:
                accepted = data["known_score"] >= fit["threshold"]
                C_metrics.append(dict(case=case, policy=name, **metrics(data["labels"], data["predictions"], accepted, data["classes"])))
        selection[case] = dict(known_photos=int(known.sum()), unknown_photos=int((~known).sum()),
                               known_species=len(np.unique(data["labels"][known])), unknown_species=len(np.unique(data["labels"][~known])),
                               normalization=stats, fits=fits)
    write(args.out/"camera_CV_summary.json", cv_summary)
    write(args.out/"camera_CV_folds.json", cv_folds)
    # The only full-C fitted candidate thresholds. No E labels or scores accessed yet.
    freeze(args.out/"thresholds_frozen_before_E.json", dict(protocol_sha256=sha256(args.out/"protocol.json"),
           CV_summary_sha256=sha256(args.out/"camera_CV_summary.json"), selection=selection))

    baseline_E = load_npz(sources["C27"]/"predictions_E.npz")
    installed_cutoff = model.threshold
    base = dict(predictions=baseline_E["predictions"], accepted=baseline_E["known_score"] >= installed_cutoff)
    np.testing.assert_array_equal(base["accepted"], baseline_E["accepted"])
    records = [dict(case="installed", policy="current_macro90", threshold=installed_cutoff,
                    **metrics(baseline_E["labels"], base["predictions"], base["accepted"], baseline_E["classes"]))]
    per_species, intervals = [], []
    for case, folder in sources.items():
        e = load_npz(folder/"predictions_E.npz")
        for key in ("indices", "labels", "cameras", "predictions", "classes", "known_score"):
            np.testing.assert_array_equal(e[key], baseline_E[key])
        for name, fit in selection[case]["fits"].items():
            if not fit["feasible"]:
                continue
            accepted = e["known_score"] >= fit["threshold"]
            result = dict(predictions=e["predictions"], accepted=accepted)
            records.append(dict(case=case, policy=name, threshold=fit["threshold"],
                                **metrics(e["labels"], e["predictions"], accepted, e["classes"])))
            intervals.extend(dict(case=case, policy=name, **r) for r in paired_interval(
                e["labels"], e["cameras"], e["classes"], base, result))
            np.savez_compressed(args.out/case/f"E_{name}.npz", indices=e["indices"], labels=e["labels"],
                                cameras=e["cameras"], classes=e["classes"], predictions=e["predictions"],
                                accepted=accepted, known_score=e["known_score"], threshold=np.asarray(fit["threshold"]))
            # Offline candidate descriptor, deliberately not a software deployment manifest.
            write(args.out/case/f"{name}.joint.json", dict(kind="offline_joint_calibration_candidate",
                  model_checkpoint=str(sources[case]/"model.neri-model"), model_checkpoint_sha256=sha256(sources[case]/"model.neri-model"),
                  active_objective=fit, score_normalization=selection[case]["normalization"],
                  requires_joint_feedback_policy_before_deployment=True))
            for label in np.unique(e["labels"]):
                mask = e["labels"] == label
                per_species.append(dict(case=case, policy=name, species=label, known=bool(label in e["classes"]),
                                        photos=int(mask.sum()), rejected=int((~accepted[mask]).sum()),
                                        correct_accepted=int(((e["predictions"][mask] == label)&accepted[mask]).sum())))
    write(args.out/"C_metrics.json", C_metrics)
    write(args.out/"E_metrics.json", records)
    csvwrite(args.out/"E_metrics.csv", records)
    csvwrite(args.out/"per_species.csv", per_species)
    csvwrite(args.out/"paired_camera_intervals.csv", intervals)
    for r in records:
        if r["case"] == "installed":
            continue
        with np.load(args.out/r["case"]/f"E_{r['policy']}.npz", allow_pickle=False) as saved:
            rebuilt = metrics(saved["labels"], saved["predictions"], saved["accepted"], saved["classes"])
            for key, value in rebuilt.items():
                if value is None:
                    assert r[key] is None
                else:
                    np.testing.assert_allclose(value, r[key], atol=1e-12, rtol=0)
        fit = selection[r["case"]]["fits"][r["policy"]]
        c = C_data[r["case"]]
        cm = metrics(c["labels"], c["predictions"], c["known_score"] >= fit["threshold"], c["classes"])
        if fit.get("min_unknown_rejection") is not None:
            assert cm["unknown_rejection_macro"] >= fit["min_unknown_rejection"]-1e-12
        if fit.get("min_known_acceptance") is not None:
            assert 1-cm["known_false_rejection_macro"] >= fit["min_known_acceptance"]-1e-12
    if sha256(source_plan["baseline_manifest"]) != source_plan["baseline_manifest_sha256"]:
        raise ValueError("Installed model changed during comparison")
    write(args.out/"verification.json", dict(passed=True, installed_model_unchanged=True, fixed_E_images=len(baseline_E["labels"]),
          thresholds_frozen_before_E=True, E_only_changes_acceptance_not_classification=True,
          raw_C_scores_reconstructed=True, camera_group_CV_with_train_only_normalization=True,
          feasible_constraints_verified_on_C=True, all_E_metrics_recomputed_from_saved_decisions=True))
    print("E RESULTS", json.dumps(records, ensure_ascii=False), flush=True)


def report(args):
    selection = read(args.out/"thresholds_frozen_before_E.json")["selection"]
    records = read(args.out/"E_metrics.json")
    cnames = dict(C27="原 C：27 张未知", C47="扩充 C：47 张未知", installed="已安装模型")
    pnames = dict(current_macro90="当前门限", macro90="未知 Macro90", joint_equal_cost="联合：两种错误等成本",
                  joint_unknown_cost2="联合：未知误接纳成本×2", joint_unknown90="联合：未知拒识下限90%",
                  joint_known80_unknown90="联合约束：已知接纳80%＋未知拒识90%")
    lines = ["# 已知／未知联合校准对照", "",
             "分类模型和分数保持不变，只用 C 选择门限。两组内部按物种等权，已知与未知的照片总量不决定权重。", "",
             "联合主目标为 0.5×已知宏误拒率＋0.5×未知宏误接纳率；保守目标把未知错误成本加倍。另固定报告未知拒识下限90%的方案，以及已知接纳80%＋未知拒识90%的可行性。所有方案均在 E 评分前固定，未根据 E 挑选权重或最好门限。", "",
             "|校准数据|方案|门限|E 已知误拒|E 正确接纳|E 未知拒识|", "|---|---|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"|{cnames[r['case']]}|{pnames[r['policy']]}|{r['threshold']:.6f}|{r['known_false_rejection_macro']*100:.2f}%|{r['known_correct_acceptance_macro']*100:.2f}%|{r['unknown_rejection_macro']*100:.2f}%|")
    lines += ["", "E 始终是相同的 43 个已知物种、14,449 张及 21 个独立未知物种、28 张；全部 E 比例为物种宏平均。分类预测没有变化。", "", "## 联合约束的可行性", ""]
    for case, block in selection.items():
        fit = block["fits"]["joint_known80_unknown90"]
        lines.append(f"{cnames[case]}：已知接纳≥80%且未知拒识≥90%，{'可行' if fit['feasible'] else '不可行'}。在未知拒识≥90%时，已知接纳最多 {fit['maximum_known_acceptance_at_unknown_constraint']*100:.2f}%；在已知接纳≥80%时，未知拒识最多 {fit['maximum_unknown_rejection_at_known_constraint']*100:.2f}%。这些是 C 上对全部可实现门限枚举的结果。")
    lines += ["", "## C 内部的相机分组交叉验证", "",
              "按相机／原图别名关联组件分成5折。每折的 median/MAD 和门限都仅使用训练折；已冻结的 Within/Ridge 不重新拟合。历史和序列证据均不跨相机折，避免把验证折的已知标签用于标准化。", "",
              "|数据|方案|OOF已知误拒|OOF未知拒识|完整覆盖|", "|---|---|---:|---:|---|"]
    for r in read(args.out/"camera_CV_summary.json"):
        if r["policy"] not in pnames:
            continue
        pct = lambda key: f"{r[key]*100:.2f}%" if key in r else "—"
        lines.append(f"|{cnames[r['case']]}|{pnames[r['policy']]}|{pct('known_false_rejection_macro')}|{pct('unknown_rejection_macro')}|{r['covered_photos']}/{r['total_photos']}|")
    lines += ["", "原 C 的 B/C/E 相机目录隔离；扩充 C 与 B 共享10个相机目录，但连拍、原图和裁图内容隔离。两个 C 均与 E 相机及 E Unknown 物种隔离。C 中已知类覆盖42类，E覆盖43类；不能把 C 或 OOF 指标当作 E 成绩。", "",
              "校准与测试未知样本仍很少，E 此前已曝光，结果为历史诊断。联合无约束目标可能放宽未知拒识；不能将降低误拒表述为同时保证未知拒识90%。相机配对区间见 paired_camera_intervals.csv。", "",
              "保存的 *.joint.json 为离线候选描述，当前软件模型和自动学习仍使用原 Macro90。若要部署联合候选，需要同时接入联合重校准策略，避免补图后退回原目标。本轮没有替换当前模型。"]
    (args.out/"REPORT.md").write_text("\n".join(lines)+"\n", "utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion", type=Path, default=REPO/"runs/expanded_calibration_20261114")
    parser.add_argument("--comparison", type=Path, default=REPO/"runs/local_protocol_comparison_20261114")
    parser.add_argument("--out", type=Path, default=REPO/"runs/joint_calibration_20261114")
    parser.add_argument("--stage", choices=("run", "report"), default="run")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        if args.stage == "run":
            run(args)
        report(args)
    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
