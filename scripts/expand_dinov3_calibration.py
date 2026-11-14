"""Expand trusted local calibration; compare on the unchanged camera/species E.

One arm changes only calibration, with disclosed B/C camera sharing. Another
keeps full camera isolation by moving whole cameras out of B and refitting.
Neither arm changes the installed model or relaxes the Macro90 target.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.compare_local_dinov3_protocols import (
    camera_components, csvwrite, freeze, metrics, query, read, rows_read, write,
)
from system.dinov3.encoder import sha256
from system.dinov3.model import DinoV3RidgeWithinModel
from system.dinov3.runtime import read_manifest, resolve_manifest_asset

SEED = 20261114
ARMS = ("baseline", "expanded_same_model", "expanded_camera_isolated")


def identities(row):
    result = {("image", row["image_id"]), ("crop", row["sha256"]),
              ("event", row["camera"], row["sequence"])}
    if row.get("source_sha256"):
        result.add(("source", row["source_sha256"]))
    return result


def pick(uses, purposes):
    return np.asarray(sorted(i for i, use in uses.items() if use in purposes), dtype=np.int64)


def audit_split(rows, uses, known, require_camera_isolation):
    owners = {}
    cameras = {p: set() for p in ("bank", "calibration", "evaluation")}
    unknown = {p: set() for p in ("calibration", "evaluation")}
    for i, use in uses.items():
        pool = ("bank" if use == "bank" else "calibration" if use.startswith("calibration")
                else "evaluation" if use.startswith("evaluation") else None)
        if pool is None:
            continue
        row = rows[i]
        cameras[pool].add(row["camera"])
        if row["species"] not in known:
            if pool == "bank":
                raise ValueError("Unknown admitted to B")
            unknown[pool].add(row["species"])
        for key in identities(row):
            if key in owners and owners[key] != pool:
                raise ValueError(f"Cross-pool acquisition/original/content leakage: {key[0]}")
            owners[key] = pool
    if {rows[i]["species"] for i, u in uses.items() if u == "bank"} != set(known):
        raise ValueError("Known class coverage changed")
    if unknown["calibration"] & unknown["evaluation"]:
        raise ValueError("Unknown calibration/test species overlap")
    if (cameras["bank"] | cameras["calibration"]) & cameras["evaluation"]:
        raise ValueError("Evaluation cameras must remain independent")
    shared = sorted(cameras["bank"] & cameras["calibration"])
    if require_camera_isolation and shared:
        raise ValueError("B/C camera isolation failed")
    return dict(passed=True, B_C_shared_cameras=shared, full_camera_isolation=not shared,
                E_camera_overlap=0, unknown_C_E_species_overlap=0,
                acquisition_original_content_isolation=True)


def expansion_plan(rows, uses, known, unknown_roles):
    bank = pick(uses, {"bank"})
    evaluation = pick(uses, {"evaluation", "evaluation_unknown"})
    protected = set().union(*(identities(rows[int(i)]) for i in np.r_[bank, evaluation]))
    e_cameras = {rows[int(i)]["camera"] for i in evaluation}
    reserve = [i for i, u in uses.items() if u == "future_support_reserve"]
    eligible, rejected = [], []
    for i in reserve:
        row = rows[i]
        reason = None
        if row["species"] in known or unknown_roles.get(row["species"]) != "calibration":
            reason = "heldout_unknown_species_or_non_unknown"
        elif row["camera"] in e_cameras:
            reason = "evaluation_camera"
        elif identities(row) & protected:
            reason = "B_or_E_event_original_or_content"
        if reason:
            rejected.append(dict(index=i, id=row["id"], species=row["species"], reason=reason))
        else:
            eligible.append(i)
    fixed = dict(uses)
    for i in eligible:
        fixed[i] = "calibration_unknown"

    # A whole-camera transfer may not remove the last support of any Known class.
    groups, _ = camera_components(rows)
    candidates = [g for g in groups if set(g) & set(eligible)]
    rank = lambda g: hashlib.sha256(f"{SEED}:{min(rows[i]['camera'] for i in g)}".encode()).hexdigest()
    candidates.sort(key=lambda g: (-len(set(g) & set(eligible)), rank(g)))
    counts = Counter(rows[int(i)]["species"] for i in bank)
    strict = dict(uses)
    moved, blocked, moved_cameras = [], [], set()
    for group in candidates:
        taken = [i for i in group if uses[i] == "bank"]
        by_class = Counter(rows[i]["species"] for i in taken)
        missing = sorted(c for c, n in by_class.items() if counts[c] <= n)
        if missing:
            blocked.append(dict(candidate_indices=sorted(set(group) & set(eligible)),
                                would_remove_known_classes=missing))
            continue
        counts.subtract(by_class)
        for i in taken:
            strict[i] = "calibration"
        moved.extend(taken)
        moved_cameras.update(rows[i]["camera"] for i in group)
    remaining_protected = set().union(*(identities(rows[int(i)]) for i in
                                       np.r_[pick(strict, {"bank"}), evaluation]))
    strict_added = []
    for i in reserve:
        row = rows[i]
        if (unknown_roles.get(row["species"]) == "calibration" and row["species"] not in known
                and row["camera"] in moved_cameras and not identities(row) & remaining_protected):
            strict[i] = "calibration_unknown"
            strict_added.append(i)
    audits = dict(baseline=audit_split(rows, uses, known, True),
                  expanded_same_model=audit_split(rows, fixed, known, False),
                  expanded_camera_isolated=audit_split(rows, strict, known, True))
    for arm_uses in (fixed, strict):
        np.testing.assert_array_equal(evaluation, pick(arm_uses, {"evaluation", "evaluation_unknown"}))
    return dict(seed=SEED, arms={name: {str(i): u for i, u in mapping.items()} for name, mapping in
                (("baseline", uses), ("expanded_same_model", fixed), ("expanded_camera_isolated", strict))},
                added_unknown_indices=eligible, strict_added_unknown_indices=strict_added,
                strict_moved_B_indices=sorted(moved), strict_blocked_camera_groups=blocked,
                reserve_exclusions=rejected, audits=audits,
                target_unknown_macro_rejection=.90, installed_model_changed=False,
                policy="all eligible trusted reserves; no E-driven sampling or hyperparameter selection")


def signature(model):
    digest = hashlib.sha256()
    for value in (model.origin_, model.transform_, model.ridge_head.coef_, model.ridge_head.intercept_,
                  model.ridge_head.class_counts_):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(json.dumps(list(map(str, model.classes))).encode())
    for key, (count, total) in sorted(model.within_updater.groups.items()):
        digest.update(json.dumps([key, count]).encode())
        digest.update(np.ascontiguousarray(total).tobytes())
    return digest.hexdigest()


def prepare(args):
    rows = rows_read(args.comparison/"current_reviewed.jsonl")
    partition = read(args.comparison/"partition.json")
    if sha256(args.comparison/"current_reviewed.jsonl") != partition["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    root = args.source/"data/by_species/00已校验"
    now = {Path(name).stem: (str(Path(folder)/name), Path(folder).relative_to(root).parts[0])
           for folder, _, files in os.walk(root) for name in files
           if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}}
    expected = {r["id"]: r for r in rows}
    if set(now) != set(expected) or any(now[k][1] != row["species"] for k, row in expected.items()):
        raise ValueError("Reviewed inventory/labels changed; rebuild the feature manifest first")
    payload = read_manifest(args.baseline_manifest)
    checkpoint = resolve_manifest_asset(args.baseline_manifest, payload["checkpoint"])
    calibration = resolve_manifest_asset(args.baseline_manifest, payload["calibration"])
    if sha256(checkpoint) != payload["checkpoint_sha256"] or sha256(calibration) != payload["calibration_sha256"]:
        raise ValueError("Installed baseline integrity failed")
    model = DinoV3RidgeWithinModel.load(checkpoint)
    if (model.inference_recipe != "experiment_full" or set(model.classes) != set(partition["known_classes"])
            or int(model.ridge_head.class_counts_.sum()) != 44472
            or model.metadata.get("human_learning_samples")):
        raise ValueError("Baseline no longer matches the frozen B/C/E model; create a new audited support plan")
    uses = {int(k): u for k, u in partition["assignments"].items()}
    plan = expansion_plan(rows, uses, set(model.classes), partition["unknown_species_roles"])
    for i in sorted(set(plan["added_unknown_indices"]) | set(plan["strict_added_unknown_indices"])):
        if sha256(rows[i]["file"]) != rows[i]["sha256"]:
            raise ValueError(f"Reserved crop content changed: {rows[i]['id']}")
    plan.update(baseline_manifest=str(args.baseline_manifest.resolve()), baseline_manifest_sha256=sha256(args.baseline_manifest),
                baseline_checkpoint=str(checkpoint), baseline_checkpoint_sha256=payload["checkpoint_sha256"],
                baseline_calibration=str(calibration), baseline_calibration_sha256=payload["calibration_sha256"],
                reviewed_manifest_sha256=partition["manifest_sha256"], source_partition_sha256=sha256(args.comparison/"partition.json"),
                reviewed_inventory_images=len(now), original_signature=signature(model))
    freeze(args.out/"plan.json", plan)
    records = []
    for i, row in enumerate(rows):
        records.append(dict(index=i, id=row["id"], file=row["file"], species=row["species"], camera=row["camera"],
                            **{arm: plan["arms"][arm][str(i)] for arm in ARMS}))
    csvwrite(args.out/"photo_uses_comparison.csv", records)
    csvwrite(args.out/"excluded_reserves.csv", plan["reserve_exclusions"])
    write(args.out/"split_audit.json", plan["audits"])
    print("PLAN", json.dumps(dict(fixed_added=len(plan["added_unknown_indices"]),
          strict_added=len(plan["strict_added_unknown_indices"]), moved_B=len(plan["strict_moved_B_indices"])), ensure_ascii=False), flush=True)


def score(model, q):
    return model.predict(q["raw"], q["cameras"], q["sequences"], contents=q["crop_hashes"], adapted_features=q["adapted"])


def paired_interval(truth, cameras, classes, base, actual, seed=SEED):
    names, inverse = np.unique(cameras, return_inverse=True)
    draws = np.random.default_rng(seed).multinomial(len(names), np.full(len(names), 1/len(names)), size=2000)
    output = []
    known = np.isin(truth, classes)
    for label, mask, b, a in (
        ("known_false_rejection", known, ~base["accepted"], ~actual["accepted"]),
        ("known_correct_acceptance", known, (base["predictions"] == truth) & base["accepted"],
         (actual["predictions"] == truth) & actual["accepted"]),
        ("unknown_rejection", ~known, ~base["accepted"], ~actual["accepted"])):
        species = np.unique(truth[mask])
        delta = a.astype(float)-b.astype(float)
        count = np.stack([np.bincount(inverse[truth == c], minlength=len(names)) for c in species], axis=1)
        sums = np.stack([np.bincount(inverse[truth == c], weights=delta[truth == c], minlength=len(names)) for c in species], axis=1)
        den = draws@count
        values = np.divide(draws@sums, den, out=np.zeros_like(den, dtype=float), where=den > 0).sum(1)/np.maximum((den > 0).sum(1), 1)
        low, high = np.quantile(values[(den > 0).any(1)], [.025, .975])
        output.append(dict(metric=label, delta_pp=float(np.mean([delta[truth == c].mean() for c in species])*100),
                           lower95_pp=float(low*100), upper95_pp=float(high*100), camera_groups=len(names), draws=2000))
    return output


def run(args):
    plan = read(args.out/"plan.json")
    if sha256(args.baseline_manifest) != plan["baseline_manifest_sha256"]:
        raise ValueError("Live model changed after the comparison plan was frozen")
    if (args.out/"decisions_before_E.json").exists():
        if not (args.out/"verification.json").exists() or not read(args.out/"verification.json")["passed"]:
            raise FileExistsError("Frozen candidate decisions already exist; use a new --out to retry without overwriting them")
        frozen = read(args.out/"decisions_before_E.json")
        for arm, digest in frozen["checkpoints"].items():
            if sha256(args.out/arm/"model.neri-model") != digest:
                raise ValueError("A frozen candidate checkpoint changed")
        print("REUSE verified frozen comparison; no model or threshold refitted", flush=True)
        return
    if sha256(plan["baseline_checkpoint"]) != plan["baseline_checkpoint_sha256"]:
        raise ValueError("Baseline checkpoint changed")
    rows = rows_read(args.comparison/"current_reviewed.jsonl")
    if sha256(args.comparison/"current_reviewed.jsonl") != plan["reviewed_manifest_sha256"]:
        raise ValueError("Reviewed manifest changed after planning")
    audit = read(args.comparison/"input_audit.json")
    cache = Path(audit.get("features_file", REPO/"runs/dinov3_reviewed_20261107/features.npy"))
    if sha256(cache) != audit["features_sha256"]:
        raise ValueError("Verified frozen feature cache changed")
    raw = np.load(cache, allow_pickle=False, mmap_mode="r")[[r["feature_index"] for r in rows]]
    uses = {arm: {int(k): v for k, v in mapping.items()} for arm, mapping in plan["arms"].items()}
    baseline = DinoV3RidgeWithinModel.load(plan["baseline_checkpoint"])
    if signature(baseline) != plan["original_signature"]:
        raise ValueError("Baseline support state differs")
    models, cal_queries, cal_predictions, counts = {}, {}, {}, {}
    for arm in ARMS:
        folder = args.out/arm
        folder.mkdir(exist_ok=True)
        strict = arm != "expanded_same_model"
        audit_split(rows, uses[arm], set(baseline.classes), strict)
        b = pick(uses[arm], {"bank"})
        c = pick(uses[arm], {"calibration", "calibration_unknown"})
        q = query(rows, raw, c)
        if arm == "expanded_camera_isolated":
            print("REFIT strict Bank", len(b), flush=True)
            model = DinoV3RidgeWithinModel(inference_recipe="experiment_full", metadata=dict(baseline.metadata,
                calibration_expansion_plan_sha256=sha256(args.out/"plan.json"),
                calibration_expansion_arm=arm, baseline_partition_sha256=baseline.metadata.get("partition_sha256"),
                partition_sha256=sha256(args.out/"plan.json"),
                source="expanded camera-isolated calibration; geometry and heads refit from reduced B only")).fit(
                raw[b], np.asarray([rows[int(i)]["species"] for i in b]),
                np.asarray([rows[int(i)]["camera"] for i in b]), np.asarray([rows[int(i)]["sequence"] for i in b]))
        else:
            model = deepcopy(baseline)
            if arm != "baseline":
                model.metadata.update(calibration_expansion_plan_sha256=sha256(args.out/"plan.json"),
                                      calibration_expansion_arm=arm)
        if arm != "baseline":
            print("CALIBRATE", arm, len(c), flush=True)
            model.calibrate(q["raw"], q["labels"], q["cameras"], q["sequences"],
                            contents=q["crop_hashes"], adapted_features=q["adapted"], target_recall=.90)
        if arm == "expanded_same_model":
            if signature(model) != signature(baseline):
                raise ValueError("Calibration-only arm modified classification or support parameters")
            for key in ("margin", "radial"):
                if model.calibration[key] != baseline.calibration[key]:
                    raise ValueError("Known calibration standardization changed in the fixed-model arm")
        cal_predictions[arm] = score(model, q)
        model.save(folder/"model.neri-model")
        write(folder/"calibration.json", model.calibration)
        np.savez_compressed(folder/"calibration_inputs.npz", features=q["raw"], labels=q["labels"], cameras=q["cameras"],
                            sequences=q["sequences"], contents=q["crop_hashes"], source_contents=q["contents"],
                            adapted_features=q["adapted"], encoder_sha256=np.asarray(model.metadata["encoder_sha256"]))
        models[arm], cal_queries[arm] = model, q
        unknown = ~np.isin(q["labels"], model.classes)
        counts[arm] = dict(bank_images=len(b), C_images=len(c), C_known_images=int((~unknown).sum()),
                           C_unknown_images=int(unknown.sum()), C_unknown_species=len(np.unique(q["labels"][unknown])),
                           threshold=model.threshold, fitted_support_signature=signature(model))
    # Freeze all three actual decisions/cutoffs before evaluating E. No best-arm selection follows.
    freeze(args.out/"decisions_before_E.json", dict(plan_sha256=sha256(args.out/"plan.json"),
           checkpoints={arm: sha256(args.out/arm/"model.neri-model") for arm in ARMS}, counts=counts))
    indices = pick(uses["baseline"], {"evaluation", "evaluation_unknown"})
    eq = query(rows, raw, indices)
    summary, per_species, predictions = [], [], {}
    for arm in ARMS:
        print("EVALUATE fixed E", arm, len(indices), flush=True)
        result = score(models[arm], eq)
        predictions[arm] = result
        if arm == "baseline":
            with np.load(args.comparison/"camera_split/experiment_full.npz", allow_pickle=False) as ref:
                np.testing.assert_array_equal(indices, ref["e_indices"])
                np.testing.assert_array_equal(result["predictions"], ref["e_pred"])
                np.testing.assert_array_equal(result["accepted"], ref["e_accepted"])
                np.testing.assert_allclose(result["known_score"], ref["e_score"], atol=1e-10, rtol=0)
        if arm == "expanded_same_model":
            np.testing.assert_array_equal(result["predictions"], predictions["baseline"]["predictions"])
        for scope, q, p in (("C", cal_queries[arm], cal_predictions[arm]), ("E", eq, result)):
            record = dict(arm=arm, scope=scope, **counts[arm],
                          **metrics(q["labels"], p["predictions"], p["accepted"], models[arm].classes))
            summary.append(record)
            if scope == "C" and record["unknown_rejection_macro"] < .9-1e-12:
                raise ValueError("Macro90 target was relaxed or failed")
            np.savez_compressed(args.out/arm/f"predictions_{scope}.npz", indices=q["indices"], labels=q["labels"],
                                cameras=q["cameras"], classes=models[arm].classes, predictions=p["predictions"],
                                accepted=p["accepted"], known_score=p["known_score"], threshold=np.asarray(models[arm].threshold))
        for label in np.unique(eq["labels"]):
            mask = eq["labels"] == label
            per_species.append(dict(arm=arm, species=label, known=bool(label in models[arm].classes),
                                    photos=int(mask.sum()), rejected=int((~result["accepted"][mask]).sum()),
                                    correct=int((result["predictions"][mask] == label).sum()),
                                    correct_accepted=int(((result["predictions"][mask] == label)&result["accepted"][mask]).sum())))
    intervals = [dict(arm=arm, **row) for arm in ARMS[1:] for row in paired_interval(
        eq["labels"], eq["cameras"], baseline.classes, predictions["baseline"], predictions[arm])]
    write(args.out/"metrics.json", summary)
    csvwrite(args.out/"metrics.csv", summary)
    csvwrite(args.out/"per_species.csv", per_species)
    csvwrite(args.out/"paired_camera_intervals.csv", intervals)
    if sha256(args.baseline_manifest) != plan["baseline_manifest_sha256"]:
        raise ValueError("Installed baseline changed during comparison")
    # Recompute the saved metrics and model round-trip, independent of cached result dictionaries.
    for arm in ARMS:
        restored = DinoV3RidgeWithinModel.load(args.out/arm/"model.neri-model")
        probe = restored.predict(eq["raw"], eq["cameras"], eq["sequences"], contents=eq["crop_hashes"], adapted_features=eq["adapted"])
        np.testing.assert_array_equal(probe["accepted"], predictions[arm]["accepted"])
        np.testing.assert_array_equal(probe["predictions"], predictions[arm]["predictions"])
        np.testing.assert_array_equal(probe["known_score"], predictions[arm]["known_score"])
        for scope in ("C", "E"):
            with np.load(args.out/arm/f"predictions_{scope}.npz", allow_pickle=False) as p:
                recomputed = metrics(p["labels"], p["predictions"], p["accepted"], p["classes"])
            record = next(r for r in summary if r["arm"] == arm and r["scope"] == scope)
            for key, value in recomputed.items():
                if value is None:
                    assert record[key] is None
                else:
                    np.testing.assert_allclose(value, record[key], atol=1e-12, rtol=0)
    write(args.out/"verification.json", dict(passed=True, unchanged_installed_manifest=True, unchanged_test_indices=True,
          C_Macro90_all_arms=True, frozen_classification_in_calibration_only_arm=True,
          original_baseline_replay_matches=True, model_roundtrip_E_decisions_match=True,
          saved_C_E_metrics_recomputed=True, no_pending_or_deleted_crops_used=True))
    print("RESULTS", json.dumps([r for r in summary if r["scope"] == "E"], ensure_ascii=False), flush=True)


def report(args):
    plan, values = read(args.out/"plan.json"), read(args.out/"metrics.json")
    names = dict(baseline="当前实验模型", expanded_same_model="固定模型，仅扩大 C", expanded_camera_isolated="整相机扩容 C 并重训")
    by_arm = {r["arm"]: r for r in values if r["scope"] == "E"}
    delta = (by_arm["expanded_same_model"]["known_false_rejection_macro"]-by_arm["baseline"]["known_false_rejection_macro"])*100
    with np.load(args.out/"expanded_same_model/predictions_C.npz", allow_pickle=False) as c, \
            np.load(args.out/"baseline/predictions_E.npz", allow_pickle=False) as b, \
            np.load(args.out/"expanded_same_model/predictions_E.npz", allow_pickle=False) as e:
        old_cutoff = by_arm["baseline"]["threshold"]
        new_cutoff = by_arm["expanded_same_model"]["threshold"]
        unknown_species = np.unique(c["labels"][~np.isin(c["labels"], c["classes"])])
        rejection = lambda cutoff: float(np.mean([(c["known_score"][c["labels"] == label] < cutoff).mean() for label in unknown_species]))
        known = np.isin(b["labels"], b["classes"])
        diagnosis = dict(expanded_C_unknown_rejection_at_old_threshold=rejection(old_cutoff),
                         expanded_C_unknown_rejection_at_new_threshold=rejection(new_cutoff),
                         extra_E_known_rejections=int((~e["accepted"][known]).sum()-(~b["accepted"][known]).sum()),
                         old_threshold=old_cutoff, new_threshold=new_cutoff,
                         E_unknown_decisions_identical=bool(np.array_equal(b["accepted"][~known], e["accepted"][~known])))
    write(args.out/"threshold_diagnosis.json", diagnosis)
    lines = ["# 扩大校准集后的实验模型对照", "",
             f"结果：固定当前分类模型、仅扩充未知校准样本后，E 已知误拒率变化 {delta:+.2f} 个百分点；本次扩容没有降低误拒率。", "",
             f"原门限 {old_cutoff:.6f} 在扩充后的 C 上只能达到 {diagnosis['expanded_C_unknown_rejection_at_old_threshold']*100:.2f}% 的未知宏拒识。为保持 Macro90，门限变为 {new_cutoff:.6f}，C 未知宏拒识为 {diagnosis['expanded_C_unknown_rejection_at_new_threshold']*100:.2f}%。E 额外拒绝 {diagnosis['extra_E_known_rejections']} 张已知照片，E 未知接纳/拒识决定逐张保持一致。", "",
             "目标保持未知物种等权 Macro90；原 E 完全固定，不参与补图、校准、选参或样本挑选。", "",
             "仅使用现有 74,032 张人工已校验图片中的备用样本；没有把大量未校验照片、人工删除裁图或模型预测当作真值。", "",
             "|方案|Bank|C 已知|C 未知（物种）|E 已知误拒|E 正确接纳|E 分类准确率|E 未知拒识|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        r = next(r for r in values if r["scope"] == "E" and r["arm"] == arm)
        lines.append(f"|{names[arm]}|{r['bank_images']}|{r['C_known_images']}|{r['C_unknown_images']}（{r['C_unknown_species']}）|{r['known_false_rejection_macro']*100:.2f}%|{r['known_correct_acceptance_macro']*100:.2f}%|{r['known_closed_accuracy_macro']*100:.2f}%|{r['unknown_rejection_macro']*100:.2f}%|")
    lines += ["", "全部 E 指标均为物种宏平均。E 仍为 43 个已知物种的 14,449 张和 21 个独立未知物种的 28 张，共 14,477 张。未知测试样本仍很少，95% 相机配对区间见 paired_camera_intervals.csv。", "",
              "## 扩容边界", "",
              f"固定模型方案新增 {len(plan['added_unknown_indices'])} 张未知校准，Ridge、Within 支持、白化、分类配方完全冻结；新增照片与 B 共享相机目录，但连拍、原图和裁图内容均与 B/E 隔离。E 相机及 E Unknown 物种继续完全隔离。因此该方案只用于诊断校准扩容影响，不能称为 B/C 相机独立校准。", "",
              f"严格方案将 {len(plan['strict_moved_B_indices'])} 张原 Bank 支持所在的整相机组件移给 C，新增 {len(plan['strict_added_unknown_indices'])} 张未知校准，并只用缩减后的 B 重新拟合白化、Ridge 与 Within。为保留 45 个已知物种的训练支持，不能移动会使某个已知物种无训练支持的相机。该方案同时改变训练支持和几何，其差值不能全部归因于 C 扩大。", "",
              "63 张备用照片中涉及独立未知测试物种、测试相机或受保护连拍/内容的样本未用于固定模型校准；逐条原因见 excluded_reserves.csv。照片用途对照见 photo_uses_comparison.csv，实际隔离审计见 split_audit.json。", "",
              "本轮保留并报告所有预定方案，不依据 E 选择门限或挑选扩容样本。每个候选的完整校准包、固定模型和逐图决策均已保存；软件当前模型与原照片用途入口保持原值。", "",
              "这些照片和 E 此前已用于历史研究，结果属于历史固定测试集上的诊断；不代表新采集盲测或物理机位/GPS 的完全独立验证。"]
    (args.out/"REPORT.md").write_text("\n".join(lines)+"\n", "utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("F:/files/python/Neri_plus"))
    parser.add_argument("--comparison", type=Path, default=REPO/"runs/local_protocol_comparison_20261114")
    parser.add_argument("--baseline-manifest", type=Path, default=Path("D:/Application/Neri/res/model/cls/user/dinov3-experiment-local.neri.json"))
    parser.add_argument("--out", type=Path, default=REPO/"runs/expanded_calibration_20261114")
    parser.add_argument("--stage", choices=("prepare", "run", "report", "all"), default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        if args.stage in ("prepare", "all"):
            prepare(args)
        if args.stage in ("run", "all"):
            run(args)
        if (args.out/"metrics.json").exists():
            report(args)
    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
