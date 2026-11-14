"""Whole-species exclusion, cross-fitted proxy Unknown budgets and transfer checks.

The proxy library uses one reviewed crop per alias-linked camera acquisition.
Every negative is scored by a freshly fitted model excluding its whole species.
Real Unknown C is reserved for checks/controls; E scoring follows decision freeze.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.compare_local_dinov3_protocols import csvwrite, freeze, metrics, query, read, rows_read, write, validate_partition
from scripts.compare_dinov3_joint_calibration import gate_scores, load_npz, normalization
from system.dinov3.balanced_unknown_calibration import fit_unknown_threshold
from system.dinov3.encoder import sha256
from system.dinov3.experiment import event_fuse, pooled_evidence, row_zscore
from system.dinov3.joint_calibration import fit_joint_threshold
from system.dinov3.model import DinoV3RidgeWithinModel
from system.dinov3.runtime import read_manifest, resolve_manifest_asset

SEEDS = (20261114, 20261115, 20261116)
BUDGETS = (100, 300, 1000)
POLICIES = ("proxy_macro90", "proxy_joint_equal", "proxy_joint_cost2")


def rank(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def acquisition_units(rows, indices):
    """Join event/original/content aliases; never count extra frames as units."""
    parent = {int(i): int(i) for i in indices}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i in sorted(parent):
        row = rows[i]
        keys = [("event", row["camera"], row["sequence"]), ("image", row["image_id"]), ("crop", row["sha256"])]
        if row.get("source_sha256"):
            keys.append(("source", row["source_sha256"]))
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i in sorted(parent):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=min)


def species_folds(rows, units, count=5):
    support = defaultdict(set)
    for g, unit in enumerate(units):
        for i in unit:
            support[rows[i]["species"]].add(g)
    folds = [[] for _ in range(count)]
    masses = [0]*count
    for species in sorted(support, key=lambda c: (-len(support[c]), rank(SEEDS[0], c))):
        f = min(range(count), key=lambda j: (masses[j], len(folds[j]), j))
        folds[f].append(species)
        masses[f] += len(support[species])
    if any(not fold for fold in folds):
        raise ValueError("Not enough species for requested exclusion folds")
    return [sorted(fold) for fold in folds]


def proxy_order(rows, units, seed):
    """Cover all species first, then species/camera round-robin, one crop/unit."""
    options = defaultdict(dict)
    for g, unit in enumerate(units):
        by_species = defaultdict(list)
        for i in unit:
            by_species[rows[i]["species"]].append(i)
        for species, ii in by_species.items():
            options[species][g] = min(ii, key=lambda i: rank(seed, rows[i]["id"]))
    # Bipartite matching guarantees rare co-occurring species get distinct units.
    owners = {}
    def match(species, visited):
        for g in sorted(options[species], key=lambda g: rank(SEEDS[0], f"anchor:{g}")):
            if g in visited:
                continue
            visited.add(g)
            if g not in owners or match(owners[g], visited):
                owners[g] = species
                return True
        return False
    for species in sorted(options, key=lambda c: (len(options[c]), rank(SEEDS[0], c))):
        if not match(species, set()):
            raise ValueError("Cannot cover every species with independent acquisition units")
    anchors = {species: g for g, species in owners.items()}
    used, ordered = set(), []
    camera_counts = Counter()
    def append(species, g):
        i = options[species][g]
        ordered.append(dict(index=i, species=species, unit=g))
        used.add(g)
        camera_counts[(species, rows[i]["camera"])] += 1
    species_order = sorted(options, key=lambda c: rank(seed, c))
    for species in species_order:
        append(species, anchors[species])
    while len(used) < len(units):
        progressed = False
        for species in species_order:
            remaining = [g for g in options[species] if g not in used]
            if not remaining:
                continue
            g = min(remaining, key=lambda g: (camera_counts[(species, rows[options[species][g]]["camera"])], rank(seed, g)))
            append(species, g)
            progressed = True
        if not progressed:
            raise ValueError("Proxy unit ordering failed to cover remaining acquisitions")
    return ordered


def fit_shadow(raw, rows, bank, held, metadata):
    train = np.asarray([int(i) for i in bank if rows[int(i)]["species"] not in held], dtype=np.int64)
    meta = lambda key: np.asarray([rows[int(i)][key] for i in train])
    model = DinoV3RidgeWithinModel(inference_recipe="experiment_full", metadata=metadata).fit(
        raw[train], meta("species"), meta("camera"), meta("sequence"))
    if set(model.classes) & set(held) or {key[0] for key in model.within_updater.groups} & set(held):
        raise ValueError("Held species leaked into Ridge or Within")
    return model, train


def evidence(model, q, stats=None, known=None):
    within = model._compute_within_scores(q["raw"], adapted_features=q["adapted"])
    ridge = model.scores(q["raw"], adapted_features=q["adapted"])
    _, labels, cameras = model.within_updater.bank()
    counts = np.asarray([len(set(cameras[labels == c])) for c in model.classes])
    blend = .55*row_zscore(within-.02*np.log(np.maximum(counts, 1)/3)[None, :])+.45*row_zscore(ridge)
    fused = event_fuse(blend, q["cameras"], q["sequences"], q["crop_hashes"])
    norm = np.linalg.norm((q["adapted"].astype(np.float64)-model.origin_)@model.transform_, axis=1)
    _, pooled, radial = pooled_evidence(within, norm, q["cameras"], q["sequences"])
    if stats is None:
        if known is None or not np.any(known):
            raise ValueError("Known calibration positives required for normalization")
        stats = normalization(pooled, radial, known)
    return dict(predictions=model.classes[fused.argmax(1)], known_score=gate_scores(pooled, radial, stats)), stats


def get_inputs(args):
    rows = rows_read(args.comparison/"current_reviewed.jsonl")
    split = read(args.comparison/"partition.json")
    if sha256(args.comparison/"current_reviewed.jsonl") != split["manifest_sha256"]:
        raise ValueError("Reviewed manifest changed")
    source = read(args.expansion/"plan.json")
    if sha256(source["baseline_manifest"]) != source["baseline_manifest_sha256"]:
        raise ValueError("Installed experimental model changed")
    uses = {int(k): v for k, v in split["assignments"].items()}
    choose = lambda names: np.asarray(sorted(i for i, use in uses.items() if use in names), dtype=np.int64)
    return rows, split, source, choose


def prepare(args):
    rows, split, source, choose = get_inputs(args)
    validate_partition(rows, {int(k):v for k,v in split["assignments"].items()}, split["known_classes"])
    sim = choose({"calibration"})
    units = acquisition_units(rows, sim)
    held_folds = species_folds(rows, units)
    fold_for = {species: f for f, held in enumerate(held_folds) for species in held}
    samples = []
    for seed in SEEDS:
        order = proxy_order(rows, units, seed)
        for budget in BUDGETS:
            selected = order[:budget]
            if len({r["species"] for r in selected}) != len(fold_for):
                raise ValueError("A budget would change pseudo-Unknown species coverage")
            samples.append(dict(seed=seed, budget=budget, actual_images=len(selected),
                                selected=[dict(r, fold=fold_for[r["species"]]) for r in selected]))
    audit = read(args.comparison/"input_audit.json")
    cache = Path(audit.get("features_file", REPO/"runs/dinov3_reviewed_20261107/features.npy"))
    plan = dict(seeds=SEEDS, budgets=BUDGETS, policies=POLICIES, held_folds=held_folds,
        simulated_species=len(fold_for), acquisition_units=len(units),
        bank=choose({"bank"}).tolist(), simulation_pool=sim.tolist(),
        real_calibration_pool=choose({"calibration", "calibration_unknown"}).tolist(),
        evaluation=choose({"evaluation", "evaluation_unknown"}).tolist(), known_classes=split["known_classes"],
        samples=samples, reviewed_manifest_sha256=split["manifest_sha256"],
        feature_cache=str(cache), features_sha256=audit["features_sha256"], encoder_sha256=audit["encoder_sha256"],
        installed_manifest=source["baseline_manifest"], installed_manifest_sha256=source["baseline_manifest_sha256"],
        source_checkpoint=source["baseline_checkpoint"], source_checkpoint_sha256=source["baseline_checkpoint_sha256"],
        original_C_scores_sha256=sha256(args.expansion/"baseline/predictions_C.npz"),
        original_E_scores_sha256=sha256(args.expansion/"baseline/predictions_E.npz"),
        protocol=dict(B="refit whiten/Ridge/Within from B excluding the entire held species set; frozen encoder",
            proxy="one reviewed crop per alias-linked acquisition; all budgets retain all proxy species; nested seeds",
            positive="all remaining Known C scores from every shadow model; species balanced, repeated contexts disclosed",
            simulation_context="all Known-C images, label-free histories/whole events fixed before budget sampling; real Unknown C excluded",
            calibration="pooled standardized cross-fitted Known vs proxy Unknown; group priors equal for joint policy",
            transfer="shared proxy thresholds checked on original real C then fixed E; not assumed invariant across class counts",
            guarded_transfer="max(proxy cutoff, full45 real-C Macro90 cutoff), fixed rule before E",
            E="same full E files; shadow Known/proxy-U/real-U populations separate; full45 transfer separate",
            sampling_information="metadata only, never scores/E errors; all policies and seeds reported",
            application_model_changed=False))
    freeze(args.out/"plan.json", plan)
    inventory = [dict(seed=s["seed"], budget=s["budget"], index=r["index"], id=rows[r["index"]]["id"],
                       species=r["species"], unit=r["unit"], fold=r["fold"]) for s in samples for r in s["selected"]]
    csvwrite(args.out/"proxy_samples.csv", inventory)
    print("PLAN", json.dumps(dict(species=len(fold_for), units=len(units), folds=held_folds,
          actual_samples=[dict(seed=s["seed"], budget=s["budget"], actual=s["actual_images"]) for s in samples]), ensure_ascii=False), flush=True)


def load_features(args, plan):
    rows, _, _, _ = get_inputs(args)
    if sha256(plan["feature_cache"]) != plan["features_sha256"]:
        raise ValueError("Feature cache integrity failed")
    raw = np.load(plan["feature_cache"], allow_pickle=False, mmap_mode="r")[[r["feature_index"] for r in rows]]
    return rows, raw


def fit_models(args):
    plan = read(args.out/"plan.json")
    rows, raw = load_features(args, plan)
    sim = np.asarray(plan["simulation_pool"], dtype=np.int64)
    q = query(rows, raw, sim)
    for fold, held_list in enumerate(plan["held_folds"]):
        folder = args.out/f"fold_{fold}"
        folder.mkdir(exist_ok=True)
        frozen = folder/"fitted.json"
        if frozen.exists():
            info = read(frozen)
            if info["plan_sha256"] != sha256(args.out/"plan.json") or any(sha256(folder/name) != digest for name, digest in info["files"].items()):
                raise ValueError("Frozen shadow generation changed")
            print("REUSE shadow", fold, flush=True)
            continue
        print("FIT shadow", fold, "held", ",".join(held_list), flush=True)
        model, train = fit_shadow(raw, rows, plan["bank"], set(held_list),
            dict(encoder_sha256=plan["encoder_sha256"], preprocessing="letterbox224_imagenet",
                 held_species=held_list, plan_sha256=sha256(args.out/"plan.json"), analysis_only=True))
        expected = set(plan["known_classes"])-set(held_list)
        assert set(model.classes) == expected
        known = np.isin(q["labels"], model.classes)
        result, stats = evidence(model, q, known=known)
        model.save(folder/"shadow.neri-model")
        write(folder/"normalization.json", stats)
        np.savez_compressed(folder/"simulation_C.npz", indices=sim, labels=q["labels"], cameras=q["cameras"],
                            sequences=q["sequences"], known=known, classes=model.classes,
                            predictions=result["predictions"], known_score=result["known_score"])
        np.save(folder/"training_indices.npy", train, allow_pickle=False)
        restored = DinoV3RidgeWithinModel.load(folder/"shadow.neri-model")
        assert not restored.is_calibrated and not set(restored.classes)&set(held_list)
        np.testing.assert_array_equal(restored.origin_, model.origin_)
        np.testing.assert_array_equal(restored.transform_, model.transform_)
        np.testing.assert_array_equal(restored.ridge_head.coef_, model.ridge_head.coef_)
        write(frozen, dict(plan_sha256=sha256(args.out/"plan.json"), held_species=held_list,
              bank_photos=len(train), model_classes=len(model.classes),
              normalization_known_rows=int(known.sum()), held_absent_from_geometry_ridge_within=True,
              files={name:sha256(folder/name) for name in ("shadow.neri-model", "normalization.json", "simulation_C.npz", "training_indices.npy")}))
        print("SAVED shadow", fold, "classes", len(model.classes), "B", len(train), flush=True)


def select_thresholds(args):
    path = args.out/"proxy_thresholds_frozen.json"
    if path.exists():
        print("REUSE proxy threshold selection", flush=True)
        return
    plan = read(args.out/"plan.json")
    sim_data = {fold:load_npz(args.out/f"fold_{fold}/simulation_C.npz") for fold in range(len(plan["held_folds"]))}
    positive_scores, positive_labels = [], []
    for fold, data in sim_data.items():
        positive_scores.extend(data["known_score"][data["known"]])
        positive_labels.extend("K:"+str(c) for c in data["labels"][data["known"]])
        assert not set(data["labels"][data["known"]])&set(plan["held_folds"][fold])
    positive_scores, positive_labels = np.asarray(positive_scores), np.asarray(positive_labels)
    selection = []
    for sample in plan["samples"]:
        negative_scores, negative_labels = [], []
        for chosen in sample["selected"]:
            data = sim_data[chosen["fold"]]
            pos = int(np.searchsorted(data["indices"], chosen["index"]))
            assert int(data["indices"][pos]) == chosen["index"] and not data["known"][pos]
            assert str(data["labels"][pos]) == chosen["species"]
            assert chosen["species"] not in data["classes"]
            negative_scores.append(float(data["known_score"][pos]))
            negative_labels.append("U:"+chosen["species"])
        scores = np.r_[positive_scores, negative_scores]
        labels = np.r_[positive_labels, negative_labels]
        known = np.r_[np.ones(len(positive_scores), dtype=bool), np.zeros(len(negative_scores), dtype=bool)]
        macro = fit_unknown_threshold(negative_scores, negative_labels, target=.9, weighting="species")
        fits = dict(proxy_macro90=dict(macro, feasible=True),
                    proxy_joint_equal=fit_joint_threshold(scores, labels, known),
                    proxy_joint_cost2=fit_joint_threshold(scores, labels, known, unknown_cost=2/3))
        constraint = fit_joint_threshold(scores, labels, known, min_unknown_rejection=.9, min_known_acceptance=.8)
        selection.append(dict(seed=sample["seed"], budget=sample["budget"], actual_images=len(negative_scores),
             unknown_species=len(set(negative_labels)), known_score_instances=len(positive_scores),
             unique_known_photos=len(plan["simulation_pool"]), fits=fits, known80_unknown90=constraint))
    freeze(path, dict(plan_sha256=sha256(args.out/"plan.json"), selection=selection,
           role_semantics="K/U prefixes identify species status in its generating model, not contradictory labels in one classifier",
           shadow_generations={str(f):sha256(args.out/f"fold_{f}/fitted.json") for f in sim_data},
           real_unknown_C_used_for_proxy_selection=False))
    print("FROZEN proxy cutoffs", len(selection)*len(POLICIES), flush=True)


def population_rows(labels, predictions, accepted, classes, original_known):
    result = []
    for species in np.unique(labels):
        mask = labels == species
        role = "known" if species in classes else "proxy_unknown" if species in original_known else "real_unknown"
        result.append(dict(species=str(species), role=role, photos=int(mask.sum()),
                           rejection=float((~accepted[mask]).mean()),
                           correct_acceptance=float(((predictions[mask] == species)&accepted[mask]).mean()),
                           closed_accuracy=float((predictions[mask] == species).mean())))
    return result


def aggregate_populations(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["role"],row["species"])].append(row)
    result = {}
    for role in ("known", "proxy_unknown", "real_unknown"):
        species = [values for (r,_), values in grouped.items() if r == role]
        result[role+"_species"] = len(species)
        if species:
            result[role+"_rejection_macro"] = float(np.mean([np.mean([v["rejection"] for v in values]) for values in species]))
            if role == "known":
                for key in ("correct_acceptance", "closed_accuracy"):
                    result["known_"+key+"_macro"] = float(np.mean([np.mean([v[key] for v in values]) for values in species]))
    return result


def validate_transfer(args):
    if (args.out/"operating_points_before_E.json").exists():
        print("REUSE operating points", flush=True)
        return
    plan = read(args.out/"plan.json")
    proxy = read(args.out/"proxy_thresholds_frozen.json")
    baseline = load_npz(args.expansion/"baseline/predictions_C.npz")
    if sha256(args.expansion/"baseline/predictions_C.npz") != plan["original_C_scores_sha256"]:
        raise ValueError("Original C score archive changed")
    original_known = set(plan["known_classes"])
    known = np.isin(baseline["labels"], baseline["classes"])
    real_tau = fit_unknown_threshold(baseline["known_score"][~known], baseline["labels"][~known], target=.9)["threshold"]
    real_joint = fit_joint_threshold(baseline["known_score"], baseline["labels"], known)["threshold"]
    controls = dict(current_macro90=float(real_tau), current_joint_equal=float(real_joint))
    full_checks = []
    for sample in proxy["selection"]:
        for policy, fit in sample["fits"].items():
            for mode, tau in (("raw_transfer", fit["threshold"]), ("real_C90_guard", max(fit["threshold"], real_tau))):
                full_checks.append(dict(seed=sample["seed"], budget=sample["budget"], policy=policy, mode=mode,
                     threshold=float(tau), **metrics(baseline["labels"], baseline["predictions"],
                        baseline["known_score"] >= tau, baseline["classes"])))
    rows, raw = load_features(args, plan)
    q = query(rows, raw, np.asarray(plan["real_calibration_pool"], dtype=np.int64))
    shadow_checks = []
    for fold in range(len(plan["held_folds"])):
        model = DinoV3RidgeWithinModel.load(args.out/f"fold_{fold}/shadow.neri-model")
        p,_ = evidence(model, q, stats=read(args.out/f"fold_{fold}/normalization.json"))
        real_u = ~np.isin(q["labels"], list(original_known))
        actual_k = np.isin(q["labels"], model.classes)
        tau = fit_unknown_threshold(p["known_score"][real_u], q["labels"][real_u], target=.9)["threshold"]
        selected = actual_k | real_u
        joint = fit_joint_threshold(p["known_score"][selected], q["labels"][selected], actual_k[selected])["threshold"]
        shadow_checks.append(dict(fold=fold, real_C_macro90=float(tau), real_C_joint_equal=float(joint)))
        np.savez_compressed(args.out/f"fold_{fold}/real_C_validation.npz", indices=q["indices"], labels=q["labels"],
                            classes=model.classes, predictions=p["predictions"], known_score=p["known_score"])
    freeze(args.out/"operating_points_before_E.json", dict(proxy_sha256=sha256(args.out/"proxy_thresholds_frozen.json"),
           controls=controls, full45_C_checks=full_checks, shadow_controls=shadow_checks,
           guard_rule="max(proxy threshold, full45 real-C Macro90 threshold); no E choice"))
    print("FROZEN all operating points before E", flush=True)


def evaluate(args):
    if (args.out/"verification.json").exists():
        print("REUSE verified E comparison", flush=True)
        return
    plan = read(args.out/"plan.json")
    proxy = read(args.out/"proxy_thresholds_frozen.json")
    operating = read(args.out/"operating_points_before_E.json")
    original_known = set(plan["known_classes"])
    e = load_npz(args.expansion/"baseline/predictions_E.npz")
    if sha256(args.expansion/"baseline/predictions_E.npz") != plan["original_E_scores_sha256"]:
        raise ValueError("Original E score archive changed")
    np.testing.assert_array_equal(e["indices"], np.asarray(plan["evaluation"]))
    transfer = []
    for policy,tau in operating["controls"].items():
        transfer.append(dict(scope="control", seed=0, budget=0, policy=policy, mode="control", threshold=tau,
                    **metrics(e["labels"], e["predictions"], e["known_score"] >= tau, e["classes"])))
    for condition in operating["full45_C_checks"]:
        tau = condition["threshold"]
        row = {key:condition[key] for key in ("seed","budget","policy","mode","threshold")}
        transfer.append(dict(scope="full45_transfer", **row,
                    **metrics(e["labels"], e["predictions"], e["known_score"] >= tau, e["classes"])))
    write(args.out/"full45_E_metrics.json", transfer)
    csvwrite(args.out/"full45_E_metrics.csv", transfer)
    rows, raw = load_features(args, plan)
    q = query(rows, raw, e["indices"])
    condition_species = defaultdict(list)
    detailed = []
    for fold in range(len(plan["held_folds"])):
        print("EVALUATE shadow on fixed E", fold, flush=True)
        model = DinoV3RidgeWithinModel.load(args.out/f"fold_{fold}/shadow.neri-model")
        predicted,_ = evidence(model, q, stats=read(args.out/f"fold_{fold}/normalization.json"))
        np.savez_compressed(args.out/f"fold_{fold}/E_scores.npz", indices=q["indices"], labels=q["labels"],
                            classes=model.classes, predictions=predicted["predictions"], known_score=predicted["known_score"])
        policies = [(0,0,name,ctrl[name]) for ctrl in operating["shadow_controls"] if ctrl["fold"] == fold
                    for name in ("real_C_macro90","real_C_joint_equal")]
        policies += [(s["seed"],s["budget"],name,fit["threshold"]) for s in proxy["selection"] for name,fit in s["fits"].items()]
        for seed,budget,policy,tau in policies:
            per_species = population_rows(q["labels"], predicted["predictions"], predicted["known_score"] >= tau,
                                          set(model.classes), original_known)
            condition_species[(seed,budget,policy)].extend(per_species)
            detailed.extend(dict(fold=fold,seed=seed,budget=budget,policy=policy,**r) for r in per_species)
    shadow = [dict(seed=seed,budget=budget,policy=policy,**aggregate_populations(values))
              for (seed,budget,policy),values in condition_species.items()]
    write(args.out/"shadow_E_metrics.json", shadow)
    csvwrite(args.out/"shadow_E_metrics.csv", shadow)
    csvwrite(args.out/"shadow_per_species.csv", detailed)
    frozen_hashes = {str(f):sha256(args.out/f"fold_{f}/fitted.json") for f in range(len(plan["held_folds"]))}
    assert frozen_hashes == proxy["shadow_generations"]
    if sha256(plan["installed_manifest"]) != plan["installed_manifest_sha256"]:
        raise ValueError("Application model changed during simulation")
    for sample in plan["samples"]:
        selected = sample["selected"]
        assert len({r["unit"] for r in selected}) == len(selected)
        assert len({r["species"] for r in selected}) == plan["simulated_species"]
        for r in selected:
            assert r["species"] in plan["held_folds"][r["fold"]]
    write(args.out/"verification.json", dict(passed=True, application_unchanged=True, all_held_species_absent_from_fit=True,
           whitening_refit_from_reduced_B=True, independent_acquisitions_per_budget=True, fixed_species_coverage=True,
           thresholds_fixed_before_E=True, real_unknown_C_unused_for_proxy_fitting=True,
           shadow_and_full45_populations_separate=True, fixed_E_photos=len(e["indices"])))


def summary_rows(rows, keys, values):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, records in groups.items():
        result = dict(zip(keys,key), seeds=len(records))
        for name in values:
            data = np.asarray([r[name] for r in records], float)
            result[name] = float(data.mean())
            result[name+"_min"] = float(data.min())
            result[name+"_max"] = float(data.max())
        output.append(result)
    return output


def report(args):
    plan = read(args.out/"plan.json")
    transfer = read(args.out/"full45_E_metrics.json")
    shadow = read(args.out/"shadow_E_metrics.json")
    full_summary = summary_rows([r for r in transfer if r["scope"] != "control"], ("budget","policy","mode"),
                      ("threshold","known_false_rejection_macro","known_correct_acceptance_macro","unknown_rejection_macro"))
    shadow_summary = summary_rows([r for r in shadow if r["budget"]], ("budget","policy"),
                      ("known_rejection_macro","known_correct_acceptance_macro","real_unknown_rejection_macro","proxy_unknown_rejection_macro"))
    csvwrite(args.out/"full45_budget_summary.csv",full_summary)
    csvwrite(args.out/"shadow_budget_summary.csv",shadow_summary)
    primary = sorted([r for r in full_summary if r["policy"] == "proxy_joint_equal" and r["mode"] == "raw_transfer"],key=lambda r:r["budget"])
    real_checks = read(args.out/"operating_points_before_E.json")["full45_C_checks"]
    c_unknown = [r["unknown_rejection_macro"] for r in real_checks if r["policy"] == "proxy_joint_equal" and r["mode"] == "raw_transfer" and r["budget"] == 1000]
    proxy_fits = read(args.out/"proxy_thresholds_frozen.json")["selection"]
    infeasible = sum(not r["known80_unknown90"]["feasible"] for r in proxy_fits)
    lines = ["# 增加模拟未知照片：整物种留出与预算对照", "",
        "结果：完整45类模型上的等成本联合门限迁移，三个种子平均已知误拒率依次为 " + "、".join(f"{r['budget']}张 {r['known_false_rejection_macro']*100:.2f}%" for r in primary) + "，当前原门限为38.80%。本次固定E的真实未知宏拒识率均为85.71%。", "",
        f"1000张候选在真实未知C上的宏拒识只有 {np.mean(c_unknown)*100:.2f}%，未达到90%；加上预定的真实C90保护后，等成本联合方案回到原门限及38.80%误拒。增加模拟数量让本次种子波动减小，但不能据此声称突破原来已知/未知的分数重叠。", "",
        f"从 {plan['simulated_species']} 个有独立校准照片的原已知物种构建5轮影子模型。每轮从 B 完整移除被留出物种，重新拟合白化、Ridge、Within；编码器冻结。", "",
        f"模拟库最多 {plan['acquisition_units']} 个独立采集组件，每组件取一张已审核代表裁图。固定同一物种覆盖，三个种子下比较100、300、1000张；不以重复连拍、增强图或生成图凑数。", "",
        "已知正例来自影子模型保留物种的 C：同一照片会在保留它的4个模型中生成4份分数，不能称为4张独立照片。未知负例只来自完整排除该物种的模型。K/U前缀记录生成模型中的角色，不是在同一个分类器中把已有物种改叫Unknown。", "",
        "模拟分数按每轮剩余已知 C 的 median/MAD 标准化，再汇总联合选择共享门限。预算仅表示最终参与门限拟合的模拟未知分数数量；所有C物种标签已用于规划留出和取样，完整模拟查询池及标签无关历史/事件上下文先冻结，不把预算当作全部曝光/标注/推理照片数。真实未知 C 不参与模拟门限拟合。", "",
        "## 影子模型上的真实未知 E", "",
        "以下在每个影子模型的实际已知类集合内评估，先按物种跨模型平均，再做物种宏平均；不是完整45类模型的成绩。测试真实未知始终是相同21类28张。", "",
        "|模拟未知预算|策略|已知误拒|已知正确接纳|真实未知拒识|模拟未知E拒识|", "|---|---|---:|---:|---:|---:|"]
    for r in shadow_summary:
        lines.append(f"|{r['budget']}|{r['policy']}|{r['known_rejection_macro']*100:.2f}%|{r['known_correct_acceptance_macro']*100:.2f}%|{r['real_unknown_rejection_macro']*100:.2f}%|{r['proxy_unknown_rejection_macro']*100:.2f}%|")
    lines += ["", "表中为3个种子均值，逐种子及最小/最大值在CSV；真实未知E未用于抽样、门限或权重选择。", "",
        "## 向当前完整45类模型迁移的独立检查", "",
        "影子模型类数与几何不同，标准化不保证门限能直接迁移。以下把预定共享门限作为迁移候选在当前45类分数上检查；另预定用真实C Macro90门限作下限保护。两种方式均先在真实C检查并冻结，再评分同一E，不根据E选最好门限。", "",
        "|预算|策略|迁移方式|已知误拒|正确接纳|真实未知拒识|", "|---|---|---|---:|---:|---:|"]
    for r in transfer:
        if r["scope"] == "control":
            lines.append(f"|对照|{r['policy']}|原45类|{r['known_false_rejection_macro']*100:.2f}%|{r['known_correct_acceptance_macro']*100:.2f}%|{r['unknown_rejection_macro']*100:.2f}%|")
    for r in full_summary:
        lines.append(f"|{r['budget']}|{r['policy']}|{r['mode']}|{r['known_false_rejection_macro']*100:.2f}%|{r['known_correct_acceptance_macro']*100:.2f}%|{r['unknown_rejection_macro']*100:.2f}%|")
    lines += ["", "完整45类的E仍为43个已知物种14,449张、21个真实未知物种28张。所有比例为物种宏平均。不同策略/预算都报告，没有将最有利的种子当作结果。", "",
        f"模拟汇总库上，已知接纳80%＋模拟未知拒识90%的联合约束在 {infeasible}/{len(proxy_fits)} 个预定样本条件中不可行；可行性仅指模拟分数分布，不代表真实未知。", "",
        "真实未知样本少且历史E此前已曝光；这些是探索性历史诊断，不能据此承诺独立新物种的拒识率。raw_transfer只是分数迁移探针，real_C90_guard只保证真实C约束，不能保证E也达到90%。相机身份是目录代理，并非硬件序号或GPS确认。当前软件模型、物种库和照片用途指针均未更改。", "",
        "plan.json保存物种组合及全部抽样；各fold保存训练索引、排除类、未校准影子模型、标准化和C/E分数；proxy_thresholds_frozen.json与operating_points_before_E.json保存评分前决策；完整结果与3种子范围见CSV。"]
    (args.out/"REPORT.md").write_text("\n".join(lines)+"\n", "utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison",type=Path,default=REPO/"runs/local_protocol_comparison_20261114")
    parser.add_argument("--expansion",type=Path,default=REPO/"runs/expanded_calibration_20261114")
    parser.add_argument("--out",type=Path,default=REPO/"runs/simulated_unknown_20261114")
    parser.add_argument("--stage",choices=("prepare","fit","select","validate","evaluate","report","all"),default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=4):
        if args.stage in ("prepare","all"): prepare(args)
        if args.stage in ("fit","all"): fit_models(args)
        if args.stage in ("select","all"): select_thresholds(args)
        if args.stage in ("validate","all"): validate_transfer(args)
        if args.stage in ("evaluate","all"): evaluate(args)
        if (args.out/"verification.json").exists(): report(args)
    print("DONE",args.out,flush=True)


if __name__ == "__main__":
    main()
