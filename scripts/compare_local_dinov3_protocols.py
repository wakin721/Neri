"""Compare production and frozen experimental recipes, then freeze camera splits.

All image files stay in place. Manifests assign uses; no production model is
replaced. Historical data are explicitly not described as a fresh blind test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from system.dinov3.context import FastCausalHistoryIndex, adapt_with_history
from system.dinov3.encoder import sha256
from system.dinov3.model import (
    DinoV3RidgeWithinModel, resolve_target_tracks, top75, sequence_mean,
)
from system.dinov3.balanced_unknown_calibration import fit_unknown_threshold

SEED = 20261114
SCHEMES = ("production_dual", "track_single_ridge", "experiment_single_ridge", "experiment_full")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text("utf8"))


def rows_read(path):
    with Path(path).open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def csvwrite(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze(path, value):
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    if Path(path).exists() and read(path) != value:
        raise ValueError(f"Frozen protocol changed: {path}; use a new output directory")
    write(path, value)


def meta(rows, indices, key):
    return np.asarray([str(rows[int(i)][key]) for i in indices])


def content_id(row):
    frame = str(row["source_sha256"] or row.get("image_id") or row["sha256"])
    suffix = row["id"].rsplit("_", 1)[-1]
    return f"{frame}:{int(suffix) if suffix.isdigit() else 0}"


def audit_inputs(source, old, out):
    """Rehash every crop, matching cache identities, not just timestamp metadata."""
    cached = rows_read(old / "reviewed.jsonl")
    by_id = {r["id"]: i for i, r in enumerate(cached)}
    root = source / "data/by_species/00已校验"
    print("ENUMERATE reviewed photos", flush=True)
    photos = sorted(Path(folder)/name for folder, _, names in os.walk(root)
                    for name in names if Path(name).suffix.lower() in EXTENSIONS)
    print("VERIFY cached features and hash current crops", len(photos), flush=True)
    if len({p.stem for p in photos}) != len(photos):
        raise ValueError("Current reviewed IDs are not unique")
    verified = read(old / "feature_verification.json")
    if not verified["passed"] or sha256(old / "features.npy") != verified["features_sha256"]:
        raise ValueError("Frozen feature verification failed")
    if sha256(old / "reviewed.jsonl") != verified["manifest_sha256"]:
        raise ValueError("Frozen feature manifest verification failed")
    current, new, changed = [], [], []
    def inspect(p):
        stat = p.stat()
        digest = sha256(p)
        after = p.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError(f"Image changed while hashing: {p}")
        index = by_id.get(p.stem)
        if index is None:
            return None, str(p), None
        old_row = cached[index]
        row = dict(old_row, file=str(p), species=p.relative_to(root).parts[0],
                   size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest,
                   feature_index=index)
        return row, None, str(p) if digest != old_row["sha256"] else None
    with ThreadPoolExecutor(max_workers=12) as pool:
        for n, (row, added, modified) in enumerate(pool.map(inspect, photos), 1):
            if row is not None:
                current.append(row)
            if added:
                new.append(added)
            if modified:
                changed.append(modified)
            if n % 10000 == 0:
                print(f"HASH AUDIT {n}/{len(photos)}", flush=True)
    audit = dict(current_photos=len(photos), cached_photos=len(cached), new_files=new, changed_content=changed,
                 missing_cached_ids=sorted(set(by_id) - {r["id"] for r in current}),
                 relabelled=sum(r["species"] != cached[r["feature_index"]]["species"] for r in current),
                 features_file=str(old/"features.npy"),features_sha256=verified["features_sha256"], encoder_sha256=verified["encoder_sha256"],
                 all_current_images_content_hashed=True)
    write(out / "input_audit.json", audit)
    if new or changed:
        raise ValueError("New/changed pixels require fresh feature extraction; see input_audit.json")
    with (out / "current_reviewed.jsonl").open("w", encoding="utf8") as stream:
        for row in current:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return current


def camera_components(rows):
    """Join cameras linked by any identical crop/original/source/sequence alias."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    by_hash = defaultdict(list)
    for i, row in enumerate(rows):
        by_hash[row["sha256"]].append(i)
        keys = [("camera", row["camera"]), ("crop", row["sha256"]), ("image", row["image_id"]),
                ("event", row["camera"], row["sequence"])]
        if row.get("source_sha256"):
            keys.append(("source", row["source_sha256"]))
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    excluded = {}
    for indices in by_hash.values():
        conflict = len({rows[i]["species"] for i in indices}) > 1
        for i in indices if conflict else indices[1:]:
            excluded[i] = "label_conflict" if conflict else "duplicate_crop"
    components = defaultdict(list)
    for i in range(len(rows)):
        if i not in excluded:
            components[find(i)].append(i)
    return list(components.values()), excluded


def make_camera_partition(rows, known_classes, seed=SEED):
    """Metadata-only MILP; whole camera/content components, no coverage repairs.

    Known classes with >=3 camera components require B/C/E coverage; those with
    two require B/E; singleton-camera classes remain training-only. Unknown
    species are deterministically divided into disjoint C-U/E-U species sets.
    Unknown crops on an incompatible camera are reserved for later use.
    """
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix
    known = set(known_classes)
    groups, excluded = camera_components(rows)
    labels = sorted({r["species"] for r in rows})
    counts = np.asarray([[sum(rows[i]["species"] == c for i in g) for g in groups] for c in labels], dtype=float)
    rank = lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest()
    unknown = sorted(set(labels) - known, key=rank)
    unknown_roles = {c: "calibration" if j % 2 == 0 else "evaluation" for j, c in enumerate(unknown)}
    # Binary group-to-pool variables + absolute deviations + unknown missing-coverage slack.
    ng, nl = len(groups), len(labels)
    nd = 3 * (nl + 1)
    nv = 3 * ng + nd + len(unknown)
    cost = np.zeros(nv)
    lo, hi = np.zeros(nv), np.full(nv, np.inf)
    hi[:3*ng] = 1
    hi[3*ng+nd:] = 1
    integrality = np.zeros(nv)
    integrality[:3*ng] = 1
    constraints, lower, upper = [], [], []
    def constraint(terms, lower_bound, upper_bound):
        constraints.append(terms)
        lower.append(lower_bound)
        upper.append(upper_bound)
    for g in range(ng):
        constraint({3*g+p: 1 for p in range(3)}, 1, 1)
    fractions = (0.60, 0.20, 0.20)
    for j, label in enumerate(labels):
        present = np.flatnonzero(counts[j])
        if label in known:
            for p in [0] + ([2] if len(present) >= 2 else []) + ([1] if len(present) >= 3 else []):
                constraint({3*int(g)+p: 1 for g in present}, 1, np.inf)
        else:
            p = 1 if unknown_roles[label] == "calibration" else 2
            u = 3*ng+nd+unknown.index(label)
            constraint({**{3*int(g)+p: 1 for g in present}, u: 1}, 1, np.inf)
            cost[u] = 10
        for p in range(3):
            d = 3*ng+3*j+p
            target = fractions[p]*counts[j].sum()
            terms = {3*g+p: counts[j,g] for g in present}
            constraint({**terms, d: -1}, -np.inf, target)
            constraint({k: -v for k,v in terms.items()} | {d: -1}, -np.inf, -target)
            cost[d] = (1 if label in known else .2) / max(counts[j].sum(), 1)
    for p in range(3):
        d = 3*ng+3*nl+p
        sizes = counts.sum(0)
        terms = {3*g+p: sizes[g] for g in range(ng)}
        target = fractions[p]*sizes.sum()
        constraint(terms | {d: -1}, -np.inf, target)
        constraint({k: -v for k,v in terms.items()} | {d: -1}, -np.inf, -target)
        cost[d] = 10/sizes.sum()
    # Deterministic small tie-breaking preference, unrelated to image appearance or scores.
    for g, indices in enumerate(groups):
        for p in range(3):
            cost[3*g+p] = int(rank(f"{rows[indices[0]]['camera']}:{p}")[:8],16)/2**32*1e-7
    a = lil_matrix((len(constraints),nv))
    for j, terms in enumerate(constraints):
        for k, value in terms.items():
            a[j,k] = value
    result = milp(cost, integrality=integrality, bounds=Bounds(lo,hi),
                  constraints=LinearConstraint(a.tocsr(),lower,upper),
                  options=dict(time_limit=30, mip_rel_gap=.01))
    if result.x is None:
        raise ValueError(f"No strict camera partition: {result.message}")
    chosen = result.x[:3*ng].reshape(ng,3)
    if not np.allclose(chosen, chosen.round(), atol=1e-6):
        raise ValueError("MILP returned nonintegral camera assignments")
    pool_names = ("bank", "calibration", "evaluation")
    assignments = {}
    camera_pool = {}
    for g, indices in enumerate(groups):
        pool = pool_names[int(chosen[g].argmax())]
        for i in indices:
            row = rows[i]
            camera_pool[row["camera"]] = pool
            assignments[i] = (pool if row["species"] in known else
                              f"{pool}_unknown" if unknown_roles[row["species"]] == pool else "future_support_reserve")
    for i, reason in excluded.items():
        assignments[i] = f"excluded_{reason}"
    info = dict(seed=seed, known_classes=sorted(known), unknown_species_roles=unknown_roles,
                camera_components=len(groups), requested_camera_pool_image_fractions=fractions,
                optimization=dict(status=int(result.status), message=result.message, objective=float(result.fun),
                                  gap=float(result.mip_gap)),
                split_policy="whole camera/source/content components; no shared-camera coverage repairs",
                unknown_policy="disjoint unknown species for calibration and evaluation; incompatible cameras reserved",
                known_policy="preserve original >=10-crop classes; insufficient cameras explicitly unevaluated",
                original_images_moved=False, historical_data_exposure=True,
                purpose_counts=dict(Counter(assignments.values())))
    return assignments, info


def validate_partition(rows, assignments, known_classes):
    known = set(known_classes)
    ownership = {}
    uses = defaultdict(list)
    for i, purpose in assignments.items():
        uses[purpose].append(i)
        if purpose.startswith("excluded"):
            continue
        pool = "bank" if purpose == "bank" else "calibration" if purpose.startswith("calibration") else "evaluation" if purpose.startswith("evaluation") else None
        if pool is None:
            continue
        r = rows[i]
        keys = [("camera",r["camera"]),("image",r["image_id"]),("content",r["sha256"]),
                ("event",r["camera"],r["sequence"])]
        if r.get("source_sha256"):
            keys.append(("source",r["source_sha256"]))
        for key in keys:
            if key in ownership and ownership[key] != pool:
                raise ValueError(f"Cross-pool leakage: {key[0]}")
            ownership[key] = pool
    bank_classes = {rows[i]["species"] for i in uses["bank"]}
    if bank_classes != known:
        raise ValueError("Bank known class coverage changed")
    cu = {rows[i]["species"] for i in uses["calibration_unknown"]}
    eu = {rows[i]["species"] for i in uses["evaluation_unknown"]}
    if not cu or not eu or cu & eu or (cu|eu) & known:
        raise ValueError("Unknown species isolation/coverage failed")
    if len(assignments) != len(rows) or set(assignments) != set(range(len(rows))):
        raise ValueError("Every crop must have exactly one purpose")
    return dict(passed=True, full_camera_isolation=True, source_event_content_isolation=True,
                calibration_unknown_species=len(cu), evaluation_unknown_species=len(eu),
                unknown_calibration_test_species_overlap=0, every_photo_has_one_purpose=True)


def save_partition(rows, old, out):
    known = read(old / "input_audit.json")["known_classes"]
    path = out / "partition.json"
    if path.exists():
        protocol = read(path)
        assignments = {int(k):v for k,v in protocol["assignments"].items()}
    else:
        assignments, info = make_camera_partition(rows, known)
        protocol = dict(**info, manifest_sha256=sha256(out / "current_reviewed.jsonl"),
                        assignments={str(k):v for k,v in assignments.items()})
        freeze(path, protocol)
    if protocol["manifest_sha256"] != sha256(out / "current_reviewed.jsonl"):
        raise ValueError("Current manifest differs from frozen partition")
    validation = validate_partition(rows,assignments,known)
    write(out / "partition_validation.json",validation)
    records = []
    handles = {}
    try:
        for i,r in enumerate(rows):
            purpose = assignments[i]
            if purpose not in handles:
                handles[purpose] = (out / f"{purpose}.jsonl").open("w",encoding="utf8")
            record = dict(r, row_index=i, purpose=purpose)
            handles[purpose].write(json.dumps(record,ensure_ascii=False)+"\n")
            records.append(dict(id=r["id"],file=r["file"],species=r["species"],camera=r["camera"],
                                sequence=r["sequence"],purpose=purpose,feature_index=r["feature_index"]))
    finally:
        for handle in handles.values():
            handle.close()
    csvwrite(out / "photo_uses.csv",records)
    per_class = []
    for c in sorted({r["species"] for r in rows}):
        ii = [i for i,r in enumerate(rows) if r["species"]==c]
        row = dict(species=c,role="known" if c in known else "unknown",photos=len(ii),
                   cameras=len({rows[i]["camera"] for i in ii}),events=len({(rows[i]["camera"],rows[i]["sequence"]) for i in ii}))
        for purpose in sorted(set(assignments.values())):
            row[purpose] = sum(assignments[i]==purpose for i in ii)
        row["evaluation_status"] = "camera_holdout" if row.get("evaluation",0) else "not_evaluated" if c in known else "unknown_holdout" if row.get("evaluation_unknown",0) else "not_test_unknown"
        per_class.append(row)
    csvwrite(out / "species_uses.csv",per_class)
    print("PARTITION",json.dumps(protocol["purpose_counts"],ensure_ascii=False),flush=True)
    return assignments


def query(rows, raw, indices):
    x = np.asarray(raw[indices])
    records = [dict(camera_component=rows[int(i)]["camera"],seq_id=rows[int(i)]["sequence"],
                    datetime=rows[int(i)]["datetime"],content_hash=rows[int(i)]["source_sha256"] or rows[int(i)]["sha256"])
               for i in indices]
    history = FastCausalHistoryIndex(records).build()
    adapted = adapt_with_history(x, history)
    return dict(raw=x,adapted=adapted,labels=meta(rows,indices,"species"),
                cameras=meta(rows,indices,"camera"),sequences=meta(rows,indices,"sequence"),
                contents=np.asarray([content_id(rows[int(i)]) for i in indices]),
                crop_hashes=meta(rows,indices,"sha256"),indices=np.asarray(indices))


def collect_scores(model,q):
    within = model._compute_within_scores(q["raw"],adapted_features=q["adapted"])
    ridge = model.scores(q["raw"],adapted_features=q["adapted"])
    top = np.sort(within,axis=1)[:,-2:]
    margin = 2*top[:,1]-top[:,0]
    norm = np.linalg.norm((q["adapted"].astype(np.float64)-model.origin_)@model.transform_,axis=1)
    return dict(within=within,ridge=ridge,margin=margin,norm=norm)


def normalize(values, known):
    median = float(np.median(values[known]))
    scale = max(float(1.4826*np.median(abs(values[known]-median))),1e-6)
    return dict(median=median,scale=scale)


def normalized(values, info):
    return (values-info["median"])/info["scale"]


def reference_recipe(model,cq,eq,cs,es,bank_cameras,bank_labels,reference,track=False,blend=False):
    # Use the exact frozen experiment helpers for class fusion and score blending.
    classes = model.classes
    uc = ~np.isin(cq["labels"],classes)
    bias,_ = reference.cardinality_bias(bank_cameras,bank_labels,classes,alpha=.02)
    cal = {}
    result = []
    for q,s in ((cq,cs),(eq,es)):
        seq = (resolve_target_tracks(q["adapted"],q["cameras"],q["sequences"],q["contents"])
               if track else q["sequences"])
        score_matrix = (.55*reference.row_zscore(s["within"]-bias[None,:]) + .45*reference.row_zscore(s["ridge"])) if blend else s["ridge"]
        # Historical experiment uses crop hashes as content IDs and event grouping.
        content = q["contents"] if track else q["crop_hashes"]
        fused = reference.adaptive_sequence_fuse(score_matrix,q["cameras"],seq,content,
                                                tau=.5 if blend else 1.,consensus_boost=True)
        # Preserve the experiment's float32 top-fraction mean before float64
        # standardization; the production top75 helper promotes before pooling.
        _, inverse = np.unique(np.column_stack((q["cameras"],seq)),axis=0,return_inverse=True)
        pooled = reference.compute_top_quantile_pooling(s["margin"],inverse,.75)
        radial = sequence_mean(s["norm"],q["cameras"],seq)
        if not cal:
            cal = dict(margin=normalize(pooled,~uc),radial=normalize(radial,~uc))
            score = normalized(pooled,cal["margin"])-.08*normalized(radial,cal["radial"])
            cal["fit"] = fit_unknown_threshold(score[uc],cq["labels"][uc],target=.9,weighting="species")
        else:
            score = normalized(pooled,cal["margin"])-.08*normalized(radial,cal["radial"])
        result.append(dict(predictions=classes[fused.argmax(1)],accepted=score>=cal["fit"]["threshold"],known_score=score))
    return result,cal


def metrics(truth,pred,accepted,classes):
    known = np.isin(truth,classes)
    correct = pred==truth
    def macro(values,mask):
        return float(np.mean([values[mask&(truth==c)].mean() for c in np.unique(truth[mask])])) if mask.any() else None
    def micro(values,mask):
        return float(values[mask].mean()) if mask.any() else None
    return dict(images=len(truth),known_images=int(known.sum()),unknown_images=int((~known).sum()),
                known_species=len(np.unique(truth[known])),unknown_species=len(np.unique(truth[~known])),
                known_closed_accuracy_macro=macro(correct,known),known_correct_acceptance_macro=macro(correct&accepted,known),
                known_false_rejection_macro=macro(~accepted,known),known_correct_but_rejected_macro=macro(correct&~accepted,known),
                unknown_rejection_macro=macro(~accepted,~known),known_closed_accuracy_micro=micro(correct,known),
                known_correct_acceptance_micro=micro(correct&accepted,known),known_false_rejection_micro=micro(~accepted,known),
                unknown_rejection_micro=micro(~accepted,~known),
                accepted_known_accuracy_micro=micro(correct,known&accepted))


def compare(model,cq,eq,bank_cameras,bank_labels,reference,out,baseline=None):
    out.mkdir(parents=True,exist_ok=True)
    model.save(out / "frozen_dual.neri-model")
    freeze(out / "protocol.json",dict(schemes=SCHEMES,encoder="frozen DINOv3-B/16; shared B-fitted geometry",
           classifiers=dict(production_dual="Ridge, tau1, gated target tracks",track_single_ridge="Ridge, tau1, all target tracks",
                            experiment_single_ridge="Ridge, tau1, whole events",experiment_full=".55 row-z Within cardinality .02 + .45 row-z Ridge; tau .5; whole events"),
           gate="all thresholds frozen on C auxiliary unknown species Macro90; production per-box prerequisite Macro80",
           test_selection=False,reference_sha256=sha256(Path(reference.__file__))))
    write(out/"source_provenance.json",{str(p):sha256(p) for p in (
        REPO/"system/dinov3/model.py",REPO/"system/dinov3/context.py",
        Path(reference.__file__).with_name("verify_classifier_head_improvements_20261107.py"))})
    print("SCORING calibration",len(cq["raw"]),flush=True)
    cdual = model.predict(cq["raw"],cq["cameras"],cq["sequences"],contents=cq["contents"],adapted_features=cq["adapted"])
    cs = collect_scores(model,cq)
    print("SCORING evaluation",len(eq["raw"]),flush=True)
    edual = model.predict(eq["raw"],eq["cameras"],eq["sequences"],contents=eq["contents"],adapted_features=eq["adapted"])
    es = collect_scores(model,eq)
    if baseline is not None:
        for actual,path in ((cdual,baseline/"calibration_replay.npz"),(edual,baseline/"evaluation_predictions.npz")):
            with np.load(path,allow_pickle=False) as expected:
                np.testing.assert_array_equal(actual["predictions"],expected["predictions"])
                np.testing.assert_array_equal(actual["accepted"],expected["accepted"])
                np.testing.assert_allclose(actual["box_known_score"],expected["box_known_score"],atol=1e-10,rtol=0)
        write(out/"baseline_parity.json",dict(passed=True,C_and_E_predictions_acceptance_match=True))
    results = {"production_dual":(cdual,edual)}
    calibration = {"production_dual":model.calibration}
    for name in SCHEMES[1:]:
        print("RECIPE",name,flush=True)
        result,cal = reference_recipe(model,cq,eq,cs,es,bank_cameras,bank_labels,reference,
                                      track=name=="track_single_ridge",blend=name=="experiment_full")
        results[name] = result
        calibration[name] = cal
    # Save all decisions before exposing E metrics, including per-row paired records.
    for name,(c,e) in results.items():
        np.savez_compressed(out/f"{name}.npz",c_indices=cq["indices"],e_indices=eq["indices"],classes=model.classes,
                            c_pred=c["predictions"],c_accepted=c["accepted"],c_score=c["known_score"],
                            e_pred=e["predictions"],e_accepted=e["accepted"],e_score=e["known_score"])
    np.savez_compressed(out/"query_metadata.npz",c_indices=cq["indices"],e_indices=eq["indices"],
                        c_labels=cq["labels"],e_labels=eq["labels"],
                        c_cameras=cq["cameras"],e_cameras=eq["cameras"])
    np.savez_compressed(out/"production_gate_components.npz",indices=eq["indices"],
                        box_accepted=edual["box_accepted"],sequence_accepted=edual["sequence_accepted"],
                        box_score=edual["box_known_score"],sequence_score=edual["sequence_known_score"])
    write(out/"calibration.json",calibration)
    write(out/"decisions_frozen.json",dict(files={p.name:sha256(p) for p in out.glob("*.npz")},
                                         calibration_sha256=sha256(out/"calibration.json")))
    summary=[]
    for name,pair in results.items():
        for split,q,p in zip(("calibration","evaluation"),(cq,eq),pair):
            summary.append(dict(scheme=name,split=split,**metrics(q["labels"],p["predictions"],p["accepted"],model.classes)))
    write(out/"metrics.json",summary)
    csvwrite(out/"metrics.csv",summary)
    species_rows=[]
    for name,(_,e) in results.items():
        for label in np.unique(eq["labels"]):
            mask = eq["labels"]==label
            species_rows.append(dict(scheme=name,species=label,known=bool(label in model.classes),photos=int(mask.sum()),
                                     correct=int((e["predictions"][mask]==label).sum()),accepted=int(e["accepted"][mask].sum()),
                                     rejected=int((~e["accepted"][mask]).sum()),
                                     correct_accepted=int(((e["predictions"][mask]==label)&e["accepted"][mask]).sum())))
    csvwrite(out/"per_species.csv",species_rows)
    records=[]
    for j,i in enumerate(eq["indices"]):
        row=dict(index=int(i),species=eq["labels"][j])
        for name,(_,e) in results.items():
            row[name+"_prediction"]=e["predictions"][j]
            row[name+"_accepted"]=bool(e["accepted"][j])
        records.append(row)
    csvwrite(out/"paired_predictions.csv",records)
    known=np.isin(eq["labels"],model.classes)
    y=eq["labels"]
    macro=lambda v: float(np.mean([v[y==c].mean() for c in np.unique(y[known])]))
    write(out/"gate_diagnosis.json",dict(known_box_failure_macro=macro(~edual["box_accepted"]),
          known_sequence_failure_after_box_pass_macro=macro(edual["box_accepted"]&~edual["sequence_accepted"]),
          known_correct_but_rejected_macro=macro((edual["predictions"]==y)&~edual["accepted"])))
    print("METRICS",json.dumps([r for r in summary if r["split"]=="evaluation"],ensure_ascii=False),flush=True)
    return summary


def run_old(source,old,out,reference):
    rows=rows_read(old/"reviewed.jsonl")
    raw=np.load(old/"features.npy",allow_pickle=False,mmap_mode="r")
    parts=read(old/"partition.json")["indices"]
    model=DinoV3RidgeWithinModel.load(old/"frozen.neri-model")
    with np.load(old/"calibration_replay.npz",allow_pickle=False) as c:
        cq=dict(raw=c["features"],adapted=c["adapted_features"],labels=c["labels"],cameras=c["cameras"],
                sequences=c["sequences"],contents=c["contents"],indices=np.asarray(parts["calibration"]),
                crop_hashes=meta(rows,parts["calibration"],"sha256"))
    eq=query(rows,raw,np.asarray(parts["evaluation"]))
    return compare(model,cq,eq,meta(rows,parts["bank"],"camera"),meta(rows,parts["bank"],"species"),reference,out/"original_split",old)


def run_new(source,old,out,reference):
    rows=rows_read(out/"current_reviewed.jsonl")
    protocol=read(out/"partition.json")
    assignments={int(k):v for k,v in protocol["assignments"].items()}
    validate_partition(rows,assignments,protocol["known_classes"])
    raw_cache=np.load(old/"features.npy",allow_pickle=False,mmap_mode="r")
    raw=raw_cache[[r["feature_index"] for r in rows]]
    choose=lambda names: np.asarray(sorted(i for i,p in assignments.items() if p in names),dtype=np.int64)
    b=choose({"bank"}); c=choose({"calibration","calibration_unknown"}); e=choose({"evaluation","evaluation_unknown"})
    print("FIT strict camera Bank",len(b),flush=True)
    model=DinoV3RidgeWithinModel(metadata=dict(encoder_sha256=read(old/"feature_verification.json")["encoder_sha256"],
            partition_sha256=sha256(out/"partition.json"),not_for_production=True)).fit(
            raw[b],meta(rows,b,"species"),meta(rows,b,"camera"),meta(rows,b,"sequence"))
    cq=query(rows,raw,c); eq=query(rows,raw,e)
    print("CALIBRATE strict camera",len(c),flush=True)
    model.calibrate(cq["raw"],cq["labels"],cq["cameras"],cq["sequences"],contents=cq["contents"],adapted_features=cq["adapted"])
    return compare(model,cq,eq,meta(rows,b,"camera"),meta(rows,b,"species"),reference,out/"camera_split")


def verify_results(out):
    """Recompute saved decisions and camera-paired intervals without refitting."""
    verification = {}
    for folder in ("original_split","camera_split"):
        root = out/folder
        if not (root/"query_metadata.npz").exists():
            continue
        frozen = read(root/"decisions_frozen.json")
        for name,digest in frozen["files"].items():
            if sha256(root/name)!=digest:
                raise ValueError(f"Frozen decision file changed: {name}")
        if sha256(root/"calibration.json")!=frozen["calibration_sha256"]:
            raise ValueError("Frozen calibration changed")
        expected = {(r["scheme"],r["split"]):r for r in read(root/"metrics.json")}
        with np.load(root/"query_metadata.npz",allow_pickle=False) as q:
            states={name:dict(np.load(root/f"{name}.npz",allow_pickle=False)) for name in SCHEMES}
            for name,z in states.items():
                for prefix,split in (("c","calibration"),("e","evaluation")):
                    np.testing.assert_array_equal(z[prefix+"_indices"],q[prefix+"_indices"])
                    actual=metrics(q[prefix+"_labels"],z[prefix+"_pred"],z[prefix+"_accepted"],z["classes"])
                    for key,value in actual.items():
                        wanted=expected[(name,split)][key]
                        if value is None:
                            assert wanted is None
                        else:
                            np.testing.assert_allclose(value,wanted,atol=1e-12,rtol=0)
                    if split=="calibration" and actual["unknown_rejection_macro"]<.9-1e-12:
                        raise ValueError("Macro90 calibration target not reached")
            # Same camera bootstrap draws for both methods; descriptive historical intervals.
            y=q["e_labels"]; classes=states["production_dual"]["classes"]
            cameras,inv=np.unique(q["e_cameras"],return_inverse=True)
            draws=np.random.default_rng(SEED).multinomial(len(cameras),np.full(len(cameras),1/len(cameras)),size=2000)
            intervals=[]
            for name in SCHEMES[1:]:
                base,other=states["production_dual"],states[name]
                for metric,mask,bv,ov in (
                    ("known_correct_acceptance",np.isin(y,classes),(base["e_pred"]==y)&base["e_accepted"],(other["e_pred"]==y)&other["e_accepted"]),
                    ("known_false_rejection",np.isin(y,classes),~base["e_accepted"],~other["e_accepted"]),
                    ("unknown_rejection",~np.isin(y,classes),~base["e_accepted"],~other["e_accepted"])):
                    target=np.unique(y[mask])
                    if not len(target): continue
                    counts=np.stack([np.bincount(inv[y==c],minlength=len(cameras)) for c in target],axis=1)
                    delta=ov.astype(float)-bv.astype(float)
                    sums=np.stack([np.bincount(inv[y==c],weights=delta[y==c],minlength=len(cameras)) for c in target],axis=1)
                    den=draws@counts; num=draws@sums; present=den>0
                    values=np.divide(num,den,out=np.zeros_like(num),where=present).sum(1)/np.maximum(present.sum(1),1)
                    valid=present.any(1); lower,upper=np.quantile(values[valid],[.025,.975])
                    point=float(np.mean([delta[y==c].mean() for c in target]))
                    intervals.append(dict(scheme=name,reference="production_dual",metric=metric,delta_percentage_points=point*100,
                                          lower95_percentage_points=float(lower*100),upper95_percentage_points=float(upper*100),
                                          bootstrap_cameras=len(cameras),bootstrap_draws=2000,
                                          min_species_present=int(present[valid].sum(1).min()),target_species=len(target)))
            csvwrite(root/"paired_camera_intervals.csv",intervals)
        verification[folder]=dict(passed=True,frozen_hashes_checked=True,all_saved_metrics_recomputed=True,
                                  C_Macro90_verified=True,paired_intervals=len(intervals))
    write(out/"verification.json",verification)
    return verification


def publish_uses(source,out):
    """Register the new data-use pointer; never alter the application model."""
    validation=read(out/"partition_validation.json")
    audit=read(out/"input_audit.json")
    if not validation["passed"] or audit["new_files"] or audit["changed_content"] or not audit["all_current_images_content_hashed"]:
        raise ValueError("Cannot register an unverified data-use split")
    pointer=source/"data/current_reviewed_training_split.json"
    if pointer.exists() and not (out/"previous_training_split_pointer.json").exists():
        write(out/"previous_training_split_pointer.json",read(pointer))
    paths={purpose:str(out/f"{purpose}.jsonl") for purpose in read(out/"partition.json")["purpose_counts"]}
    payload=dict(version="reviewed_camera_isolated_20261114",root=str(out),
                 partition=str(out/"partition.json"),partition_sha256=sha256(out/"partition.json"),
                 photo_uses=str(out/"photo_uses.csv"),species_uses=str(out/"species_uses.csv"),
                 manifests=paths,manifest_sha256={purpose:sha256(path) for purpose,path in paths.items()},
                 feature_cache=dict(path=audit.get("features_file",str(REPO/"runs/dinov3_reviewed_20261107/features.npy")),
                                    sha256=audit["features_sha256"],row_index_field="feature_index",
                                    encoder_sha256=audit["encoder_sha256"]),
                 known_classes=read(out/"partition.json")["known_classes"],
                 note="Photo-use manifests only. Original photos stay in place; application model and older dataset pointers are unchanged.")
    write(pointer,payload)
    write(out/"registered_training_split_pointer.json",dict(path=str(pointer),sha256=sha256(pointer)))


def report(out):
    labels=dict(production_dual="当前双门控＋Ridge",track_single_ridge="目标轨迹单门控＋Ridge",
                experiment_single_ridge="实验事件单门控＋Ridge",experiment_full="完整实验配方：Within＋Ridge")
    lines=["# 本地数据：双门控与实验配方比较、照片用途重划分", "",
           "固定 DINOv3-B/16 编码特征。全部方案在每一划分内共享 Bank、白化几何和历史输入；每个门控只用 C 重新校准 Macro90。未替换软件当前模型。", "",
           "原划分用于诊断；新划分按整相机及原图/内容关联组件隔离，未知校准与未知测试按物种隔离。所有图片都曾在历史训练/校准/研究中曝光，因此新划分是重新训练后的相机留出诊断，不是新采集的盲测。", ""]
    if (out/"camera_split/metrics.json").exists():
        values={r["scheme"]:r for r in read(out/"camera_split/metrics.json") if r["split"]=="evaluation"}
        dual,exp=values["production_dual"],values["experiment_full"]
        lines += [f"新划分上，完整实验配方比双门控的正确接纳率高 {(exp['known_correct_acceptance_macro']-dual['known_correct_acceptance_macro'])*100:.2f} 个百分点；未知拒识率分别为 {exp['unknown_rejection_macro']*100:.2f}% 和 {dual['unknown_rejection_macro']*100:.2f}%，均未达到 90%。原划分上双门控反而有更高正确接纳率，因此不能笼统断言双门控导致退化。", ""]
        interval_path=out/"camera_split/paired_camera_intervals.csv"
        if interval_path.exists():
            with interval_path.open(encoding="utf-8-sig") as stream:
                interval=next(r for r in csv.DictReader(stream) if r["scheme"]=="experiment_full" and r["metric"]=="known_correct_acceptance")
            lines += [f"实验配方正确接纳收益的相机配对 95% 区间为 [{float(interval['lower95_percentage_points']):+.2f}, {float(interval['upper95_percentage_points']):+.2f}] 个百分点，跨过 0；当前只能描述为本次划分上的小幅点估计优势，不能认定稳定优于双门控。", ""]
    for folder,title in (("original_split","原有划分：同一批本地图片上的配对对照"),("camera_split","新用途划分：相机隔离＋独立未知物种测试")):
        path=out/folder/"metrics.json"
        if not path.exists():
            continue
        lines += [f"## {title}","","全部比例为物种宏平均；正确接纳要求分类正确且通过门控。","",
                  "|方案|已知类数/图片|未知类数/图片|闭集准确率|正确接纳率|已知误拒率|未知拒识率|",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in read(path):
            if r["split"]!="evaluation": continue
            pct=lambda k: "—" if r[k] is None else f"{r[k]*100:.2f}%"
            lines.append(f"|{labels[r['scheme']]}|{r['known_species']}/{r['known_images']}|{r['unknown_species']}/{r['unknown_images']}|{pct('known_closed_accuracy_macro')}|{pct('known_correct_acceptance_macro')}|{pct('known_false_rejection_macro')}|{pct('unknown_rejection_macro')}|")
        lines += ["", "单框移除对照也重建轨迹和重新校准；不能将差值全部解释为 AND 判定本身。事件门控沿用历史实验整事件聚合，可能混合多目标，属于诊断对照。", ""]
    partition=read(out/"partition.json")
    lines += ["## 新照片用途", "", "|用途|图片数|", "|---|---:|"]
    for purpose,count in sorted(partition["purpose_counts"].items()):
        lines.append(f"|{purpose}|{count}|")
    lines += ["", "bank：拟合 Ridge/Within/白化；calibration：已知校准及误拒监测；calibration_unknown：未知门限校准；evaluation：已知相机留出；evaluation_unknown：未参与校准的未知物种测试；future_support_reserve：相机归属不符合本轮未知角色的照片，保留供未来补图，不参与本轮拟合/历史/评分；excluded_duplicate_crop：重复裁图，仅留一个代表。", "",
              "逐图用途在 photo_uses.csv 与各用途 JSONL；逐物种覆盖在 species_uses.csv；完整划分与种子在 partition.json；隔离核验在 partition_validation.json。仅改用途清单，不移动/删除原始照片。", "",
              "单相机已知物种仅用于 Bank；两相机已知物种用于 Bank/E。缺少独立相机的物种不计入相应测试分母。新 E 不参与选门限、超参数、划分或挑选方案；不得根据本轮 E 结果继续调参后复用它作盲测。", "",
              "相机隔离依据恢复的相机目录 ID；没有按硬件序号或 GPS 进一步合并跨目录的同一设备/机位，因此不声称全新站点泛化。保留 45 个原已知物种以控制类别变化；红嘴鸥与赤麻鸭仅入 Bank，红翅旋壁雀有 Bank/E、无独立 C。未知校准仅 18 类 27 张、未知测试 21 类 28 张，不能据此声称精确且稳定的未知拒识率。", "",
              "当前安装模型在 11 月 13 日已自动补图；本轮生产基线是可复现的 11 月 7 日冻结初始模型，并在 C/E 逐项重放一致，不将初始模型指标冒充自动学习后的当前实例成绩。", "",
              "本轮比较固定 B/16；L/16 需要独立提取本地特征，未将历史 iWildCam L/16 成绩当成本地结果。"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("F:/files/python/Neri_plus"))
    parser.add_argument("--old",type=Path,default=REPO/"runs/dinov3_reviewed_20261107")
    parser.add_argument("--out",type=Path,default=REPO/"runs/local_protocol_comparison_20261114")
    parser.add_argument("--stage",choices=("prepare","old","new","evaluate-new","report","verify","publish-uses","all"),default="all")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.source))
    from training import evaluate_improved_head_gate_sizes_20261107 as reference
    with threadpool_limits(limits=4):
        if args.stage in ("prepare","all"):
            rows=audit_inputs(args.source,args.old,args.out)
            save_partition(rows,args.old,args.out)
        if args.stage in ("old","all"):
            run_old(args.source,args.old,args.out,reference)
        if args.stage in ("new","evaluate-new","all"):
            run_new(args.source,args.old,args.out,reference)
        if args.stage in ("verify","new","evaluate-new","all"):
            verify_results(args.out)
        if args.stage in ("publish-uses","new","all"):
            publish_uses(args.source,args.out)
        if (args.out/"partition.json").exists():
            report(args.out)
    print("DONE",args.out,flush=True)


if __name__=="__main__":
    main()
