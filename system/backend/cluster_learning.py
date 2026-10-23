"""Explicit human confirmation of a rejected cluster and its saved features."""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path


def _schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS cluster_feedback_operations(
        operation_id TEXT PRIMARY KEY,request_json TEXT NOT NULL,result_json TEXT)""")
    conn.commit()


def confirm_rejection_cluster(request):
    from .dinov2_registry_service import load_checkpoint_for_model, open_registry_for_model, _open_candidate_feedback
    checkpoint = load_checkpoint_for_model(request.classification_model_path)
    registry = open_registry_for_model(request.classification_model_path)
    try:
        feedback = _open_candidate_feedback(registry, checkpoint)
        try:
            return _confirm(request, checkpoint, registry, feedback)
        finally:
            feedback.close()
    finally:
        registry.close()


def _verified_cluster_evidence(request, checkpoint, registry, feedback):
    from system.dinov2.rejection_clustering import MIN_REJECTION_CLUSTER_EVENTS
    from .dinov2_registry_service import _candidate_feedback_observations
    from .rejection_clusters import _resolve, cluster_assignment

    # Verify density and freeze its evidence in one WAL read snapshot. Runtime
    # writes may continue during a long fit, but newly arriving, unverified
    # events must not be swept into the subsequent human-learning operation.
    registry._conn.execute("BEGIN")
    try:
        entry = registry.get(int(request.cluster_id.split(":")[2]))
        if entry.status != "candidate":
            raise ValueError("已注册物种请通过物种注册界面修改；此入口用于确认拒识候选簇")
        if entry.event_count < MIN_REJECTION_CLUSTER_EVENTS:
            raise ValueError(f"该候选不足 {MIN_REJECTION_CLUSTER_EVENTS} 个独立事件，暂按噪声/证据不足处理；可逐框人工校验")
        links = _resolve(feedback, registry._conn, list(set(request.observation_ids)))
        expected = cluster_assignment(checkpoint.fingerprint, entry)["cluster_id"]
        if any(links.get(oid, {}).get("cluster_id") != expected for oid in request.observation_ids):
            raise ValueError("所选检测框不全部属于当前簇，未修改其他簇")
        if any(value.get("is_noise", True) for value in links.values()):
            raise ValueError("该候选未通过 HDBSCAN 密度成簇校验，暂按噪声/证据不足处理；可逐框人工校验")
        observations = _candidate_feedback_observations(registry, entry.id, checkpoint=checkpoint)
        if not observations:
            raise ValueError("该簇没有可学习的已保存特征")
        return entry, observations
    finally:
        registry._conn.rollback()


def _confirm(request, checkpoint, registry, feedback):
    from . import services
    from .dinov2_registry_service import _feature_center_array
    from .dinov2_feedback_service import (
        _journal_ecological_state, _raw_box_for_observation, _recompute_detection_summary,
        revert_feedback_operation,
    )

    species = request.species_name.strip()
    if not species or species.lower() in {"unknown", "empty"} or species == "空" or "拒识" in species:
        raise ValueError("请填写人工确认的实际物种名称")
    if any(separator in species for separator in (",", "，", "、")):
        raise ValueError("每个簇只能确认一个物种")
    species = next((name for name in checkpoint.classes if name.casefold() == species.casefold()), species)
    root = Path(request.input_path).expanduser().resolve()
    if not root.exists():
        raise ValueError("输入路径不存在")
    parts = request.cluster_id.split(":")
    if len(parts) != 4 or parts[0] != "registry" or parts[1] != checkpoint.fingerprint or parts[3] != "0":
        raise ValueError("簇编号与当前分类模型不一致")
    entry = registry.get(int(parts[2]))
    operation_id = request.feedback_operation_id
    signature = json.dumps({"cluster": request.cluster_id, "species": species, "root": str(root),
                            "ids": sorted(set(request.observation_ids))}, ensure_ascii=False, sort_keys=True)
    _schema(feedback._conn)
    cached = feedback._conn.execute("SELECT * FROM cluster_feedback_operations WHERE operation_id=?", (operation_id,)).fetchone()
    if cached is not None:
        if cached["request_json"] != signature:
            raise ValueError("同一反馈操作不能用于不同的簇或物种")
        operation = feedback._conn.execute("SELECT reverted FROM feedback_operations WHERE operation_id=?", (operation_id,)).fetchone()
        if operation is not None and operation[0]:
            raise ValueError("该学习操作已撤回")
        if cached["result_json"]:
            return json.loads(cached["result_json"])
        raise ValueError("该学习操作未完成，请撤回后重试")
    entry, observations = _verified_cluster_evidence(request, checkpoint, registry, feedback)

    # Validate and prepare every visible file before learning or writing labels.
    updates = {}
    for oid in sorted(set(request.observation_ids)):
        observation = feedback.get_observation(oid)
        path = Path(observation.source_path).expanduser().resolve()
        if not (path == root or (root.is_dir() and path.is_relative_to(root))):
            raise ValueError("检测框不属于当前输入目录")
        if not path.is_file():
            raise ValueError(f"文件不存在: {path}")
        if path not in updates:
            item = services._reload_validation_item(path, root)
            before = services._load_detection_data_for_path(path, services._detection_db_search_roots(root, None))
            if not before:
                raise ValueError(f"缺少检测结果: {path.name}")
            updates[path] = {"before": before, "data": copy.deepcopy(before), "item": item, "oid": oid}
        box = _raw_box_for_observation(updates[path]["data"], oid)
        box["物种"] = species
        box["feedback_status"] = "confirmed"
    feedback._conn.execute("INSERT INTO cluster_feedback_operations VALUES(?,?,NULL)", (operation_id, signature))
    feedback._conn.commit()
    renamed = False
    try:
        affected = {species} if species in checkpoint.classes else set()
        for path, context in updates.items():
            _journal_ecological_state(feedback, operation_id=operation_id, observation_id=context["oid"],
                                      input_path=str(root), file_path=str(path), detection_data=context["before"],
                                      validated=context["item"].validated)
        for observation in observations:
            previous = feedback._conn.execute(
                "SELECT positive_species,hard_negative_species FROM human_feedback WHERE observation_id=? AND active=1",
                (observation.id,),
            ).fetchone()
            if previous is not None:
                affected.update(value for value in previous if value in checkpoint.classes)
            feedback.persist_observation(observation)
            if species in checkpoint.classes:
                feedback.record_feedback(observation.id, operation_id=operation_id, action="update", confirmed_species=species)
            else:
                feedback.record_registry_feedback(observation.id, operation_id=operation_id, registration_id=entry.id,
                                                  previous_common_name=entry.common_name, previous_scientific_name=entry.scientific_name,
                                                  confirmed_species=species)
        if species in checkpoint.classes:
            states = {name: feedback.recompute_species(name, _feature_center_array(checkpoint)) for name in sorted(affected)}
            state = states[species]
            learning = {"status": state.status, "active": bool(state.active_generation),
                        "formal": state.status in {"confirmed", "mature"} and bool(state.active_generation),
                        "needs_review": state.needs_review}
        else:
            for name in sorted(affected):
                feedback.recompute_species(name, _feature_center_array(checkpoint))
            registry.set_identity(entry.id, common_name=species, scientific_name=entry.scientific_name)
            renamed = True
            learning = {"status": "candidate", "active": False, "formal": False, "needs_review": False}
        for path, context in updates.items():
            data = context["data"]
            _recompute_detection_summary(data)
            boxes = data.get("检测框", [])
            validated = context["item"].validated or bool(boxes) and all(box.get("feedback_status") == "confirmed" for box in boxes)
            services._persist_validation_update(path, root, data, bool(validated))
        result = {"operation_id": operation_id, "feature_events": len(observations), "learning": learning,
                  "items": [services._reload_validation_item(path, root).model_dump(mode="json") for path in updates]}
        feedback._conn.execute("UPDATE cluster_feedback_operations SET result_json=? WHERE operation_id=?",
                               (json.dumps(result, ensure_ascii=False), operation_id))
        feedback._conn.commit()
        return result
    except Exception:
        # The existing undo journal restores visible labels and feedback
        # generations. Original encoder/checkpoint weights are never modified.
        try:
            revert_feedback_operation(request)
        except Exception:
            logging.getLogger(__name__).exception("Cluster feedback rollback needs recovery: %s", operation_id)
        if renamed:
            try:
                if registry.get(entry.id).common_name == species:
                    registry.restore_identity(entry.id, common_name=entry.common_name, scientific_name=entry.scientific_name)
            except Exception:
                logging.getLogger(__name__).exception("Could not restore cluster identity: %s", entry.id)
        raise
