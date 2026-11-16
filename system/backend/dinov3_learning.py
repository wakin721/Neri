"""Durable, automatic human-reviewed learning for DINOv3 joint models.

Annotations stage source-bound crops; a background worker coalesces confirmations,
replays calibration, and atomically publishes a new generation.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing, contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
from system.dinov3.display import model_display_name
import sqlite3
import threading
import zipfile
from datetime import datetime, timezone

import numpy as np

from system.dino.state import default_state_root

_LOCK = threading.RLock()
_UPDATE_LOCK = threading.Lock()
_WORKERS_LOCK = threading.Lock()
_WORKERS = {}
_CONTEXT = threading.local()
_CALIBRATION_CHECKS = set()
_UNKNOWN = {"", "空", "Unknown", "unknown", "拒识/Unknown", "无"}
_UNKNOWN_FOLDED = {name.casefold() for name in _UNKNOWN} | {"unknow", "empty", "未知", "未知物种", "待确认", "待核实"}
_REQUIRED = {"features", "labels", "cameras", "sequences"}
_OPTIONAL = {"sites", "contents", "source_contents", "adapted_features", "datetimes"}


class _Superseded(RuntimeError):
    pass


def _key(path) -> str:
    if isinstance(path,str) and path.startswith('family:'):
        return path
    from system.dinov3.shared_learning import family_descriptor
    descriptor=family_descriptor(path)
    if descriptor:
        return descriptor['key']
    return str(Path(path).expanduser().resolve()).replace("\\", "/").casefold()


def _path_key(path):
    return str(Path(path).expanduser().resolve()).replace('\\','/').casefold()


def _state_root():
    return getattr(_CONTEXT,'root',None) or default_state_root()


def _connect():
    root = _state_root()
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(root / "dinov3_learning.sqlite3"), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS samples(
          model TEXT NOT NULL, observation TEXT NOT NULL, data TEXT NOT NULL,
          operation TEXT NOT NULL, applied INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(model, observation));
        CREATE TABLE IF NOT EXISTS operations(
          model TEXT NOT NULL, observation TEXT NOT NULL, operation TEXT NOT NULL,
          previous TEXT, PRIMARY KEY(model, observation, operation));
        CREATE TABLE IF NOT EXISTS learning_runs(
          model TEXT PRIMARY KEY, state TEXT NOT NULL, message TEXT NOT NULL,
          last_update TEXT, last_error TEXT);
        CREATE TABLE IF NOT EXISTS shared_learning_aliases(
          alias TEXT PRIMARY KEY, family TEXT NOT NULL, checkpoint_sha256 TEXT NOT NULL,
          imported_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS shared_learning_committed(
          model TEXT NOT NULL, observation TEXT NOT NULL, data TEXT NOT NULL,
          operation TEXT NOT NULL, feature BLOB, PRIMARY KEY(model,observation));
    """)
    if "feature" not in {row[1] for row in conn.execute("PRAGMA table_info(samples)")}:
        conn.execute("ALTER TABLE samples ADD COLUMN feature BLOB")
        conn.commit()
    return conn


def _snapshot_row(row,key):
    if row is None:
        return None
    snapshot=dict(row)
    snapshot['model']=key
    snapshot['feature']=bytes(snapshot['feature']).hex() if snapshot.get('feature') is not None else None
    return json.dumps(snapshot,ensure_ascii=False)


def _import_alias(conn,source,payload,target,current,descriptor):
    """Copy legacy rows and undo chains; originals are retained as backups."""
    from system.dinov3 import shared_learning as shared
    from system.dinov3.runtime import resolve_manifest_asset
    from system.dinov3.encoder import sha256
    alias=_path_key(source)
    key=descriptor['key']
    receipt=conn.execute('SELECT family FROM shared_learning_aliases WHERE alias=?',(alias,)).fetchone()
    if receipt:
        if receipt['family']!=key:
            raise ValueError('模型路径曾绑定另一套学习档案，未合并。')
        return current
    checkpoint=resolve_manifest_asset(source,payload['checkpoint'])
    if sha256(checkpoint)!=payload['checkpoint_sha256']:
        raise ValueError('旧模型校验失败，未迁移学习记录。')
    calibration=resolve_manifest_asset(source,payload['calibration'])
    if sha256(calibration)!=payload['calibration_sha256']:
        raise ValueError('旧模型校准包校验失败，未迁移。')
    if shared.calibration_identity(calibration)!=current['learning_base_calibration_id']:
        raise ValueError('两版模型的校准数据不同，无法自动合并；原模型及记录已保留。')
    rows=conn.execute('SELECT * FROM samples WHERE model=? ORDER BY observation',(alias,)).fetchall()
    learned=shared.checkpoint_info(source,payload).get('metadata',{}).get('human_learning_samples',{})
    recorded={json.loads(row['data']).get('sample_id') for row in rows}
    if set(learned)-recorded:
        raise ValueError('旧模型包含缺少补图记录的已学样本，无法安全合并；原模型已保留。')
    current=shared.register_encoder(source,payload,target,current)
    previous={}
    for row in rows:
        existing=conn.execute('SELECT * FROM samples WHERE model=? AND observation=?',(key,row['observation'])).fetchone()
        if existing is not None:
            left,right=json.loads(existing['data']),json.loads(row['data'])
            fields=('file_path','bbox','species','size','mtime_ns')
            if any(left.get(field)!=right.get(field) for field in fields):
                raise ValueError('CPU/GPU 对同一检测框存在不同校验，未自动覆盖；请先统一标签。')
            if existing['operation']==row['operation']:
                continue
            previous[row['observation']]=_snapshot_row(existing,key)
        feature=row['feature'] if row['feature'] is not None else existing['feature'] if existing is not None else None
        conn.execute('INSERT OR REPLACE INTO samples(model,observation,data,operation,applied,feature) VALUES(?,?,?,?,0,?)',
                     (key,row['observation'],row['data'],row['operation'],feature))
        # Keep the last published support separately from a pending correction.
        # This also makes "discard pending" preserve existing learned species.
        data=json.loads(row['data'])
        committed=dict(row)
        if data.get('sample_id') in learned:
            for operation in conn.execute('SELECT previous FROM operations WHERE model=? AND observation=? AND previous IS NOT NULL ORDER BY rowid DESC',
                                          (alias,row['observation'])).fetchall():
                candidate=json.loads(operation['previous'])
                old_data=json.loads(candidate['data'])
                if old_data.get('sample_id') in learned and old_data.get('species')==learned[old_data['sample_id']]:
                    committed=candidate
                    committed['feature']=bytes.fromhex(candidate['feature']) if candidate.get('feature') else None
                    data=old_data
                    break
            data['species']=learned[data['sample_id']]
            conn.execute('INSERT OR REPLACE INTO shared_learning_committed VALUES(?,?,?,?,?)',
                         (key,row['observation'],json.dumps(data,ensure_ascii=False),committed['operation'],committed['feature']))
    for row in conn.execute('SELECT * FROM operations WHERE model=? ORDER BY rowid',(alias,)).fetchall():
        old=row['previous']
        if old is not None:
            value=json.loads(old)
            value['model']=key
            old=json.dumps(value,ensure_ascii=False)
        elif row['observation'] in previous:
            old=previous[row['observation']]
        conn.execute('INSERT OR IGNORE INTO operations VALUES(?,?,?,?)',(key,row['observation'],row['operation'],old))
    conn.execute('INSERT INTO shared_learning_aliases VALUES(?,?,?,?)',
                 (alias,key,payload['checkpoint_sha256'],datetime.now(timezone.utc).isoformat()))
    return current


def prepare_shared_learning(model_path,payload=None,*,connection=None):
    """Return a family working manifest while preserving the selected encoder."""
    from system.dinov3 import shared_learning as shared
    from system.dinov3.runtime import read_manifest
    source=Path(model_path).expanduser().resolve()
    descriptor=shared.family_descriptor(source,payload)
    if descriptor is None:
        return source,payload
    target=shared.family_manifest(_state_root(),descriptor['id'])
    if source==target.resolve():
        return source,payload if payload is not None else read_manifest(source)
    payload=payload if payload is not None else read_manifest(source)
    with (nullcontext(connection) if connection is not None else _write_lock()) as conn:
        candidates=[(source,payload)]
        registered=conn.execute('SELECT 1 FROM shared_learning_aliases WHERE family=? LIMIT 1',(descriptor['key'],)).fetchone()
        if not target.is_file() or registered is None:
            for peer in shared.peers(source,descriptor['id']):
                candidates.append((peer,read_manifest(peer)))
            # Upgrades may live in another folder. The durable feedback DB
            # knows the previous installation paths even when the new catalog
            # only lists the current installation's resources.
            legacy_paths={row[0] for row in conn.execute(
                'SELECT model FROM samples UNION SELECT model FROM operations UNION SELECT model FROM learning_runs')}
            known={_path_key(alias) for alias,_ in candidates}
            for legacy in sorted(legacy_paths):
                if legacy.startswith('family:') or legacy in known:
                    continue
                candidate=Path(legacy)
                if candidate.is_file() and shared.family_id(candidate)==descriptor['id']:
                    candidates.append((candidate,read_manifest(candidate)))
                    known.add(legacy)
        primary=max(candidates,key=lambda pair:len(shared.checkpoint_info(*pair).get('metadata',{}).get('human_learning_samples',{})))
        target,current,_created=shared.initialize(*primary,descriptor,_state_root())
        # When duplicate confirmations agree, the explicitly selected legacy
        # deployment becomes the current operation and its undo restores the
        # previously imported confirmation rather than erasing both histories.
        for alias,alias_payload in [*candidates[1:],candidates[0]]:
            current=_import_alias(conn,alias,alias_payload,target,current,descriptor)
        current=shared.register_encoder(source,payload,target,current)
        if not target.is_file() or json.loads(target.read_text('utf8'))!=current:
            shared.atomic_manifest(target,current)
    return target,current


def _same_publication(left,right):
    if left.get('learning_family_id'):
        # Choosing an encoder changes no learned evidence. UI polling and a
        # CPU/GPU switch must not repeatedly supersede an otherwise valid fit.
        keys=('learning_family_id','checkpoint_sha256','calibration_sha256','learning_base_sha256')
        return all(left.get(key)==right.get(key) for key in keys)
    return left==right


def resolve_learning_snapshot(model_path,payload):
    """Finish a one-time legacy merge before inference sees the shared head."""
    manifest,current=prepare_shared_learning(model_path,payload)
    if current is None or not current.get('learning_family_id'):
        return manifest,current or payload
    with closing(_connect()) as conn:
        pending=conn.execute('SELECT 1 FROM samples WHERE model=? AND applied=0 LIMIT 1',(_key(manifest),)).fetchone()
    if pending is not None:
        update_model(model_path)
    # Another model can select a backend during a coalesced update. Reapply the
    # caller's choice; immutable generation files make this snapshot safe to use.
    return prepare_shared_learning(model_path,payload)


@contextmanager
def _write_lock():
    """Serialize publication and annotation changes, also across backend processes."""
    with _LOCK, closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


def stage_feedback(model_path, input_path, file_path, box, action, operation_id, *, connection=None):
    """Stage only an explicit per-box confirmation; repeated marks replace a row."""
    from system.dinov3.observation_support import _box_bbox, make_dinov3_observation_id
    from system.dinov3.context import camera_id_for_path

    if not model_path:
        return
    prepare_shared_learning(model_path,connection=connection)
    file_path = Path(file_path).resolve()
    bbox = _box_bbox(box)
    observation = str(box.get("observation_id") or "")
    if not observation:
        raise ValueError("补图缺少检测框身份，请重新识别该照片。")
    species = str(box.get("物种") or "").strip()
    eligible = (action in {"update", "correct"} and species.casefold() not in _UNKNOWN_FOLDED
                and box.get("learning_eligible") is not False and box.get("learning_excluded") is not True)
    if eligible and (len(bbox) != 4 or not np.isfinite(bbox).all() or
                     bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
        raise ValueError("补图需要有效的目标检测框。")
    if eligible and box.get("frame_index") is not None:
        raise ValueError("联合模型补图暂只支持照片。")
    stat = file_path.stat()
    primary = (box.get("候选项") or [{}])[0]
    camera = camera_id_for_path(file_path, root=input_path)
    sequence = str(box.get("event_id") or primary.get("event_id") or
                   f"single:{make_dinov3_observation_id(file_path, 0)}")
    data = dict(file_path=str(file_path), bbox=bbox, species=species,
                camera=camera, sequence=sequence, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns)
    model = _key(model_path)
    with (nullcontext(connection) if connection is not None else _write_lock()) as conn:
        old = conn.execute("SELECT * FROM samples WHERE model=? AND observation=?",
                           (model, observation)).fetchone()
        feature = old["feature"] if old is not None else None
        if old is not None and old["applied"]:
            old_data = json.loads(old["data"])
            if (eligible and old_data["species"] == species and old_data["mtime_ns"] == stat.st_mtime_ns
                    and old_data["bbox"] == bbox and old_data["size"] == stat.st_size):
                return
            if feature is None:
                raise ValueError("该检测框属于旧版学习记录，缺少可撤回特征，请重新导出模型后修改训练标签。")
        if old is not None:
            old_data = json.loads(old["data"])
            if eligible and (old_data["mtime_ns"] != stat.st_mtime_ns or old_data["bbox"] != bbox):
                feature = None
            if feature is not None and old_data.get("sample_id"):
                data["sample_id"] = old_data["sample_id"]
                data["encoder_sha256"] = old_data.get("encoder_sha256")
                if old_data.get('encoder_litert_sha256'):
                    data['encoder_litert_sha256']=old_data['encoder_litert_sha256']
            snapshot = dict(old)
            snapshot["feature"] = bytes(snapshot["feature"]).hex() if snapshot["feature"] is not None else None
            previous = json.dumps(snapshot, ensure_ascii=False)
        else:
            previous = None
        conn.execute("INSERT OR IGNORE INTO operations VALUES(?,?,?,?)",
                     (model, observation, operation_id, previous))
        if eligible or feature is not None:
            if not eligible:
                data["species"] = ""
            conn.execute("INSERT OR REPLACE INTO samples(model,observation,data,operation,applied,feature) VALUES(?,?,?,?,0,?)",
                         (model, observation, json.dumps(data, ensure_ascii=False), operation_id, feature))
        else:
            conn.execute("DELETE FROM samples WHERE model=? AND observation=?", (model, observation))


def revert_staged_feedback(model_path, operation_id):
    """Undo pending supplements alongside annotation undo; never pretend to untrain."""
    prepare_shared_learning(model_path)
    with _write_lock() as conn:
        rows = conn.execute("SELECT * FROM operations WHERE model=? AND operation=?",
                            (_key(model_path), operation_id)).fetchall()
        for row in rows:
            current = conn.execute("SELECT * FROM samples WHERE model=? AND observation=?",
                                   (row["model"], row["observation"])).fetchone()
            if current is not None and current["operation"] != operation_id:
                raise ValueError("该检测框已有较新的校验，请先撤销较新的操作。")
            if row["previous"] is not None:
                old = json.loads(row["previous"])
                feature = bytes.fromhex(old["feature"]) if old.get("feature") is not None else None
                conn.execute("INSERT OR REPLACE INTO samples(model,observation,data,operation,applied,feature) VALUES(?,?,?,?,0,?)",
                             (old["model"], old["observation"], old["data"], old["operation"], feature))
            elif current is not None and current["feature"] is not None:
                data = json.loads(current["data"])
                data["species"] = ""
                conn.execute("UPDATE samples SET data=?,applied=0 WHERE model=? AND observation=?",
                             (json.dumps(data, ensure_ascii=False), row["model"], row["observation"]))
            else:
                conn.execute("DELETE FROM samples WHERE model=? AND observation=?",
                             (row["model"], row["observation"]))
        conn.execute("DELETE FROM operations WHERE model=? AND operation=?", (_key(model_path), operation_id))


def reconcile_image_samples(model_path, input_path, file_path, observation_ids, operation_id, *, connection):
    """Withdraw this image's old crops if re-detection removed or replaced them."""
    source = str(Path(file_path).resolve())
    prepare_shared_learning(model_path,connection=connection)
    rows = connection.execute(
        "SELECT observation,data FROM samples WHERE model=? AND json_extract(data,'$.file_path')=?",
        (_key(model_path), source),
    ).fetchall()
    withdrawn = 0
    for row in rows:
        old = json.loads(row["data"])
        if row["observation"] in observation_ids or not old.get("species"):
            continue
        stage_feedback(model_path, input_path, file_path,
                       {"observation_id": row["observation"], "边界框": old["bbox"], "物种": old["species"], "learning_eligible": False},
                       "unverified", operation_id, connection=connection)
        withdrawn += 1
    return withdrawn


def _load_model(path):
    from system.dinov3.runtime import read_manifest, resolve_manifest_asset
    from system.dinov3.encoder import sha256
    from system.dinov3.model import DinoV3RidgeWithinModel

    manifest = Path(path).expanduser().resolve()
    payload = read_manifest(manifest)
    manifest,payload=prepare_shared_learning(manifest,payload)
    checkpoint = resolve_manifest_asset(manifest, payload["checkpoint"])
    if sha256(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError("模型校验失败，未更新。")
    model = DinoV3RidgeWithinModel.load(checkpoint)
    if model.inference_recipe != payload.get("inference_recipe", "production_dual"):
        raise ValueError("模型推理配方与清单不一致，未更新。")
    if model.dimension != 768 or model.provenance.get("encoder_sha256") != payload["encoder_sha256"]:
        raise ValueError("模型与编码器来源不一致，未更新。")
    return manifest, payload, model


def _read_calibration(path, digest):
    with np.load(path, allow_pickle=False) as archive:
        if not _REQUIRED <= set(archive.files):
            raise ValueError("校准数据缺少 features/labels/cameras/sequences。")
        if str(archive.get("encoder_sha256", "")) != digest:
            raise ValueError("校准数据缺少编码器 SHA-256 或与当前模型不一致，请重新导出校准数据。")
        data = {k: archive[k].copy() for k in _REQUIRED | _OPTIONAL if k in archive}
    features = data["features"]
    if features.ndim != 2 or features.shape[1] != 768 or not len(features) or not np.isfinite(features).all():
        raise ValueError("校准特征必须是非空的有限 [N,768] 数组。")
    for key, value in data.items():
        if key in {"features", "adapted_features"}:
            if value.shape != features.shape or not np.isfinite(value).all():
                raise ValueError("校准特征维度不一致。")
        elif value.shape != (len(features),) or value.dtype.kind not in "US":
            raise ValueError(f"校准字段 {key} 必须为与特征对齐的字符串数组。")
    return data


def _calibration(manifest, payload):
    from system.dinov3.runtime import resolve_manifest_asset
    from system.dinov3.encoder import sha256

    if not payload.get("calibration") or not payload.get("calibration_sha256"):
        raise ValueError("当前模型未附带完整校准数据，请先导入与该编码器匹配的校准 NPZ。")
    path = resolve_manifest_asset(manifest, payload["calibration"])
    if sha256(path) != payload["calibration_sha256"]:
        raise ValueError("校准数据 SHA-256 不匹配，未更新模型。")
    return _read_calibration(path, payload["encoder_sha256"])


def _recalibrate(model, data):
    from system.dinov3.context import FastCausalHistoryIndex, adapt_with_history

    adapted = data.get("adapted_features")
    if adapted is None and "datetimes" in data:
        count = len(data["features"])
        sites = data.get("sites", data["cameras"])
        contents = data.get("contents", np.asarray([f"cal:{i}" for i in range(count)]))
        records = [dict(camera_component=str(sites[i]), seq_id=str(data["sequences"][i]),
                        datetime=str(data["datetimes"][i]), content_hash=str(contents[i])) for i in range(count)]
        adapted = adapt_with_history(data["features"], FastCausalHistoryIndex(records).build())
    model.calibrate(data["features"], data["labels"], data["cameras"], data["sequences"],
                    sites=data.get("sites"), contents=data.get("contents"), adapted_features=adapted)


def learning_status(model_path):
    from system.dinov3.runtime import read_manifest

    manifest = Path(model_path).expanduser().resolve()
    payload = read_manifest(manifest)
    manifest,payload=prepare_shared_learning(manifest,payload)
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT data FROM samples WHERE model=? AND applied=0", (_key(manifest),)).fetchall()
        run = conn.execute("SELECT * FROM learning_runs WHERE model=?", (_key(manifest),)).fetchone()
    counts = Counter(json.loads(row["data"])["species"] for row in rows if json.loads(row["data"])["species"])
    learned = {}
    classes = []
    rejection = {}
    from system.dinov3.runtime import resolve_manifest_asset
    with zipfile.ZipFile(resolve_manifest_asset(manifest, payload["checkpoint"])) as archive:
        info = json.loads(archive.read("manifest.json"))
        learned = info.get("metadata", {}).get("human_learning_samples", {})
        classes = info.get("classes", [])
        if 'calibration.json' in archive.namelist():
            calibration_info = json.loads(archive.read('calibration.json'))
            experiment = info.get('inference_recipe') == 'experiment_full'
            rejection = dict(box_threshold=None if experiment else calibration_info.get('box_threshold', calibration_info.get('threshold')),
                             sequence_threshold=calibration_info.get('threshold'),
                             gate_mode='sequence' if experiment else 'box_and_sequence',
                             inference_recipe=info.get('inference_recipe', 'production_dual'),
                             target_unknown_recall=calibration_info.get('target'), direction='higher_is_known')
    try:
        if not payload.get("calibration"):
            raise ValueError("当前模型缺少完整校准数据，请在设置中导入后自动继续学习。")
        calibration = resolve_manifest_asset(manifest, payload.get("calibration"))
        stat = calibration.stat()
        stamp = (str(calibration), stat.st_mtime_ns, stat.st_size, payload.get("calibration_sha256"))
        if stamp not in _CALIBRATION_CHECKS:
            _calibration(manifest, payload)
            _CALIBRATION_CHECKS.add(stamp)
        available, message = True, "人工确认或修改物种后自动学习。"
    except (OSError, ValueError, KeyError) as exc:
        available, message = False, str(exc)
    state = run["state"] if run is not None else "idle"
    if rows and state in {"ready", "idle"}:
        state = "queued"
    if not available:
        state = "blocked"
    elif run is not None:
        message = run["message"]
    if state == "queued":
        message = "人工校验已保存，等待后台自动学习。"
    return dict(pending_count=len(rows), species_counts=dict(counts),
                calibration_available=available, message=message, state=state,
                learned_count=len(learned), learned_species_counts=dict(Counter(learned.values())),
                class_count=len(classes), last_update=run["last_update"] if run else None,
                last_error=run["last_error"] if run else None, rejection=rejection)


def discard_pending(model_path):
    """Clear staged supports without changing ecological labels or learned data."""
    prepare_shared_learning(model_path)
    if _key(model_path).startswith('family:'):
        key=_key(model_path)
        with _write_lock() as conn:
            pending=conn.execute('SELECT * FROM samples WHERE model=? AND applied=0',(key,)).fetchall()
            for row in pending:
                saved=conn.execute('SELECT * FROM shared_learning_committed WHERE model=? AND observation=?',
                                   (key,row['observation'])).fetchone()
                if saved is None:
                    conn.execute('DELETE FROM operations WHERE model=? AND observation=?',(key,row['observation']))
                    conn.execute('DELETE FROM samples WHERE model=? AND observation=?',(key,row['observation']))
                    continue
                keep=set()
                operation_id=saved['operation']
                while operation_id and operation_id not in keep:
                    keep.add(operation_id)
                    operation=conn.execute('SELECT previous FROM operations WHERE model=? AND observation=? AND operation=?',
                                           (key,row['observation'],operation_id)).fetchone()
                    operation_id=json.loads(operation['previous'])['operation'] if operation and operation['previous'] else None
                for operation in conn.execute('SELECT operation FROM operations WHERE model=? AND observation=?',(key,row['observation'])).fetchall():
                    if operation['operation'] not in keep:
                        conn.execute('DELETE FROM operations WHERE model=? AND observation=? AND operation=?',
                                     (key,row['observation'],operation['operation']))
                conn.execute('INSERT OR IGNORE INTO operations VALUES(?,?,?,NULL)',(key,row['observation'],saved['operation']))
                conn.execute('INSERT OR REPLACE INTO samples(model,observation,data,operation,applied,feature) VALUES(?,?,?,?,1,?)',
                             (key,row['observation'],saved['data'],saved['operation'],saved['feature']))
        return learning_status(model_path)
    with _write_lock() as conn:
        key = _key(model_path)
        conn.execute("DELETE FROM operations WHERE model=? AND observation IN "
                     "(SELECT observation FROM samples WHERE model=? AND applied=0)", (key, key))
        conn.execute("DELETE FROM samples WHERE model=? AND applied=0", (key,))
    return learning_status(model_path)


def import_calibration(model_path, calibration_path):
    from system.dinov3.deployment import publish_model

    manifest, payload, model = _load_model(model_path)
    data = _read_calibration(calibration_path, payload["encoder_sha256"])
    _recalibrate(model, data)
    with _write_lock():
        if not _same_publication(json.loads(manifest.read_bytes()),payload):
            raise ValueError('校准期间学习成果已更新，请重试导入；当前模型已保留。')
        publish_model(model, manifest, existing_manifest=payload,
                      display_name=model_display_name(payload), calibration_data=data)
    schedule_learning(model_path)
    return learning_status(manifest)


def _load_crop(data):
    """Use the same 10% expansion and square RGB114 padding as inference."""
    import cv2

    path = Path(data["file_path"])
    stat = path.stat()
    if stat.st_size != data["size"] or stat.st_mtime_ns != data["mtime_ns"]:
        raise ValueError(f"补图源文件已改变，请重新人工确认：{path.name}")
    image_bytes = path.read_bytes()
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"无法读取补图照片：{path.name}")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, data["bbox"])
    pad_w, pad_h = int((x2-x1)*0.1), int((y2-y1)*0.1)
    left, top = max(0, x1-pad_w), max(0, y1-pad_h)
    right, bottom = min(width, x2+pad_w), min(height, y2+pad_h)
    if right <= left or bottom <= top:
        raise ValueError(f"补图检测框超出照片范围：{path.name}")
    crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    ch, cw = crop.shape[:2]
    size = max(ch, cw)
    pt, pl = (size-ch)//2, (size-cw)//2
    crop = cv2.copyMakeBorder(crop, pt, size-ch-pt, pl, size-cw-pl,
                              cv2.BORDER_CONSTANT, value=(114,114,114))
    sample_id = hashlib.sha256(str(crop.shape).encode()+crop.tobytes()).hexdigest()
    return crop, sample_id, hashlib.sha256(image_bytes).hexdigest()


def update_model(model_path):
    from system.dinov3.deployment import publish_model
    from system.dinov3.encoder import sha256
    from system.dinov3.runtime import load_manifest_encoder, resolve_manifest_asset
    from system.dinov3.shared_learning import encoder_compatible
    from threadpoolctl import threadpool_limits

    from system.dinov3.model import DinoV3RidgeWithinModel
    requested_model_path=str(Path(model_path).expanduser().resolve())

    # Heavy encoding and calibration happen outside the annotation transaction.
    # Before publishing, compare the snapshot again so a correction can never
    # race an older training label into the active model.
    with _UPDATE_LOCK, threadpool_limits(limits=4):
        manifest, payload, current_model = _load_model(model_path)
        manifest_snapshot = manifest.read_bytes()
        if not _same_publication(json.loads(manifest_snapshot),payload):
            return dict(updated_count=0, superseded=True)
        with _write_lock() as conn:
            if payload.get('learning_family_id'):
                conn.execute('INSERT OR IGNORE INTO samples(model,observation,data,operation,applied,feature) '
                    'SELECT model,observation,data,operation,1,feature FROM shared_learning_committed WHERE model=?',(_key(manifest),))
            rows = conn.execute("SELECT * FROM samples WHERE model=? ORDER BY observation", (_key(manifest),)).fetchall()
        pending = [row for row in rows if not row["applied"]]
        if not pending:
            return dict(updated_count=0, species_counts={}, model_path=str(manifest), message="没有待补图样本。")
        previous_samples = dict(current_model.metadata.get("human_learning_samples", {}))
        cal = _calibration(manifest, payload)
        base = current_model.metadata.get("human_learning_base")
        if base is None:
            model = current_model
            base = {"checkpoint": payload["checkpoint"], "sha256": payload["checkpoint_sha256"]}
        else:
            base_path = resolve_manifest_asset(manifest, base["checkpoint"])
            if sha256(base_path) != base["sha256"]:
                raise ValueError("学习基础模型校验失败。")
            model = DinoV3RidgeWithinModel.load(base_path)
        legacy_applied = dict(model.metadata.get("human_learning_samples", {}))
        if payload.get('learning_family_id'):
            recorded={json.loads(row['data']).get('sample_id') for row in rows}
            with closing(_connect()) as conn:
                recorded.update(json.loads(row['data']).get('sample_id') for row in conn.execute(
                    'SELECT data FROM shared_learning_committed WHERE model=?',(_key(manifest),)))
            if set(previous_samples)-recorded-set(legacy_applied):
                raise ValueError('共享档案缺少已学样本记录，保留当前模型并停止更新，避免丢失学习成果。')
        unique = {}
        vectors = {}
        calibration_contents = {str(c).split(":")[0] for key in ("contents", "source_contents") for c in cal.get(key, [])}
        prepared = {}
        for row in rows:
            data = json.loads(row["data"])
            prepared[row["observation"]] = data
            if not data["species"]:
                continue
            reused=row['feature'] is not None and encoder_compatible(data,payload)
            if reused:
                sample_id = data["sample_id"]
                feature = np.frombuffer(row["feature"], dtype=np.float32).copy()
                if feature.shape != (768,) or not np.isfinite(feature).all():
                    raise ValueError("已保存的学习特征无效。")
                vectors[sample_id] = feature
            else:
                crop, sample_id, file_digest = _load_crop(data)
                del crop
                data["sample_id"] = sample_id
                if calibration_contents & {file_digest, sample_id}:
                    raise ValueError("补图与校准集内容重叠，请使用独立的人工确认训练样本。")
            if not reused:
                data["encoder_sha256"] = payload["encoder_sha256"]
                if "encoder_litert_sha256" in payload:
                    data["encoder_litert_sha256"] = payload["encoder_litert_sha256"]
                else:
                    data.pop("encoder_litert_sha256", None)
            if sample_id in legacy_applied:
                if legacy_applied[sample_id] != data["species"]:
                    raise ValueError("旧版学习记录存在标签冲突，请重新导出模型。")
                continue
            if sample_id in unique and unique[sample_id]["species"] != data["species"]:
                raise ValueError("待补图中存在相同内容、不同物种的冲突，请先修正标注。")
            unique.setdefault(sample_id, data)
        missing = [sid for sid in unique if sid not in vectors]
        if missing:
            encoder = load_manifest_encoder(manifest, payload)
            try:
                for begin in range(0, len(missing), 32):
                    ids = missing[begin:begin+32]
                    features = encoder.encode([_load_crop(unique[sid])[0] for sid in ids], array_color="rgb")
                    vectors.update(zip(ids, features))
            finally:
                del encoder
        if unique:
            values = list(unique.values())
            model.partial_fit(np.stack([vectors[sid] for sid in unique]), [d["species"] for d in values],
                              [d["camera"] for d in values], [d["sequence"] for d in values])
        new_samples = {**legacy_applied, **{sid: d["species"] for sid, d in unique.items()}}
        changed = new_samples != previous_samples
        if changed:
            _recalibrate(model, cal)
        model.metadata["human_learning_base"] = base
        model.metadata["human_learning_samples"] = new_samples
        model.metadata["human_learning_operations"] = sorted({row["operation"] for row in rows})
        @contextmanager
        def commit_guard():
            with _write_lock() as conn:
                latest = conn.execute("SELECT * FROM samples WHERE model=? ORDER BY observation", (_key(manifest),)).fetchall()
                if (not _same_publication(json.loads(manifest.read_bytes()),json.loads(manifest_snapshot))
                        or [tuple(r) for r in latest] != [tuple(r) for r in rows]):
                    raise _Superseded("有新的人工校验，合并后重新学习。")
                yield
                for row in rows:
                    data = prepared[row["observation"]]
                    if data["species"]:
                        feature = vectors.get(data["sample_id"])
                        encoded = feature.astype(np.float32).tobytes() if feature is not None else None
                    else:
                        encoded = row["feature"]
                    conn.execute("UPDATE samples SET applied=1,data=?,feature=? WHERE model=? AND observation=?",
                                 (json.dumps(data, ensure_ascii=False), encoded, row["model"], row["observation"]))
                    if payload.get('learning_family_id'):
                        conn.execute('INSERT OR REPLACE INTO shared_learning_committed VALUES(?,?,?,?,?)',
                                     (row['model'],row['observation'],json.dumps(data,ensure_ascii=False),row['operation'],encoded))
                conn.execute("INSERT OR REPLACE INTO learning_runs VALUES(?,?,?,?,NULL)",
                             (_key(manifest), "ready", "人工确认样本已自动学习，下次识别生效。", datetime.now(timezone.utc).isoformat()))
        try:
            if changed:
                publish_model(model, manifest, existing_manifest=payload,
                              display_name=model_display_name(payload), calibration_data=cal,
                              publication_guard=commit_guard)
            else:
                with commit_guard():
                    pass
        except _Superseded:
            return dict(updated_count=0, superseded=True, message="有新的人工校验，合并后重新学习。")
    counts = Counter(json.loads(row["data"])["species"] for row in pending if json.loads(row["data"])["species"])
    updated_count = sum(previous_samples.get(sid) != label for sid, label in new_samples.items()) + len(set(previous_samples)-set(new_samples))
    return dict(updated_count=updated_count, species_counts=dict(counts), model_path=requested_model_path,
                message="已自动学习并重新校准，下次识别生效。")


def _run_state(model, state, message, error=None):
    with _write_lock() as conn:
        conn.execute("INSERT INTO learning_runs(model,state,message,last_error) VALUES(?,?,?,?) "
                     "ON CONFLICT(model) DO UPDATE SET state=excluded.state,message=excluded.message,last_error=excluded.last_error",
                     (_key(model), state, message, error))


def _learning_worker(model, root, control):
    _CONTEXT.root = root
    while not control["stop"]:
        control["event"].wait()
        if control["stop"]:
            break
        control["event"].clear()
        # Coalesce rapid confirmations without holding up annotation writes.
        while control["event"].wait(1.0):
            control["event"].clear()
            if control["stop"]:
                return
        model=control.get('model_path',model)
        try:
            _run_state(model, "running", "正在自动学习人工确认样本…")
            result = update_model(model)
            if result.get("superseded"):
                control["event"].set()
                continue
            _run_state(model, "ready", "人工确认样本已自动学习，下次识别生效。")
        except Exception as exc:
            _run_state(model, "blocked" if isinstance(exc, (ValueError, FileNotFoundError)) else "failed",
                       f"自动学习未完成：{exc}", str(exc))
        if not control["event"].is_set():
            control["idle"].set()


def schedule_learning(model_path):
    """Notify one background worker after annotation persistence has committed."""
    if not model_path:
        return
    model = str(Path(model_path).expanduser().resolve())
    root = _state_root()
    key = (_key(model), str(root))
    with _WORKERS_LOCK:
        control = _WORKERS.get(key)
        if control is None or control["stop"] or not control["thread"].is_alive():
            control = {"event": threading.Event(), "idle": threading.Event(), "stop": False}
            thread = threading.Thread(target=_learning_worker, args=(model, root, control),
                                      name="neri-model-learning", daemon=True)
            control["thread"] = thread
            _WORKERS[key] = control
            thread.start()
        control["idle"].clear()
        control['model_path']=model
        control["event"].set()


def resume_learning():
    """Resume committed pending confirmations after a backend restart."""
    with closing(_connect()) as conn:
        paths = [row[0] for row in conn.execute("SELECT DISTINCT model FROM samples WHERE applied=0")]
        migrated={row[0] for row in conn.execute('SELECT alias FROM shared_learning_aliases')}
    for path in paths:
        if path in migrated:
            continue
        if path.startswith('family:'):
            import importlib.util
            from system.dinov3.shared_learning import family_manifest,atomic_manifest
            path=family_manifest(_state_root(),path.removeprefix('family:'))
            if path.is_file():
                with _write_lock():
                    payload=json.loads(path.read_text('utf8'))
                    if payload.get('encoder_backend')=='pytorch' and importlib.util.find_spec('torch') is None:
                        cpu=next((value for value in payload.get('learning_encoder_variants',{}).values()
                                  if value.get('encoder_backend')=='litert'),None)
                        if cpu:
                            payload.update(cpu)
                            payload.pop('encoder_code',None)
                            atomic_manifest(path,payload)
        if Path(path).is_file():
            schedule_learning(path)


def stop_learning():
    with _WORKERS_LOCK:
        for control in _WORKERS.values():
            control["stop"] = True
            control["event"].set()
