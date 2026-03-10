"""
SQLite 检测数据管理器
用于替代 temp_photo_dir 下散落的 JSON 文件 + validation.json，
实现批量读取、毫秒级加载。
"""
import sqlite3
import json
import os
import logging

logger = logging.getLogger(__name__)

DB_FILENAME = "detections.db"


def get_db_path(temp_photo_dir: str) -> str:
    return os.path.join(temp_photo_dir, DB_FILENAME)


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")   # 允许并发读写
    conn.execute("PRAGMA synchronous=NORMAL") # 平衡性能与安全
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str):
    """初始化表结构（幂等）"""
    with _get_conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS detections (
                base_name      TEXT PRIMARY KEY,
                image_filename TEXT NOT NULL,
                detection_json TEXT NOT NULL,
                updated_at     INTEGER DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS validation (
                image_filename TEXT PRIMARY KEY,
                is_validated   INTEGER NOT NULL   -- 1=True / 0=False
            );
            CREATE INDEX IF NOT EXISTS idx_det_imgfile
                ON detections(image_filename);
        """)


# ──────────────────────────────────────────────
# detections 表操作
# ──────────────────────────────────────────────

def upsert_detection(db_path: str, base_name: str,
                     image_filename: str, detection_data: dict):
    """写入/更新单条检测记录"""
    with _get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO detections (base_name, image_filename, detection_json, updated_at)
            VALUES (?, ?, ?, strftime('%s','now'))
            ON CONFLICT(base_name) DO UPDATE SET
                image_filename = excluded.image_filename,
                detection_json = excluded.detection_json,
                updated_at     = excluded.updated_at
        """, (base_name, image_filename,
              json.dumps(detection_data, ensure_ascii=False)))


def get_detection(db_path: str, base_name: str) -> dict | None:
    """读取单条检测记录"""
    with _get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT detection_json FROM detections WHERE base_name=?",
            (base_name,)
        ).fetchone()
    return json.loads(row["detection_json"]) if row else None


def update_detection(db_path: str, base_name: str, detection_data: dict):
    """更新已有检测记录"""
    with _get_conn(db_path) as conn:
        conn.execute("""
            UPDATE detections
            SET detection_json=?, updated_at=strftime('%s','now')
            WHERE base_name=?
        """, (json.dumps(detection_data, ensure_ascii=False), base_name))


def delete_detection(db_path: str, base_name: str):
    """删除检测记录及对应校验状态"""
    with _get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT image_filename FROM detections WHERE base_name=?",
            (base_name,)
        ).fetchone()
        if row:
            conn.execute("DELETE FROM validation WHERE image_filename=?",
                         (row["image_filename"],))
        conn.execute("DELETE FROM detections WHERE base_name=?", (base_name,))


def get_all_detections_with_validation(db_path: str) -> list[dict]:
    """
    一次性读取全部检测记录 + 校验状态（LEFT JOIN）。
    是 _load_species_data_core 的核心查询，替代 N 次 JSON 读取。
    返回列表，每项包含:
        base_name / image_filename / detection_json(str) / is_validated(int|None)
    """
    with _get_conn(db_path) as conn:
        rows = conn.execute("""
            SELECT d.base_name,
                   d.image_filename,
                   d.detection_json,
                   v.is_validated
            FROM detections d
            LEFT JOIN validation v ON d.image_filename = v.image_filename
        """).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────
# validation 表操作
# ──────────────────────────────────────────────

def upsert_validation(db_path: str, image_filename: str, is_validated: bool):
    with _get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO validation (image_filename, is_validated) VALUES (?,?)
            ON CONFLICT(image_filename) DO UPDATE SET is_validated=excluded.is_validated
        """, (image_filename, 1 if is_validated else 0))


def delete_validation(db_path: str, image_filename: str):
    with _get_conn(db_path) as conn:
        conn.execute("DELETE FROM validation WHERE image_filename=?",
                     (image_filename,))


def delete_validation_bulk(db_path: str, image_filenames: list[str]):
    with _get_conn(db_path) as conn:
        conn.executemany("DELETE FROM validation WHERE image_filename=?",
                         [(f,) for f in image_filenames])


def get_all_validation(db_path: str) -> dict:
    """返回 {image_filename: bool}"""
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT image_filename, is_validated FROM validation"
        ).fetchall()
    return {r["image_filename"]: bool(r["is_validated"]) for r in rows}


# ──────────────────────────────────────────────
# 迁移：将已有 JSON 文件一次性导入数据库
# ──────────────────────────────────────────────

def migrate_from_json(db_path: str, photo_dir: str,
                      image_basename_map: dict) -> int:
    """
    将 temp_photo_dir 下所有散落的 .json 文件（及 validation.json）
    一次性导入 SQLite。已存在的记录跳过，不覆盖。
    返回本次迁移的条数。
    """
    # 读取 validation.json
    validation_data: dict = {}
    val_path = os.path.join(photo_dir, "validation.json")
    if os.path.exists(val_path):
        try:
            with open(val_path, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
        except Exception as e:
            logger.error(f"[迁移] 读取 validation.json 失败: {e}")

    json_files = [
        f for f in os.listdir(photo_dir)
        if f.lower().endswith('.json') and f != 'validation.json'
    ]

    migrated = 0
    with _get_conn(db_path) as conn:
        existing = {
            r[0] for r in
            conn.execute("SELECT base_name FROM detections").fetchall()
        }
        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0]
            if base_name in existing:
                continue
            image_filename = image_basename_map.get(base_name)
            if not image_filename:
                continue
            try:
                with open(os.path.join(photo_dir, json_file),
                          'r', encoding='utf-8') as f:
                    data = json.load(f)
                conn.execute("""
                    INSERT OR IGNORE INTO detections
                        (base_name, image_filename, detection_json)
                    VALUES (?,?,?)
                """, (base_name, image_filename,
                      json.dumps(data, ensure_ascii=False)))
                migrated += 1
            except Exception as e:
                logger.error(f"[迁移] {json_file} 失败: {e}")

        # 迁移 validation 状态
        for filename, value in validation_data.items():
            conn.execute("""
                INSERT OR IGNORE INTO validation (image_filename, is_validated)
                VALUES (?,?)
            """, (filename, 1 if value else 0))

    logger.info(f"[迁移] 共迁移 {migrated} 条检测记录到 SQLite")
    return migrated