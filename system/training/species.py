"""Resolve user-facing Chinese species names to YOLO-safe training class names."""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

from system.utils import resource_path


class SpeciesNameResolver:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else Path(
            resource_path("res/species_database.db")
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(value or "").strip())
        normalized = re.sub(r"\s+", "_", normalized)
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_.-")
        return normalized

    @staticmethod
    def _pinyin(value: str) -> str:
        try:
            from pypinyin import Style, lazy_pinyin

            syllables = lazy_pinyin(
                value,
                style=Style.NORMAL,
                errors=lambda chars: list(chars),
            )
        except ImportError as error:
            raise RuntimeError("pypinyin_dependency_missing") from error
        parts = []
        for syllable in syllables:
            safe = SpeciesNameResolver._safe_name(syllable).lower()
            if safe:
                parts.append(safe)
        return "_".join(parts)

    def resolve(self, chinese_name: str) -> str:
        name = str(chinese_name or "").strip()
        if name == "空照片":
            return "background"
        if not name:
            return "unknown"

        scientific = ""
        if self.db_path.exists():
            try:
                with sqlite3.connect(str(self.db_path)) as db:
                    row = db.execute(
                        "SELECT 学名 FROM species WHERE 中文名=? LIMIT 1",
                        (name,),
                    ).fetchone()
                if row:
                    scientific = str(row[0] or "").strip()
            except (sqlite3.Error, OSError):
                scientific = ""

        safe_scientific = self._safe_name(scientific)
        if safe_scientific:
            return safe_scientific

        fallback = self._pinyin(name)
        return fallback or "unknown"

    def resolve_many(self, names: list[str]) -> dict[str, str]:
        return {name: self.resolve(name) for name in names}
