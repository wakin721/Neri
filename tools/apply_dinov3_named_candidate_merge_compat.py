from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor in {path}, found {count}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "system/dinov3/feedback.py",
    '''    def registry_assignments(self, operation_id: str) -> list[dict[str, object]]:\n        rows = self._conn.execute(\n            """\n            SELECT registration_id,previous_common_name,previous_scientific_name,\n                   assigned_common_name,identity_restore_allowed\n            FROM feedback_registry_assignments\n            WHERE operation_id=? ORDER BY registration_id\n            """,\n            (operation_id,),\n        ).fetchall()\n        return [\n            {\n                "registration_id": int(row["registration_id"]),\n                "previous_common_name": str(row["previous_common_name"]),\n                "previous_scientific_name": str(row["previous_scientific_name"]),\n                "assigned_common_name": str(row["assigned_common_name"]),\n                "identity_restore_allowed": bool(row["identity_restore_allowed"]),\n            }\n            for row in rows\n        ]\n''',
    '''    def registry_assignments(self, operation_id: str) -> list[dict[str, object]]:\n        """Return the stable public assignment shape used by existing callers."""\n        rows = self._conn.execute(\n            """\n            SELECT registration_id,previous_common_name,previous_scientific_name,\n                   assigned_common_name\n            FROM feedback_registry_assignments\n            WHERE operation_id=? ORDER BY registration_id\n            """,\n            (operation_id,),\n        ).fetchall()\n        return [\n            {\n                "registration_id": int(row["registration_id"]),\n                "previous_common_name": str(row["previous_common_name"]),\n                "previous_scientific_name": str(row["previous_scientific_name"]),\n                "assigned_common_name": str(row["assigned_common_name"]),\n            }\n            for row in rows\n        ]\n\n    def registry_assignments_for_restore(\n        self, operation_id: str\n    ) -> list[dict[str, object]]:\n        """Return assignment rows plus the internal identity-restore policy."""\n        rows = self._conn.execute(\n            """\n            SELECT registration_id,previous_common_name,previous_scientific_name,\n                   assigned_common_name,identity_restore_allowed\n            FROM feedback_registry_assignments\n            WHERE operation_id=? ORDER BY registration_id\n            """,\n            (operation_id,),\n        ).fetchall()\n        return [\n            {\n                "registration_id": int(row["registration_id"]),\n                "previous_common_name": str(row["previous_common_name"]),\n                "previous_scientific_name": str(row["previous_scientific_name"]),\n                "assigned_common_name": str(row["assigned_common_name"]),\n                "identity_restore_allowed": bool(row["identity_restore_allowed"]),\n            }\n            for row in rows\n        ]\n''',
)

replace_once(
    "system/backend/dinov3_feedback_service.py",
    '''        registry_assignments = feedback.registry_assignments(\n            request.feedback_operation_id\n        )\n''',
    '''        registry_assignments = feedback.registry_assignments_for_restore(\n            request.feedback_operation_id\n        )\n''',
)
