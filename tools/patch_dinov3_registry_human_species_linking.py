from __future__ import annotations

from pathlib import Path


path = Path("system/dinov3/registry.py")
text = path.read_text(encoding="utf-8")
old = '''        candidate_ids = []
        if preferred_entry_id is not None:
            candidate_ids.append(int(preferred_entry_id))
        matched = self.match(vector)
        if matched is not None:
            candidate_ids.append(int(matched["id"]))
        for candidate_id in dict.fromkeys(candidate_ids):
            try:
                detail = self.get(candidate_id)
            except RegistryEntryNotFound:
                continue
            if detail.status == "candidate" and (
                not detail.common_name or detail.common_name == common
            ):
                entry_id = candidate_id
                break
'''
new = '''        # An explicit human species label is stronger evidence than automatic
        # embedding similarity. Reuse an already named Registry species first,
        # including provisional/confirmed/mature entries, so diverse views of
        # the same manually confirmed species do not create duplicate candidates.
        candidate_ids = [
            int(row["id"])
            for row in self._conn.execute(
                """
                SELECT id FROM registrations
                WHERE common_name=?
                ORDER BY CASE status
                    WHEN 'mature' THEN 0
                    WHEN 'confirmed' THEN 1
                    WHEN 'provisional' THEN 2
                    ELSE 3
                END, candidate_number
                """,
                (common,),
            ).fetchall()
        ]
        if preferred_entry_id is not None:
            candidate_ids.append(int(preferred_entry_id))
        matched = self.match(vector)
        if matched is not None:
            candidate_ids.append(int(matched["id"]))
        for candidate_id in dict.fromkeys(candidate_ids):
            try:
                detail = self.get(candidate_id)
            except RegistryEntryNotFound:
                continue
            if detail.common_name == common:
                entry_id = candidate_id
                break
            if detail.status == "candidate" and not detail.common_name:
                entry_id = candidate_id
                break
'''
if new in text:
    print("same-name Registry linking patch already applied")
elif old not in text:
    raise RuntimeError("record_human_species anchor not found")
else:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("applied same-name Registry linking patch")
