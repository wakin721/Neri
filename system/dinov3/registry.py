"""Compatibility facade for the DINOv3 species registry.

The implementation lives in ``registry_impl`` so this facade can keep the
public module stable while applying small behavioural fixes.
"""
from __future__ import annotations

from . import registry_impl as _impl
from .registry_examples import (
    persist_registry_event_example,
    registry_event_example_path,
)

# Preserve the historical module surface, including private helpers used by
# nearby DINOv3 modules/tests, then override only SpeciesRegistry.
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


class SpeciesRegistry(_impl.SpeciesRegistry):
    def set_identity(self, entry_id, *, common_name, scientific_name=""):
        """Save a Candidate identity without implicitly merging or deleting it."""
        # Empty registries historically had no migration marker. Mark the first
        # modern identity write as post-migration only when no other named entry
        # already exists. A real legacy non-empty database is migrated on open;
        # the extra guard also lets migration tests explicitly construct old
        # duplicate rows through registry_impl without being reclassified here.
        marker = self._conn.execute(
            "SELECT value FROM metadata WHERE key='migration_named_candidate_merge_v1'"
        ).fetchone()
        if marker is None:
            existing_named = self._conn.execute(
                """
                SELECT 1 FROM registrations
                WHERE id<>? AND TRIM(common_name)<>''
                LIMIT 1
                """,
                (int(entry_id),),
            ).fetchone()
            if existing_named is None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                    ("migration_named_candidate_merge_v1", "1"),
                )
                self._conn.commit()
        return super().set_identity(
            entry_id,
            common_name=common_name,
            scientific_name=scientific_name,
        )

    def record_observation(
        self,
        entry_id,
        embedding,
        *,
        camera_id,
        captured_at,
        source_path,
        bbox=None,
        frame_index=None,
        timestamp_seconds=None,
    ):
        event_key, *_ = self._event_key(
            entry_id,
            camera_id,
            captured_at,
            source_path,
        )
        updated = super().record_observation(
            entry_id,
            embedding,
            camera_id=camera_id,
            captured_at=captured_at,
            source_path=source_path,
            bbox=bbox,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )
        if bbox is not None:
            row = self._conn.execute(
                "SELECT id FROM events WHERE registration_id=? AND event_key=?",
                (entry_id, event_key),
            ).fetchone()
            if row is not None:
                persist_registry_event_example(
                    self.path,
                    int(row["id"]),
                    source_path=source_path,
                    bbox=bbox,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                )
        return updated

    def merge_candidate_into(self, source_id: int, target_id: int):
        """Merge one explicitly named Candidate into a same-name Registry entry."""
        source_id = int(source_id)
        target_id = int(target_id)
        if source_id == target_id:
            raise RegistrationConditionError("Source and target Registry entries must differ")

        source = self.get(source_id)
        target = self.get(target_id)
        if source.status != "candidate":
            raise RegistrationConditionError("Only Candidate species can be merged")
        if target.status not in {"candidate", "provisional", "confirmed", "mature"}:
            raise RegistrationConditionError("Target Registry species cannot accept Candidate evidence")

        source_name = source.common_name.strip()
        target_name = target.common_name.strip()
        if not source_name or not target_name or source_name.casefold() != target_name.casefold():
            raise RegistrationConditionError("Registry species names must match before merging")

        source_scientific = source.scientific_name.strip()
        target_scientific = target.scientific_name.strip()
        if (
            source_scientific
            and target_scientific
            and source_scientific.casefold() != target_scientific.casefold()
        ):
            raise RegistrationConditionError("Scientific names conflict for same-name Registry entries")
        scientific_name = target_scientific or source_scientific
        cluster_purity = min(source.cluster_purity, target.cluster_purity)

        from .feedback import feedback_path_for_registry, redirect_registry_assignments_file

        redirect_registry_assignments_file(
            feedback_path_for_registry(self.path),
            {source_id: target_id},
        )

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                "DELETE FROM prototypes WHERE registration_id IN (?,?)",
                (source_id, target_id),
            )
            self._conn.execute(
                "UPDATE events SET registration_id=? WHERE registration_id=?",
                (target_id, source_id),
            )
            self._conn.execute(
                """
                UPDATE registrations
                SET scientific_name=?,cluster_purity=?,updated_at=?
                WHERE id=?
                """,
                (scientific_name, cluster_purity, _now(), target_id),
            )
            self._conn.execute("DELETE FROM registrations WHERE id=?", (source_id,))
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        self._refresh(target_id)
        return self.get(target_id)

    def list_events(self, entry_id):
        events = super().list_events(entry_id)
        for event in events:
            if registry_event_example_path(self.path, int(event["id"])).is_file():
                event["has_example"] = True
        return events

    def list(self, *, status=None):
        """List named Registry entries first, then unnamed entries, by evidence count."""
        entries = super().list(status=status)
        return sorted(
            entries,
            key=lambda entry: (
                0 if entry.common_name.strip() else 1,
                -entry.event_count,
                entry.candidate_number,
                entry.id,
            ),
        )
