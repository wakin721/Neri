"""Compatibility facade for the DINOv3 species registry.

The implementation lives in ``registry_impl`` so this facade can keep the
public module stable while applying small behavioural fixes.
"""
from __future__ import annotations

from . import registry_impl as _impl

# Preserve the historical module surface, including private helpers used by
# nearby DINOv3 modules/tests, then override only SpeciesRegistry.
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


class SpeciesRegistry(_impl.SpeciesRegistry):
    def set_identity(self, entry_id, *, common_name, scientific_name=""):
        """Save an identity and collapse only duplicate non-registerable Candidates."""
        updated = super().set_identity(
            entry_id,
            common_name=common_name,
            scientific_name=scientific_name,
        )
        # A registration-ready Candidate must remain addressable long enough to
        # be promoted to provisional. Legacy/low-evidence named Candidates can
        # still be collapsed immediately so duplicate Registry slots disappear.
        if updated.can_register:
            return updated
        merge = self.merge_duplicate_named_candidates()
        redirects = merge.get("redirects", {})
        survivor_id = int(redirects.get(int(entry_id), updated.id))
        return self.get(survivor_id)

    def list(self, *, status=None):
        """Sort only unnamed Candidate slots by event count, descending."""
        entries = super().list(status=status)
        slots = [
            index
            for index, entry in enumerate(entries)
            if entry.status == "candidate" and not entry.common_name.strip()
        ]
        if len(slots) < 2:
            return entries
        ordered = sorted(
            (entries[index] for index in slots),
            key=lambda entry: (-entry.event_count, entry.candidate_number),
        )
        result = list(entries)
        for index, entry in zip(slots, ordered, strict=True):
            result[index] = entry
        return result
