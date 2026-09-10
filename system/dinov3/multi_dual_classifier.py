"""Manifest-enabled cosine + L2 rejection for DINOv3 Multi-prototype heads."""
from __future__ import annotations

from .classifier import DinoV3Classifier
from .rejection import MultiDualRejectionConfig


class MultiDualDinoV3Classifier(DinoV3Classifier):
    """Multi-prototype classifier using cosine AND squared-distance rejection."""

    def __init__(self, checkpoint, *, rejection, encoder=None, feedback=None, registry=None):
        super().__init__(checkpoint, encoder=encoder, feedback=feedback, registry=registry)
        if not isinstance(rejection, MultiDualRejectionConfig):
            raise TypeError("rejection must be MultiDualRejectionConfig")
        self.rejection = rejection

    @property
    def rejection_metadata(self):
        return self.rejection.as_dict()

    def _classify_multi_prototype(self, array):
        bank = self._effective_bank()
        if not bank.formal:
            raise ValueError("Formal Multi-prototype bank cannot be empty")
        centered_rows = array - self._feature_center[None, :]
        results = []
        for row_index, centered in enumerate(centered_rows):
            formal_index, formal, formal_distance, formal_score = self._winning_record(centered, bank.formal)
            candidates = self._species_candidates(centered, bank.formal)
            if self.rejection.accepts(cosine_score=formal_score, squared_distance=formal_distance):
                from .classifier import DinoV3Prediction
                results.append(DinoV3Prediction(
                    species=formal.species,
                    accepted=True,
                    best_known_species=formal.species,
                    head_species=formal.species,
                    prototype_species=formal.species,
                    head_prototype_consistent=True,
                    known_score=formal_score,
                    threshold=self.rejection.cosine_threshold,
                    candidates=candidates,
                    embedding=array[row_index].copy(),
                    source=formal.source,
                    registry_id=formal.registry_id,
                    registration_status=formal.registration_status,
                    nearest_prototype_index=formal_index,
                    squared_distance=formal_distance,
                ))
            else:
                from .classifier import DinoV3Prediction
                results.append(DinoV3Prediction(
                    species="Unknown",
                    accepted=False,
                    best_known_species=formal.species,
                    head_species=formal.species,
                    prototype_species=formal.species,
                    head_prototype_consistent=True,
                    known_score=formal_score,
                    threshold=self.rejection.cosine_threshold,
                    candidates=candidates,
                    embedding=array[row_index].copy(),
                    source=formal.source,
                    registry_id=formal.registry_id,
                    registration_status=formal.registration_status,
                    nearest_prototype_index=formal_index,
                    squared_distance=formal_distance,
                ))
        return results
