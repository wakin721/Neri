"""Manifest-enabled cosine + L2 rejection for DINOv3 Multi-prototype heads."""
from __future__ import annotations

from typing import Any

from .classifier import DinoV3Classifier, DinoV3Prediction, _normalize_rows
from .rejection import MultiDualRejectionConfig


class MultiDualDinoV3Classifier(DinoV3Classifier):
    """Multi-prototype classifier with a jointly calibrated dual rejection gate.

    Closed-set classification uses the same raw prototypes, but Dual mode applies
    the full CL2N transform used by training/ablation: subtract the stored center,
    then L2-normalize. A formal prediction is accepted only when BOTH the cosine
    and squared-distance gates pass. Provisional registry/feedback prototypes use
    the same gate before being surfaced as assistive matches.
    """

    def __init__(
        self,
        checkpoint,
        *,
        rejection: MultiDualRejectionConfig,
        encoder=None,
        feedback=None,
        registry=None,
    ) -> None:
        super().__init__(
            checkpoint,
            encoder=encoder,
            feedback=feedback,
            registry=registry,
        )
        if not isinstance(rejection, MultiDualRejectionConfig):
            raise TypeError("rejection must be MultiDualRejectionConfig")
        self.rejection = rejection

    @property
    def rejection_metadata(self) -> dict[str, float | str]:
        return self.rejection.as_dict()

    def _center_features(self, array):
        return _normalize_rows(super()._center_features(array))

    def explain_feature(self, feature) -> dict[str, Any]:
        explanation = super().explain_feature(feature)
        explanation["rejection"] = self.rejection_metadata
        return explanation

    def _classify_multi_prototype(self, array) -> list[DinoV3Prediction]:
        bank = self._effective_bank()
        if not bank.formal:
            raise ValueError("Formal Multi-prototype bank cannot be empty")
        centered_rows = self._center_features(array)

        results: list[DinoV3Prediction] = []
        for row_index, centered in enumerate(centered_rows):
            formal_index, formal, formal_distance, formal_score = self._winning_record(
                centered,
                bank.formal,
            )
            candidates = self._species_candidates(centered, bank.formal)
            if self.rejection.accepts(
                cosine_score=formal_score,
                squared_distance=formal_distance,
            ):
                results.append(
                    DinoV3Prediction(
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
                    )
                )
                continue

            if bank.provisional:
                provisional_index, provisional, provisional_distance, provisional_score = (
                    self._winning_record(centered, bank.provisional)
                )
                if self.rejection.accepts(
                    cosine_score=provisional_score,
                    squared_distance=provisional_distance,
                ):
                    results.append(
                        DinoV3Prediction(
                            species=provisional.species,
                            accepted=False,
                            best_known_species=formal.species,
                            head_species=formal.species,
                            prototype_species=provisional.species,
                            head_prototype_consistent=False,
                            known_score=provisional_score,
                            threshold=self.rejection.cosine_threshold,
                            candidates=candidates,
                            embedding=array[row_index].copy(),
                            source=provisional.source,
                            registry_id=provisional.registry_id,
                            registration_status=provisional.registration_status,
                            assistive_match=True,
                            nearest_prototype_index=len(bank.formal) + provisional_index,
                            squared_distance=provisional_distance,
                        )
                    )
                    continue

            results.append(
                DinoV3Prediction(
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
                )
            )
        return results
