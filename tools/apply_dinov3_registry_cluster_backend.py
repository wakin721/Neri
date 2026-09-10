from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str, marker: str | None = None) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if marker and marker in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:100]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


# API response schema: expose feedback learning state and per-prototype clusters.
replace_once(
    "system/backend/models.py",
    '''class DinoV3RegistryEntryResponse(BaseModel):
    id: int
    candidate_number: int
    status: str
    common_name: str = ""
    scientific_name: str = ""
    event_count: int
    camera_count: int
    prototype_count: int
    cluster_purity: float
    embedding_consistency: float
    conditions: dict[str, bool] = Field(default_factory=dict)
    can_register: bool
    display_name: str
''',
    '''class DinoV3ClusterExampleRefResponse(BaseModel):
    kind: Literal["registry", "observation"]
    registration_id: int | None = None
    event_id: int | None = None
    observation_id: str | None = None


class DinoV3RegistryClusterResponse(BaseModel):
    id: str
    label: str
    source: str
    prototype_index: int
    event_count: int = 0
    camera_count: int = 0
    sample_count: int = 0
    mean_squared_distance: float | None = None
    active: bool = True
    learning_status: str | None = None
    example_refs: list[DinoV3ClusterExampleRefResponse] = Field(default_factory=list)


class DinoV3RegistryEntryResponse(BaseModel):
    id: int
    candidate_number: int
    status: str
    common_name: str = ""
    scientific_name: str = ""
    event_count: int
    camera_count: int
    prototype_count: int
    cluster_purity: float
    embedding_consistency: float
    conditions: dict[str, bool] = Field(default_factory=dict)
    can_register: bool
    display_name: str
    feedback_event_count: int = 0
    feedback_prototype_count: int = 0
    learning_status: str | None = None
    clusters: list[DinoV3RegistryClusterResponse] = Field(default_factory=list)
''',
    marker="class DinoV3RegistryClusterResponse",
)

# Keep observation ids when grouping feedback evidence into independent events.
replace_once(
    "system/dinov3/feedback.py",
    '''@dataclass(frozen=True)
class _EvidenceObservation:
    row_id: int
    source_path: str
    camera_id: str
    captured_at: datetime | None
    embedding: np.ndarray


@dataclass(frozen=True)
class _EvidenceEvent:
    key: str
    camera_id: str
    embedding: np.ndarray
''',
    '''@dataclass(frozen=True)
class _EvidenceObservation:
    row_id: int
    observation_id: str
    source_path: str
    camera_id: str
    captured_at: datetime | None
    embedding: np.ndarray


@dataclass(frozen=True)
class _EvidenceEvent:
    key: str
    camera_id: str
    embedding: np.ndarray
    observation_ids: tuple[str, ...]
''',
    marker="observation_ids: tuple[str, ...]",
)
replace_once(
    "system/dinov3/feedback.py",
    "SELECT h.id,o.payload,o.embedding\n",
    "SELECT h.id,h.observation_id,o.payload,o.embedding\n",
    marker="SELECT h.id,h.observation_id,o.payload,o.embedding",
)
replace_once(
    "system/dinov3/feedback.py",
    '''                _EvidenceObservation(
                    row_id=int(row["id"]),
                    source_path=str(payload.get("source_path") or ""),
''',
    '''                _EvidenceObservation(
                    row_id=int(row["id"]),
                    observation_id=str(row["observation_id"]),
                    source_path=str(payload.get("source_path") or ""),
''',
    marker='observation_id=str(row["observation_id"]),',
)
replace_once(
    "system/dinov3/feedback.py",
    '''                    _EvidenceEvent(
                        key=f"missing|{camera_id}|{identity}",
                        camera_id=camera_id,
                        embedding=embedding,
                    )
''',
    '''                    _EvidenceEvent(
                        key=f"missing|{camera_id}|{identity}",
                        camera_id=camera_id,
                        embedding=embedding,
                        observation_ids=tuple(item.observation_id for item in group),
                    )
''',
    marker="observation_ids=tuple(item.observation_id for item in group)",
)
replace_once(
    "system/dinov3/feedback.py",
    '''        return _EvidenceEvent(
            key=f"timed|{camera_id}|{marker:.6f}|{first.row_id}",
            camera_id=camera_id,
            embedding=embedding,
        )
''',
    '''        return _EvidenceEvent(
            key=f"timed|{camera_id}|{marker:.6f}|{first.row_id}",
            camera_id=camera_id,
            embedding=embedding,
            observation_ids=tuple(item.observation_id for item in observations),
        )
''',
    marker="observation_ids=tuple(item.observation_id for item in observations)",
)

# Public, read-only cluster summaries for known-species human feedback.
feedback_cluster_method = '''    def cluster_details(
        self,
        species: str,
        feature_center: np.ndarray,
    ) -> list[dict[str, object]]:
        """Describe current feedback clusters without exposing 768-D embeddings."""
        positive_events = self._group_events(
            self._evidence_observations(species, column="positive_species")
        )
        if not positive_events:
            return []

        center = feature_center
        if hasattr(center, "numpy"):
            center = center.numpy()
        center = np.asarray(center, dtype=np.float32)
        if center.shape != (DINO_DIM,) or not np.isfinite(center).all():
            raise ValueError("Expected finite feature_center with shape (768,)")

        state = self.learning_state(species)
        active_generation = self._active_generation(species)
        active = active_generation is not None
        if active_generation is not None:
            prototype_values = self._generation_prototypes(int(active_generation["id"]))
            prototypes = np.stack(prototype_values).astype(np.float32, copy=False)
        else:
            centered = np.stack(
                [event.embedding - center for event in positive_events]
            ).astype(np.float32)
            prototypes = deterministic_k_means(centered, max_k=1).astype(
                np.float32,
                copy=False,
            )

        event_vectors = np.stack(
            [event.embedding - center for event in positive_events]
        ).astype(np.float32)
        deltas = event_vectors[:, None, :] - prototypes[None, :, :]
        distances = np.einsum("nkd,nkd->nk", deltas, deltas, optimize=True)
        labels = np.argmin(distances, axis=1)
        result: list[dict[str, object]] = []
        for prototype_index in range(len(prototypes)):
            member_indices = np.flatnonzero(labels == prototype_index).tolist()
            if not member_indices:
                continue
            ordered = sorted(
                member_indices,
                key=lambda index: (float(distances[index, prototype_index]), index),
            )
            refs: list[dict[str, object]] = []
            seen: set[str] = set()
            for event_index in ordered:
                for observation_id in positive_events[event_index].observation_ids:
                    if observation_id in seen:
                        continue
                    seen.add(observation_id)
                    refs.append(
                        {
                            "kind": "observation",
                            "observation_id": observation_id,
                        }
                    )
                    if len(refs) >= 3:
                        break
                if len(refs) >= 3:
                    break
            member_distances = distances[member_indices, prototype_index]
            result.append(
                {
                    "id": f"feedback:{species}:{prototype_index}",
                    "label": (
                        f"Feedback Cluster #{prototype_index + 1}"
                        if active
                        else "反馈证据（尚未形成 prototype）"
                    ),
                    "source": "feedback" if active else "feedback_evidence",
                    "prototype_index": prototype_index,
                    "event_count": len(member_indices),
                    "camera_count": len(
                        {positive_events[index].camera_id for index in member_indices}
                    ),
                    "sample_count": sum(
                        len(positive_events[index].observation_ids)
                        for index in member_indices
                    ),
                    "mean_squared_distance": float(np.mean(member_distances)),
                    "active": active,
                    "learning_status": state.status,
                    "example_refs": refs,
                }
            )
        return result

'''
replace_once(
    "system/dinov3/feedback.py",
    "    def prototype_bank(self, feature_center: np.ndarray) -> PrototypeBank:\n",
    feedback_cluster_method
    + "    def prototype_bank(self, feature_center: np.ndarray) -> PrototypeBank:\n",
    marker="Describe current feedback clusters without exposing 768-D embeddings",
)

# Per-registry-entry cluster membership is derived from current prototypes on read.
registry_cluster_method = '''    def cluster_details(self, entry_id: int) -> list[dict[str, object]]:
        """Assign Registry events to their nearest current prototype for inspection."""
        detail = self.get(entry_id)
        rows = self._event_rows(entry_id)
        if not rows:
            return []
        embeddings = np.stack([_from_blob(row["embedding"]) for row in rows]).astype(
            np.float32,
            copy=False,
        )
        prototypes = self._prototypes(entry_id)
        if len(prototypes) == 0:
            prototypes = build_prototype(embeddings)[None, :]
        deltas = embeddings[:, None, :] - prototypes[None, :, :]
        distances = np.einsum("nkd,nkd->nk", deltas, deltas, optimize=True)
        labels = np.argmin(distances, axis=1)

        result: list[dict[str, object]] = []
        for prototype_index in range(len(prototypes)):
            member_indices = np.flatnonzero(labels == prototype_index).tolist()
            if not member_indices:
                continue
            ordered = sorted(
                member_indices,
                key=lambda index: (float(distances[index, prototype_index]), int(rows[index]["id"])),
            )
            refs: list[dict[str, object]] = []
            for event_index in ordered:
                row = rows[event_index]
                coordinates = (
                    row["box_x1"],
                    row["box_y1"],
                    row["box_x2"],
                    row["box_y2"],
                )
                if not str(row["source_path"]) or not all(
                    value is not None for value in coordinates
                ):
                    continue
                refs.append(
                    {
                        "kind": "registry",
                        "registration_id": entry_id,
                        "event_id": int(row["id"]),
                    }
                )
                if len(refs) >= 3:
                    break
            member_distances = distances[member_indices, prototype_index]
            result.append(
                {
                    "id": f"registry:{entry_id}:{prototype_index}",
                    "label": f"Cluster #{prototype_index + 1}",
                    "source": "registry",
                    "prototype_index": prototype_index,
                    "event_count": len(member_indices),
                    "camera_count": len(
                        {str(rows[index]["camera_id"]) for index in member_indices}
                    ),
                    "sample_count": sum(
                        int(rows[index]["sample_count"]) for index in member_indices
                    ),
                    "mean_squared_distance": float(np.mean(member_distances)),
                    "active": True,
                    "learning_status": detail.status,
                    "example_refs": refs,
                }
            )
        return result

'''
replace_once(
    "system/dinov3/registry.py",
    "    def prototype_bank(self, feature_center: np.ndarray) -> PrototypeBank:\n",
    registry_cluster_method
    + "    def prototype_bank(self, feature_center: np.ndarray) -> PrototypeBank:\n",
    marker="Assign Registry events to their nearest current prototype for inspection",
)

# Unified catalog: checkpoint base clusters + known-species feedback + Registry clusters.
replace_once(
    "system/dinov3/api.py",
    '''def build_registry_catalog(checkpoint: Any, registry: Any) -> list[dict[str, Any]]:
    """Merge immutable checkpoint classes with mutable local Registry entries."""
    result: list[dict[str, Any]] = []
    counts = tuple(int(value) for value in checkpoint.prototypes_per_class)
    for index, species in enumerate(checkpoint.classes):
        result.append(
            {
                "id": -(index + 1),
                "candidate_number": index + 1,
                "status": "checkpoint",
                "common_name": species,
                "scientific_name": "",
                "event_count": 0,
                "camera_count": 0,
                "prototype_count": counts[index],
                "cluster_purity": 1.0,
                "embedding_consistency": 1.0,
                "conditions": {},
                "can_register": False,
                "display_name": species,
            }
        )
    result.extend(entry.as_dict() for entry in registry.list())
    return result
''',
    '''def build_registry_catalog(
    checkpoint: Any,
    registry: Any,
    *,
    feedback: Any | None = None,
) -> list[dict[str, Any]]:
    """Merge checkpoint, human-feedback, and local Registry state for inspection."""
    result: list[dict[str, Any]] = []
    counts = tuple(int(value) for value in checkpoint.prototypes_per_class)
    raw_class_indices = checkpoint.prototype_class_indices
    if hasattr(raw_class_indices, "tolist"):
        class_indices = [int(value) for value in raw_class_indices.tolist()]
    else:
        class_indices = [int(value) for value in raw_class_indices]

    for index, species in enumerate(checkpoint.classes):
        base_indices = [
            prototype_index
            for prototype_index, class_index in enumerate(class_indices)
            if class_index == index
        ]
        clusters: list[dict[str, Any]] = [
            {
                "id": f"checkpoint:{index}:{local_index}",
                "label": f"Base #{local_index + 1}",
                "source": "checkpoint",
                "prototype_index": prototype_index,
                "event_count": 0,
                "camera_count": 0,
                "sample_count": 0,
                "mean_squared_distance": None,
                "active": True,
                "learning_status": None,
                "example_refs": [],
            }
            for local_index, prototype_index in enumerate(base_indices)
        ]
        feedback_event_count = 0
        feedback_prototype_count = 0
        learning_status = None
        if feedback is not None:
            state = feedback.learning_state(species)
            if state.positive_events > 0:
                feedback_event_count = int(state.positive_events)
                feedback_prototype_count = int(state.prototype_count)
                learning_status = str(state.status)
                clusters.extend(
                    feedback.cluster_details(species, checkpoint.feature_center)
                )
        result.append(
            {
                "id": -(index + 1),
                "candidate_number": index + 1,
                "status": "checkpoint",
                "common_name": species,
                "scientific_name": "",
                "event_count": 0,
                "camera_count": 0,
                "prototype_count": counts[index],
                "cluster_purity": 1.0,
                "embedding_consistency": 1.0,
                "conditions": {},
                "can_register": False,
                "display_name": species,
                "feedback_event_count": feedback_event_count,
                "feedback_prototype_count": feedback_prototype_count,
                "learning_status": learning_status,
                "clusters": clusters,
            }
        )

    for entry in registry.list():
        data = entry.as_dict()
        cluster_reader = getattr(registry, "cluster_details", None)
        if callable(cluster_reader):
            data.update(
                {
                    "feedback_event_count": 0,
                    "feedback_prototype_count": 0,
                    "learning_status": entry.status,
                    "clusters": cluster_reader(entry.id),
                }
            )
        result.append(data)
    return result


def _catalog_with_feedback(checkpoint: Any, registry: Any) -> list[dict[str, Any]]:
    from .feedback import HumanFeedbackStore, feedback_path_for_registry

    feedback_path = feedback_path_for_registry(registry.path)
    if not feedback_path.exists():
        return build_registry_catalog(checkpoint, registry)
    feedback = HumanFeedbackStore(
        feedback_path,
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        threshold=checkpoint.threshold,
    )
    try:
        return build_registry_catalog(checkpoint, registry, feedback=feedback)
    finally:
        feedback.close()
''',
    marker="def _catalog_with_feedback",
)
replace_once(
    "system/dinov3/api.py",
    "            lambda registry: build_registry_catalog(checkpoint, registry),\n",
    "            lambda registry: _catalog_with_feedback(checkpoint, registry),\n",
    marker="lambda registry: _catalog_with_feedback(checkpoint, registry)",
)

# Explanation axes are now defined only by the selected species' prototypes.
replace_once(
    "system/dinov3/classifier.py",
    '''    @staticmethod
    def _projection_y_axis(
        axis_x: np.ndarray,
        origin: np.ndarray,
        current: np.ndarray,
        prototypes: Sequence[np.ndarray],
    ) -> np.ndarray:
        residual = current - origin - float(np.dot(current - origin, axis_x)) * axis_x
        norm = float(np.linalg.norm(residual))
        if norm > 1e-8:
            return residual / norm

        best = None
        best_norm = 0.0
        for prototype in prototypes:
            candidate = prototype - origin
            candidate = candidate - float(np.dot(candidate, axis_x)) * axis_x
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm > best_norm:
                best = candidate
                best_norm = candidate_norm
        if best is not None and best_norm > 1e-8:
            return best / best_norm

        basis_index = int(np.argmin(np.abs(axis_x)))
        basis = np.zeros(DINO_FEATURE_DIM, dtype=np.float32)
        basis[basis_index] = 1.0
        basis = basis - float(np.dot(basis, axis_x)) * axis_x
        basis_norm = float(np.linalg.norm(basis))
        if basis_norm <= 1e-8:
            raise ValueError("Unable to construct local projection axis")
        return basis / basis_norm
''',
    '''    @staticmethod
    def _projection_y_axis(
        axis_x: np.ndarray,
        origin: np.ndarray,
        prototypes: Sequence[np.ndarray],
    ) -> np.ndarray:
        residuals: list[np.ndarray] = []
        for prototype in prototypes:
            candidate = prototype - origin
            candidate = candidate - float(np.dot(candidate, axis_x)) * axis_x
            if float(np.linalg.norm(candidate)) > 1e-8:
                residuals.append(candidate.astype(np.float32, copy=False))

        if residuals:
            matrix = np.stack(residuals).astype(np.float32, copy=False)
            _u, _s, vh = np.linalg.svd(matrix, full_matrices=False)
            axis_y = vh[0].astype(np.float32, copy=False)
            axis_y = axis_y - float(np.dot(axis_y, axis_x)) * axis_x
            axis_norm = float(np.linalg.norm(axis_y))
            if axis_norm > 1e-8:
                axis_y = axis_y / axis_norm
                for residual in residuals:
                    alignment = float(np.dot(residual, axis_y))
                    if abs(alignment) <= 1e-8:
                        continue
                    if alignment < 0:
                        axis_y = -axis_y
                    return axis_y

        basis_index = int(np.argmin(np.abs(axis_x)))
        basis = np.zeros(DINO_FEATURE_DIM, dtype=np.float32)
        basis[basis_index] = 1.0
        basis = basis - float(np.dot(basis, axis_x)) * axis_x
        basis_norm = float(np.linalg.norm(basis))
        if basis_norm <= 1e-8:
            raise ValueError("Unable to construct local projection axis")
        return basis / basis_norm
''',
    marker="residuals: list[np.ndarray] = []",
)
replace_once(
    "system/dinov3/classifier.py",
    '''        first_index = int(nearest_species[0]["nearest_prototype_index"])
        first = records[first_index].embedding
        if len(nearest_species) > 1:
            second_index = int(nearest_species[1]["nearest_prototype_index"])
            second = records[second_index].embedding
            origin = (first + second) * 0.5
            axis_x = second - first
            axis_norm = float(np.linalg.norm(axis_x))
            if axis_norm <= 1e-8:
                axis_x = np.zeros(DINO_FEATURE_DIM, dtype=np.float32)
                axis_x[0] = 1.0
            else:
                axis_x = axis_x / axis_norm
        else:
            origin = first.copy()
            axis_x = np.zeros(DINO_FEATURE_DIM, dtype=np.float32)
            axis_x[0] = 1.0

        selected_prototypes = [records[index].embedding for index in selected_indices]
        axis_y = self._projection_y_axis(axis_x, origin, centered, selected_prototypes)
''',
    '''        species_prototypes = {
            species: [
                records[index].embedding
                for index in selected_indices
                if records[index].species == species
            ]
            for species in selected_species
        }
        first = np.stack(species_prototypes[selected_species[0]]).mean(axis=0)
        if len(nearest_species) > 1:
            second = np.stack(species_prototypes[selected_species[1]]).mean(axis=0)
            origin = (first + second) * 0.5
            axis_x = second - first
            axis_norm = float(np.linalg.norm(axis_x))
            if axis_norm <= 1e-8:
                axis_x = np.zeros(DINO_FEATURE_DIM, dtype=np.float32)
                axis_x[0] = 1.0
            else:
                axis_x = axis_x / axis_norm
        else:
            origin = first.copy()
            axis_x = np.zeros(DINO_FEATURE_DIM, dtype=np.float32)
            axis_x[0] = 1.0

        selected_prototypes = [records[index].embedding for index in selected_indices]
        axis_y = self._projection_y_axis(axis_x, origin, selected_prototypes)
''',
    marker="species_prototypes = {",
)
replace_once(
    "system/dinov3/classifier.py",
    '''        The x-axis joins the closest prototype of the two nearest species; the
        y-axis is an orthogonal residual direction chosen deterministically.
''',
    '''        The x-axis joins the prototype centroids of the two nearest species; the
        y-axis is derived only from nearby prototype residuals. The current sample
        is projected into this fixed local frame and never defines either axis.
''',
    marker="current sample\n        is projected into this fixed local frame",
)

print("Applied DINOv3 registry cluster backend implementation")
