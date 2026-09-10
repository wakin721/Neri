from __future__ import annotations

from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def require_replace(text: str, old: str, new: str, *, count: int | None = 1) -> str:
    found = text.count(old)
    if count is not None and found != count:
        raise RuntimeError(f"expected {count} occurrences, found {found}: {old[:80]!r}")
    if found == 0:
        raise RuntimeError(f"anchor not found: {old[:80]!r}")
    return text.replace(old, new)


def patch_classifier() -> None:
    path = "system/dinov3/classifier.py"
    text = read(path)
    if "def explain_feature(" in text:
        return
    marker = "    def _classify_multi_prototype(self, array: np.ndarray) -> list[DinoV3Prediction]:\n"
    addition = '''    @staticmethod
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

    def explain_feature(self, feature: np.ndarray) -> dict[str, Any]:
        """Return a deterministic local 2-D explanation around the nearest two species.

        Classification still uses the full 768-dimensional centered feature space.
        The x-axis joins the closest prototype of the two nearest species; the
        y-axis is an orthogonal residual direction chosen deterministically.
        """
        value = np.asarray(feature, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("Expected one feature vector")
        array = self._validate_features(value[None, :])
        centered = array[0] - self._feature_center
        bank = self._effective_bank()
        records = tuple(bank.formal) + tuple(bank.provisional)
        if not records:
            raise ValueError("Prototype bank cannot be empty")

        nearest_species = list(self._species_candidates(centered, records)[:2])
        selected_species = [str(item["name"]) for item in nearest_species]
        selected_indices = [
            index
            for index, record in enumerate(records)
            if record.species in selected_species
        ]
        if not selected_indices:
            raise ValueError("No nearby prototypes are available")

        first_index = int(nearest_species[0]["nearest_prototype_index"])
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

        def project(vector: np.ndarray) -> tuple[float, float]:
            delta = vector - origin
            return float(np.dot(delta, axis_x)), float(np.dot(delta, axis_y))

        points: list[dict[str, Any]] = []
        for index in selected_indices:
            record = records[index]
            x, y = project(record.embedding)
            points.append(
                {
                    "kind": "prototype",
                    "species": record.species,
                    "source": record.source,
                    "registry_id": record.registry_id,
                    "registration_status": record.registration_status,
                    "prototype_index": index,
                    "x": x,
                    "y": y,
                }
            )

        prediction = self._classify_multi_prototype(array)[0]
        current_x, current_y = project(centered)
        points.append(
            {
                "kind": "current",
                "species": prediction.species,
                "source": prediction.source,
                "registry_id": prediction.registry_id,
                "registration_status": prediction.registration_status,
                "prototype_index": prediction.nearest_prototype_index,
                "x": current_x,
                "y": current_y,
            }
        )
        return {
            "species": prediction.species,
            "accepted": prediction.accepted,
            "best_known_species": prediction.best_known_species,
            "known_score": prediction.known_score,
            "threshold": prediction.threshold,
            "nearest_prototype_index": prediction.nearest_prototype_index,
            "squared_distance": prediction.squared_distance,
            "nearest_species": nearest_species,
            "projection": {
                "method": "nearest_two_species_axis",
                "species": selected_species,
                "points": points,
            },
        }

'''
    if marker not in text:
        raise RuntimeError("classifier insertion anchor missing")
    write(path, text.replace(marker, addition + marker, 1))


def patch_registry() -> None:
    path = "system/dinov3/registry.py"
    text = read(path)
    if "def record_human_species(" not in text:
        marker = "    def set_cluster_purity(self, entry_id, value):\n"
        addition = '''    def restore_identity(self, entry_id, *, common_name="", scientific_name=""):
        self._row(entry_id)
        self._conn.execute(
            "UPDATE registrations SET common_name=?,scientific_name=?,updated_at=? WHERE id=?",
            (str(common_name).strip(), str(scientific_name).strip(), _now(), entry_id),
        )
        self._conn.commit()
        return self.get(entry_id)

    def record_human_species(
        self,
        embedding,
        *,
        common_name,
        camera_id,
        captured_at,
        source_path,
        bbox=None,
        frame_index=None,
        timestamp_seconds=None,
        preferred_entry_id=None,
    ):
        """Attach explicit human identity to a candidate without renaming formal entries."""
        common = str(common_name).strip()
        if not common:
            raise ValueError("common_name is required")
        vector = normalize_embedding(embedding)
        entry_id = None

        candidate_ids = []
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

        if entry_id is None:
            entry_id = self._create()
        before = self.get(entry_id)
        self.record_observation(
            entry_id,
            vector,
            camera_id=camera_id,
            captured_at=captured_at,
            source_path=source_path,
            bbox=bbox,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )
        updated = self.set_identity(
            entry_id,
            common_name=common,
            scientific_name=before.scientific_name,
        )
        return updated, before.common_name, before.scientific_name

'''
        if marker not in text:
            raise RuntimeError("registry method insertion anchor missing")
        text = text.replace(marker, addition + marker, 1)

    old = '''        display = (
            f"未知物种 #{row['candidate_number']}"
            if status == "candidate"
            else (
'''
    new = '''        display = (
            (
                f"{common}（候选 #{row['candidate_number']}）"
                if common
                else f"未知物种 #{row['candidate_number']}"
            )
            if status == "candidate"
            else (
'''
    if old in text:
        text = text.replace(old, new, 1)
    write(path, text)


def patch_feedback_store() -> None:
    path = "system/dinov3/feedback.py"
    text = read(path)
    if "CREATE TABLE IF NOT EXISTS feedback_registry_assignments" not in text:
        old = '''            CREATE TABLE IF NOT EXISTS feedback_learning_states(
              species TEXT PRIMARY KEY,
              payload TEXT NOT NULL
            );
'''
        new = old + '''            CREATE TABLE IF NOT EXISTS feedback_registry_assignments(
              operation_id TEXT NOT NULL,
              registration_id INTEGER NOT NULL,
              previous_common_name TEXT NOT NULL DEFAULT '',
              previous_scientific_name TEXT NOT NULL DEFAULT '',
              assigned_common_name TEXT NOT NULL,
              PRIMARY KEY(operation_id, registration_id)
            );
'''
        text = require_replace(text, old, new)

    if "def record_registry_feedback(" not in text:
        marker = "    @staticmethod\n    def _feedback_record(row: sqlite3.Row) -> FeedbackRecord:\n"
        addition = '''    def record_registry_feedback(
        self,
        observation_id: str,
        *,
        operation_id: str,
        registration_id: int,
        previous_common_name: str,
        previous_scientific_name: str,
        confirmed_species: str,
    ) -> FeedbackRecord:
        """Record an auditable Registry assignment without training checkpoint classes."""
        observation = self.get_observation(observation_id)
        species = str(confirmed_species).strip()
        if not species or species in {"Unknown", "空"}:
            raise ValueError("confirmed registry species is required")
        operation = self._conn.execute(
            "SELECT reverted FROM feedback_operations WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if operation is not None and bool(operation["reverted"]):
            raise ValueError("feedback operation has already been reverted")
        self._conn.execute(
            "INSERT OR IGNORE INTO feedback_operations(operation_id,reverted) VALUES(?,0)",
            (operation_id,),
        )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO feedback_registry_assignments(
              operation_id,registration_id,previous_common_name,
              previous_scientific_name,assigned_common_name
            ) VALUES(?,?,?,?,?)
            """,
            (
                operation_id,
                int(registration_id),
                str(previous_common_name),
                str(previous_scientific_name),
                species,
            ),
        )

        previous = self._conn.execute(
            "SELECT id FROM human_feedback WHERE observation_id=? AND active=1",
            (observation_id,),
        ).fetchone()
        supersedes_id = int(previous["id"]) if previous is not None else None
        if supersedes_id is not None:
            self._conn.execute(
                "UPDATE human_feedback SET active=0 WHERE id=?",
                (supersedes_id,),
            )
        cursor = self._conn.execute(
            """
            INSERT INTO human_feedback(
              observation_id,operation_id,feedback_type,predicted_species,
              confirmed_species,positive_species,hard_negative_species,active,
              supersedes_id
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                observation_id,
                operation_id,
                "registry",
                observation.predicted_species,
                species,
                None,
                None,
                1,
                supersedes_id,
            ),
        )
        self._conn.commit()
        return FeedbackRecord(
            id=int(cursor.lastrowid),
            observation_id=observation_id,
            operation_id=operation_id,
            feedback_type="registry",
            predicted_species=observation.predicted_species,
            confirmed_species=species,
            positive_species=None,
            hard_negative_species=None,
            active=True,
        )

    def registry_assignments(self, operation_id: str) -> list[dict[str, object]]:
        rows = self._conn.execute(
            """
            SELECT registration_id,previous_common_name,previous_scientific_name,
                   assigned_common_name
            FROM feedback_registry_assignments
            WHERE operation_id=? ORDER BY registration_id
            """,
            (operation_id,),
        ).fetchall()
        return [
            {
                "registration_id": int(row["registration_id"]),
                "previous_common_name": str(row["previous_common_name"]),
                "previous_scientific_name": str(row["previous_scientific_name"]),
                "assigned_common_name": str(row["assigned_common_name"]),
            }
            for row in rows
        ]

    def representative_observation_id(self, species: str) -> str | None:
        row = self._conn.execute(
            """
            SELECT h.observation_id
            FROM human_feedback h
            JOIN observations o ON o.id=h.observation_id
            WHERE h.active=1 AND h.positive_species=?
            ORDER BY h.id DESC LIMIT 1
            """,
            (species,),
        ).fetchone()
        return None if row is None else str(row["observation_id"])

'''
        if marker not in text:
            raise RuntimeError("feedback store insertion anchor missing")
        text = text.replace(marker, addition + marker, 1)
    write(path, text)


def patch_registry_service() -> None:
    path = "system/backend/dinov3_registry_service.py"
    text = read(path)
    if "def render_media_example(" in text:
        return
    old = '''def render_registry_example(
    registry: SpeciesRegistry, registration_id: int, event_id: int
) -> bytes:
    event = next(
        (item for item in registry.list_events(registration_id) if item["id"] == event_id),
        None,
    )
    if event is None:
        raise FileNotFoundError("DINOv3 registry event not found")
    bbox = event.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise FileNotFoundError("该历史事件没有裁切框信息")
    frame = _read_registry_frame(event)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
'''
    new = '''def render_media_example(
    *,
    source_path: str,
    bbox,
    frame_index: int | None = None,
    timestamp_seconds: float | None = None,
) -> bytes:
    values = list(bbox) if bbox is not None else []
    if len(values) != 4:
        raise FileNotFoundError("该历史事件没有裁切框信息")
    event = {
        "source_path": source_path,
        "bbox": values,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
    }
    frame = _read_registry_frame(event)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in values)
'''
    text = require_replace(text, old, new)
    old_tail = '''    if not ok:
        raise RuntimeError("裁切例图编码失败")
    return encoded.tobytes()
'''
    new_tail = old_tail + '''

def render_registry_example(
    registry: SpeciesRegistry, registration_id: int, event_id: int
) -> bytes:
    event = next(
        (item for item in registry.list_events(registration_id) if item["id"] == event_id),
        None,
    )
    if event is None:
        raise FileNotFoundError("DINOv3 registry event not found")
    return render_media_example(
        source_path=str(event.get("source_path") or ""),
        bbox=event.get("bbox"),
        frame_index=(
            int(event["frame_index"]) if event.get("frame_index") is not None else None
        ),
        timestamp_seconds=(
            float(event["timestamp_seconds"])
            if event.get("timestamp_seconds") is not None
            else None
        ),
    )
'''
    # The original tail now belongs to render_media_example and appears once.
    text = require_replace(text, old_tail, new_tail)
    write(path, text)


def patch_feedback_service() -> None:
    path = "system/backend/dinov3_feedback_service.py"
    text = read(path)
    if "def record_registry_species_feedback(" not in text:
        marker = "def _raw_box_for_observation(detection_data: dict, observation_id: str) -> dict:\n"
        addition = '''def _assign_registry_species(
    feedback: HumanFeedbackStore,
    classification_model_path: str,
    observation: FeedbackObservation,
    *,
    operation_id: str,
    confirmed_species: str,
):
    from .dinov3_registry_service import open_registry_for_model

    registry = open_registry_for_model(classification_model_path)
    try:
        updated, previous_common, previous_scientific = registry.record_human_species(
            observation.embedding,
            common_name=confirmed_species,
            camera_id=observation.camera_id,
            captured_at=observation.captured_at,
            source_path=observation.source_path,
            bbox=observation.bbox,
            frame_index=observation.frame_index,
            timestamp_seconds=observation.timestamp_seconds,
            preferred_entry_id=observation.registry_id,
        )
        try:
            feedback.record_registry_feedback(
                observation.id,
                operation_id=operation_id,
                registration_id=updated.id,
                previous_common_name=previous_common,
                previous_scientific_name=previous_scientific,
                confirmed_species=confirmed_species,
            )
        except Exception:
            registry.restore_identity(
                updated.id,
                common_name=previous_common,
                scientific_name=previous_scientific,
            )
            raise
        return updated
    finally:
        registry.close()


def record_registry_species_feedback(
    classification_model_path: str,
    observation: FeedbackObservation,
    operation_id: str,
    confirmed_species: str,
):
    feedback, _feature_center = _open_feedback_state(classification_model_path)
    try:
        return _assign_registry_species(
            feedback,
            classification_model_path,
            observation,
            operation_id=operation_id,
            confirmed_species=confirmed_species,
        )
    finally:
        feedback.close()


def explain_feedback_observation(
    classification_model_path: str,
    observation_id: str,
) -> dict:
    from system.dinov3.classifier import DinoV3Classifier
    from .dinov3_registry_service import load_checkpoint_for_model, open_registry_for_model

    feedback, _feature_center = _open_feedback_state(classification_model_path)
    registry = open_registry_for_model(classification_model_path)
    try:
        observation = feedback.get_observation(observation_id)
        checkpoint = load_checkpoint_for_model(classification_model_path)
        classifier = DinoV3Classifier(checkpoint, feedback=feedback, registry=registry)
        result = classifier.explain_feature(observation.embedding)
        result["current_example_available"] = bool(
            observation.source_path and Path(observation.source_path).expanduser().is_file()
        )
        nearest_example = None
        nearest = result.get("nearest_species") or []
        if nearest:
            closest = nearest[0]
            registry_id = closest.get("registry_id")
            if registry_id is not None:
                try:
                    events = registry.list_events(int(registry_id))
                except KeyError:
                    events = []
                event = next((item for item in events if item.get("has_example")), None)
                if event is not None:
                    nearest_example = {
                        "kind": "registry",
                        "species": str(closest.get("name") or ""),
                        "registration_id": int(registry_id),
                        "event_id": int(event["id"]),
                    }
            if nearest_example is None:
                representative_id = feedback.representative_observation_id(
                    str(closest.get("name") or "")
                )
                if representative_id:
                    nearest_example = {
                        "kind": "observation",
                        "species": str(closest.get("name") or ""),
                        "observation_id": representative_id,
                    }
        result["nearest_example"] = nearest_example
        return result
    finally:
        registry.close()
        feedback.close()


def render_feedback_observation_example(
    classification_model_path: str,
    observation_id: str,
) -> bytes:
    from .dinov3_registry_service import render_media_example

    feedback, _feature_center = _open_feedback_state(classification_model_path)
    try:
        observation = feedback.get_observation(observation_id)
        return render_media_example(
            source_path=observation.source_path,
            bbox=observation.bbox,
            frame_index=observation.frame_index,
            timestamp_seconds=observation.timestamp_seconds,
        )
    finally:
        feedback.close()

'''
        if marker not in text:
            raise RuntimeError("feedback service insertion anchor missing")
        text = text.replace(marker, addition + marker, 1)

    old = '''        record = feedback.record_feedback(
            request.observation_id,
            operation_id=request.feedback_operation_id,
            action=action,
            confirmed_species=species_name,
        )
        affected = _affected_learning_species(record)
        for species in sorted(affected):
            feedback.recompute_species(species, feature_center)

        item = services._reload_validation_item(file_path, input_path)
        return {
            "item": item,
            "operation_id": request.feedback_operation_id,
            "affected_species": sorted(affected),
        }
'''
    new = '''        registry_entry = None
        if (
            action == "update"
            and species_name
            and species_name not in feedback.checkpoint_classes
        ):
            registry_entry = _assign_registry_species(
                feedback,
                request.classification_model_path,
                observation,
                operation_id=request.feedback_operation_id,
                confirmed_species=species_name,
            )
            affected: set[str] = set()
        else:
            record = feedback.record_feedback(
                request.observation_id,
                operation_id=request.feedback_operation_id,
                action=action,
                confirmed_species=species_name,
            )
            affected = _affected_learning_species(record)
            for species in sorted(affected):
                feedback.recompute_species(species, feature_center)

        item = services._reload_validation_item(file_path, input_path)
        return {
            "item": item,
            "operation_id": request.feedback_operation_id,
            "affected_species": sorted(affected),
            "registry_id": registry_entry.id if registry_entry is not None else None,
        }
'''
    if old in text:
        text = text.replace(old, new, 1)

    old_revert = '''        affected = feedback.revert_operation(request.feedback_operation_id)
        for species in sorted(affected):
            if species in feedback.checkpoint_classes:
                feedback.recompute_species(species, feature_center)

        _ensure_ecological_journal(feedback)
'''
    new_revert = '''        registry_assignments = feedback.registry_assignments(
            request.feedback_operation_id
        )
        affected = feedback.revert_operation(request.feedback_operation_id)
        for species in sorted(affected):
            if species in feedback.checkpoint_classes:
                feedback.recompute_species(species, feature_center)

        if registry_assignments:
            from .dinov3_registry_service import open_registry_for_model

            registry = open_registry_for_model(request.classification_model_path)
            try:
                for assignment in registry_assignments:
                    registration_id = int(assignment["registration_id"])
                    try:
                        current = registry.get(registration_id)
                    except KeyError:
                        continue
                    if current.common_name != assignment["assigned_common_name"]:
                        continue
                    registry.restore_identity(
                        registration_id,
                        common_name=str(assignment["previous_common_name"]),
                        scientific_name=str(assignment["previous_scientific_name"]),
                    )
            finally:
                registry.close()

        _ensure_ecological_journal(feedback)
'''
    if old_revert in text:
        text = text.replace(old_revert, new_revert, 1)
    write(path, text)


def patch_services() -> None:
    path = "system/backend/services.py"
    text = read(path)
    if "def _checkpoint_species_for_model(" not in text:
        marker = "def _record_validation_feedback(\n"
        addition = '''def _checkpoint_species_for_model(classification_model_path: str) -> set[str]:
    from .dinov3_registry_service import load_checkpoint_for_model

    return set(load_checkpoint_for_model(classification_model_path).classes)


def _record_validation_registry_feedback(
    classification_model_path: str,
    observation,
    operation_id: str,
    confirmed_species: str,
):
    from .dinov3_feedback_service import record_registry_species_feedback

    return record_registry_species_feedback(
        classification_model_path,
        observation,
        operation_id,
        confirmed_species,
    )


'''
        if marker not in text:
            raise RuntimeError("services helper insertion anchor missing")
        text = text.replace(marker, addition + marker, 1)

    old_confirmed = '''        confirmed_species = (
            (request.species_name or "").strip() or None
            if request.action == "update"
            else None
        )
        try:
'''
    new_confirmed = '''        confirmed_species = (
            (request.species_name or "").strip() or None
            if request.action == "update"
            else None
        )
        checkpoint_species = (
            _checkpoint_species_for_model(request.classification_model_path)
            if request.action == "update" and confirmed_species
            else set()
        )
        try:
'''
    if old_confirmed in text:
        text = require_replace(text, old_confirmed, new_confirmed, count=2)

    old_call = '''                _record_validation_feedback(
                    request.classification_model_path,
                    observation,
                    operation_id,
                    request.action,
                    confirmed_species,
                )
'''
    new_call = '''                if (
                    request.action == "update"
                    and confirmed_species
                    and confirmed_species not in checkpoint_species
                ):
                    _record_validation_registry_feedback(
                        request.classification_model_path,
                        observation,
                        operation_id,
                        confirmed_species,
                    )
                else:
                    _record_validation_feedback(
                        request.classification_model_path,
                        observation,
                        operation_id,
                        request.action,
                        confirmed_species,
                    )
'''
    if old_call in text:
        text = require_replace(text, old_call, new_call, count=2)
    write(path, text)


def patch_registry_api() -> None:
    path = "system/dinov3/api.py"
    text = read(path)
    if "def build_registry_catalog(" not in text:
        marker = "def dinov3_registry_router() -> APIRouter:\n"
        addition = '''def build_registry_catalog(checkpoint: Any, registry: Any) -> list[dict[str, Any]]:
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


'''
        if marker not in text:
            raise RuntimeError("registry API insertion anchor missing")
        text = text.replace(marker, addition + marker, 1)

    if '@router.get("/registry/catalog"' not in text:
        marker = '''    @router.get("/registry/{registration_id}", response_model=DinoV3RegistryEntryResponse)
'''
        addition = '''    @router.get("/registry/catalog", response_model=list[DinoV3RegistryEntryResponse])
    def list_registry_catalog(
        classification_model_path: str = Query(..., min_length=1),
    ):
        checkpoint = load_checkpoint_for_model(classification_model_path)
        return _run_with_registry(
            classification_model_path,
            lambda registry: build_registry_catalog(checkpoint, registry),
        )

'''
        if marker not in text:
            raise RuntimeError("registry catalog route anchor missing")
        text = text.replace(marker, addition + marker, 1)
    write(path, text)


def patch_feedback_api() -> None:
    path = "system/backend/dinov3_feedback_api.py"
    text = read(path)
    if "explain_feedback_observation" in text:
        return
    text = text.replace(
        "from fastapi import APIRouter, HTTPException\n",
        "from fastapi import APIRouter, HTTPException, Query, Response\n",
        1,
    )
    old_import = "from .dinov3_feedback_service import apply_box_feedback, revert_feedback_operation\n"
    new_import = '''from .dinov3_feedback_service import (
    apply_box_feedback,
    explain_feedback_observation,
    render_feedback_observation_example,
    revert_feedback_operation,
)
'''
    text = require_replace(text, old_import, new_import)
    marker = '''    @router.post("/revert")
'''
    addition = '''    @router.get("/observations/{observation_id}/explain")
    def observation_explanation(
        observation_id: str,
        classification_model_path: str = Query(..., min_length=1),
    ):
        try:
            return explain_feedback_observation(
                classification_model_path,
                observation_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="DINOv3 observation not found") from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/observations/{observation_id}/example")
    def observation_example(
        observation_id: str,
        classification_model_path: str = Query(..., min_length=1),
    ):
        try:
            content = render_feedback_observation_example(
                classification_model_path,
                observation_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="DINOv3 observation not found") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=60"},
        )

'''
    if marker not in text:
        raise RuntimeError("feedback API insertion anchor missing")
    text = text.replace(marker, addition + marker, 1)
    write(path, text)


if __name__ == "__main__":
    patch_classifier()
    patch_registry()
    patch_feedback_store()
    patch_registry_service()
    patch_feedback_service()
    patch_services()
    patch_registry_api()
    patch_feedback_api()
