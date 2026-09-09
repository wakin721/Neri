# DINOv3 Human-Feedback Multi-Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist box-level DINOv3 observations and explicit human feedback, then use independently confirmed events to build reversible local Multi-prototype overlays without modifying the shipped checkpoint.

**Architecture:** Add a fingerprint-scoped `feedback.sqlite3` beside the existing DINOv3 `registry.sqlite3`. `ImageProcessor` generates opaque observation ids but remains persistence-free; backend orchestration persists observations and feedback, while `HumanFeedbackStore` owns event aggregation, prototype generations, quality gates, and undo. `DinoV3Classifier` composes checkpoint prototypes, human-feedback overlays, and existing new-species registry overlays.

**Tech Stack:** Python 3.13, SQLite/WAL, NumPy, FastAPI/Pydantic, Flutter/Dart, existing DINOv3 Multi-prototype code and Flutter validation UI.

**Spec:** `docs/superpowers/specs/2026-09-09-dinov3-human-feedback-multiprototype-design.md`

## Global Constraints

- `multi_prototype.pt` is immutable; no feedback path may rewrite checkpoint bytes, checkpoint fingerprint, encoder weights, or cloud manifests.
- Human feedback is isolated by exact checkpoint fingerprint and stored locally in `feedback.sqlite3` beside that fingerprint's `registry.sqlite3`.
- Only explicit human feedback trains the overlay; model predictions alone never self-train.
- `correct` and corrected samples are positive evidence for the confirmed checkpoint species.
- Accepted `A -> B` corrections create `positive(B)` plus `hard_negative(A)`; rejected Unknown -> B creates only `positive(B)`.
- `empty` is rejection-negative only; `unverified`/ignored/reverted feedback does not participate in learning.
- Independent-event boundary is exactly 1,800 seconds.
- 0-3 events: Collecting/no inference overlay; >=4: Provisional/max 1 assistive prototype; >=10 + >=2 cameras + quality gate: Confirmed/max 2 formal prototypes; >=20 + >=3 cameras + quality gate: Mature/max 3 formal prototypes.
- Quality gate is `positive_coverage >= 0.80` and, when hard-negative events exist, `hard_negative_false_accept_rate <= 0.20`, evaluated with the checkpoint acceptance threshold.
- Failed generations never replace the last known-good Confirmed/Mature generation; set `needs_review=1` instead.
- Whole-file/batch auto-feedback is allowed only for media with exactly one learnable DINOv3 observation for the selected fingerprint.
- High-dimensional embeddings never enter detection JSON, Flutter DTO payloads, logs, or public API responses.
- Undo/revert must reverse both ecological validation state and model-learning state.
- Existing `SpeciesRegistry` remains responsible only for Unknown/new-species Candidate -> Provisional -> Confirmed -> Mature registration.

---

## File Structure

- Create `system/dinov3/feedback.py`: fingerprint-scoped SQLite store, observation persistence, feedback revisions, event aggregation, prototype generations, learning state, quality gate, and `PrototypeBank` provider.
- Create `system/backend/dinov3_feedback_service.py`: open/resolve feedback store for a selected DINOv3 model, persist runtime observations, apply box feedback, revert operations, and bridge ecological detection updates.
- Create `system/backend/dinov3_feedback_api.py`: FastAPI router for box feedback and operation revert.
- Modify `system/dinov3/classifier.py`: compose feedback overlay with checkpoint and registry banks; extend runtime observation metadata.
- Modify `system/dinov3/runtime.py`: own/open/close the feedback store alongside the registry.
- Modify `system/image_processor.py`: generate opaque observation ids and attach them to candidate metadata + `DinoV3Observation`; no SQLite access.
- Modify `system/backend/services.py`: persist drained observations at the existing post-inference boundary; preserve original-video paths; add conservative file/batch auto-feedback and operation ids.
- Modify `system/backend/models.py`: box-feedback/revert request and response models; extend validation requests with optional model/operation metadata without breaking existing callers.
- Modify `system/backend/main.py`: include the new DINOv3 feedback router.
- Modify `frontend/lib/src/models/job.dart`: decode `observation_id`, `registry_id`, `predicted_species`, and `feedback_status` on `DetectionBox`.
- Modify `frontend/lib/src/api_client_core.dart`: box-feedback and revert methods.
- Modify `frontend/lib/src/widgets/detection_media_viewer.dart`: hit-test DINOv3 boxes, highlight the selected observation, support current-frame video selection.
- Modify `frontend/lib/src/screens/species_validation_screen.dart`: box-feedback strip, safe file/batch auto-feedback messaging, feedback operation ids in undo history.
- Add Python tests under `tests/test_dinov3_feedback_*.py` and focused Flutter tests under `frontend/test/dinov3_*feedback*_test.dart`.

---

### Task 1: Create the fingerprint-scoped feedback store and observation persistence

**Files:**
- Create: `system/dinov3/feedback.py`
- Create: `tests/test_dinov3_feedback_store.py`

**Interfaces:**
- Produces: `feedback_path_for_registry(registry_path: str | Path) -> Path`
- Produces: `HumanFeedbackStore(path: str | Path, *, model_fingerprint: str, checkpoint_classes: tuple[str, ...], threshold: float)`
- Produces: `HumanFeedbackStore.persist_observation(observation: FeedbackObservation) -> str`
- Produces: `FeedbackObservation` dataclass with `id`, `source_path`, `media_kind`, `box_index`, `frame_index`, `timestamp_seconds`, `bbox`, `camera_id`, `captured_at`, `predicted_species`, `best_known_species`, `accepted`, `prediction_source`, `registry_id`, `known_score`, `threshold`, and normalized 768-d `embedding`.
- Produces: `HumanFeedbackStore.close() -> None`

- [ ] **Step 1: Write failing storage tests**

```python
# tests/test_dinov3_feedback_store.py
from datetime import datetime
import sqlite3
import numpy as np
import pytest

from system.dinov3.feedback import (
    FeedbackObservation,
    HumanFeedbackStore,
    feedback_path_for_registry,
)

FP = "a" * 64


def vec(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def make_observation(observation_id="obs-1") -> FeedbackObservation:
    return FeedbackObservation(
        id=observation_id,
        source_path="C:/camera/IMG_0001.JPG",
        media_kind="image",
        box_index=0,
        frame_index=None,
        timestamp_seconds=None,
        bbox=(1.0, 2.0, 30.0, 40.0),
        camera_id="camera-a",
        captured_at=datetime(2026, 9, 9, 10, 0, 0),
        predicted_species="盘羊",
        best_known_species="盘羊",
        accepted=True,
        prediction_source="base",
        registry_id=None,
        known_score=0.8,
        threshold=0.31,
        embedding=vec(0),
    )


def test_feedback_path_is_registry_sibling(tmp_path):
    assert feedback_path_for_registry(tmp_path / "registry.sqlite3") == tmp_path / "feedback.sqlite3"


def test_persist_observation_round_trips_without_public_embedding(tmp_path):
    store = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("盘羊", "家牛"),
        threshold=0.31,
    )
    observation = make_observation()
    assert store.persist_observation(observation) == "obs-1"
    loaded = store.get_observation("obs-1")
    assert loaded.id == "obs-1"
    assert loaded.embedding.shape == (768,)
    assert np.allclose(loaded.embedding, vec(0))
    store.close()


def test_feedback_store_rejects_fingerprint_mismatch(tmp_path):
    path = tmp_path / "feedback.sqlite3"
    HumanFeedbackStore(path, model_fingerprint=FP, checkpoint_classes=("盘羊",), threshold=0.31).close()
    with pytest.raises(ValueError, match="fingerprint"):
        HumanFeedbackStore(path, model_fingerprint="b" * 64, checkpoint_classes=("盘羊",), threshold=0.31)


def test_feedback_store_enables_wal_and_foreign_keys(tmp_path):
    path = tmp_path / "feedback.sqlite3"
    store = HumanFeedbackStore(path, model_fingerprint=FP, checkpoint_classes=("盘羊",), threshold=0.31)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    store.close()
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
pytest tests/test_dinov3_feedback_store.py -q
```

Expected: collection/import fails because `system.dinov3.feedback` does not exist.

- [ ] **Step 3: Implement schema, serialization, and observation persistence**

```python
# system/dinov3/feedback.py - required public skeleton
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import numpy as np

from .simple_shot import normalize_embedding

DINO_DIM = 768


def feedback_path_for_registry(registry_path: str | Path) -> Path:
    return Path(registry_path).expanduser().resolve().with_name("feedback.sqlite3")


@dataclass(frozen=True)
class FeedbackObservation:
    id: str
    source_path: str
    media_kind: str
    box_index: int
    frame_index: int | None
    timestamp_seconds: float | None
    bbox: tuple[float, float, float, float]
    camera_id: str
    captured_at: datetime | None
    predicted_species: str
    best_known_species: str | None
    accepted: bool
    prediction_source: str
    registry_id: int | None
    known_score: float
    threshold: float
    embedding: np.ndarray


class HumanFeedbackStore:
    def __init__(self, path, *, model_fingerprint, checkpoint_classes, threshold):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_fingerprint = str(model_fingerprint)
        self.checkpoint_classes = tuple(checkpoint_classes)
        self.threshold = float(threshold)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def close(self):
        self._conn.close()
```

The schema must implement the four tables from the spec: `observations`, `feedback_operations`, `human_feedback`, `species_learning_state`, plus `overlay_prototypes` and metadata. Store normalized float32 embeddings as little-endian blobs; `get_observation()` reconstructs them only for backend use.

- [ ] **Step 4: Run the focused test and verify GREEN**

```bash
pytest tests/test_dinov3_feedback_store.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the storage foundation**

```bash
git add system/dinov3/feedback.py tests/test_dinov3_feedback_store.py
git commit -m "feat: add DINOv3 human feedback store"
```

---

### Task 2: Implement feedback semantics, revision history, and operation revert

**Files:**
- Modify: `system/dinov3/feedback.py`
- Create: `tests/test_dinov3_feedback_semantics.py`

**Interfaces:**
- Consumes: `HumanFeedbackStore`, `FeedbackObservation` from Task 1.
- Produces: `HumanFeedbackStore.record_feedback(observation_id: str, *, operation_id: str, action: str, confirmed_species: str | None) -> FeedbackRecord`
- Produces: `HumanFeedbackStore.revert_operation(operation_id: str) -> set[str]` returning affected checkpoint species.
- Produces: `HumanFeedbackStore.active_feedback(observation_id: str) -> FeedbackRecord | None`.

- [ ] **Step 1: Write failing feedback-semantics tests**

```python
# tests/test_dinov3_feedback_semantics.py
from tests.test_dinov3_feedback_store import FP, make_observation
from system.dinov3.feedback import HumanFeedbackStore


def test_correct_creates_positive_only(tmp_path):
    store = HumanFeedbackStore(tmp_path / "feedback.sqlite3", model_fingerprint=FP, checkpoint_classes=("盘羊", "家牛"), threshold=0.31)
    store.persist_observation(make_observation())
    record = store.record_feedback("obs-1", operation_id="op-1", action="correct", confirmed_species="盘羊")
    assert record.positive_species == "盘羊"
    assert record.hard_negative_species is None
    store.close()


def test_accepted_correction_creates_positive_and_hard_negative(tmp_path):
    store = HumanFeedbackStore(tmp_path / "feedback.sqlite3", model_fingerprint=FP, checkpoint_classes=("盘羊", "家牛"), threshold=0.31)
    store.persist_observation(make_observation())
    record = store.record_feedback("obs-1", operation_id="op-1", action="update", confirmed_species="家牛")
    assert record.positive_species == "家牛"
    assert record.hard_negative_species == "盘羊"
    store.close()


def test_rejected_unknown_correction_has_no_species_hard_negative(tmp_path):
    obs = make_observation()
    obs = obs.__class__(**{**obs.__dict__, "predicted_species": "Unknown", "best_known_species": "盘羊", "accepted": False})
    store = HumanFeedbackStore(tmp_path / "feedback.sqlite3", model_fingerprint=FP, checkpoint_classes=("盘羊", "家牛"), threshold=0.31)
    store.persist_observation(obs)
    record = store.record_feedback("obs-1", operation_id="op-1", action="update", confirmed_species="家牛")
    assert record.positive_species == "家牛"
    assert record.hard_negative_species is None
    store.close()


def test_revert_restores_superseded_feedback(tmp_path):
    store = HumanFeedbackStore(tmp_path / "feedback.sqlite3", model_fingerprint=FP, checkpoint_classes=("盘羊", "家牛"), threshold=0.31)
    store.persist_observation(make_observation())
    store.record_feedback("obs-1", operation_id="op-1", action="correct", confirmed_species="盘羊")
    store.record_feedback("obs-1", operation_id="op-2", action="update", confirmed_species="家牛")
    assert store.active_feedback("obs-1").confirmed_species == "家牛"
    assert store.revert_operation("op-2") == {"盘羊", "家牛"}
    assert store.active_feedback("obs-1").confirmed_species == "盘羊"
    store.close()
```

- [ ] **Step 2: Run semantics tests and verify RED**

```bash
pytest tests/test_dinov3_feedback_semantics.py -q
```

Expected: failures because feedback-record/revert APIs are not implemented.

- [ ] **Step 3: Implement explicit feedback classification and append-preserving revisions**

```python
@dataclass(frozen=True)
class FeedbackRecord:
    id: int
    observation_id: str
    operation_id: str
    feedback_type: str
    predicted_species: str
    confirmed_species: str | None
    positive_species: str | None
    hard_negative_species: str | None
    active: bool


def _feedback_semantics(observation: FeedbackObservation, action: str, confirmed_species: str | None):
    if action == "correct":
        if observation.predicted_species == "Unknown" or not observation.accepted:
            raise ValueError("Rejected observations require an explicit confirmed species")
        return "correct", observation.predicted_species, None
    if action == "update":
        if not confirmed_species:
            raise ValueError("confirmed species is required")
        hard_negative = (
            observation.predicted_species
            if observation.accepted and observation.predicted_species != confirmed_species
            else None
        )
        return "corrected", confirmed_species, hard_negative
    if action == "empty":
        return "empty", None, None
    if action == "unverified":
        return "ignored", None, None
    raise ValueError(f"Unsupported feedback action: {action}")
```

`record_feedback()` must reject a corrected species not present in `checkpoint_classes`; new species remain owned by `SpeciesRegistry`. Each new feedback revision deactivates the previous row, stores `supersedes_id`, and records/creates the operation row in one transaction. `revert_operation()` marks the operation reverted, deactivates its active revisions, and reactivates the nearest non-reverted superseded revision.

- [ ] **Step 4: Run storage + semantics tests**

```bash
pytest tests/test_dinov3_feedback_store.py tests/test_dinov3_feedback_semantics.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit feedback semantics**

```bash
git add system/dinov3/feedback.py tests/test_dinov3_feedback_semantics.py
git commit -m "feat: record reversible DINOv3 feedback"
```

---

### Task 3: Add independent-event aggregation, prototype generations, and quality gates

**Files:**
- Modify: `system/dinov3/feedback.py`
- Create: `tests/test_dinov3_feedback_learning.py`

**Interfaces:**
- Consumes: active positive/hard-negative feedback from Task 2.
- Produces: `HumanFeedbackStore.learning_state(species: str) -> SpeciesLearningState`
- Produces: `HumanFeedbackStore.recompute_species(species: str, feature_center: np.ndarray) -> SpeciesLearningState`
- Produces: `HumanFeedbackStore.prototype_bank(feature_center: np.ndarray) -> PrototypeBank`
- Produces: `SpeciesLearningState.status` in `collecting|provisional|confirmed|mature` and `needs_review`.

- [ ] **Step 1: Write failing lifecycle tests**

```python
# tests/test_dinov3_feedback_learning.py
from datetime import timedelta
import numpy as np
from system.dinov3.feedback import HumanFeedbackStore
from tests.test_dinov3_feedback_store import FP, make_observation


def add_positive(store, index, *, camera, minutes):
    obs = make_observation(f"obs-{index}")
    obs = obs.__class__(**{
        **obs.__dict__,
        "camera_id": camera,
        "captured_at": obs.captured_at + timedelta(minutes=minutes),
        "source_path": f"C:/camera/{index}.JPG",
    })
    store.persist_observation(obs)
    store.record_feedback(obs.id, operation_id=f"op-{index}", action="correct", confirmed_species="盘羊")


def test_four_events_create_assistive_provisional_overlay(tmp_path):
    store = HumanFeedbackStore(tmp_path / "feedback.sqlite3", model_fingerprint=FP, checkpoint_classes=("盘羊", "家牛"), threshold=0.31)
    for i in range(4):
        add_positive(store, i, camera="cam-a", minutes=i * 31)
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    bank = store.prototype_bank(np.zeros(768, dtype=np.float32))
    assert state.status == "provisional"
    assert len(bank.provisional) == 1
    assert bank.formal == ()
    store.close()


def test_ten_events_two_cameras_promote_when_quality_gate_passes(tmp_path):
    store = HumanFeedbackStore(tmp_path / "feedback.sqlite3", model_fingerprint=FP, checkpoint_classes=("盘羊", "家牛"), threshold=0.31)
    for i in range(10):
        add_positive(store, i, camera=f"cam-{i % 2}", minutes=i * 31)
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    assert state.status == "confirmed"
    assert state.positive_coverage >= 0.80
    assert 1 <= len(store.prototype_bank(np.zeros(768, dtype=np.float32)).formal) <= 2
    store.close()


def test_twenty_events_three_cameras_reach_mature(tmp_path):
    store = HumanFeedbackStore(tmp_path / "feedback.sqlite3", model_fingerprint=FP, checkpoint_classes=("盘羊", "家牛"), threshold=0.31)
    for i in range(20):
        add_positive(store, i, camera=f"cam-{i % 3}", minutes=i * 31)
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    assert state.status == "mature"
    assert 1 <= state.prototype_count <= 3
    store.close()
```

Add a second fixture vector for hard negatives and a test asserting a >20% hard-negative false-accept rate blocks Provisional -> Confirmed. Add a test where a Confirmed generation later fails and `active_generation` remains unchanged while `needs_review=True`.

- [ ] **Step 2: Run lifecycle tests and verify RED**

```bash
pytest tests/test_dinov3_feedback_learning.py -q
```

Expected: failures because event aggregation and prototype bank are not implemented.

- [ ] **Step 3: Implement deterministic event grouping and generation activation**

```python
EVENT_GAP_SECONDS = 1800


def _event_key(species: str, camera_id: str, captured_at: datetime | None, source_path: str) -> str:
    if captured_at is None:
        return f"missing:{species}:{camera_id}:{Path(source_path).resolve()}"
    bucket = int(captured_at.timestamp())
    return f"timed:{species}:{camera_id}:{bucket}"
```

Do not use the simple timestamp bucket above as the final grouping algorithm; implement the same gap-based sequential grouping semantics as `SpeciesRegistry`: sort by camera/time, merge only when the next observation is `<1800` seconds after the current event end, and derive source-path identity when timestamps are missing. Aggregate each event as normalized mean of normalized embeddings.

For prototype computation, subtract the supplied 768-d `feature_center`, call existing `deterministic_k_means(centered, max_k=limit)`, evaluate `positive_coverage` and `hard_negative_false_accept_rate` at `self.threshold`, then atomically activate or reject the new generation according to the spec.

- [ ] **Step 4: Run feedback store/semantics/lifecycle tests**

```bash
pytest tests/test_dinov3_feedback_store.py tests/test_dinov3_feedback_semantics.py tests/test_dinov3_feedback_learning.py -q
```

Expected: all tests pass, including 29:59/30:00 boundary, hard-negative gate, and last-known-good generation preservation.

- [ ] **Step 5: Commit learning lifecycle**

```bash
git add system/dinov3/feedback.py tests/test_dinov3_feedback_learning.py
git commit -m "feat: build DINOv3 feedback prototypes"
```

---

### Task 4: Compose the human-feedback overlay into DINOv3 runtime/classification

**Files:**
- Modify: `system/dinov3/classifier.py`
- Modify: `system/dinov3/runtime.py`
- Modify: `system/dinov3/__init__.py`
- Modify: `tests/test_dinov3_overlay_bank.py`
- Create: `tests/test_dinov3_feedback_runtime.py`

**Interfaces:**
- Consumes: `HumanFeedbackStore.prototype_bank(feature_center)`.
- Changes: `DinoV3Classifier(checkpoint, *, encoder=None, registry=None, feedback=None)`.
- Changes: `DinoV3Runtime` adds `feedback: HumanFeedbackStore` and `owns_feedback: bool`.
- Produces: runtime classifier effective bank = checkpoint base + feedback formal + registry formal; provisional = feedback provisional + registry provisional.

- [ ] **Step 1: Write failing composition tests**

```python
# tests/test_dinov3_feedback_runtime.py
import numpy as np
from system.dinov3.checkpoint import validate_checkpoint
from system.dinov3.classifier import DinoV3Classifier
from system.dinov3.prototype_bank import PrototypeBank, PrototypeRecord
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload


class FakeFeedback:
    def prototype_bank(self, _feature_center):
        return PrototypeBank(
            formal=(PrototypeRecord(species="盘羊", embedding=np.eye(1, 768, 7, dtype=np.float32)[0], source="feedback", registration_status="confirmed"),),
            provisional=(),
        )


def test_classifier_merges_feedback_formal_records_with_base():
    checkpoint = validate_checkpoint(make_multi_prototype_payload(threshold=0.3))
    classifier = DinoV3Classifier(checkpoint, feedback=FakeFeedback())
    bank = classifier._effective_bank()
    assert any(record.source == "base" for record in bank.formal)
    assert any(record.source == "feedback" for record in bank.formal)
```

Also add a runtime test monkeypatching `default_dinov3_state_root()` and verifying `feedback.sqlite3` is created beside the fingerprint registry and shares the checkpoint fingerprint.

- [ ] **Step 2: Run focused composition tests and verify RED**

```bash
pytest tests/test_dinov3_feedback_runtime.py tests/test_dinov3_overlay_bank.py -q
```

Expected: failures because `feedback` is not accepted/composed.

- [ ] **Step 3: Implement three-source bank composition and runtime ownership**

```python
# classifier.py target shape
class DinoV3Classifier:
    def __init__(self, checkpoint, *, encoder=None, registry=None, feedback=None):
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.registry = registry
        self.feedback = feedback
        # existing initialization unchanged

    def _effective_bank(self) -> PrototypeBank:
        registry_bank = self.registry.prototype_bank(self._feature_center) if self.registry is not None else PrototypeBank(())
        feedback_bank = self.feedback.prototype_bank(self._feature_center) if self.feedback is not None else PrototypeBank(())
        return PrototypeBank(
            formal=self._base_records + tuple(feedback_bank.formal) + tuple(registry_bank.formal),
            provisional=tuple(feedback_bank.provisional) + tuple(registry_bank.provisional),
        )
```

`load_dinov3_model()` must open `HumanFeedbackStore(feedback_path_for_registry(registry.path), model_fingerprint=checkpoint.fingerprint, checkpoint_classes=checkpoint.classes, threshold=checkpoint.threshold)` and pass it to the classifier. Preserve caller-injected registry support and close owned feedback at the same lifecycle boundary as owned registry.

- [ ] **Step 4: Run DINOv3 classifier/overlay/runtime tests**

```bash
pytest tests/test_dinov3_multi_prototype_classifier.py tests/test_dinov3_overlay_bank.py tests/test_dinov3_feedback_runtime.py -q
```

Expected: all pass and existing new-species overlay behavior is unchanged.

- [ ] **Step 5: Commit runtime composition**

```bash
git add system/dinov3/classifier.py system/dinov3/runtime.py system/dinov3/__init__.py tests/test_dinov3_overlay_bank.py tests/test_dinov3_feedback_runtime.py
git commit -m "feat: compose DINOv3 feedback overlay"
```

---

### Task 5: Give every DINOv3 box a persistent observation id and persist observations after inference

**Files:**
- Modify: `system/dinov3/classifier.py`
- Modify: `system/image_processor.py`
- Create: `system/backend/dinov3_feedback_service.py`
- Modify: `system/backend/services.py`
- Modify: `tests/test_dinov3_image_processor.py`
- Modify: `tests/test_dinov3_backend_integration.py`
- Create: `tests/test_dinov3_feedback_persistence.py`

**Interfaces:**
- Changes: `DinoV3Observation` adds `observation_id: str` and `best_known_species: str`.
- Candidate metadata written by `ImageProcessor` includes `observation_id`, `registry_id`, `predicted_species`, `accepted`, and existing DINO metadata.
- Produces: `persist_runtime_observations(detector, paths, items, input_path, *, source_paths=None) -> None` in `dinov3_feedback_service.py`.
- Consumes: existing `_persist_dinov3_observations()` call sites in `services.py`; replace their internals with the new service while preserving existing registry accumulation.

- [ ] **Step 1: Write failing observation-id tests**

```python
# add to tests/test_dinov3_image_processor.py

def test_dinov3_candidate_and_runtime_observation_share_observation_id(fake_processor_setup):
    processor, result = fake_processor_setup
    processor.detect_batch_species(["image.jpg"])
    observations = processor.drain_dinov3_observations()
    candidate = result.candidates_data[0][0]
    assert candidate["observation_id"]
    assert observations[0].observation_id == candidate["observation_id"]
    assert observations[0].best_known_species
```

```python
# tests/test_dinov3_feedback_persistence.py

def test_backend_persists_known_and_rejected_dinov3_observations(tmp_path, fake_detector):
    # arrange detector.dinov3_feedback and drain output with one accepted + one rejected observation
    persist_runtime_observations(fake_detector, [tmp_path / "a.jpg"], [fake_item], tmp_path)
    assert fake_detector.dinov3_feedback.get_observation("obs-known").accepted is True
    assert fake_detector.dinov3_feedback.get_observation("obs-unknown").accepted is False
```

Add a fast-video test asserting persisted `source_path` is the original video while `frame_index`/timestamp distinguish observations.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
pytest tests/test_dinov3_image_processor.py tests/test_dinov3_feedback_persistence.py -q
```

Expected: failures because observation ids/feedback persistence do not exist.

- [ ] **Step 3: Generate ids in ImageProcessor and persist all DINO observations in backend**

```python
# image_processor.py inside DINO crop loop
observation_id = uuid.uuid4().hex
candidate = prediction.as_candidate(detection_confidence=det_conf)
candidate.update({
    "observation_id": observation_id,
    "predicted_species": prediction.species,
})
self._dinov3_observations.append(
    DinoV3Observation(
        observation_id=observation_id,
        best_known_species=prediction.best_known_species,
        # existing fields preserved
    )
)
```

`persist_runtime_observations()` must use the same drained observation once for both responsibilities: persist every DINO observation to `feedback.sqlite3`, and preserve existing base-rejected/registry-matched accumulation into `SpeciesRegistry`. Do not drain twice. The existing backend call sites already sit immediately after image/fast-video inference; preserve original-video path substitution for sampled video.

- [ ] **Step 4: Run image/backend integration tests**

```bash
pytest tests/test_dinov3_image_processor.py tests/test_dinov3_backend_integration.py tests/test_dinov3_feedback_persistence.py -q
```

Expected: all pass; registry accumulation tests remain green and every DINO box has an observation id in serialized small metadata.

- [ ] **Step 5: Commit observation persistence**

```bash
git add system/dinov3/classifier.py system/image_processor.py system/backend/dinov3_feedback_service.py system/backend/services.py tests/test_dinov3_image_processor.py tests/test_dinov3_backend_integration.py tests/test_dinov3_feedback_persistence.py
git commit -m "feat: persist DINOv3 box observations"
```

---

### Task 6: Add backend box-feedback and revert APIs

**Files:**
- Modify: `system/backend/models.py`
- Modify: `system/backend/dinov3_feedback_service.py`
- Create: `system/backend/dinov3_feedback_api.py`
- Modify: `system/backend/main.py`
- Create: `tests/test_dinov3_feedback_api.py`

**Interfaces:**
- Produces request `DinoV3BoxFeedbackRequest` with `input_path`, `file_path`, `classification_model_path`, `observation_id`, `action`, optional `species_name`, and `feedback_operation_id`.
- Produces request `DinoV3FeedbackRevertRequest` with `classification_model_path` and `feedback_operation_id`.
- Produces endpoint `POST /api/dinov3/feedback/box` returning `{item: DetectionItem, operation_id: str, affected_species: list[str]}`.
- Produces endpoint `POST /api/dinov3/feedback/revert` returning `{operation_id: str, affected_species: list[str]}`.

- [ ] **Step 1: Write failing FastAPI tests**

```python
# tests/test_dinov3_feedback_api.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
from system.backend.dinov3_feedback_api import dinov3_feedback_router


def test_box_feedback_requires_matching_observation(monkeypatch):
    app = FastAPI()
    app.include_router(dinov3_feedback_router())
    client = TestClient(app)
    response = client.post("/api/dinov3/feedback/box", json={
        "input_path": "C:/camera",
        "file_path": "C:/camera/a.jpg",
        "classification_model_path": "model.neri.json",
        "observation_id": "missing",
        "action": "correct",
        "feedback_operation_id": "op-1",
    })
    assert response.status_code in {400, 404}
```

Add service-mocked success tests for `correct`, `update`, `empty`, and revert; assert returned payload never contains an `embedding` key.

- [ ] **Step 2: Run API tests and verify RED**

```bash
pytest tests/test_dinov3_feedback_api.py -q
```

Expected: import/route failures.

- [ ] **Step 3: Implement request models, service validation, and router**

```python
# system/backend/dinov3_feedback_api.py
router = APIRouter(prefix="/api/dinov3/feedback", tags=["dinov3-feedback"])

@router.post("/box")
def box_feedback(request: DinoV3BoxFeedbackRequest):
    try:
        return apply_box_feedback(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="DINOv3 observation not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/revert")
def revert_feedback(request: DinoV3FeedbackRevertRequest):
    try:
        return revert_feedback_operation(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="DINOv3 feedback operation not found") from exc
```

`apply_box_feedback()` must verify model fingerprint + source path, update the corresponding persisted detection box ecological species for `update/empty`, recompute file-level species/count from boxes, save the detection DB transaction, then write feedback and recompute affected known-species learning state. On feedback-store failure return an explicit error; never claim model improvement silently.

- [ ] **Step 4: Run API + feedback tests**

```bash
pytest tests/test_dinov3_feedback_api.py tests/test_dinov3_feedback_semantics.py tests/test_dinov3_feedback_learning.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit feedback API**

```bash
git add system/backend/models.py system/backend/dinov3_feedback_service.py system/backend/dinov3_feedback_api.py system/backend/main.py tests/test_dinov3_feedback_api.py
git commit -m "feat: add DINOv3 box feedback API"
```

---

### Task 7: Add safe file/batch automatic feedback and reversible validation operation ids

**Files:**
- Modify: `system/backend/models.py`
- Modify: `system/backend/services.py`
- Modify: `system/backend/main_core.py` only if route response typing requires it; otherwise leave unchanged.
- Create: `tests/test_dinov3_feedback_validation_bridge.py`

**Interfaces:**
- Extends `ValidationMarkRequest` and `ValidationBatchMarkRequest` with optional `classification_model_path: str | None` and `feedback_operation_id: str | None`.
- Existing callers without those fields remain valid.
- `mark_validation_item()` / `mark_validation_items()` auto-feed only when exactly one learnable DINO observation exists for a file.
- A batch operation uses one shared operation id for all eligible files and returns normal `DetectionItem` payloads unchanged.

- [ ] **Step 1: Write failing single/multi-box eligibility tests**

```python
# tests/test_dinov3_feedback_validation_bridge.py

def test_single_box_file_mark_correct_creates_feedback(monkeypatch, request_factory):
    request = request_factory(file_paths=["a.jpg"], action="correct", operation_id="op-1")
    monkeypatch.setattr(service_module, "_learnable_observations_for_file", lambda *args, **kwargs: [fake_observation("obs-1")])
    service_module.mark_validation_items(request)
    assert recorded_feedback == [("obs-1", "op-1", "correct", None)]


def test_multi_box_file_mark_does_not_auto_feed(monkeypatch, request_factory):
    request = request_factory(file_paths=["a.jpg"], action="correct", operation_id="op-1")
    monkeypatch.setattr(service_module, "_learnable_observations_for_file", lambda *args, **kwargs: [fake_observation("obs-1"), fake_observation("obs-2")])
    service_module.mark_validation_items(request)
    assert recorded_feedback == []
```

Add `update` and `empty` coverage and a test that an omitted `classification_model_path` preserves old validation behavior without opening feedback state.

- [ ] **Step 2: Run validation bridge tests and verify RED**

```bash
pytest tests/test_dinov3_feedback_validation_bridge.py -q
```

Expected: request fields/helpers do not exist.

- [ ] **Step 3: Implement conservative eligibility and operation grouping**

```python
def _eligible_auto_feedback(observations):
    learnable = [obs for obs in observations if obs.id]
    return learnable[0] if len(learnable) == 1 else None
```

For each marked file, write ecological validation first using the existing bulk DB path. Only after that succeeds, resolve the selected DINO model and exactly-one observation rule, record feedback with the shared operation id, and recompute affected species. If feedback fails, surface a DINO-specific error and leave checkpoint assets untouched. Preserve the current validation DB bulk transaction behavior.

- [ ] **Step 4: Run existing backend integration plus new bridge tests**

```bash
pytest tests/test_dinov3_feedback_validation_bridge.py tests/test_dinov3_backend_integration.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit validation bridge**

```bash
git add system/backend/models.py system/backend/services.py tests/test_dinov3_feedback_validation_bridge.py
git commit -m "feat: bridge validation into DINOv3 feedback"
```

---

### Task 8: Expose DINOv3 feedback metadata and box APIs to Flutter

**Files:**
- Modify: `frontend/lib/src/models/job.dart`
- Modify: `frontend/lib/src/api_client_core.dart`
- Create: `frontend/lib/src/models/dinov3_feedback.dart`
- Create: `frontend/test/dinov3_feedback_api_test.dart`

**Interfaces:**
- `DetectionBox` gains nullable `observationId`, `registryId`, `predictedSpecies`, `feedbackStatus`.
- Produces `DinoV3BoxFeedbackResult` and `DinoV3FeedbackRevertResult`.
- Produces `NeriApiClient.markDinoV3BoxFeedback(...)` and `NeriApiClient.revertDinoV3Feedback(...)`.

- [ ] **Step 1: Write failing Dart model/API tests**

```dart
// frontend/test/dinov3_feedback_api_test.dart
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/testing.dart';
import 'package:http/http.dart' as http;
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';

void main() {
  test('DetectionBox decodes DINO feedback metadata', () {
    final box = DetectionBox.fromJson({
      'species': '盘羊',
      'bbox': [1, 2, 3, 4],
      'observation_id': 'obs-1',
      'registry_id': 7,
      'predicted_species': '家牛',
      'feedback_status': 'corrected',
    });
    expect(box.observationId, 'obs-1');
    expect(box.registryId, 7);
    expect(box.predictedSpecies, '家牛');
  });

  test('box feedback posts exact observation and operation ids', () async {
    late Map<String, dynamic> body;
    final client = NeriApiClient(httpClient: MockClient((request) async {
      body = jsonDecode(request.body) as Map<String, dynamic>;
      return http.Response(jsonEncode({'item': {'filename':'a.jpg','path':'a.jpg','file_type':'jpg'}, 'operation_id':'op-1','affected_species':['盘羊']}), 200);
    }));
    await client.markDinoV3BoxFeedback(
      inputPath: 'C:/camera',
      filePath: 'C:/camera/a.jpg',
      classificationModelPath: 'model.neri.json',
      observationId: 'obs-1',
      action: 'correct',
      feedbackOperationId: 'op-1',
    );
    expect(body['observation_id'], 'obs-1');
    expect(body['feedback_operation_id'], 'op-1');
  });
}
```

- [ ] **Step 2: Run the Dart test and verify RED**

```bash
cd frontend && flutter test test/dinov3_feedback_api_test.dart
```

Expected: compile failures for missing fields/methods.

- [ ] **Step 3: Implement Flutter DTOs and API methods**

```dart
class DinoV3BoxFeedbackResult {
  const DinoV3BoxFeedbackResult({required this.item, required this.operationId, required this.affectedSpecies});
  final DetectionItem item;
  final String operationId;
  final List<String> affectedSpecies;
}
```

`markDinoV3BoxFeedback()` POSTs `/api/dinov3/feedback/box`; `revertDinoV3Feedback()` POSTs `/api/dinov3/feedback/revert`; both use `_ensureSuccess` and decode server-authoritative state.

- [ ] **Step 4: Run Dart API/model tests**

```bash
cd frontend && flutter test test/dinov3_feedback_api_test.dart test/dinov3_registry_test.dart
```

Expected: all pass.

- [ ] **Step 5: Commit Flutter feedback API foundation**

```bash
git add frontend/lib/src/models/job.dart frontend/lib/src/models/dinov3_feedback.dart frontend/lib/src/api_client_core.dart frontend/test/dinov3_feedback_api_test.dart
git commit -m "feat: expose DINOv3 feedback to Flutter"
```

---

### Task 9: Add box hit-testing, selected-box highlighting, feedback controls, batch messaging, and undo

**Files:**
- Modify: `frontend/lib/src/widgets/detection_media_viewer.dart`
- Modify: `frontend/lib/src/screens/species_validation_screen.dart`
- Create: `frontend/test/dinov3_box_feedback_ui_test.dart`
- Create: `frontend/test/dinov3_box_hit_test_test.dart`

**Interfaces:**
- `DetectionMediaViewer` gains `selectedObservationId: String?` and `onDetectionBoxSelected: ValueChanged<DetectionBox?>?`.
- `SpeciesValidationScreen` stores current selected DINO box, generates one UUID-like feedback operation id per single/batch action, and records that id in `_MarkHistoryEntry` for revert.
- Multi-box whole-file/batch actions show learning-skipped summary while ecological validation still succeeds.

- [ ] **Step 1: Write failing box-selection and feedback-strip tests**

```dart
// frontend/test/dinov3_box_hit_test_test.dart
void main() {
  testWidgets('clicking a fitted DINO box selects its observation id', (tester) async {
    DetectionBox? selected;
    await tester.pumpWidget(MaterialApp(home: SizedBox(
      width: 400,
      height: 300,
      child: DetectionMediaViewer(
        item: testItem,
        visibleBoxes: [testDinoBox],
        showDetections: true,
        onOpenExternal: () {},
        onDetectionBoxSelected: (box) => selected = box,
      ),
    )));
    await tester.tapAt(const Offset(200, 150));
    expect(selected?.observationId, 'obs-1');
  });
}
```

```dart
// frontend/test/dinov3_box_feedback_ui_test.dart
void main() {
  testWidgets('selected DINO box shows correct/update/empty/ignore controls', (tester) async {
    await pumpValidationScreen(tester, item: itemWithOneDinoBox);
    await selectBox(tester);
    expect(find.text('检测框校验'), findsOneWidget);
    expect(find.text('正确'), findsWidgets);
    expect(find.text('修改物种'), findsOneWidget);
    expect(find.text('空 / 误检'), findsOneWidget);
    expect(find.text('不参与学习'), findsOneWidget);
  });
}
```

Add a video test using current-frame visible boxes only, a multi-box batch summary test, and an undo test asserting the stored `feedbackOperationId` is sent to `revertDinoV3Feedback()`.

- [ ] **Step 2: Run focused UI tests and verify RED**

```bash
cd frontend && flutter test test/dinov3_box_hit_test_test.dart test/dinov3_box_feedback_ui_test.dart
```

Expected: compile/widget failures because viewer callbacks and feedback controls do not exist.

- [ ] **Step 3: Implement hit-testing with the painter's `BoxFit.contain` transform**

```dart
Rect fittedMediaRect(Size viewport, Size media) {
  final scale = math.min(viewport.width / media.width, viewport.height / media.height);
  final size = Size(media.width * scale, media.height * scale);
  return Rect.fromLTWH(
    (viewport.width - size.width) / 2,
    (viewport.height - size.height) / 2,
    size.width,
    size.height,
  );
}
```

Use the same transform for painting and hit-testing. For videos, hit-test only `_currentBoxes(position)`. Selected boxes receive a distinct stroke/label emphasis without changing file/group selection.

- [ ] **Step 4: Implement feedback strip, operation ids, eligibility message, and undo bridge**

```dart
class _MarkHistoryEntry {
  _MarkHistoryEntry(Iterable<DetectionItem> items, {this.feedbackOperationId, Iterable<String>? quickMarkSpecies})
      : items = items.toList(), quickMarkSpecies = List<String>.from(quickMarkSpecies ?? const []);
  final List<DetectionItem> items;
  final List<String> quickMarkSpecies;
  final String? feedbackOperationId;
}
```

When a box is selected, `正确/修改物种/空/不参与学习` calls `markDinoV3BoxFeedback()` and merges the returned `DetectionItem`. File/batch actions continue through existing callbacks but pass a generated operation id/model path once MainWindow wiring is available; if the backend reports/skips multiple-box learning, show the summary without failing ecological validation. Undo reverts ecological validation first and then calls `revertDinoV3Feedback()` for that history entry's operation id.

- [ ] **Step 5: Run focused Flutter validation tests**

```bash
cd frontend && flutter test test/dinov3_box_hit_test_test.dart test/dinov3_box_feedback_ui_test.dart test/dinov3_feedback_api_test.dart
```

Expected: all pass.

- [ ] **Step 6: Run full Python and Flutter regressions**

```bash
pytest -q
cd frontend && flutter test
```

Expected: complete Python and Flutter suites pass; non-DINO validation behavior is unchanged.

- [ ] **Step 7: Verify checkpoint immutability in an integration test**

```bash
pytest tests/test_dinov3_feedback_learning.py -q
```

The test suite must include SHA-256 before/after assertions around feedback/prototype operations and pass without modifying the checkpoint file.

- [ ] **Step 8: Commit the box-level validation UI**

```bash
git add frontend/lib/src/widgets/detection_media_viewer.dart frontend/lib/src/screens/species_validation_screen.dart frontend/test/dinov3_box_hit_test_test.dart frontend/test/dinov3_box_feedback_ui_test.dart
git commit -m "feat: add DINOv3 box validation feedback"
```

---

## Plan Self-Review Notes

- Spec coverage: observation persistence, fingerprint isolation, feedback semantics, hard negatives, 1,800-second events, 4/10/20 lifecycle, quality gate, last-known-good generations, effective-bank composition, box UI, single-box auto-feedback, batch safety, undo, corruption/failure boundaries, no checkpoint mutation, and fast-video original-path behavior are all assigned to tasks above.
- No new online/cloud feedback synchronization is introduced.
- `SpeciesRegistry` remains independent; Task 5 explicitly preserves its existing accumulation while draining observations only once.
- Type consistency: the same `observation_id`, `feedback_operation_id`, `classification_model_path`, and `registry_id` names are used across Python storage, FastAPI DTOs, serialized detection metadata, Dart models, and UI history.
- Execution order matters: complete Tasks 1-7 before Flutter Tasks 8-9 so the UI consumes a stable backend contract.
