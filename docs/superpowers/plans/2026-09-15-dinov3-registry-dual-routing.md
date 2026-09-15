# DINOv3 Registry Dual Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement calibrated dual-signal Candidate routing, human-confirmed-only prototype updates, and the identified Registry correctness and scalability fixes.

**Architecture:** Classification owns the routing decision because it already has both calibrated signals. Runtime persistence may create Candidate evidence but cannot mutate an existing learned prototype; the human-feedback path remains the only update authority. Registry storage gains order-independent event merging, computed quality, serialized transactions, bulk reads, and lazy cluster APIs.

**Tech Stack:** Python 3, NumPy, SQLite, FastAPI, Flutter/Dart, pytest

**Spec:** `docs/superpowers/specs/2026-09-15-dinov3-registry-dual-routing-design.md`

## Global Constraints

- Base checkpoint files remain immutable.
- Existing Registry databases migrate additively in place.
- The same winning prototype supplies both cosine and squared-distance signals.
- Learned prototype updates require explicit human confirmation.
- Manual registration requires four independent events.

---

### Task 1: Dual routing and update authority

**Files:**
- Modify: `system/dinov3/rejection.py`
- Modify: `system/dinov3/classifier.py`
- Modify: `system/dinov3/multi_dual_classifier.py`
- Modify: `system/image_processor.py`
- Modify: `system/backend/dinov3_feedback_service.py`
- Test: `tests/test_dinov3_registry_routing.py`

**Interfaces:**
- Produces: `route_observation(cosine_score, squared_distance) -> str`
- Produces: `DinoV3Prediction.registry_action` and `DinoV3Observation.registry_action`

- [x] **Step 1: Write failing routing and persistence tests**

```python
assert config.route(cosine_score=.9, squared_distance=.1) == "update_prototype"
assert config.route(cosine_score=.9, squared_distance=.9) == "candidate"
assert config.route(cosine_score=.1, squared_distance=.1) == "candidate"
assert config.route(cosine_score=.1, squared_distance=.9) == "new_mode_candidate"
# Persisting an inference-only registry match must not add an event.
```

- [x] **Step 2: Run the focused tests and observe the missing routing contract**

Run: `python -m pytest -q tests/test_dinov3_registry_routing.py`
Expected: FAIL because routing metadata and the manual-update guard do not exist.

- [x] **Step 3: Implement routing metadata and block inference-only prototype updates**

```python
def route(self, *, cosine_score: float, squared_distance: float) -> str:
    cosine_pass = cosine_score >= self.cosine_threshold
    distance_pass = squared_distance <= self.squared_distance_threshold
    if cosine_pass and distance_pass:
        return "update_prototype"
    if not cosine_pass and not distance_pass:
        return "new_mode_candidate"
    return "candidate"
```

- [x] **Step 4: Run focused routing and backend persistence tests**

Run: `python -m pytest -q tests/test_dinov3_registry_routing.py tests/test_dinov3_backend_integration.py tests/test_dinov3_feedback_persistence.py`
Expected: PASS.

### Task 2: Event correctness and computed quality

**Files:**
- Modify: `system/dinov3/registry_impl.py`
- Modify: `system/dinov3/registry.py`
- Test: `tests/test_dinov3_registry.py`

**Interfaces:**
- Produces: order-independent event insertion/bridging
- Produces: persisted `cluster_purity` derived from prototype coverage

- [x] **Step 1: Add failing tests for out-of-order bridges and heterogeneous purity**

```python
# 00:00, 00:40, 00:20 becomes one transitive event.
assert entry.event_count == 1
# A deliberately heterogeneous four-event candidate fails the purity gate.
assert entry.conditions["cluster_purity"] is False
```

- [x] **Step 2: Run the focused Registry tests and observe both failures**

Run: `python -m pytest -q tests/test_dinov3_registry.py`
Expected: FAIL with three events and an always-passing purity condition.

- [x] **Step 3: Implement interval-neighbor merging and prototype coverage quality**

```python
purity = float(np.mean(np.max(cosine_similarity_matrix, axis=1) >= self.join_threshold))
start = min(captured_at, *matching_starts)
end = max(captured_at, *matching_ends)
```

- [x] **Step 4: Run Registry lifecycle, merge, and migration tests**

Run: `python -m pytest -q tests/test_dinov3_registry.py tests/test_dinov3_registry_management.py tests/test_dinov3_registry_same_name_merge.py tests/test_dinov3_registry_named_candidate_migration.py`
Expected: PASS.

### Task 3: SQLite concurrency and bulk reads

**Files:**
- Modify: `system/dinov3/registry_impl.py`
- Test: `tests/test_dinov3_registry_concurrency.py`
- Test: `tests/test_dinov3_registry.py`

**Interfaces:**
- Produces: one re-entrant writer lock per resolved Registry path
- Produces: bounded-query `list()` and prototype-only `match()`

- [x] **Step 1: Add failing concurrent-create and query-budget tests**

```python
assert len({entry.candidate_number for entry in concurrent_entries}) == len(concurrent_entries)
assert traced_query_count_for_list <= 4
assert traced_query_count_for_match <= 4
```

- [x] **Step 2: Run the focused tests and observe uniqueness/query-budget failures**

Run: `python -m pytest -q tests/test_dinov3_registry_concurrency.py tests/test_dinov3_registry.py`
Expected: FAIL because writers are not serialized and reads use N+1 queries.

- [x] **Step 3: Add WAL, busy timeout, path locks, indexes, aggregate summaries, and bulk prototype matching**

```sql
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
CREATE INDEX IF NOT EXISTS idx_events_registry_camera_time
ON events(registration_id,camera_id,started_at,ended_at);
```

- [x] **Step 4: Run the concurrency and Registry suites**

Run: `python -m pytest -q tests/test_dinov3_registry_concurrency.py tests/test_dinov3_registry.py tests/test_dinov3_registry_human_species_linking.py`
Expected: PASS.

### Task 4: Lazy catalog details and documentation alignment

**Files:**
- Modify: `system/dinov3/api.py`
- Modify: `system/backend/models.py`
- Modify: `frontend/lib/src/widgets/dinov3_registry_dialog.dart`
- Modify: `frontend/lib/src/api_client.dart`
- Modify: `docs/superpowers/specs/2026-09-09-dinov3-pipeline-registry-ui-design.md`
- Test: `tests/test_dinov3_registry_catalog_explain.py`
- Test: `frontend/test/dinov3_registry_test.dart`

**Interfaces:**
- Produces: `GET /api/dinov3/registry/{id}/clusters`
- Catalog summaries omit expensive clusters; selected-entry details load them lazily.

- [x] **Step 1: Add failing API and widget tests for lazy clusters**

```python
assert catalog_registry_entry["clusters"] == []
assert client.get(f"/api/dinov3/registry/{entry_id}/clusters").status_code == 200
```

- [x] **Step 2: Run focused backend/frontend tests and observe eager behavior**

Run: `python -m pytest -q tests/test_dinov3_registry_catalog_explain.py`
Expected: FAIL because the lazy endpoint does not exist.

- [x] **Step 3: Add the lazy endpoint, load selected clusters on demand, and align the four-event documentation**

```dart
final results = await Future.wait(cluster.exampleRefs.take(3).map(loadExample));
```

- [x] **Step 4: Run API, widget, and analyzer checks**

Run: `python -m pytest -q tests/test_dinov3_api.py tests/test_dinov3_registry_catalog_explain.py`
Run: `flutter test test/dinov3_registry_test.dart && flutter analyze`
Expected: PASS.

Result: Python API checks passed locally. Flutter SDK was unavailable locally, so widget/analyzer verification is delegated to GitHub Actions.

### Task 5: Broad verification

**Files:**
- Verify only

**Interfaces:**
- Consumes all prior tasks.

- [x] **Step 1: Run all DINOv3 Registry and dual-rejection tests**

Run: `python -m pytest -q tests/test_dinov3_registry*.py tests/test_dinov3_multi_dual*.py tests/test_dinov3_backend_integration.py tests/test_dinov3_feedback_persistence.py`
Expected: PASS.

- [x] **Step 2: Inspect the final diff for accidental checkpoint mutation or unrelated changes**

Run: `git diff --check && git diff --stat && git status --short`
Expected: no whitespace errors and only planned files changed.
