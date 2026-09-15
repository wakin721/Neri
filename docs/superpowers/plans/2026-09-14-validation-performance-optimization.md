# Validation and Processing Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove scale-dependent full scans and redundant persistence/UI work from validation and processing paths without changing recognition or learning semantics.

**Architecture:** Reuse the branch's `system/backend/preview_fast.py` as the indexed SQLite access boundary, add an indexed source-path projection to DINOv3 feedback observations, batch observation persistence while caching decoded still-image frames per persistence call, and add a conservative local-update fast path to the Flutter validation grouping cache. Every fast path falls back to the existing behavior when its preconditions cannot be proven.

**Tech Stack:** Python 3, `sqlite3`, FastAPI backend services, NumPy/OpenCV, Flutter/Dart, `pytest`, `flutter_test`.

**Spec:** `docs/superpowers/specs/2026-09-14-validation-performance-optimization-design.md`

## Global Constraints

- Keep the branch's `拒识/Unknown` semantics and serializer patch unchanged.
- Do not change recognition thresholds, checkpoints, Registry status rules, event-gap rules, default `imgsz`, augmentation, FP16, batch size, or classification thresholds.
- Existing `detections.db` and DINOv3 feedback databases must migrate in place. Do not introduce the deferred canonical-database redesign.
- Calls that intentionally request a complete index must retain legacy behavior.
- Performance tests must assert query/work shape, not wall-clock thresholds.
- Preserve first-database-wins semantics for duplicate detection/validation rows.
- Preserve per-observation non-fatal DINOv3 persistence behavior unless a step explicitly introduces an atomic unit and tests its rollback contract.

---

## Task 1: Generalize indexed detection and validation lookups

**Files:**
- Modify: `system/backend/preview_fast.py`
- Modify: `system/backend/runtime_patches.py`
- Modify: `system/backend/services.py`
- Modify: `tests/test_preview_fast.py`
- Modify: `tests/test_runtime_patches.py`

**Target interfaces:**

```python
# system/backend/preview_fast.py

def load_detection_index_for_filenames(
    db_paths: Iterable[Path],
    filenames: set[str],
) -> dict[str, dict[str, Any]]: ...


def load_validation_index_for_filenames(
    db_paths: Iterable[Path],
    filenames: set[str],
) -> dict[str, bool]: ...


def load_preview_indexes(
    db_paths: Iterable[Path],
    filenames: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]: ...
```

The runtime wrappers must keep the historical function signatures and only take the indexed path when a filename set is supplied.

- [ ] **Step 1.1: Add RED tests for targeted detection/validation lookup.**

Extend `tests/test_preview_fast.py` with separate tests for the two targeted helpers. Populate a database with requested rows plus unrelated rows, install a SQLite trace callback, and assert every targeted `SELECT` contains `WHERE image_filename IN`.

Include a 1,200-filename case to verify chunking. The expected result must still use the first matching database when the same base name or validation filename exists in multiple DBs.

- [ ] **Step 1.2: Add RED tests for runtime fallback behavior.**

Extend `tests/test_runtime_patches.py` with a fake services module exposing legacy `_load_detection_index` and `_load_validation_index` functions. Assert:

```python
patched_detection(roots, recursive=False, filenames={'a.jpg'})
```

uses the indexed helper, while:

```python
patched_detection(roots, recursive=False, filenames=None)
```

delegates exactly once to the original loader.

For validation, introduce an optional `filenames` argument in the compatibility wrapper and assert `filenames=None` remains the legacy full-index path.

- [ ] **Step 1.3: Run the focused tests and verify they fail for the expected missing interfaces.**

Run:

```bash
python -m pytest -q tests/test_preview_fast.py tests/test_runtime_patches.py
```

Expected RED reason: missing generalized helpers/wrappers, not unrelated import errors.

- [ ] **Step 1.4: Refactor `preview_fast.py` without changing preview semantics.**

Split the current combined query loop into the two target helpers. Keep `_SQLITE_IN_CHUNK = 400`, malformed JSON tolerance, and first-database-wins behavior. `load_preview_indexes()` should compose the two helpers rather than duplicate SQL.

For each DB, use parameterized SQL only:

```python
SELECT base_name, image_filename, detection_json
FROM detections
WHERE image_filename IN (...)
```

and:

```python
SELECT image_filename, is_validated
FROM validation
WHERE image_filename IN (...)
```

If a legacy DB lacks a requested table, catch `sqlite3.Error` for that DB and continue exactly as current preview loading does.

- [ ] **Step 1.5: Patch the historical service loaders conservatively.**

In `runtime_patches.install_runtime_patches()`:

1. Capture the original `_load_detection_index` and `_load_validation_index` once.
2. For `_load_detection_index`, if `filenames is None`, call the original unchanged. Otherwise resolve DB candidates with the service's `_candidate_detection_dbs_for_roots()` and call `load_detection_index_for_filenames()`.
3. For `_load_validation_index`, support a new optional `filenames` keyword in the wrapper; if omitted/`None`, call the original unchanged. Otherwise use `load_validation_index_for_filenames()`.
4. Mark wrappers with idempotence attributes so repeated `install_runtime_patches()` calls do not double-wrap them.

- [ ] **Step 1.6: Pass target filenames at single-item service call sites.**

In `system/backend/services.py`, change only the single-item/reload paths that already know the filename:

```python
validation_index = _load_validation_index(
    roots,
    recursive=False,
    filenames={path.name},
)
```

Apply the same targeted call in `_reload_validation_item()`. Do not change export paths that intentionally need complete validation state.

The existing processing startup and mark paths already supply `filenames` to `_load_detection_index`; after the runtime patch they automatically become indexed.

- [ ] **Step 1.7: Run GREEN tests.**

```bash
python -m pytest -q tests/test_preview_fast.py tests/test_runtime_patches.py
python -m compileall -q system/backend tests/test_preview_fast.py tests/test_runtime_patches.py
```

- [ ] **Step 1.8: Commit.**

Commit message:

```text
perf: index validation and detection lookups
```

---

## Task 2: Add an indexed source-path projection to DINOv3 feedback observations

**Files:**
- Modify: `system/dinov3/feedback.py`
- Modify: `tests/test_dinov3_feedback_store.py`
- Verify: `tests/test_dinov3_feedback_learning.py`

**Target interfaces:**

```python
# system/dinov3/feedback.py

def normalize_observation_source_path(value: str | Path) -> str: ...


class HumanFeedbackStore:
    def persist_observations(
        self,
        observations: Iterable[FeedbackObservation],
    ) -> tuple[str, ...]: ...

    def observations_for_source_path(
        self,
        source_path: str | Path,
    ) -> list[FeedbackObservation]: ...
```

- [ ] **Step 2.1: Add RED migration tests.**

In `tests/test_dinov3_feedback_store.py`, create a legacy feedback SQLite file manually with the old `observations(id,payload,embedding)` schema plus valid metadata. Insert:

- one valid historical payload containing `source_path`;
- one malformed JSON payload;
- one valid row for another source path.

Open it with `HumanFeedbackStore` and assert the intended post-migration contract:

```sql
PRAGMA table_info(observations)
```

contains `source_path`, and:

```sql
PRAGMA index_list(observations)
```

contains `idx_observations_source_path`.

Assert the valid row is backfilled with the normalized path, malformed payload remains `NULL`, and metadata contains a one-time migration marker such as `migration_observations_source_path_v1=1`.

Close/reopen and assert the stored projection remains unchanged, proving idempotence.

- [ ] **Step 2.2: Add RED indexed-lookup tests.**

Populate a migrated store with thousands of noise rows and one target row. Install a SQLite trace callback on `store._conn`, call:

```python
store.observations_for_source_path(target_path)
```

and assert the observation query contains:

```sql
WHERE source_path = ?
```

or an equivalent indexed predicate. Assert only the target `FeedbackObservation` is materialized.

Also assert `EXPLAIN QUERY PLAN` for the lookup references `idx_observations_source_path`; do not assert execution time.

- [ ] **Step 2.3: Run RED tests.**

```bash
python -m pytest -q tests/test_dinov3_feedback_store.py
```

Expected RED reason: source-path column/index/API absent.

- [ ] **Step 2.4: Implement stable source-path normalization.**

Add `normalize_observation_source_path()` near the feedback-store helpers. Its semantics must match the backend's current `_path_identity`: trim, `expanduser()`, resolve when possible, replace backslashes with `/`, and `casefold()`.

Do not import backend service code into `system/dinov3/feedback.py`; keep this normalization at the DINO store layer and later make the backend delegate to it.

- [ ] **Step 2.5: Implement idempotent schema migration/backfill.**

During `_init_schema()`:

1. Create fresh `observations` tables with `source_path TEXT` included.
2. For existing tables, inspect `PRAGMA table_info(observations)` and `ALTER TABLE ... ADD COLUMN source_path TEXT` only when missing.
3. Create `idx_observations_source_path` unconditionally with `IF NOT EXISTS`.
4. If migration marker is absent, select only rows whose `source_path IS NULL`, parse each payload once, normalize valid payload paths, update those rows, tolerate malformed JSON, set the marker, and commit.

Do not remove `source_path` from `payload`; it remains the compatibility representation.

- [ ] **Step 2.6: Implement batch observation persistence.**

Make `persist_observation()` delegate to:

```python
return self.persist_observations((observation,))[0]
```

`persist_observations()` should serialize all valid rows, execute one `executemany()` using `(id, source_path, payload, embedding)`, and commit once for that call. Preserve `INSERT OR REPLACE` behavior.

Factor row decoding into one private helper so both `get_observation()` and `observations_for_source_path()` construct observations identically.

- [ ] **Step 2.7: Run GREEN and learning-regression tests.**

```bash
python -m pytest -q \
  tests/test_dinov3_feedback_store.py \
  tests/test_dinov3_feedback_learning.py
```

- [ ] **Step 2.8: Commit.**

Commit message:

```text
perf: index DINOv3 observations by source path
```

---

## Task 3: Make file-level DINOv3 validation feedback use one indexed store context

**Files:**
- Modify: `system/backend/services.py`
- Modify: `system/backend/dinov3_feedback_service.py`
- Add: `tests/test_validation_performance.py`
- Verify: `tests/test_dinov3_backend_integration.py`

**Target backend interface:**

```python
# system/backend/dinov3_feedback_service.py

def apply_validation_feedback_batch(
    classification_model_path: str,
    paths: Sequence[Path],
    *,
    operation_id: str,
    action: str,
    confirmed_species: str | None,
    assign_registry_species: bool,
) -> int: ...
```

The function owns exactly one `_open_feedback_state()` call for the request.

- [ ] **Step 3.1: Add RED tests for lookup work shape.**

Create `tests/test_validation_performance.py`. Build a feedback store with a large set of noise observations plus two target files. Exercise the batch helper and trace SQL. Assert there is no query matching the old pattern:

```sql
SELECT id,payload FROM observations ORDER BY id
```

and that target lookup queries are constrained by the indexed `source_path` column.

- [ ] **Step 3.2: Add RED test for one feedback-store open per batch.**

Monkeypatch `_open_feedback_state` to count calls, submit multiple paths, and assert one open/close lifecycle for the entire batch rather than one per file.

Cover all existing feedback decisions:

- exactly one learnable observation -> feedback applied;
- zero observations -> skipped;
- more than one observation -> skipped;
- `action='update'` to an out-of-checkpoint species -> Registry assignment path;
- checkpoint species/correct/empty -> standard feedback path.

- [ ] **Step 3.3: Run RED tests.**

```bash
python -m pytest -q tests/test_validation_performance.py
```

- [ ] **Step 3.4: Replace `_learnable_observations_for_file()` internals with indexed lookup.**

Keep `_learnable_observations_for_file()` as a compatibility wrapper, but have it call `HumanFeedbackStore.observations_for_source_path()` rather than selecting all observations. If it opens its own store for legacy callers, ensure the new batch path does not call it repeatedly.

Make backend `_path_identity()` delegate to `normalize_observation_source_path()` to prevent future normalization drift.

- [ ] **Step 3.5: Implement `apply_validation_feedback_batch()`.**

Open feedback state once, then for each requested path:

```python
observations = feedback.observations_for_source_path(path)
```

Use the existing conservative rule: apply file-level learning only when there is exactly one learnable observation.

For standard feedback, call `feedback.record_feedback()` and accumulate `_affected_learning_species(record)` in a set. Recompute each affected species once after all files are recorded rather than after every file.

For Registry assignment, reuse the already-open feedback object and call `_assign_registry_species()`; do not reopen feedback state from `record_registry_species_feedback()`.

- [ ] **Step 3.6: Wire `mark_validation_items()` to the batch helper.**

In `services.mark_validation_items()` compute `assign_registry_species` from the existing checkpoint-membership logic, then call `apply_validation_feedback_batch()` once. Preserve the current `RuntimeError("DINOv3 自动反馈失败: ...")` surface on failure.

Add a `time.perf_counter()` measurement around the DINO lookup/feedback block and one total latency measurement for `mark_validation_items()`; log counts and elapsed seconds, not embeddings or image contents.

- [ ] **Step 3.7: Run GREEN and backend integration tests.**

```bash
python -m pytest -q \
  tests/test_validation_performance.py \
  tests/test_dinov3_backend_integration.py \
  tests/test_dinov3_feedback_store.py
```

- [ ] **Step 3.8: Commit.**

Commit message:

```text
perf: batch DINOv3 validation feedback lookup
```

---

## Task 4: Batch inference observation persistence and reuse still-image decodes

**Files:**
- Modify: `system/backend/dinov3_feedback_service.py`
- Modify: `system/dinov3/registry_examples.py`
- Modify: `system/dinov3/registry.py` only if required to pass the per-call cache through Registry example persistence
- Modify: `tests/test_dinov3_feedback_persistence.py`
- Add or modify: `tests/test_dinov3_registry_examples.py` if no existing focused example-cache test file exists
- Verify: `tests/test_dinov3_registry.py`

**Target example-cache behavior:**

A persistence call may hold a local cache keyed by normalized source path for still images. It must be discarded before `persist_runtime_observations()` returns. Video frames must remain lazy; do not cache `VideoCapture` frames across observations/batches unless the key includes an exact frame/timestamp and a test proves correctness.

- [ ] **Step 4.1: Add RED test for one feedback observation transaction per inference batch.**

Extend `tests/test_dinov3_feedback_persistence.py` with several runtime observations. Spy on `feedback.persist_observations()` and assert `persist_runtime_observations()` passes the valid observations as one batch instead of calling `persist_observation()` once per item.

The test must still verify observation IDs, bbox, camera/capture metadata, accepted/rejected state, and embeddings are persisted correctly.

- [ ] **Step 4.2: Add RED test for repeated still-image decode elimination.**

Create a temporary image and persist multiple observation examples from different boxes on that same still. Monkeypatch `cv2.imread` and assert the source still is decoded once within the persistence call while every requested crop file is written.

Add a control test showing two different source images cause two reads.

- [ ] **Step 4.3: Run RED tests.**

```bash
python -m pytest -q \
  tests/test_dinov3_feedback_persistence.py \
  tests/test_dinov3_registry.py
```

- [ ] **Step 4.4: Batch the feedback observation insert phase.**

Refactor `persist_runtime_observations()` into two phases:

1. Validate observation indices and build `(FeedbackObservation, source context)` records in memory.
2. Call `feedback.persist_observations(...)` once for all records with non-empty observation IDs.

After that, persist examples and Registry evidence per observation. Preserve the current per-observation exception handling for example/Registry work so one bad observation does not invalidate already-persisted inference evidence.

Add timing fields using `time.perf_counter()` for:

- observation-row preparation/write;
- feedback example persistence;
- Registry persistence.

- [ ] **Step 4.5: Add a per-call still-frame cache to registry example helpers.**

In `system/dinov3/registry_examples.py`, allow `_read_frame()` / `_encode_example()` / `_persist_example()` / `persist_feedback_observation_example()` and, if needed, `persist_registry_event_example()` to receive an optional mutable cache.

Only cache a successful `cv2.imread()` result for a still image. If `cv2.imread()` fails and the function falls back to `VideoCapture`, do not put that decoded frame into the still cache.

This avoids frame/timestamp aliasing for video while eliminating repeat full-image decode for several boxes on one photo.

- [ ] **Step 4.6: Thread one cache through each `persist_runtime_observations()` call.**

Create the cache at function entry after determining observations exist. Pass it to feedback example persistence. If Registry facade changes are needed, add only an optional cache keyword with a default of `None` so all existing Registry callers remain source-compatible.

- [ ] **Step 4.7: Run GREEN persistence/regression tests.**

```bash
python -m pytest -q \
  tests/test_dinov3_feedback_persistence.py \
  tests/test_dinov3_feedback_store.py \
  tests/test_dinov3_registry.py \
  tests/test_dinov3_feedback_learning.py
```

- [ ] **Step 4.8: Commit.**

Commit message:

```text
perf: batch DINOv3 observation persistence
```

---

## Task 5: Add a conservative incremental Flutter validation-cache path

**Files:**
- Add: `frontend/lib/src/utils/validation_cache_delta.dart`
- Modify: `frontend/lib/src/screens/species_validation_screen.dart`
- Add: `frontend/test/species_validation_incremental_cache_test.dart`
- Verify: `frontend/test/species_validation_group_expansion_test.dart`
- Verify: `frontend/test/dinov3_validation_selected_box_learning_test.dart`

**Target utility interface:**

```dart
class ValidationCacheDelta {
  static bool canAdoptPendingEcho<T>({
    required List<T> nextItems,
    required int cachedLength,
    required Map<String, int> cachedIndexByPath,
    required Set<String> pendingPaths,
    required String Function(T item) pathOf,
    required bool groupingSettingsUnchanged,
    required bool refreshVersionUnchanged,
  });
}
```

The utility must inspect only pending paths by index on the fast path. It must not compute a full path signature.

- [ ] **Step 5.1: Write RED unit tests for O(changed-items) echo validation.**

In `frontend/test/species_validation_incremental_cache_test.dart`, create 10,000 lightweight test items and a complete cached path->index map. Record how many times `pathOf` is invoked.

With one pending path, assert `canAdoptPendingEcho()` returns true and `pathOf` is called only for the pending entry, not all 10,000 items.

Add fallback tests for:

- different list length;
- pending path missing from the cached map;
- item at cached index now has a different path;
- grouping settings changed;
- refresh version changed;
- empty pending set.

- [ ] **Step 5.2: Run the new test and confirm RED.**

```bash
cd frontend
flutter test test/species_validation_incremental_cache_test.dart
```

- [ ] **Step 5.3: Implement `ValidationCacheDelta`.**

Keep the utility generic and independent from widgets. It should validate only the known pending indices. It must return false whenever correctness cannot be established cheaply.

- [ ] **Step 5.4: Track cached indices and pending parent echoes in the screen.**

In `_SpeciesValidationScreenState`, add:

```dart
Map<String, int> _bucketCacheItemIndexByPath = const <String, int>{};
final Set<String> _pendingValidationEchoPaths = <String>{};
```

Populate `_bucketCacheItemIndexByPath` during full cache rebuilds, when an O(N) pass is already occurring.

When `_markSelected`, `_markBatch`, or undo receives authoritative updated items, add those paths to `_pendingValidationEchoPaths` before the parent-provided `widget.items` echo arrives.

- [ ] **Step 5.5: Add a local cache adoption method.**

Implement a private method similar to:

```dart
bool _tryAdoptValidationEchoWithoutGlobalSignatures()
```

Its preconditions are:

1. grouping settings and `refreshVersion` match the cached values;
2. `ValidationCacheDelta.canAdoptPendingEcho(...)` returns true;
3. each pending item belongs to an existing cached group;
4. replacing pending items does not change the group's species bucket label;
5. replacing pending items does not newly complete a deferred group that must move between validated/unvalidated buckets.

For pending items only:

- replace references in `_bucketCacheItemByPath`;
- replace the matching item inside the owning group and bucket lists;
- invalidate the owning group's cached species label;
- update only affected group/bucket state.

If any precondition fails, clear/retain the appropriate dirty markers and fall through to the existing `_itemsPathSignature()` / `_itemsGroupingSignature()` rebuild path.

- [ ] **Step 5.6: Enter the fast path before global signatures.**

At the top of `_currentBuckets()`, after the existing identity fast path but before:

```dart
final pathSignature = _itemsPathSignature(widget.items);
final groupingSignature = _itemsGroupingSignature(widget.items);
```

try `_tryAdoptValidationEchoWithoutGlobalSignatures()`. If it succeeds, return `_bucketCache` immediately.

Do not weaken the existing full-regroup conditions for metadata reorder, auto-group settings, confidence/min-frame ratio, group completion, or species/bucket changes.

- [ ] **Step 5.7: Add widget regression coverage.**

Extend the new test or add a focused harness that marks one item without changing its species/group boundary and verifies the selected group/list remains correct. Add a second case where a species change or group completion forces the full regroup and lands the item in the correct bucket.

Keep `species_validation_group_expansion_test.dart` unchanged unless a fixture update is strictly required; it protects the metadata-reorder fallback that must remain full rebuild behavior.

- [ ] **Step 5.8: Run GREEN Flutter tests.**

```bash
cd frontend
flutter test \
  test/species_validation_incremental_cache_test.dart \
  test/species_validation_group_expansion_test.dart \
  test/dinov3_validation_selected_box_learning_test.dart
```

- [ ] **Step 5.9: Commit.**

Commit message:

```text
perf: update validation grouping cache incrementally
```

---

## Task 6: Add scale regressions and complete performance instrumentation

**Files:**
- Modify: `system/backend/preview_fast.py`
- Modify: `system/backend/services.py`
- Modify: `system/backend/dinov3_feedback_service.py`
- Modify: `tests/test_preview_fast.py`
- Modify: `tests/test_validation_performance.py`
- Modify as needed: `tests/test_dinov3_feedback_store.py`

- [ ] **Step 6.1: Add a 20k-row detection regression without timing assertions.**

In `tests/test_preview_fast.py`, populate approximately 20,000 unrelated detection/validation rows and a small requested set. Trace statements and assert targeted helpers issue only `WHERE image_filename IN (...)` selects. Assert only requested rows are JSON-decoded/materialized.

- [ ] **Step 6.2: Add a 50k-row observation lookup regression without large embedding memory.**

In `tests/test_validation_performance.py` or `test_dinov3_feedback_store.py`, insert approximately 50,000 noise observation rows directly into SQLite with valid `source_path` projections but minimal/dummy blobs that will never be decoded. Insert one fully valid target observation. Call `observations_for_source_path(target)` and assert only the target row is materialized and the query plan uses the source-path index.

- [ ] **Step 6.3: Finish timing instrumentation.**

Use module loggers and `time.perf_counter()` to emit concise INFO timing lines for:

- targeted detection/validation index lookup: requested filename count, DB count, elapsed;
- validation mark request: target count, persistence elapsed, DINO feedback elapsed, total elapsed;
- DINO runtime persistence: observation count, observation DB write elapsed, feedback-example elapsed, Registry elapsed.

Do not include embeddings or image bytes. Do not add per-item INFO logs inside large loops.

- [ ] **Step 6.4: Run all focused Python verification.**

```bash
python -m pytest -q \
  tests/test_preview_fast.py \
  tests/test_runtime_patches.py \
  tests/test_dinov3_feedback_store.py \
  tests/test_dinov3_feedback_learning.py \
  tests/test_dinov3_feedback_persistence.py \
  tests/test_dinov3_registry.py \
  tests/test_dinov3_backend_integration.py \
  tests/test_validation_performance.py
python -m compileall -q system tests
```

If an existing test name differs because the repository already has a more specific validation-performance test, use that actual file and record the substitution in the commit/PR notes rather than inventing a duplicate test suite.

- [ ] **Step 6.5: Run focused Flutter verification.**

```bash
cd frontend
flutter test \
  test/species_validation_incremental_cache_test.dart \
  test/species_validation_group_expansion_test.dart \
  test/dinov3_validation_selected_box_learning_test.dart
```

If the environment has the Flutter SDK and dependencies available, also run the broader validation-screen test group. If it does not, record that limitation explicitly; do not claim it passed.

- [ ] **Step 6.6: Commit.**

Commit message:

```text
test: cover validation performance regressions
```

---

## Task 7: Final verification, branch review, and PR update

**Files:**
- No production changes expected unless verification finds a defect.
- Update PR #24 description or add a clear top-level PR comment if the connector cannot edit the body.

- [ ] **Step 7.1: Invoke the verification-before-completion workflow.**

Run fresh verification after the final code change. Do not rely on earlier green output. Capture exact pass/fail counts for the focused Python and Flutter commands.

- [ ] **Step 7.2: Review the final diff against the approved spec.**

Compare:

```text
nightly...fix/validation-rejection-preview-cache
```

Check specifically that no changes landed in:

- model thresholds/configuration;
- checkpoint files;
- DINOv3 rejection math;
- Registry registration/status thresholds;
- default `imgsz`/augment/FP16/batch settings;
- canonical database layout.

- [ ] **Step 7.3: Recheck PR state and mergeability.**

Verify PR #24 is still open, head points to the final branch SHA, base remains `nightly`, and GitHub reports it mergeable. If `nightly` advanced, compare behind/ahead state before making any merge claim.

- [ ] **Step 7.4: Update PR #24 with the expanded scope.**

Document:

- indexed detection/validation lookups now cover mark/startup/single-item paths, not only `/api/preview`;
- DINOv3 observation path index and migration;
- batch validation feedback lookup and observation persistence;
- still-image example decode reuse;
- Flutter incremental validation cache;
- exact focused verification results;
- any environment limitation that prevented broader/full-suite execution.

Keep the existing `拒识/Unknown` root-cause explanation in the PR.

- [ ] **Step 7.5: Do not merge without an explicit user request.**

Stop with the branch and PR ready for review. Integration into `nightly` is a separate user decision.

---

## Plan Self-Review

- No task changes recognition thresholds, model checkpoint contents, Registry learning mathematics, or ecological grouping rules.
- The existing `拒识/Unknown` serializer fix remains isolated and covered by `tests/test_runtime_patches.py`.
- Every indexed fast path has a legacy fallback for callers that genuinely request a complete index.
- DINOv3 migration is additive, idempotent, and retains the historical JSON payload.
- Batch feedback preserves the conservative “exactly one observation per file” learning rule.
- Still-frame caching is scoped to one persistence call and does not retain video frames across calls.
- Flutter cache adoption is explicitly conservative: any uncertain path falls back to the existing global rebuild.
- Scale regression tests validate query/work shape rather than hardware-dependent elapsed time.
- Final merge remains out of scope until the user explicitly requests it.
