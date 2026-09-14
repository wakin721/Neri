# Validation and Processing Performance Optimization Design

Date: 2026-09-14
Branch: `fix/validation-rejection-preview-cache`
Target base: `nightly`

## Goal

Reduce the latency of validation marking, next-item loading, and batch-processing startup on large camera-trap datasets without changing recognition thresholds, DINOv3 rejection semantics, ecological grouping rules, or learned-species decisions.

The change extends the existing preview optimization already present on this branch. It does not replace the storage layout with a new canonical database and does not change the default inference resolution or augmentation settings in this PR.

## Confirmed Bottlenecks

### Detection and validation lookup

`services._load_detection_index()` currently selects every row from `detections` and filters requested filenames in Python. `services._load_validation_index()` always selects the whole `validation` table. These functions are still used outside the optimized `/api/preview` path, including validation marking, single-item metadata loading, and processing startup.

### DINOv3 validation lookup

`_learnable_observations_for_file()` selects every persisted DINOv3 observation, parses each JSON payload, resolves every historical source path, and then filters to the requested file. Batch marking repeats that scan per target file.

### DINOv3 persistence

Observation and registry persistence commits SQLite state repeatedly at observation granularity. Example persistence can also reopen the same source image for multiple observations. These costs are synchronous with the processing batch.

### Flutter validation cache

A validation update replaces the `widget.items` list identity. `_currentBuckets()` then recomputes full path/grouping signatures across all items before deciding whether grouping can be reused. The grouping signature runs confidence/species logic across every item. Updating the cached buckets and lookups also walks the full cache.

## Design

### 1. General indexed detection/validation reader

Extend `system/backend/preview_fast.py` into a reusable indexed lookup module rather than a preview-only implementation.

Provide helpers that:

- query detections by `image_filename IN (...)` in bounded chunks;
- query validation rows by `image_filename IN (...)` in the same connection where practical;
- preserve current first-database-wins behavior;
- preserve malformed-row tolerance;
- fall back safely when a legacy database does not contain one of the expected tables or indexes.

`runtime_patches.install_runtime_patches()` will install compatible replacements for:

- `_load_detection_index` when `filenames` is supplied;
- a targeted validation-index helper that accepts filenames;
- `preview_media_items`;
- the single-item validation-state lookup used by `preview_media_item` and validation reloads.

Calls that intentionally request a complete index continue to use the historical implementation. This keeps export and legacy compatibility behavior unchanged unless a target filename set is available.

### 2. DINOv3 observation source-path index

Add a schema-compatible migration to `HumanFeedbackStore`:

```sql
ALTER TABLE observations ADD COLUMN source_path TEXT;
CREATE INDEX IF NOT EXISTS idx_observations_source_path
ON observations(source_path);
```

For new writes, `persist_observation()` stores a normalized source-path identity in this column while retaining `source_path` inside the JSON payload for backward compatibility.

For existing databases, migration backfills `source_path` from `payload` once. The migration must be idempotent and tolerate malformed historical payloads. A metadata marker records successful backfill so normal startup does not rescan the table indefinitely.

The normalized identity must use the same semantic normalization as `_path_identity`: slash-normalized, case-folded, resolved where possible.

Expose a store method such as `observations_for_source_path(path)` that performs indexed SQL lookup and constructs only matching `FeedbackObservation` objects.

`_learnable_observations_for_file()` becomes a thin compatibility wrapper around this indexed API.

For batch validation, open one feedback store for the request and resolve all requested paths through indexed queries rather than reopening the store and scanning history once per file.

### 3. Transaction reduction for DINOv3 processing

Keep classifier, Registry, and learning semantics unchanged, but reduce transaction frequency.

Introduce explicit batch transaction support at the persistence boundary:

- observation inserts for one inference batch are accumulated and committed once;
- registry mutations may share a transaction only where existing refresh/prototype semantics remain identical;
- failures roll back the current unit and preserve the existing non-fatal inference-persistence behavior.

Do not change the mathematical prototype construction, event grouping thresholds, candidate matching, or rejection rules.

Example image persistence will avoid reopening the same still image repeatedly within a batch. A small per-call source-frame cache may be used while persisting examples and discarded immediately afterward. Video frames remain lazy and are not retained across batches.

### 4. Incremental Flutter validation cache

Avoid full-dataset signature recomputation for ordinary mark operations.

The screen already knows which paths are being marked through `_deferRegroupForItems()`. Extend this state to track dirty validation paths.

On a widget update where:

- grouping settings are unchanged;
- path membership/order is unchanged;
- the update only touches known dirty paths;

update the cached `DetectionItem` references and affected groups/buckets directly without calling `_itemsGroupingSignature()` across the full dataset.

A full regroup remains mandatory when:

- paths are added, removed, or reordered;
- auto-group settings change;
- confidence/min-frame-ratio settings change;
- an affected group becomes complete and its validated/unvalidated bucket changes;
- the updated species label changes the bucket assignment or changes a group boundary-relevant state.

The implementation must prefer correctness over reuse: if the incremental preconditions cannot be proven, fall back to the current full rebuild.

### 5. Performance instrumentation

Keep and extend existing timing logs so remaining bottlenecks are observable.

Add separate timings for:

- indexed detection/validation lookup;
- DINOv3 observation lookup;
- DINOv3 observation persistence;
- registry persistence/example persistence;
- validation API total latency.

Do not log image contents, embeddings, or sensitive path payloads beyond the existing filename/path logging policy.

## Compatibility and Migration

- Existing `detections.db` files remain valid.
- Existing DINOv3 feedback databases migrate in place.
- `payload` remains the authoritative compatibility representation; `source_path` is an indexed projection used for lookup.
- If migration cannot parse a legacy observation payload, that row remains accessible by observation ID but cannot participate in source-path indexed lookup until rewritten.
- `拒识/Unknown` behavior introduced on this branch is unchanged.
- No recognition threshold, model checkpoint, Registry status rule, or ecological event-gap rule changes.

## Testing

### Python regression tests

Add tests that prove:

1. `_load_detection_index(..., filenames=...)` emits SQL constrained by requested filenames and never loads unrelated rows into the result.
2. targeted validation lookup uses `WHERE image_filename IN (...)` and chunks large requests.
3. legacy full-index calls retain old behavior.
4. DINOv3 schema migration adds/backfills/indexes `source_path` idempotently.
5. indexed observation lookup returns only the requested file and does not parse unrelated observation payloads after migration.
6. batch validation reuses one feedback-store context and does not perform one full observation scan per file.
7. transaction batching preserves persisted observations/registry state on success and rolls back the batch on failure where atomicity is promised.
8. existing rejection-preservation tests continue to pass.

Use synthetic scale tests with approximately 20k-50k detection/observation rows to catch accidental reintroduction of full scans. Timing assertions should be avoided because CI hardware is variable; assert query shape and work performed instead.

### Flutter regression tests

Add a focused test or testable cache helper showing that a one-item validation update does not recompute grouping state for all items when incremental preconditions hold, while species/bucket-changing updates still trigger a correct regroup.

Existing validation grouping tests must remain green.

## Out of Scope

The following optimizations are intentionally deferred:

- replacing per-directory/root `detections.db` files with one canonical database;
- migrating keys from basename/filename to canonical relative paths;
- changing default `imgsz`, augmentation, FP16, batch size, or classification thresholds;
- redesigning DINOv3 Registry/prototype mathematics;
- broad UI virtualization/refactoring unrelated to validation-cache updates.

These can be measured after this PR lands using the new timing instrumentation.

## Success Criteria

The implementation is complete when:

- validation and processing paths no longer perform full detection/validation table scans when target filenames are known;
- DINOv3 per-file validation lookup is index-backed rather than history-scan-backed;
- ordinary single-item marks can update validation UI state without full-dataset grouping-signature recomputation;
- DINOv3 persistence performs materially fewer synchronous commits/source-image decodes while preserving output semantics;
- targeted regression tests and the repository's available Python/Flutter verification pass;
- the branch remains mergeable into `nightly` and PR #24 accurately describes the expanded performance scope.
