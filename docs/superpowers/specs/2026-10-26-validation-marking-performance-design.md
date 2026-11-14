# Validation marking performance

Integrate `origin/codex/litert-upload-privacy` and `codex/rejected-cluster-validation` on `codex/litert-validation-performance`, preserving both feature sets. The user authorized integration and the marking performance fix.

## Approach
Reuse prepared buckets when an immutable item-list replacement only changes a bounded number of items and leaves effective ordering, group species, validation completeness, and rejection-cluster memberships unchanged. Update affected cached item references before deciding whether preparation is necessary. Real structural/configuration changes retain cancellable preparation. Equivalent parsed timestamps and EXIF corrections that preserve adjacent chronological order and every burst-gap predicate do not change boundaries. Cache ordered groups to inspect only the adjacent edges; unchanged image/video counts and gap predicates preserve inferred burst sizes.

Compared alternatives: skipping every metadata update would lose real grouping changes; moving all grouping into an isolate would be a much broader refactor and would still repeat the work. This scoped change removes unnecessary work while retaining correctness. It does not change default directory selection or initial full-scan architecture.

## Constraints and verification
Preserve manual mark/undo deferral, rejection density checks, privacy processing, directory-switch cancellation, selection, and expansion. Add regression tests for metadata updates with and without a preceding mark, equivalent timestamps, actual timestamp changes, and rejected-cluster membership changes. Run existing frontend validation tests, the Flutter suite, analyzer, and Python suite. Do not deploy or change the running installation.

## Field validation
The running installation returns date_taken=2025-09-17T11:24:49 and modified_at=2025-09-17T11:24:48 for IMAG0001.JPG. Therefore timestamp equality alone is insufficient: the one-second correction must reuse grouping when it changes neither order nor burst boundaries. A 23,310-item regression reproduces this case before the adjacency check and passes afterward.

## Follow-up scope from real marking failures
Incremental adoption must handle repeated bucket membership, completed groups, and changed species labels without recomputing unaffected groups. Mutate each affected group's shared item references once, replace its memberships across all affected buckets, retain unfinished manual-review group deferral, remove empty buckets, and retain global chronological group indexes. Preserve authoritative cache-only per-box feedback when an unrelated parent metadata update is adopted. Real structure and burst-boundary changes still use cancellable preparation.
