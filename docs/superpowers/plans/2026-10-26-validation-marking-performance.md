# Validation Marking Performance Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task in this session.

**Goal:** Integrate LiteRT privacy and rejection validation, then prevent equivalent per-item metadata updates from restarting full validation preparation.

**Architecture:** Extend the existing incremental echo adoption path to recognize bounded metadata deltas before the large-list preparation gate. Compare effective timestamps and affected group memberships, then publish replacements atomically. Fall back to the existing preparation path when structure or grouping really changes.

**Tech Stack:** Flutter/Dart, Python, pytest.

**Spec:** docs/superpowers/specs/2026-10-26-validation-marking-performance-design.md

## Global Constraints
- Preserve both merged feature sets, manual marking/undo behavior, and real group-boundary changes.
- Keep original working directory and the running installation unchanged.

## Review Focus
- A metadata response after marking must not block the next action.
- Equivalent timestamps must preserve groups; actual timestamp changes must regroup.
- New observation IDs and changed rejection memberships must not reuse stale buckets.
- Large or reordered snapshots must retain cancellable preparation.
- All affected cache references must use the latest item, including subsequent marks.

## Tasks
- [x] Create isolated integration branch, fetch, and merge both requested branches. Baseline: 16 validation tests pass.
- [x] Add failing widget tests for metadata replacement and post-mark metadata loading, asserting no absorbing validation content and bounded metadata reads.
- [x] Extend incremental adoption in `frontend/lib/src/screens/species_validation_screen.dart`, including effective timestamps and rejection membership validation; pass regression tests.
- [x] Verify actual timestamp/species/membership changes and existing directory/preparation/undo behavior.
- [x] Run Flutter suite/analyzer and Python suite, review the diff, and commit the verified change.

## Execution record
- Integration merge: 1b03b774. Original checkout stays on codex/rejected-cluster-validation.
- Scope ruling: implement the previously diagnosed metadata invalidation fix; retain full-scan architecture and directory defaults for this change.

- Review: independent reviewer found the metadata adopter must use single-item species rules with autoGroup disabled. Confirmed using ['豹猫', '未检测'] -> ['豹猫']; regression failed with a stale bucket before the conditional fix. The proposed Unknown example was corrected because Unknown is treated as rejection by this codebase.
- Metadata regression RED: one-item replacement enabled AbsorbPointer; post-mark metadata read grouping data 4,872 times for 600 items. GREEN: 23,310-item metadata and post-mark cases remain interactive, read fewer than 500 metadata fields, and next mark receives updated width=640.
- Rejection safety: known cluster reassignment updates bucket counts; a new observation replacing unresolved metadata triggers lookup and moves to its verified cluster.

- Field-evidence ruling: extend timestamp reuse from exact equality to preserved adjacent order/gap boundaries because the actual sample has a one-second EXIF correction. Strict equality still caused 186,555 grouping-data reads at 23,310 items; regression failed before adjacency validation and passes afterward. True reordered timestamps and a gap reaching the configured threshold still prepare.
- Verification so far: 26 focused frontend tests pass; full Python suite 511 passed, 1 skipped, 57 subtests passed. Dependency deprecation warning comes from the installed Starlette/httpx combination.

## Final verification
- `flutter test --reporter expanded`: 234 passed (including 23,310-item boundary-preserving metadata cases). Existing native media_kit initialization diagnostics remain in widget-test output; exit code is 0.
- `python -B -m pytest -q`: 511 passed, 1 skipped, 57 subtests passed, 1 installed Starlette/httpx deprecation warning. No Python production code changed after this run.
- `dart analyze` on all three changed Dart files: no issues. `dart format --output=none --set-exit-if-changed` and `git diff --check`: pass. Earlier full `flutter analyze --no-pub`: no issues.
- Independent code review: single-item species comparison finding fixed with a failing then passing regression. Narrow review of the subsequently added field-evidence adjacency logic found no actionable issues.
- Both requested feature tips are ancestors of the integration merge. Preserve the new local branch/worktree for review; no push, deployment, or modification of the running installation is part of this request.
