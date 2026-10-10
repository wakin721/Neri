# Validation Export Format and Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the format selector with one export button, persist a CSV/Excel/ask preference, and let a selected child-folder export target itself or produce one table per media directory under the root.

**Architecture:** The main window passes root and selected directory context to the validation screen. The screen resolves scope, format, and existing follow-up choices before one API request. A backward-compatible request flag makes the backend split a root export by media parent directory and return all output paths.

**Tech Stack:** Flutter/Dart, FastAPI/Pydantic, Python/pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-validation-export-format-and-scope-design.md`

## Global Constraints

- Settings values: `ask`, `csv`, `excel`; absent or invalid values normalize to `ask`.
- Existing export requests default to the current single-table behavior.
- A per-directory export writes tables only in directories containing supported media, including the root when applicable.
- A cancelled dialog sends no export request.

## Review Focus

- A selected nested directory must offer root scope and keep current-only exports recursive; Task 3 test.
- A root containing media alongside nested folders must receive its own table without mixing records; Task 1 test.
- A directory without supported media must receive no empty table; Task 1 test.
- A partial batch failure must identify already written output paths in the error; Task 1 test.
- An invalid saved format mode must ask rather than silently choose CSV; Task 2 test.

---

### Task 1: Backend per-directory export

**Files:**
- Modify: `system/backend/models_legacy.py`
- Modify: `system/backend/services_legacy.py`
- Test: `tests/test_directory_results.py`

**Interfaces:**
- Consumes: `ValidationExportRequest.input_path`, `file_format`, existing export options.
- Produces: optional `per_directory: bool = False` request field; `output_paths: list[str]` response field alongside compatible `output_path: str` and aggregate counters.

- [ ] **Step 1: Write failing pytest cases** in `tests/test_directory_results.py`: `test_export_per_directory_writes_root_and_nested_tables` asserts one table in every media parent, correct per-table source paths and extensions, and no table in empty folders; `test_export_per_directory_failure_lists_written_paths` forces a later write to fail and checks the earlier path appears in the error; `test_export_per_directory_limits_favorite_and_delete_side_effects` checks requested paths outside the input root are untouched. Keep existing `test_export_respects_directory_scope_and_separate_camera_events` as the single-table compatibility test.
- [ ] **Step 2: Run red tests:** `python -m pytest tests/test_directory_results.py -q`; expect new cases to fail on the absent `per_directory` field or outputs.
- [ ] **Step 3: Implement** `per_directory` and `output_paths` in `models_legacy.py`; update `export_validation_data(request: ValidationExportRequest) -> ValidationExportResponse` in `services_legacy.py` to reuse the existing per-camera processing but write each direct media-parent group separately when requested. Keep a single timestamp for the batch; aggregate rows, favorite copies and deletions; include already written paths in a `RuntimeError` if a later table fails.
- [ ] **Step 4: Run green tests:** `python -m pytest tests/test_directory_results.py -q`; expect all cases to pass.
- [ ] **Step 5: Commit:** `git add system/backend/models_legacy.py system/backend/services_legacy.py tests/test_directory_results.py && git commit -m "feat: export validation tables per media directory"`.

### Task 2: Persist export format setting and API result

**Files:**
- Modify: `frontend/lib/src/screens/settings_screen.dart`
- Modify: `frontend/lib/src/main_window.dart`
- Modify: `frontend/lib/src/api_client_core.dart`
- Modify: `frontend/lib/src/models/export_result.dart`
- Test: `frontend/test/export_settings_test.dart`

**Interfaces:**
- Consumes: Task 1's `per_directory` request key and `output_paths` response key.
- Produces: validation widget properties `rootInputPath: String`, `rootItems: List<DetectionItem>`, `exportFormatMode: String`, `onExportFormatModeChanged: Future<void> Function(String)`; API `exportValidationData(..., bool perDirectory = false)`; result `outputPaths: List<String>`.

- [ ] **Step 1: Write failing Flutter tests** in `export_settings_test.dart` for settings normalization (`ask`, `csv`, `excel`, invalid→`ask`), default ask, API decoding of `output_paths`, and the single-path fallback; update existing validation widget fixtures to pass the new required properties.
- [ ] **Step 2: Run red tests:** from `frontend`, `flutter test test/export_settings_test.dart test/species_validation_box_dialog_test.dart`; expect new assertions or missing properties to fail.
- [ ] **Step 3: Implement** a “导出格式” `_SettingsPanel` in basic settings, using `_SettingsMenuButton<String>`; add `export_format_mode` to draft loading and main-window save callback via `_saveSilentSettingsPatch`; pass the root path, full root item list, and normalized mode to the validation screen; serialize `per_directory` and decode output paths with a single-path fallback.
- [ ] **Step 4: Run green tests:** `flutter test test/export_settings_test.dart test/species_validation_box_dialog_test.dart`; expect pass.
- [ ] **Step 5: Commit:** `git add frontend/lib/src/screens/settings_screen.dart frontend/lib/src/main_window.dart frontend/lib/src/api_client_core.dart frontend/lib/src/models/export_result.dart frontend/test/export_settings_test.dart frontend/test/species_validation_box_dialog_test.dart && git commit -m "feat: persist validation export format"`.

### Task 3: Validation export dialogs and button

**Files:**
- Modify: `frontend/lib/src/screens/species_validation_screen.dart`
- Test: `frontend/test/species_validation_box_dialog_test.dart`
- Test: `frontend/test/validation_preparation_test.dart`

**Interfaces:**
- Consumes: Task 2's root path, root item list, format mode, persistence callback, API flag and result paths.
- Produces: `_resolveExportScope()` returning the selected path and `perDirectory`; `_resolveExportFormat()` returning `csv`, `excel`, or null on cancel.

- [ ] **Step 1: Write failing widget tests**: selected child shows three scope actions; root selection skips scope prompt; nested current-only selection stays recursive; format ask shows CSV, Excel and “不再询问”; remembering saves the chosen fixed mode; cancel at either dialog sends no request; fixed mode skips format prompt; button alone has the filter panel height.
- [ ] **Step 2: Run red tests:** from `frontend`, `flutter test test/species_validation_box_dialog_test.dart test/validation_preparation_test.dart`; expect the new UI assertions to fail.
- [ ] **Step 3: Implement** the single-button `_buildExportControls`, scope and format dialogs, and `_exportData` ordering: scope → format → empty-photo choice → favorite-media choice → API. Derive favorite and empty-photo candidates from `rootItems` or current `items` according to the chosen scope; use `outputPaths` for aggregate completion text and a usable open action.
- [ ] **Step 4: Run green tests:** `flutter test test/species_validation_box_dialog_test.dart test/validation_preparation_test.dart`; expect pass.
- [ ] **Step 5: Commit:** `git add frontend/lib/src/screens/species_validation_screen.dart frontend/test/species_validation_box_dialog_test.dart frontend/test/validation_preparation_test.dart && git commit -m "feat: choose validation export scope and format"`.

### Task 4: Final integration verification

**Files:** All changed source and test files.

**Interfaces:** Consumes Tasks 1–3; produces verified behavior.

- [ ] **Step 1: Format:** `dart format lib/src/screens/settings_screen.dart lib/src/screens/species_validation_screen.dart lib/src/main_window.dart lib/src/api_client_core.dart lib/src/models/export_result.dart test/export_settings_test.dart test/species_validation_box_dialog_test.dart test/validation_preparation_test.dart` from `frontend`.
- [ ] **Step 2: Verify backend:** `python -m pytest tests/test_directory_results.py -q` from repo root; expect pass.
- [ ] **Step 3: Verify frontend:** `flutter test test/export_settings_test.dart test/species_validation_box_dialog_test.dart test/validation_preparation_test.dart` and `flutter analyze` from `frontend`; expect pass.
- [ ] **Step 4: Inspect `git diff` and report any remaining risk; commit any format-only adjustment.**
