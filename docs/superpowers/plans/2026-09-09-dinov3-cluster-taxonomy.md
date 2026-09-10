# DINOv3 Cluster Species List and Taxonomy Autofill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show current-folder DINOv3 registry clusters in the validation species list and provide deterministic Chinese-common-name to scientific-name autofill from the bundled taxonomy database.

**Architecture:** Extend the existing validation bucketing layer with stable registry identities (`registry:<id>`) rather than visible labels. Add a small backend taxonomy lookup endpoint backed by `res/species_database.db`; Flutter registration UI consumes it and only autofills when the scientific name is still system-managed.

**Tech Stack:** Python/FastAPI/SQLite, Flutter/Dart, existing `DinoV3RegistryDialog`, existing species database.

**Spec:** `docs/superpowers/specs/2026-09-09-dinov3-cluster-species-list-taxonomy-design.md`

## Global Constraints

- Current-folder species list is a projection of currently loaded media, not a replacement registry.
- Registry identity keys are stable; visible labels may change after naming/promotion.
- Unnamed Candidate labels are `未知物种 #<candidate_number>`.
- Registry cluster browsing never injects historical files from other folders.
- Taxonomy lookup is offline and only uses `res/species_database.db`.
- Exact trimmed Chinese-name matching only; no fuzzy/network lookup.
- Taxonomy matches never automatically confirm or register species.
- Manual scientific-name edits are never overwritten by later autofill.

---

## File Structure

- Modify: `system/backend/services.py` or create `system/backend/species_taxonomy.py` for exact local lookup.
- Modify: `system/backend/models.py` for taxonomy lookup DTOs.
- Modify: `system/backend/main_core.py` for `/api/species/lookup`.
- Modify: `frontend/lib/src/widgets/dinov3_registry_dialog.dart` for autofill state.
- Modify: `frontend/lib/src/screens/species_validation_screen.dart` for registry-aware buckets.
- Modify: `frontend/lib/src/models/job.dart` for registry metadata on boxes.
- Modify: `frontend/lib/src/api_client_core.dart` for taxonomy lookup.
- Add Python tests for taxonomy and registry filtering.
- Add Flutter tests for labels, filtering, and autofill.

---

### Task 1: Add exact taxonomy lookup service

**Files:**
- Create: `system/backend/species_taxonomy.py`
- Modify: `system/backend/models.py`
- Modify: `system/backend/main_core.py`
- Create: `tests/test_species_taxonomy.py`

**Interfaces:**
- Produces `SpeciesTaxonomyLookupResponse(found: bool, common_name: str, scientific_name: str | None, ambiguous: bool, message: str | None)`.
- Produces `lookup_species_taxonomy(common_name: str) -> SpeciesTaxonomyLookupResponse`.

- [ ] **Step 1: Write failing tests**

```python
def test_exact_common_name_returns_scientific_name(tmp_path, monkeypatch):
    # fake species_database.db with 中文名/学名
    response = lookup_species_taxonomy(" 毛冠鹿 ")
    assert response.found is True
    assert response.scientific_name == "Elaphodus cephalophus"


def test_missing_name_allows_manual_entry():
    response = lookup_species_taxonomy("不存在物种")
    assert response.found is False
    assert response.scientific_name is None
```

- [ ] **Step 2: Implement exact SQLite query**

```python
normalized = common_name.strip()
SELECT 中文名, 学名 FROM species WHERE 中文名 = ?
```

Return ambiguity when multiple exact rows disagree on scientific names.

- [ ] **Step 3: Add endpoint**

```python
@app.get("/api/species/lookup")
def species_lookup(common_name: str = Query(...)):
    return lookup_species_taxonomy(common_name)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_species_taxonomy.py -q
```

- [ ] **Step 5: Commit**

```bash
git add system/backend/species_taxonomy.py system/backend/models.py system/backend/main_core.py tests/test_species_taxonomy.py
git commit -m "feat: add local taxonomy lookup"
```

---

### Task 2: Add registry-aware current-folder species buckets

**Files:**
- Modify: `frontend/lib/src/models/job.dart`
- Modify: `frontend/lib/src/screens/species_validation_screen.dart`
- Create: `frontend/test/dinov3_cluster_species_list_test.dart`

**Interfaces:**
- `DetectionBox` gains nullable `registryId`.
- Bucket keys become `known:<species>` and `registry:<id>`.

- [ ] **Step 1: Write failing tests**

```dart
expect(bucketForUnnamedCandidate.registryLabel, '未知物种 #3');
expect(bucketForDifferentUnknownIds, isNot(equals));
expect(mixedPhotoAppearsInKnownAndRegistryBuckets, isTrue);
```

- [ ] **Step 2: Add registry metadata decoding**

```dart
final int? registryId;
```

Read `registry_id` from detection boxes.

- [ ] **Step 3: Replace visible-label grouping with stable identity grouping**

Do not group by `species == 'Unknown'`.

```dart
String bucketKey(DetectionItem item, DetectionBox box) {
  if (box.registryId != null) return 'registry:${box.registryId}';
  return 'known:${box.species}';
}
```

- [ ] **Step 4: Apply registry labels**

Fetch registry metadata for the selected DINOv3 model. Label rules:

```text
candidate + no name -> 未知物种 #N
candidate + name -> name（候选，待注册）
provisional -> name（临时注册，待确认）
confirmed/mature -> name
```

Keep selection by registry id, not label.

- [ ] **Step 5: Run Flutter tests**

```bash
cd frontend && flutter test test/dinov3_cluster_species_list_test.dart
```

- [ ] **Step 6: Commit**

```bash
git add frontend/lib/src/models/job.dart frontend/lib/src/screens/species_validation_screen.dart frontend/test/dinov3_cluster_species_list_test.dart
git commit -m "feat: show DINOv3 clusters in species list"
```

---

### Task 3: Add Chinese-name scientific-name autofill UI

**Files:**
- Modify: `frontend/lib/src/api_client_core.dart`
- Modify: `frontend/lib/src/widgets/dinov3_registry_dialog.dart`
- Create: `frontend/lib/src/models/species_taxonomy.dart`
- Create: `frontend/test/species_taxonomy_autofill_test.dart`

**Interfaces:**
- Produces `NeriApiClient.lookupSpeciesTaxonomy(String commonName)`.
- Produces `SpeciesTaxonomyResult` model.

- [ ] **Step 1: Write failing Dart tests**

```dart
expect(await lookup('毛冠鹿').scientificName, 'Elaphodus cephalophus');
expect(manuallyEditedScientificName, remainsAfterCommonNameChange);
```

- [ ] **Step 2: Add API client method**

```dart
Future<SpeciesTaxonomyResult> lookupSpeciesTaxonomy(String commonName)
```

GET `/api/species/lookup`.

- [ ] **Step 3: Add dialog autofill state**

Track:

```dart
bool _scientificNameWasAutofilled = false;
bool _scientificNameUserEdited = false;
```

On common-name debounce:

```dart
if (!_scientificNameUserEdited) {
  controller.text = result.scientificName ?? controller.text;
  _scientificNameWasAutofilled = true;
}
```

On scientific-name manual edit:

```dart
_scientificNameUserEdited = true;
_scientificNameWasAutofilled = false;
```

- [ ] **Step 4: Run Flutter tests**

```bash
cd frontend && flutter test test/species_taxonomy_autofill_test.dart
```

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/src/api_client_core.dart frontend/lib/src/widgets/dinov3_registry_dialog.dart frontend/lib/src/models/species_taxonomy.dart frontend/test/species_taxonomy_autofill_test.dart
git commit -m "feat: autofill species taxonomy names"
```

---

### Task 4: Full regression

**Files:**
- Existing Python/Flutter suites.

- [ ] **Step 1: Run Python tests**

```bash
pytest -q
```

- [ ] **Step 2: Run Flutter tests**

```bash
cd frontend && flutter test
```

- [ ] **Step 3: Verify non-DINO behavior**

Confirm:

- YOLO model lists unchanged.
- Existing species validation still groups normal species correctly.
- Registry UI still uses registry ids.
- Taxonomy lookup failure does not block registration.

- [ ] **Step 4: Commit regression fixes if needed**

```bash
git add .
git commit -m "test: stabilize DINOv3 cluster taxonomy integration"
```
