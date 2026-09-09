# DINOv3 Cluster Species List and Taxonomy Autofill Design

This is a normative extension to:

- `2026-09-09-dinov3-pipeline-registry-ui-design.md`
- `2026-09-09-dinov3-human-feedback-multiprototype-design.md`

It adds two approved behaviors:

1. DINOv3 clusters represented in the currently opened validation folder must appear as first-class entries in the validation species list, including unnamed Candidate clusters.
2. In the DINOv3 registration dialog, entering an exact Chinese common name should automatically fill the scientific name from Neri's local `res/species_database.db` when a unique match exists.

## 1. Current-folder cluster visibility

The species list on `SpeciesValidationScreen` must combine two identity types:

```text
known species identity
registry cluster identity
```

A registry cluster is included only when at least one DINOv3 detection box in the currently loaded folder/items references that registry id. Historical observations from other folders must not make a cluster appear in the current-folder species list.

The registry remains global and fingerprint-scoped; the validation list is a current-folder projection of that registry state.

## 2. Stable bucket identity

Species-list bucketing must not use the visible label as the bucket key.

Use stable logical keys such as:

```text
known:<canonical species name>
registry:<registration_id>
```

The visible label may change after human naming or status promotion without invalidating selection state or mixing two different registry clusters.

For DINOv3 boxes, `registry_id` is authoritative for registry-cluster membership. If a box has no registry id, it is not assigned to a Candidate bucket merely because its visible species text is `Unknown`.

## 3. Candidate display labels

For a registry entry represented in the current folder:

```text
candidate + no common name
-> 未知物种 #<candidate_number>

candidate + human-confirmed common name
-> <common_name>（候选，待注册）

provisional
-> <common_name>（临时注册，待确认）

confirmed / mature
-> <common_name>
```

The current `SpeciesRegistry` display-name helper may be reused or adjusted, but the validation species list must follow the labels above even if registry-internal display text differs.

When a user saves a Chinese common name in the registration dialog, any currently visible Candidate bucket using the same registry id should refresh to the new label without requiring the application to restart.

## 4. Current-folder filtering semantics

Clicking a registry cluster entry filters the validation workspace to media from the currently loaded folder/items that contain at least one DINOv3 box with the matching `registry_id`.

It must not fetch and inject historical registry event files from other folders. Historical events remain accessible only through the existing `继续验证` registry workflow.

A media item may appear in multiple species buckets when it contains multiple identities. Example:

```text
photo.jpg
  box 1 -> 盘羊
  box 2 -> registry:7 / 未知物种 #3

species list membership:
  盘羊
  未知物种 #3
```

Selecting either bucket still opens the same media file, but the relevant box should be visually emphasized when practical. This avoids losing mixed-species or mixed-known/unknown images.

## 5. Interaction with grouping

Existing auto-group/collapse behavior remains file/group-oriented. Cluster bucketing is an additional filtering dimension and must not change independent-event grouping rules.

When collapsed groups are shown, a group is eligible for a registry bucket if at least one member contains a box with the matching registry id. Expanding the group must preserve the same registry filter.

## 6. Registry metadata delivery

The frontend needs a lightweight mapping from registry id to:

```text
registration_id
candidate_number
status
common_name
scientific_name
display label
```

The existing DINOv3 registry API is the authoritative source. The validation screen should not infer candidate numbers or names from stale detection JSON.

The validation screen may cache the registry mapping for the currently selected DINOv3 model, but it must refresh when:

- the selected DINOv3 classification model changes;
- a registry identity is edited;
- a Candidate is registered/promoted through the UI;
- validation refresh is explicitly requested and registry-backed boxes are present.

Failure to load registry metadata must not hide the media. The fallback label is:

```text
未知物种（Registry <id>）
```

and a non-blocking warning should be surfaced.

## 7. Taxonomy lookup source

Scientific-name autofill uses only Neri's bundled/local taxonomy database:

```text
res/species_database.db
```

The relevant schema already contains at least:

```text
中文名
学名
```

No web lookup is required for registration. This keeps behavior offline, deterministic, and consistent with Neri export taxonomy.

## 8. Backend taxonomy lookup API

Add a small backend lookup service rather than loading the entire taxonomy table into Flutter.

Recommended API:

```text
GET /api/species/lookup?common_name=<中文名>
```

Response for a unique exact match:

```json
{
  "found": true,
  "common_name": "毛冠鹿",
  "scientific_name": "Elaphodus cephalophus"
}
```

No match:

```json
{
  "found": false,
  "common_name": "...",
  "scientific_name": null
}
```

If the database unexpectedly contains more than one exact row for the same Chinese name with conflicting scientific names, the endpoint must not guess. It should return `found=false` plus an ambiguity flag/message, and the user may fill the scientific name manually.

Lookup is exact after trimming whitespace. V1 does not add fuzzy taxonomy matching.

## 9. Registration-dialog autofill behavior

The existing `人工确认物种` and `学名` fields remain editable.

When the Chinese common-name field changes:

1. debounce briefly to avoid a backend request per keystroke;
2. after the text stabilizes, perform the exact local taxonomy lookup;
3. if exactly one match exists and the scientific-name field is eligible for autofill, populate the returned scientific name;
4. if no match exists, leave the scientific-name field unchanged/empty and allow manual entry.

The dialog tracks whether the scientific-name value is system-managed or user-managed.

Autofill is allowed when:

```text
scientific name is empty
OR
scientific name was last filled automatically by the current common-name lookup
```

Autofill must not overwrite a value after the user manually edits the scientific-name field.

If the user later clears the manually edited scientific name, the field becomes eligible for autofill again on the next common-name lookup.

## 10. Save and registration authority

Autofill is a convenience only. The backend identity update remains authoritative.

Saving a registry identity sends the actual current field values:

```text
common_name
scientific_name
```

A taxonomy lookup failure does not block saving or registration if existing registry conditions are otherwise satisfied. Scientific name remains optional unless future policy explicitly changes that requirement.

## 11. Candidate rename propagation

After a successful identity update:

```text
registry entry response
-> refresh registry cache
-> rebuild current-folder registry bucket labels
-> preserve selected bucket by stable `registry:<id>` key
```

Therefore:

```text
未知物种 #3
```

can become:

```text
毛冠鹿（候选，待注册）
```

without changing which files are selected or moving the user to another bucket.

After registration to Provisional the same key is retained and only the label changes to:

```text
毛冠鹿（临时注册，待确认）
```

Confirmed/Mature removes the status suffix.

## 12. Relationship to human-feedback learning

Current-folder cluster display does not itself create learning feedback.

Viewing or selecting:

```text
未知物种 #N
```

is purely a navigation/filter operation.

Only explicit human validation/registration actions defined by the human-feedback and registry designs create model-learning or registration state changes.

Likewise, taxonomy autofill only fills metadata and does not treat the database match as proof that the visual cluster is correctly identified. The user must still explicitly save/confirm the identity.

## 13. Recommended implementation surfaces

Likely touched components:

- `frontend/lib/src/screens/species_validation_screen.dart`: add registry-aware bucket identities and current-folder filtering.
- `frontend/lib/src/models/job.dart`: ensure DINOv3 box metadata exposes `registryId`/`observationId` needed for bucketing.
- `frontend/lib/src/widgets/dinov3_registry_dialog.dart`: common-name lookup, scientific-name autofill state, and post-save refresh callback.
- Flutter API client: taxonomy lookup method and registry-refresh integration.
- `system/backend/services.py` or a small taxonomy service module: exact Chinese-name lookup against `species_database.db`.
- backend API router/models: expose `/api/species/lookup` response DTO.

Do not add taxonomy logic to `SpeciesRegistry`; taxonomy lookup and registry clustering are separate responsibilities.

## 14. Python tests

Python tests must prove:

1. exact Chinese-name lookup returns the bundled scientific name;
2. surrounding whitespace is trimmed;
3. unknown Chinese name returns `found=false` without raising a 500;
4. conflicting duplicate exact matches are reported as ambiguous rather than guessed;
5. taxonomy lookup does not mutate `species_database.db`;
6. registry identity save still accepts a manually supplied scientific name when lookup fails;
7. existing registry APIs remain unchanged.

## 15. Flutter tests

Flutter tests must prove:

1. an unnamed Candidate represented in current items appears as `未知物种 #N`;
2. a registry Candidate absent from current items does not appear merely because it exists in global registry state;
3. selecting `registry:<id>` shows only current items containing that registry id;
4. a mixed known/Candidate media item can belong to both relevant buckets;
5. two Candidate clusters never merge because they share the visible word `Unknown`;
6. identity rename changes the label while preserving selection by registry id;
7. Candidate, Provisional, Confirmed, and Mature labels follow the specified suffix rules;
8. registry metadata failure falls back to `未知物种（Registry <id>）` without dropping media;
9. entering an exact Chinese common name autofills the scientific name;
10. a user-edited scientific name is not overwritten by later common-name lookups;
11. clearing a manually edited scientific name makes it eligible for autofill again;
12. no-match lookup leaves manual registration usable;
13. non-DINO species-list behavior remains unchanged.

## 16. Out of scope

V1 does not:

- run a second independent clustering algorithm only for the currently opened folder;
- duplicate the persistent `SpeciesRegistry` into folder-local registries;
- fetch taxonomy from the internet;
- perform fuzzy Chinese-name matching;
- automatically register a species because a taxonomy name matched;
- show historical files from other folders in the current-folder species bucket;
- change the existing 4/10/20 prototype learning thresholds.

## 17. Acceptance criteria

The extension is complete when:

1. opening a folder with clustered DINOv3 unknowns shows each represented Candidate as a separate species-list entry;
2. unnamed Candidates use `未知物种 #N`;
3. clicking a Candidate lists only matching media from the current folder/items;
4. mixed images remain discoverable under every represented identity;
5. naming/registering a Candidate updates the visible label without losing list selection;
6. typing a Chinese common name such as `毛冠鹿` can autofill its local scientific name from `species_database.db`;
7. manual scientific-name edits are never overwritten automatically;
8. neither cluster browsing nor taxonomy autofill creates implicit model-learning confirmation.
