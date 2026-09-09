# DINOv3 Human-Feedback Multi-Prototype Learning Design

Approved scope: extend the existing native DINOv3 Multi-prototype pipeline so human validation of individual detection boxes can improve the selected classifier locally without modifying the shipped `multi_prototype.pt`. Both correctly recognized boxes and manually corrected boxes contribute positive evidence. A corrected accepted prediction also contributes a hard negative for the originally predicted species. Learned evidence is isolated by model fingerprint and only becomes eligible to affect inference after independent-event thresholds are met.

This design is additive to the existing DINOv3 pipeline and new-species registry. It does not replace `SpeciesRegistry`: the registry continues to own Unknown -> Candidate -> Provisional -> Confirmed -> Mature new-species discovery, while this design adds a separate human-feedback store for adapting already-known checkpoint species.

## 1. Fixed decisions

The following decisions are normative for the first implementation:

1. Validation for model improvement is detection-box level, not whole-image level.
2. The base `multi_prototype.pt` is immutable and is never rewritten by local learning.
3. Human learning is stored in a fingerprint-scoped local overlay.
4. Both `correct` and manually corrected samples are positive evidence for the human-confirmed species.
5. If an officially accepted prediction `A` is corrected to `B`, the sample is also a hard negative for `A`.
6. An `empty`/false-detection decision is a rejection negative, not a hard negative for a species.
7. `unverified`, ignored, or reverted feedback does not participate in learning.
8. At least 4 independent human-confirmed events are required before a local prototype is created and exposed as Provisional assistance.
9. At least 10 independent events from at least 2 cameras plus the quality gate are required before a local overlay becomes Confirmed and can participate in formal acceptance.
10. At least 20 independent events from at least 3 cameras plus the quality gate are required for Mature maintenance with up to 3 local prototypes.
11. Whole-file and batch validation may auto-feed learning only for media with exactly one learnable DINOv3 observation. Multi-box media require explicit box-level confirmation.
12. Undo/revert must also undo the corresponding model feedback and recompute affected overlay state.

## 2. Architectural boundary

The final DINOv3 classifier remains a composed prototype bank:

```text
immutable checkpoint Multi-prototypes
        +
human-feedback overlay for known checkpoint species
        +
SpeciesRegistry overlay for learned new species
        ->
DinoV3Classifier effective bank
```

The three sources have distinct ownership:

- checkpoint prototypes: distributed model assets, read-only;
- human-feedback overlay: local adaptation of checkpoint classes from explicit human confirmation;
- `SpeciesRegistry`: discovery and registration of species not represented by the checkpoint.

No feedback operation changes checkpoint bytes, checkpoint fingerprint, encoder weights, or the cloud model manifest.

## 3. Why a separate Human Feedback Store is required

Current `DinoV3Observation` values are ephemeral in `ImageProcessor._dinov3_observations`. They contain the 768-dimensional embedding and result/box mapping but disappear after the inference handoff. The existing detection JSON intentionally does not store high-dimensional embeddings, and `SpeciesRegistry` is semantically dedicated to unknown/new-species registration.

Therefore human feedback needs a separate persistent local store. The store is a sibling of the registry for the same classifier fingerprint. Its path is derived from the existing fingerprint registry path rather than rebuilding platform-specific state paths independently:

```text
registry_path_for_fingerprint(...)/../registry.sqlite3
registry_path_for_fingerprint(...)/../feedback.sqlite3
```

In implementation terms, `feedback.sqlite3` should be resolved as `registry_path.with_name("feedback.sqlite3")`. This guarantees that registry and feedback state have identical fingerprint isolation without duplicating path policy.

## 4. Observation persistence

Every DINOv3-classified detection box receives a persistent `observation_id`. The backend owns persistence; `ImageProcessor` remains an inference component and exposes only the in-memory observation transfer API.

The feedback store has an `observations` table with at least:

```text
id TEXT PRIMARY KEY
model_fingerprint TEXT NOT NULL
source_path TEXT NOT NULL
media_kind TEXT NOT NULL
box_index INTEGER NOT NULL
frame_index INTEGER NULL
timestamp_seconds REAL NULL
bbox_json TEXT NOT NULL
camera_id TEXT NOT NULL
captured_at TEXT NULL
predicted_species TEXT NOT NULL
best_known_species TEXT NULL
accepted INTEGER NOT NULL
prediction_source TEXT NOT NULL
registry_id INTEGER NULL
known_score REAL NOT NULL
threshold REAL NOT NULL
embedding BLOB NOT NULL
created_at TEXT NOT NULL
```

The embedding is normalized 768-dimensional float32 serialized as little-endian bytes. It is never included in detection JSON, Flutter DTOs, logs, or API responses.

The DINOv3 runtime observation must expose enough metadata to populate this row. In addition to the current result index, box index, embedding, accepted flag, species, source, registry id, score, threshold, and detection confidence, it must preserve `best_known_species` so rejected observations retain their nearest formal identity for diagnostics without treating that identity as an official prediction.

After a batch is classified, backend orchestration:

1. drains DINOv3 observations;
2. maps every observation to the original source media path;
3. derives camera identity using the existing camera-id logic;
4. derives capture time from media metadata and, for sampled video, its frame/timestamp context when available;
5. persists each observation;
6. receives `(result_index, box_index) -> observation_id` mappings;
7. injects only the `observation_id` and small prediction metadata into the persisted detection box record.

For fast video, `source_path` is always the original video, never a temporary extracted frame. `frame_index`/`timestamp_seconds` distinguish sampled observations inside the video.

Observation ids are opaque identifiers. The application must not reconstruct identity later from a mutable box index.

## 5. Flutter detection-box model

`DetectionBox` gains optional DINOv3 feedback metadata:

```text
observationId
predictedSpecies
feedbackStatus
```

Existing non-DINO detection boxes leave these fields null. The `observationId` is the authoritative link to `feedback.sqlite3`.

The displayed species, confidence, box coordinates, candidates, frame metadata, and track id remain unchanged.

## 6. Human feedback data model

The feedback store uses append-preserving history so correction and undo are reversible.

### 6.1 `feedback_operations`

```text
id TEXT PRIMARY KEY
kind TEXT NOT NULL              -- single, batch, box
created_at TEXT NOT NULL
reverted_at TEXT NULL
```

One UI action, including a batch action, owns one operation id.

### 6.2 `human_feedback`

```text
id INTEGER PRIMARY KEY AUTOINCREMENT
observation_id TEXT NOT NULL REFERENCES observations(id)
operation_id TEXT NOT NULL REFERENCES feedback_operations(id)
feedback_type TEXT NOT NULL     -- correct, corrected, empty, ignored
predicted_species TEXT NOT NULL
confirmed_species TEXT NULL
supersedes_id INTEGER NULL
active INTEGER NOT NULL DEFAULT 1
created_at TEXT NOT NULL
```

Only the active feedback revision for an observation contributes to learning. A later edit deactivates the previous revision and records `supersedes_id`; reverting the newer operation can reactivate the prior valid revision.

### 6.3 `species_learning_state`

```text
species TEXT PRIMARY KEY
status TEXT NOT NULL            -- collecting, provisional, confirmed, mature
event_count INTEGER NOT NULL
camera_count INTEGER NOT NULL
positive_event_count INTEGER NOT NULL
hard_negative_event_count INTEGER NOT NULL
prototype_count INTEGER NOT NULL
positive_coverage REAL NULL
hard_negative_false_accept_rate REAL NULL
needs_review INTEGER NOT NULL DEFAULT 0
active_generation INTEGER NULL
updated_at TEXT NOT NULL
```

### 6.4 `overlay_prototypes`

```text
species TEXT NOT NULL
generation INTEGER NOT NULL
prototype_index INTEGER NOT NULL
embedding BLOB NOT NULL
event_count INTEGER NOT NULL
camera_count INTEGER NOT NULL
created_at TEXT NOT NULL
PRIMARY KEY(species, generation, prototype_index)
```

Prototype embeddings are stored in the centered feature space expected by the current Multi-prototype classifier.

## 7. Feedback semantics

Feedback is interpreted as follows.

### Correct prediction

```text
predicted = A
confirmed = A
-> positive(A)
```

No hard negative is produced.

### Accepted prediction corrected to another species

```text
predicted = A
accepted = true
confirmed = B
-> positive(B)
-> hard_negative(A)
```

Hard negatives are generated only from an officially accepted wrong prediction. A rejected `Unknown` or purely assistive Provisional suggestion does not become a species-specific hard negative by default.

### Rejected/Unknown corrected to a known checkpoint species

```text
predicted = Unknown
confirmed = B
-> positive(B)
```

### Empty / false detector box

```text
feedback_type = empty
-> rejection_negative
-> no species positive
-> no species hard negative
```

Rejection negatives are retained for future rejection calibration but do not affect prototype geometry in v1.

### Ignored / unverified

The row is audit-only and does not enter any event/prototype calculation.

## 8. Scope of known-species adaptation

The Human Feedback Store adapts species present in the selected checkpoint's class list. A corrected name must resolve unambiguously to a checkpoint class before it can create a known-species overlay.

New species continue through the existing `SpeciesRegistry` workflow. This avoids representing the same new identity independently in both the feedback store and registry.

Feedback metadata may still record a correction involving a registry species for audit purposes, but v1 does not create a second human-feedback overlay for a registry-owned species.

## 9. Independent-event aggregation

Learning thresholds count independent events, not photos or boxes.

The event boundary remains 1,800 seconds to match the existing DINOv3 registry semantics:

- same confirmed species + same camera + gap `< 1800 s` -> same event;
- gap `>= 1800 s` -> new event;
- different camera -> different event;
- missing timestamps -> source-path-derived event identity.

All active positive box embeddings within one event are L2-normalized, averaged, and normalized again to produce one event embedding. Multiple animals/boxes or burst photos from the same event increase sample support but do not inflate `event_count`.

Hard negatives are also evaluated at independent-event level so a long burst of one confusion cannot dominate the quality metric.

## 10. Prototype construction and lifecycle

Positive event embeddings remain in normalized raw encoder space in storage. To build classifier overlay prototypes:

1. load positive event embeddings for a species;
2. subtract the checkpoint `feature_center` from each event embedding;
3. run the existing deterministic K-means splitter in centered space;
4. store the resulting centered prototypes as a new generation.

Prototype limits are:

```text
Collecting   0-3 events                         -> 0 prototypes in inference
Provisional  >=4 events                         -> max 1 local prototype
Confirmed    >=10 events and >=2 cameras        -> max 2 local prototypes
Mature       >=20 events and >=3 cameras        -> max 3 local prototypes
```

The 4-event threshold creates a Provisional prototype immediately when valid embeddings are available. Provisional prototypes are assistive only: they are evaluated only after formal rejection and cannot independently produce `accepted=true`.

Confirmed and Mature local prototypes enter the formal bank and therefore compete directly with checkpoint prototypes. For a base species such as `盘羊`, checkpoint and local records all carry the same species name; they do not create duplicate user-visible classes.

## 11. Quality gate and hard-negative use

Hard negatives do not subtract from prototypes and are not used for gradient training. In v1 they are used only as a promotion/replacement guard and for diagnostics.

For a candidate local prototype generation, evaluate event-level centered embeddings with the same cosine acceptance score and checkpoint threshold used by the Multi-prototype classifier.

Define:

```text
positive_coverage =
  fraction of positive event embeddings whose best local-prototype score >= checkpoint threshold

hard_negative_false_accept_rate =
  fraction of hard-negative event embeddings whose best local-prototype score >= checkpoint threshold
```

The quality gate is fixed at:

```text
positive_coverage >= 0.80
AND
(no hard-negative events OR hard_negative_false_accept_rate <= 0.20)
```

No new per-species threshold is learned in v1.

Promotion behavior:

- 4 events: create/refresh Provisional regardless of hard-negative availability; it remains assistive only.
- 10 events + 2 cameras: promote to Confirmed only if the quality gate passes.
- 20 events + 3 cameras: promote/refresh Mature only if the quality gate passes.

For a species that is already Confirmed or Mature, a newly computed generation that fails the quality gate never replaces the active generation. The last known-good generation remains active, `needs_review=1` is set, and the failed generation is not exposed to inference. Status remains Confirmed/Mature so inference semantics do not change merely because review is needed.

For a Provisional species that fails the Confirmed quality gate, it simply remains Provisional.

## 12. Effective-bank composition

`DinoV3Classifier._effective_bank()` must combine three sources without changing the nearest-prototype decision rule:

```text
formal = checkpoint base
       + human-feedback Confirmed/Mature overlay
       + SpeciesRegistry Confirmed/Mature overlay

provisional = human-feedback Provisional overlay
            + SpeciesRegistry Provisional overlay
```

Candidate/Collecting data never enters the classifier bank.

Because local overlays for a checkpoint species use the exact checkpoint class name, existing per-species candidate collapsing continues to expose a single species identity while allowing multiple local and base prototypes to compete geometrically.

## 13. Box-level validation UX

The existing file-level validation toolbar retains its current meaning. Box-level learning controls are added separately so users cannot confuse whole-file ecological validation with model feedback.

### Image interaction

The media viewer gains box hit-testing using the same `BoxFit.contain` geometry as the painter. Clicking a DINOv3 box:

- highlights the selected box;
- stores `selectedObservationId`;
- shows a compact `检测框校验` control strip below the viewer;
- does not alter the selected file/group.

### Video interaction

For fast-video results, only boxes visible for the current frame/timestamp are hit-testable. Selection is by `observationId`, not current list index or track id.

### Box feedback controls

The strip provides:

```text
正确
修改物种
空 / 误检
不参与学习
退出框级校验
```

`修改物种` reuses the existing species selection UI where practical.

A box correction updates both:

1. the persisted detection result used for ecological validation/export; and
2. the human-feedback store used for local model adaptation.

The backend recomputes the file-level species/count summary from the updated boxes before returning the refreshed `DetectionItem`.

## 14. Validation API boundary

Whole-file ecological validation remains supported and backward compatible. Add a dedicated box-validation operation for DINOv3 feedback rather than overloading existing file semantics.

The box request contains at least:

```text
input_path
file_path
classification_model_path
observation_id
action                 -- correct, update, empty, unverified
species_name            -- required for update
feedback_operation_id
```

The box response returns:

```text
updated DetectionItem
feedback result/state for the affected species
```

The backend validates that:

- the selected classification model is the DINOv3 model matching the observation fingerprint;
- the observation belongs to the requested source media;
- the observation id exists;
- a corrected known species resolves to the checkpoint class list.

Ecological detection-data persistence is primary. Human-feedback persistence is local auxiliary model state. If learning-state persistence fails, the API must return an explicit DINOv3 learning error rather than silently claiming model improvement; it must never rewrite checkpoint assets.

## 15. File-level automatic feedback

Whole-file `correct`, `update`, and `empty` actions remain fast for normal validation, but automatic model feedback is conservative.

A file-level action may automatically create DINOv3 feedback only when the file has exactly one learnable DINOv3 observation for the selected fingerprint.

Examples:

```text
one DINOv3 box + file marked correct
-> positive for that box's confirmed species

one DINOv3 box + file changed A -> B
-> positive(B), and hard_negative(A) when A was officially accepted

one DINOv3 box + file marked empty
-> rejection negative

multiple DINOv3 boxes
-> ecological validation succeeds
-> no automatic model feedback
-> explicit box-level confirmation required
```

This rule is enforced in the backend, not only in Flutter.

## 16. Batch validation safety

Batch validation uses the same per-file eligibility rule.

For each selected media item:

- exactly one learnable DINOv3 observation -> eligible for automatic feedback;
- zero observations -> no DINOv3 feedback;
- two or more observations -> skip automatic learning.

The UI should report a summary such as:

```text
已标记 100 张
模型改进：82 个单框样本已加入人工确认池；18 个多框样本未自动学习
```

The frontend may pre-compute the expected eligible/skipped counts for immediate UX, but backend eligibility remains authoritative.

V1 does not add bulk selection of arbitrary individual boxes across different media.

## 17. Undo and revision behavior

Every file-level, batch, or box-level validation action that creates model feedback carries a `feedback_operation_id` also stored by the validation UI's undo history.

When an operation is reverted:

1. ecological validation is restored through the existing undo path;
2. the matching feedback operation is marked reverted;
3. feedback rows created by that operation are deactivated;
4. a superseded prior feedback revision is reactivated where applicable;
5. all affected species independent events and candidate overlay generations are recomputed;
6. active known-good generation selection is reevaluated.

Undo therefore restores both visible validation state and model-learning state.

## 18. Failure and consistency rules

- The base checkpoint is read-only under all feedback operations.
- Feedback DB writes use transactions, foreign keys, WAL journal mode, and explicit rollback on failure.
- A malformed/corrupt feedback database produces a DINOv3-specific local-state error; the application must not delete or recreate it silently.
- High-dimensional embeddings never enter detection JSON/API responses.
- A failed feedback write does not mutate prototype-bank state in memory.
- A failed prototype recomputation leaves the previous active known-good generation intact.
- Observation/feedback fingerprint mismatch is a hard error.
- Human feedback is local and is not uploaded by the existing model-distribution or training-upload path.

## 19. Model updates and fingerprint isolation

Feedback state is tied to the checkpoint fingerprint. When a distributed DINOv3 checkpoint changes fingerprint, its prior `feedback.sqlite3` is not automatically loaded into the new model.

V1 performs no automatic migration, because old embeddings, feature center, prototype geometry, and thresholds may not be compatible with the new checkpoint. Source paths remain available in the old store for a future explicit re-embedding/migration feature.

## 20. Expected implementation components

The implementation should preserve existing module boundaries and avoid expanding the already-large historical backend service surface unnecessarily.

Recommended components:

- `system/dinov3/feedback.py`: SQLite schema, observation persistence, feedback history, event aggregation, prototype generations, quality-gate logic, overlay `PrototypeBank` provider.
- `system/backend/dinov3_feedback_service.py`: model/fingerprint resolution, feedback-store lifecycle, persistence orchestration, box-validation integration helpers.
- `system/backend/dinov3_feedback_routes.py`: box-feedback/revert API surface if the box operation is not placed directly under the existing validation router.
- `system/dinov3/classifier.py`: compose the human-feedback overlay with the current checkpoint and registry banks.
- `system/image_processor.py`: extend the ephemeral observation DTO only with metadata required for persistence; do not add SQLite responsibilities.
- `system/backend/services.py`: persist drained observations and inject observation ids at the existing inference-to-detection persistence boundary.
- `system/backend/models.py`: request/response DTOs for box validation and feedback operation ids.
- `system/backend/main.py`: include any new router without requiring `main_core.py` churn unless unavoidable.
- `frontend/lib/src/models/job.dart`: optional observation/feedback fields on `DetectionBox`.
- `frontend/lib/src/widgets/detection_media_viewer.dart`: hit-testing and selected-box highlighting.
- `frontend/lib/src/screens/species_validation_screen.dart`: box-level feedback strip, operation-id wiring, batch eligibility messaging.
- Flutter API client: box feedback and feedback-revert calls.

Exact file placement may be adjusted during planning if current branch structure provides a smaller dependency surface, but the architectural boundaries above remain normative.

## 21. Required Python tests

Python tests must prove at least:

1. every DINOv3 box can be persisted with a unique `observation_id` and mapped back to the correct result/box;
2. embeddings are stored only in the feedback DB and never emitted in detection JSON/API DTOs;
3. fingerprint mismatches are rejected;
4. `correct A` creates positive(A) and no hard negative;
5. accepted `A -> B` creates positive(B) plus hard_negative(A);
6. rejected Unknown -> B creates positive(B) without a species hard negative;
7. `empty` creates only rejection-negative feedback;
8. ignored/unverified feedback is excluded from learning;
9. 29:59 remains one independent event and 30:00 starts a new event;
10. multiple same-event positives aggregate to one normalized event embedding;
11. hard negatives are also evaluated at event level;
12. 0-3 events produce no inference overlay;
13. 4 events produce at most one Provisional prototype and cannot independently yield formal acceptance;
14. 10 events + 2 cameras + passing quality gate promotes to Confirmed with at most 2 prototypes;
15. failing the Confirmed quality gate leaves the species Provisional;
16. 20 events + 3 cameras + passing gate promotes to Mature with at most 3 prototypes;
17. a failed new generation for a previously Confirmed/Mature species preserves the prior active generation and sets `needs_review`;
18. feedback overlay uses the same checkpoint class name and does not create duplicate class identities;
19. base checkpoint file SHA-256 is unchanged after all feedback/prototype operations;
20. file-level single-box validation auto-feeds learning;
21. file-level multi-box validation never auto-feeds learning;
22. batch feedback applies only to single-observation media;
23. reverting an operation removes its learning contribution and restores the superseded revision where applicable;
24. fast-video observations reference original video paths and remain uniquely selectable by observation id;
25. existing SpeciesRegistry, DINOv3 classifier, YOLO-only, and YOLO-classification tests remain green.

## 22. Required Flutter tests

Flutter tests must prove at least:

1. `DetectionBox` decodes optional `observation_id` without affecting non-DINO boxes;
2. image box hit-testing selects the correct observation under `BoxFit.contain` scaling;
3. fast-video hit-testing targets only currently visible boxes;
4. selected-box highlighting is rendered without changing whole-file selection;
5. the box-level validation strip appears only for a selectable DINOv3 observation;
6. `正确`, `修改物种`, `空/误检`, and `不参与学习` send the correct box-feedback request;
7. whole-file validation controls retain their existing semantics;
8. one-box file validation is eligible for automatic learning;
9. multi-box file validation is marked as skipped for automatic learning;
10. batch UI reports eligible and skipped counts;
11. undo sends the matching feedback operation id for reversal;
12. non-DINO validation behavior remains unchanged.

## 23. Out of scope

This design does not:

- fine-tune DINOv3 encoder weights;
- train or rewrite the checkpoint classification head;
- rewrite `multi_prototype.pt`;
- create gradient-based hard-negative training;
- learn new per-species acceptance thresholds in v1;
- use unverified model predictions as self-training labels;
- automatically learn from multi-box whole-file validation;
- add cross-device/cloud synchronization of feedback state;
- upload feedback embeddings or local validation history;
- automatically migrate feedback across checkpoint fingerprints;
- train or adapt the YOLO detector from `empty` feedback;
- replace the existing new-species `SpeciesRegistry` lifecycle.

## 24. Acceptance criteria

The feature is complete when a user can:

1. run YOLO + DINOv3 normally;
2. later reopen validation and select an individual DINOv3 detection box using its persistent observation id;
3. confirm a correct prediction or correct a wrong one;
4. have that feedback survive process restart without exposing raw embeddings to the UI;
5. accumulate independent events without burst-photo inflation;
6. obtain an assistive local prototype after 4 events;
7. obtain a formal local overlay only after 10 events, 2 cameras, and the fixed quality gate;
8. grow to at most 3 Mature local prototypes after 20 events and 3 cameras;
9. use corrected accepted predictions as hard-negative quality guards without subtractive prototype updates;
10. undo validation and remove the corresponding learning contribution;
11. continue using the immutable cloud-distributed checkpoint with no fingerprint or file-hash mutation;
12. preserve existing new-species registry and non-DINO validation behavior.
