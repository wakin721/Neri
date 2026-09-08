# DINOv3 Pipeline, Registry API, and Flutter Integration Design

Approved scope: extend the existing DINOv3 core on branch `DINOv3` so the reviewed native DINOv3 ViT-B/16 classifier is used by Neri's real batch inference path, unknown observations are persisted into the fingerprint-scoped registry, registry management is exposed through FastAPI, and Flutter exposes registration state, validation, registration, and DINOv3-specific video restrictions.

This design builds on commit `b5ea3b8d581b326f736c52cafa8131a98e202e31`. The model contract remains fixed at `dinov3_vitb16`, 768-dimensional CLS embeddings, `letterbox224_imagenet`, `mean_l2_normalized_crop_embeddings`, frozen encoder weights verified by SHA-256, the reviewed 17-class linear head, base prototypes, and the reviewed checkpoint threshold. Dynamic learning never modifies the supplied checkpoint.

## 1. Runtime boundaries

DINOv3 is a second-stage classifier. YOLO always performs detection and the existing `_crop_single_box()` implementation remains the only crop/padding path.

```text
image
-> YOLO detection
-> existing expanded square crop
-> native DINOv3 ViT-B/16 encoder
-> normalized 768-d embedding
-> reviewed head + base prototypes
-> base known / base rejected
-> dynamic registry matching for rejected embeddings
-> normal Neri detection result
```

`ImageProcessor` keeps `cls_model` exclusively for Ultralytics classification and gains a separate `dinov3_classifier` attribute. It gains `load_dinov3_model(manifest_path, registry)`; `load_cls_model()` retains its existing YOLO-classification responsibility. A DINOv3 model is never loaded as an Ultralytics `YOLO()` classifier.

The backend service owns registry location/lifecycle. `ImageProcessor` performs inference and exposes only in-memory observations to the service.

## 2. `detect_batch_species()` integration

When `dinov3_classifier` is active, `detect_batch_species()`:

1. runs the existing YOLO detector;
2. collects crops using `_crop_single_box()` and the existing `(result_index, box_index)` mapping;
3. encodes all valid crops through the native DINOv3 encoder;
4. classifies each normalized embedding with `DinoV3Classifier.classify_features()`;
5. maps predictions back to their original boxes;
6. preserves YOLO detection confidence as `raw_det_conf`;
7. adds DINOv3 metadata to candidate/detection data: `known_score`, threshold, head species, nearest base-prototype species, head/prototype consistency, source, registry id, registration status, and accepted flag;
8. allows only `accepted=true` predictions to become the official final species for a box.

Base-checkpoint acceptance requires both reviewed-threshold passage and head/prototype class agreement. If the base checkpoint rejects a crop, dynamic registry matching is attempted. `provisional`, `confirmed`, and `mature` matches may be accepted according to the registry threshold. `candidate` matches remain auxiliary identities and are never emitted as official Known species.

High-dimensional embeddings must not appear in detection JSON, SQLite detection records, or API responses.

`ImageProcessor` gains an exact in-memory transfer API:

```text
drain_dinov3_observations() -> tuple[DinoV3RuntimeObservation, ...]
```

`DinoV3RuntimeObservation` carries result index, box index, normalized embedding, prediction source/status/registry id, accepted flag, and detection confidence. The buffer is cleared at the start of each DINOv3 batch, cleared after draining, and cleared on inference failure.

## 3. Registry accumulation

Immediately after `_detect_image_batch()` finishes inference, the backend drains DINOv3 observations.

For every base-rejected observation:

- map result index to the original media path;
- derive camera id with `camera_id_for_path(path, input_root)`;
- parse `DetectionItem.date_taken` when available;
- call `record_unknown()` if no registry id exists;
- call `record_observation()` when the prediction already matched Candidate/Provisional/Confirmed/Mature.

This lets registered identities continue accumulating evidence toward Confirmed and Mature. Base-known species are never copied into the dynamic registry.

The independent-event boundary remains exactly 1,800 seconds. Same camera + same registry identity with gap `<1800` seconds updates one event; gap `>=1800` creates a new event. Multiple matching crops in one event update that event's normalized mean and still count as one event. Missing timestamps use source-path-derived keys.

Fast-video sampled frames are special: registry `source_path` and camera identity use the original video path, never the temporary extracted-frame path. Multiple unknown sampled frames from one video therefore cannot leave stale temporary-file references in the registry.

Registry-write errors are logged without rewriting checkpoint output. A write failure must not transform a valid base-known prediction into another species.

## 4. Registry state and state directory

Existing state semantics remain normative:

- Candidate: unregistered identity.
- At 4 independent events: build/refresh a temporary prototype; public state remains Candidate.
- Candidate is manually registrable only when: events `>=5`, cameras `>=2`, cluster purity `>=0.90`, embedding consistency `>=0.75`, and common name is non-empty.
- Successful manual registration changes Candidate to Provisional.
- Provisional automatically becomes Confirmed at events `>=10` and cameras `>=2`.
- Confirmed may become Mature at events `>=20`, cameras `>=3`, and successful deterministic two-cluster splitting; Mature stores multiple prototypes.

Registry isolation remains classifier-fingerprint based.

State root is resolved exactly as follows:

- `NERI_DINOV3_STATE_DIR` when set;
- Windows: `%LOCALAPPDATA%/Neri/dinov3`;
- other platforms: `${XDG_STATE_HOME:-~/.local/state}/neri/dinov3`.

`registry_path_for_fingerprint()` creates the fingerprint subdirectory and `registry.sqlite3` below that root.

## 5. Backend model loading and mode validation

`system/backend/services.py` resolves classification models through the model catalog. For `backend=dinov3`:

- `model_path` for a detector is mandatory;
- `classification_model_path` is the DINOv3 manifest path returned by the catalog;
- manifest/checkpoint/fingerprint are validated;
- the fingerprint-scoped `SpeciesRegistry` is opened;
- `ImageProcessor.load_dinov3_model()` is called with that registry;
- classification-only/full-image operation is rejected.

The backend rejects DINOv3 + `video_mode=all` before expensive model loading. `fast` remains supported and uses sampled frames -> YOLO -> crops -> DINOv3. `skip` remains unchanged.

## 6. FastAPI registry layer

Create `system/backend/dinov3_services.py` for model resolution, registry opening, DTO conversion, identity updates, registration, and event listing. Create `system/backend/dinov3_routes.py` with an `APIRouter`, then include it from `main_core.py`.

Public API:

```text
GET   /api/dinov3/registry
GET   /api/dinov3/registry/{registration_id}
GET   /api/dinov3/registry/{registration_id}/events
PATCH /api/dinov3/registry/{registration_id}/identity
POST  /api/dinov3/registry/{registration_id}/register
```

All calls are scoped to `classification_model_path`. GET calls receive it as a query parameter. PATCH/POST request bodies include it explicitly.

Registry summary/detail fields are: `id`, `candidate_number`, `status`, `display_name`, `common_name`, `scientific_name`, `event_count`, `camera_count`, `prototype_count`, `cluster_purity`, `embedding_consistency`, `conditions`, and `can_register`.

Event rows expose only `source_path`, `camera_id`, `started_at`, `ended_at`, `timestamp_missing`, and `sample_count`; embedding blobs are never returned.

Error mapping is fixed:

- unknown registration id -> HTTP 404;
- non-DINOv3/invalid model path -> HTTP 400;
- unmet manual registration conditions -> HTTP 409;
- corrupt/incompatible DINOv3 checkpoint -> HTTP 400 with a DINOv3-specific message.

## 7. Flutter model metadata

Extend Flutter `ModelInfo` to decode `backend`, `architecture`, `feature_dim`, `requires_detector`, `supports_video_fast`, `supports_video_all`, and `checkpoint_path`. UI capability decisions must use those fields, never filename matching.

## 8. Registration status UI

Create `frontend/lib/src/models/dinov3_registry.dart` for registry DTOs and `frontend/lib/src/widgets/dinov3_registry_dialog.dart` for the UI.

When the selected classification model has `backend == 'dinov3'`, Settings shows a `物种注册状态` row directly below the classification-model selector. Other backends do not show it. The row loads `/api/dinov3/registry` and displays compact counts such as `Candidate 3 · Provisional 2 · Confirmed 18 · Mature 6`.

The dialog supports status filtering and detail display. Candidate detail includes editable common/scientific names, event count, camera count, purity, consistency, every registration condition, `继续验证`, and `注册为新物种`.

Identity edits call PATCH and then replace the displayed detail with the server response. Registration is enabled only when `can_register=true`; POST remains authoritative. Successful registration immediately displays the returned Provisional state.

## 9. “继续验证” flow

`SettingsScreen` gains an exact callback:

```text
onOpenDinoCandidateValidation(Set<String> sourcePaths)
```

The registry dialog calls the events endpoint, collects one representative `source_path` per independent event, closes, and invokes this callback.

`MainWindow` stores a normalized temporary DINOv3 validation-path filter, switches to the existing species-validation tab, and filters the normal validation item list in `_buildValidationPage()`. `SpeciesValidationScreen` remains the only validation editor.

The validation page displays a visible Candidate filter indicator with a clear action. If historical paths are absent from currently loaded jobs/preview data, the UI displays the available subset and reports the unavailable count; it never fabricates `DetectionItem` objects.

## 10. Settings and Start video restrictions

Both Settings and Start resolve the selected classification `ModelInfo` by exact path.

For DINOv3:

- detector selection is mandatory;
- `all` is disabled;
- an existing/persisted `all` value is normalized to `fast` before saving or job creation;
- `fast` and `skip` remain selectable;
- helper text states that only sampled-frame video recognition is supported.

Start blocks submission when DINOv3 is selected without a detector and shows a specific message. Backend validation remains authoritative.

## 11. Flutter API client

Add methods to `NeriApiClient`/`api_client_core.dart` for all five registry endpoints. Registry responses are decoded into `dinov3_registry.dart` models. Network failures use the existing API error mechanism.

After PATCH/POST, Flutter refreshes from server responses rather than optimistically inventing state. Automatic Confirmed/Mature transitions appear on subsequent registry refreshes after more inference observations are processed.

## 12. Resource lifecycle and failure rules

- Encoder SHA-256 mismatch is fatal to DINOv3 loading.
- Manifest/checkpoint/architecture/feature-dimension mismatch is fatal to DINOv3 loading.
- Registry API open failures are explicit and do not modify model assets.
- Registry connections owned by a processing job are closed when that processor/job is released.
- Observation buffers are cleared on success and exceptions.
- `video_mode=all` fails before native DINOv3 encoder loading when possible.

## 13. Required tests

Python tests must prove:

1. `detect_batch_species()` uses existing YOLO crops and routes them to DINOv3, not Ultralytics classification.
2. Crop predictions map to the correct result/box indexes.
3. Candidate matches are not official species; Provisional/Confirmed/Mature registry matches can be accepted.
4. Base-rejected observations automatically create/update registry entries.
5. 29:59 remains one event and 30:00 starts a new event.
6. Repeated crops in one event update its mean without increasing event count.
7. Fast-video registry paths refer to the original video, not temporary frames.
8. Registry list/detail/events/identity/register success paths work.
9. API 404/400/409 mappings work.
10. DINOv3 without detector is rejected.
11. DINOv3 + `video_mode=all` is rejected and `fast` remains valid.
12. Existing YOLO-only and YOLO-classification suites remain green.

Flutter tests must prove:

1. DINOv3 capability metadata decodes correctly.
2. Registration row appears only for DINOv3.
3. Registry dialog renders status and conditions.
4. Identity update and registration call the right API methods and refresh from responses.
5. `继续验证` passes event source paths to MainWindow and activates validation filtering.
6. Settings and Start disable/normalize `all` to `fast` for DINOv3.
7. DINOv3 without detector blocks job creation.
8. Non-DINOv3 behavior remains unchanged.

## 14. Out of scope

This integration does not fine-tune the backbone, retrain the supplied 17-class head, modify the built-in China species catalog, synchronize registries between computers, add cloud registry storage, or enable DINOv3 full tracked-video mode.
