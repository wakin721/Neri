# DINOv3 Pipeline, Registry API, and Flutter Integration Design

Approved scope: extend the existing DINOv3 core on branch `DINOv3` so the reviewed native DINOv3 ViT-B/16 classifier is used by Neri's real batch inference path, unknown observations are persisted into the fingerprint-scoped registry, registry management is exposed through FastAPI, and Flutter exposes registration state, validation, registration, and DINOv3-specific video restrictions.

This design builds on commit `b5ea3b8d581b326f736c52cafa8131a98e202e31`. The existing model contract remains unchanged: `dinov3_vitb16`, 768-dimensional CLS embeddings, `letterbox224_imagenet`, `mean_l2_normalized_crop_embeddings`, frozen encoder weights verified by SHA-256, reviewed linear head, base prototypes, and checkpoint threshold. The supplied checkpoint is never modified when new species are learned.

## 1. Runtime architecture

DINOv3 remains a second-stage classifier. YOLO is always responsible for object detection and crop extraction.

The supported image path is:

```text
input image
-> existing YOLO detector
-> existing 10% expanded square/padded crop
-> native DINOv3 ViT-B/16 encoder
-> normalized 768-d crop embedding
-> reviewed linear head + reviewed base prototypes
-> base known / base rejected
-> dynamic registry prototype matching when base rejected
-> normal Neri detection result
```

`ImageProcessor` keeps the existing YOLO classification path intact. DINOv3 is represented separately from `cls_model`, for example as `dinov3_classifier`, so existing `.pt/.onnx/.engine` YOLO classification behavior does not change.

`ImageProcessor.load_cls_model()` or a focused companion loader resolves whether the selected classification model is `backend=yolo` or `backend=dinov3` from the model manifest. A DINOv3 manifest must resolve to a valid reviewed checkpoint and encoder assets. A DINOv3 classifier cannot be loaded without an active detector model.

Backend service code owns persistence and registry lifecycle. `ImageProcessor` performs inference only; it must not decide filesystem locations for SQLite state or write the registry directly.

## 2. Batch crop inference in `detect_batch_species()`

`ImageProcessor.detect_batch_species()` continues to run YOLO detection first and continues to use `_crop_single_box()` for crop generation. The existing crop mapping `(result_index, box_index)` is retained.

When `dinov3_classifier` is active:

1. collect all valid YOLO crops in the batch;
2. encode the crops in batches with the native DINOv3 encoder;
3. L2-normalize each crop embedding;
4. classify each embedding with `DinoV3Classifier.classify_features()`;
5. map the prediction back to the original detection box;
6. preserve the existing detection confidence as `raw_det_conf`;
7. expose DINOv3 metadata in the candidate/detection structure, including `known_score`, checkpoint threshold, head species, nearest base prototype species, head/prototype consistency, source, registry id, and registration status;
8. only an accepted prediction may become the official final species for a detection box.

A base checkpoint result is accepted only when its prototype score reaches the reviewed threshold and the linear-head class equals the nearest base-prototype class. A base-rejected embedding is then checked against the dynamic registry. `provisional`, `confirmed`, and `mature` registry matches may be accepted according to the registry threshold; a `candidate` match remains an unregistered auxiliary identity and must not be emitted as an official known species.

The normal return value of `detect_batch_species()` remains compatible with current callers. High-dimensional embeddings must never be serialized into ordinary detection JSON.

To transfer embeddings to the backend service without leaking them into API results, `ImageProcessor` maintains an ephemeral DINOv3 observation buffer for the current call. A focused method such as `drain_dinov3_observations()` returns immutable in-memory observations and clears the buffer. Each observation contains only runtime objects needed by the service: source result index, box index, normalized embedding, prediction source/status/registry id, and detection confidence. The buffer is cleared at the start of every DINOv3 inference call and after draining, preventing stale observations from crossing batches.

## 3. Unknown and registered-species accumulation

After `_detect_image_batch()` receives normal detection results, the backend immediately drains DINOv3 observations from the processor.

For each base-rejected observation:

- resolve `source_path` from the batch index;
- derive `camera_id` with the existing `camera_id_for_path()` rule using the job input root when available;
- use EXIF capture time from the corresponding `DetectionItem.date_taken` when it can be parsed;
- call `SpeciesRegistry.record_unknown()` when no registry id exists;
- call `SpeciesRegistry.record_observation()` when the embedding already matched an existing Candidate/Provisional/Confirmed/Mature entry.

This means registered species continue accumulating events after registration, allowing automatic progression to Confirmed and Mature.

The independent-event boundary remains exactly 1,800 seconds. Same camera + same registry identity with a gap below 1,800 seconds updates the same event embedding; a gap at or above 1,800 seconds creates a new event. Missing timestamps are isolated using source-path-derived event keys. Multiple crops belonging to the same candidate inside one independent event are averaged in the registry and count as one event.

Base known species are not copied into the dynamic registry.

Registry accumulation happens after inference, so SQLite failures cannot corrupt the original checkpoint or encoder. A registry write failure is logged and exposed as non-fatal DINOv3 registration metadata where possible; it does not silently convert a known model prediction into another species.

## 4. Registry state and persistence

The existing state semantics are retained:

- `Candidate`: newly discovered identity; not an official known species.
- At 4 independent events: create/refresh a temporary prototype while the public status remains Candidate.
- Manual registration becomes available when all conditions are true: at least 5 independent events, at least 2 cameras, cluster purity at or above `0.90`, embedding consistency at or above `0.75`, and a confirmed common name.
- Manual registration changes Candidate to `Provisional`.
- `Provisional` automatically becomes `Confirmed` at at least 10 independent events from at least 2 cameras.
- `Confirmed` may become `Mature` at at least 20 events from at least 3 cameras when deterministic two-cluster prototype splitting succeeds; Mature uses multiple prototypes.

The SQLite database remains isolated by classifier fingerprint. Replacing the checkpoint, encoder hash, architecture, feature dimension, preprocessing contract, or aggregation contract must produce a different fingerprint and therefore a different registry namespace.

Persistent state root:

- if `NERI_DINOV3_STATE_DIR` is set, use it;
- on Windows, default to `%LOCALAPPDATA%/Neri/dinov3`;
- on other platforms, default to `${XDG_STATE_HOME:-~/.local/state}/neri/dinov3`.

The existing `registry_path_for_fingerprint()` then creates the fingerprint-specific SQLite path below that root.

## 5. Backend model resolution and job restrictions

`system/backend/services.py` resolves the selected classification model through the model catalog rather than treating every classification path as an Ultralytics model.

For a DINOv3 manifest:

- a detection model path is mandatory;
- the manifest and checkpoint are validated before inference;
- a fingerprint-scoped registry is opened;
- the native encoder/classifier is attached to `ImageProcessor`;
- the registry is attached to `DinoV3Classifier` for dynamic prototype matching;
- full-image classification-only operation is rejected.

A DINOv3 job with `video_mode=all` is rejected before processing with a clear user-facing message. `video_mode=fast` is supported and continues to use sampled frames followed by YOLO detection, crops, and DINOv3 classification. `video_mode=skip` remains unchanged.

These restrictions are enforced in the backend even if an old or custom client bypasses Flutter.

## 6. FastAPI registry service and routes

Create a focused backend registry service, separate from image processing, that resolves a DINOv3 manifest, validates the checkpoint, derives the model fingerprint, opens the correct SQLite registry, and converts registry records to API schemas.

The public API is:

```text
GET   /api/dinov3/registry
GET   /api/dinov3/registry/{registration_id}
GET   /api/dinov3/registry/{registration_id}/events
PATCH /api/dinov3/registry/{registration_id}/identity
POST  /api/dinov3/registry/{registration_id}/register
```

All endpoints are explicitly scoped to `classification_model_path`. Read endpoints receive it as a query parameter. Write requests include it in the request body so a stale UI cannot accidentally mutate the registry for another DINOv3 model.

Registry summary/detail responses include:

- `id`;
- `candidate_number`;
- `status`;
- `display_name`;
- `common_name`;
- `scientific_name`;
- `event_count`;
- `camera_count`;
- `prototype_count`;
- `cluster_purity`;
- `embedding_consistency`;
- individual registration conditions;
- `can_register`.

The events endpoint returns one row per independent event with `source_path`, `camera_id`, `started_at`, `ended_at`, `timestamp_missing`, and sample count. It never returns embedding blobs.

`PATCH .../identity` accepts `common_name` and optional `scientific_name`. Empty common names are rejected. `POST .../register` re-checks all registration conditions server-side and returns HTTP 409 when the entry is not currently eligible. Missing entries return 404. Non-DINOv3 model paths return 400.

## 7. Flutter model metadata

Extend Flutter `ModelInfo` to parse the backend metadata already exposed for DINOv3:

- `backend`;
- `architecture`;
- `feature_dim`;
- `requires_detector`;
- `supports_video_fast`;
- `supports_video_all`;
- `checkpoint_path`.

A helper resolves the selected classification `ModelInfo` by path. UI behavior must use these explicit capabilities instead of filename substring matching.

## 8. Settings-page registration entry and dialog

When the currently selected classification model has `backend == 'dinov3'`, the classification-model section shows a `物种注册状态` row directly below the selector. Non-DINOv3 models do not show this row.

The row displays compact counts such as:

```text
Candidate 3 · Provisional 2 · Confirmed 18 · Mature 6
```

Opening it shows a dedicated DINOv3 registry dialog/widget. The dialog supports status filtering and shows each registration's name/number, status, independent-event count, camera count, purity, consistency, and registration conditions.

A detail view follows the established example:

```text
未知物种 #17
状态：Candidate
人工确认物种
豹猫
Prionailurus bengalensis
5 个独立事件
3 台相机

注册条件
✓ ≥5 个独立事件
✓ ≥2 台相机
✓ cluster purity ≥ threshold
✓ embedding consistency ≥ threshold
✓ 已确认物种名称

[继续验证] [注册为新物种]
```

The identity fields are editable for Candidate entries and are saved through the identity API. The register button is enabled only when the API reports `can_register=true`; the backend still performs the authoritative re-check.

## 9. “继续验证” integration

The registry dialog loads the candidate's event endpoint and obtains representative `source_path` values, one per independent event.

`SettingsScreen` receives a callback from `MainWindow`, for example `onOpenDinoCandidateValidation(Set<String> sourcePaths)`. When `继续验证` is pressed:

1. close the registry dialog;
2. send the event source paths to `MainWindow`;
3. `MainWindow` stores a normalized temporary validation-path filter;
4. switch to the existing species-validation tab;
5. `_buildValidationPage()` filters the current job/preview items to those source paths;
6. show a visible filter indicator with an action to clear the Candidate filter and return to the normal validation list.

No second validation system is created. Existing marking, species editing, quick marks, export, metadata loading, and media viewing remain provided by `SpeciesValidationScreen`.

If some historical event paths are not present in the currently loaded job/preview data, the UI shows the available subset and states how many requested files were unavailable rather than fabricating `DetectionItem` records.

## 10. Settings-page and start-page video restrictions

Both Settings and Start pages inspect the selected classification model's capability metadata.

When DINOv3 is selected:

- the detection-model selector remains required;
- `完整识别` (`all`) is disabled in the video-mode selector;
- if the persisted/current mode is `all`, it is immediately normalized to `fast` before saving or creating a job;
- `快速识别` (`fast`) remains selectable;
- `跳过视频` (`skip`) remains selectable;
- helper text explains that DINOv3 currently supports sampled-frame video recognition only.

The Start page prevents job submission when DINOv3 is selected without a detector and presents a specific message rather than a generic model error.

Backend validation remains authoritative and rejects invalid combinations even if Flutter state becomes stale.

## 11. API client and Flutter data models

Add Flutter models for registry summary/detail/event responses and API-client methods for the five registry endpoints. Network errors use the existing Neri API error surface.

After identity updates or registration, the dialog refreshes the single detail entry and summary counts from backend responses rather than mutating local status optimistically.

After a successful registration, the UI shows the returned status (`Provisional`) immediately. Later automatic upgrades to Confirmed/Mature appear on the next registry refresh after additional observations have been processed.

## 12. Error handling and resource lifecycle

- Encoder/checkpoint/manifest incompatibility fails model loading with a clear DINOv3-specific message.
- Encoder SHA-256 mismatch remains fatal for DINOv3 model loading.
- Registry database opening errors fail registry API requests clearly but do not rewrite model assets.
- Per-batch registry write failures are logged and do not crash already completed base inference for unrelated files.
- Registry connections created for a processing job are closed when the processor/job is released.
- The ephemeral observation buffer is always cleared on success and on inference exceptions.
- DINOv3 full-video requests fail before expensive model loading when possible.

## 13. Required tests

Python tests must cover at least:

1. `detect_batch_species()` uses the existing YOLO crop path and routes crops to DINOv3 instead of Ultralytics classification.
2. Crop predictions map back to the correct result/box indexes.
3. Candidate matches are not emitted as official known species; Provisional/Confirmed/Mature registry matches may be accepted.
4. Base-rejected observations automatically create/update registry entries.
5. Same candidate + same camera at 29:59 stays one event; 30:00 creates a second event.
6. Repeated crops in one event update the event mean without increasing `event_count`.
7. Registry API list/detail/events/identity/register success paths.
8. Registry API 404, invalid model, and unmet registration-condition behavior.
9. DINOv3 without detector is rejected.
10. DINOv3 + `video_mode=all` is rejected; fast remains valid.
11. Existing YOLO-only and YOLO-classification tests remain green.

Flutter tests must cover at least:

1. DINOv3 model metadata decoding.
2. Registration row appears only for DINOv3.
3. Registry dialog renders status and registration conditions.
4. Identity update and register actions invoke the correct API methods and refresh state.
5. `继续验证` passes representative source paths to MainWindow and activates validation filtering.
6. DINOv3 disables `all` and normalizes an existing `all` selection to `fast` in both Settings and Start flows.
7. DINOv3 without a detector prevents job submission.
8. Non-DINOv3 behavior is unchanged.

## 14. Out of scope

This integration does not fine-tune the DINOv3 backbone, retrain the supplied 17-class linear head, modify the built-in China species catalog, synchronize registries across computers, enable cloud registry storage, or enable DINOv3 full tracked-video mode. Those require separate designs and calibration work.
