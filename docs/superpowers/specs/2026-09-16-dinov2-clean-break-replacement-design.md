# DINOv2 Clean-Break Replacement Design

## Goal

Replace Neri's DINOv3 classification subsystem with a DINOv2-only runtime based on `facebook/dinov2-base` (ViT-B/14, 768-dimensional CLS embeddings), and make the validated Adaptive Multi-prototype + Fusion-rejection pipeline the canonical open-set classifier.

This is an intentional breaking change. Neri will not execute DINOv3 manifests, DINOv3 checkpoints, DINOv3 prototype banks, or DINOv3 embedding feedback after the migration.

## Scope

The replacement covers the complete DINO classification path:

- local DINO encoder component installation and health checks;
- classifier checkpoint validation and manifest discovery;
- detector-crop preprocessing and embedding inference;
- Adaptive Multi-prototype closed-set classification;
- Fusion-rejection / safe prototype fallback open-set rejection;
- Registry, feedback, observation persistence, and prototype regeneration;
- model catalog and saved-model path resolution;
- backend API names used by the Flutter client;
- Flutter model/environment UI copy and model metadata;
- packaging and CI/runtime verification.

YOLO detector behavior, ordinary YOLO classification models, tracker behavior, and unrelated model-sync semantics remain out of scope.

## Breaking-change policy

DINOv3 is removed rather than retained behind an adapter layer.

The following become unsupported runtime inputs:

- `backend: "dinov3"` manifests;
- `dinov3_vitb16` classifier checkpoints;
- DINOv3 encoder weights and source mirrors;
- persisted 768-D DINOv3 prototype vectors;
- persisted DINOv3 observation embeddings and prototype overlays.

When an old DINOv3 manifest is selected, Neri must fail with an explicit migration error explaining that the current release supports DINOv2 only. It must not reinterpret a DINOv3 checkpoint as DINOv2 merely because both feature dimensions are 768.

Human-confirmed semantic labels may be reused only by re-encoding their source crops/images with DINOv2. Existing DINOv3 embedding vectors themselves are never migrated.

## Canonical DINOv2 model

The only supported backbone in this migration is:

- architecture id: `dinov2_vitb14`;
- model family: `facebook/dinov2-base`;
- feature dimension: `768`;
- embedding source: CLS token (`last_hidden_state[:, 0]`);
- encoder output transform: per-image L2 normalization;
- preprocessing: `letterbox224_imagenet`;
- event aggregation: `mean_l2_normalized_crop_embeddings`.

The deployed encoder must reproduce the benchmark feature path rather than approximate it with a different DINOv2 implementation.

## Local component layout

The canonical component root changes from `res/model/DINOv3` to `res/model/DINOv2`.

Expected layout:

```text
res/model/DINOv2/
├── model/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── classifier.pt
├── classifier.neri.json
├── LICENSE_DINOv2.md
└── install.json
```

`install.json` remains the integrity authority and contains a complete file inventory with SHA-256 hashes and sizes. The runtime must use local assets only and must not download from Hugging Face during classification.

The optional ML environment keeps the shared PyTorch installation. The DINOv2 component additionally requires compatible `transformers` and `safetensors` packages. The component installer installs those dependencies only when required by the DINOv2 runtime.

## Preprocessing contract

The existing Neri crop preprocessing contract is retained:

1. RGB conversion;
2. aspect-ratio preserving resize into a 224 x 224 canvas;
3. existing Neri letterbox fill value;
4. ImageNet mean/std normalization;
5. float tensor in CHW order.

No `AutoImageProcessor` resize/crop policy may replace this path at runtime, because doing so would change the feature distribution relative to the benchmark and trained prototype checkpoint.

## Encoder contract

`system/dinov2/encoder.py` owns the only supported encoder implementation.

Its public behavior is:

```text
encode(crops) -> float32 ndarray shaped (N, 768)
```

Every returned row must be finite and L2-normalized.

The encoder loads the local DINOv2 model with `local_files_only=True`, verifies the declared model asset fingerprints before inference, runs under `torch.inference_mode()`, and optionally uses CUDA FP16 only when enabled and available.

The DINOv3-specific source injection path (`NERI_DINOV3_SOURCE`, `dinov3.hub.backbones`, mirrored Meta source code) is deleted.

## Classifier comparison space

The classifier uses the same feature geometry as the validated benchmark:

1. encoder emits an L2-normalized 768-D vector `x`;
2. subtract the train-only stored feature center `c`;
3. L2-normalize again to obtain the comparison vector `z` (CL2N);
4. compare `z` only with prototypes and linear parameters trained in the same DINOv2 feature space.

A checkpoint lacking the required feature center is invalid.

## Adaptive Multi-prototype head

The checkpoint stores a variable number of prototypes per class rather than a global fixed `k`.

Required fields include:

- `classes`;
- `feature_center`;
- `prototypes`;
- `prototype_class_indices`;
- `prototypes_per_class`;
- the prototype-selection/training metadata needed to reproduce the frozen head;
- DINOv2 encoder identity and hashes.

Closed-set prediction remains nearest-prototype classification in the CL2N comparison space.

For each class, the class prototype score is the score of that class's nearest prototype. The winning class is determined by the frozen checkpoint policy; deployed inference does not re-fit prototypes or choose a new `k` automatically.

## Fusion-rejection head

The canonical open-set head is represented by one checkpoint schema that supports both a real linear fusion and a validation-selected safe prototype fallback.

Required rejection metadata includes:

- prototype score scaling;
- optional linear score scaling;
- `class_alpha`;
- `score_alpha`;
- `margin_weight`;
- cosine threshold;
- fused rejection score threshold;
- linear type: `none`, `ridge`, or another explicitly supported frozen linear head;
- optional linear coefficient matrix and intercept vector.

For the current `Fusion-rejection` policy, `class_alpha` is zero: prototype geometry decides the closed-set class. Linear logits may affect rejection confidence but must not silently change the predicted species.

The linear head is stored as numeric tensors (`coef`, `intercept`) rather than as a pickled scikit-learn estimator. Runtime inference therefore does not require scikit-learn.

A validation-selected fallback is encoded with `linear_type: none`. It still uses the advanced prototype/margin rejection parameters from the frozen checkpoint. This preserves the benchmark's safe-selection behavior and avoids claiming a Ridge/SVM contribution where validation selected none.

## Rejection calculation

Inference must mirror the frozen benchmark formula and parameterization. At minimum, the runtime computes:

- nearest-prototype class scores;
- selected-class prototype cosine similarity;
- selected-class / runner-up score gap used by the configured margin term;
- optional frozen linear logits;
- the configured fused rejection score;
- acceptance using the frozen cosine and rejection-score thresholds.

Thresholds are loaded from the checkpoint/manifest pair and are never recalibrated on user inference data.

## Registry and human feedback

Registry and feedback remain fingerprint-scoped. The model fingerprint includes at least the DINOv2 architecture identity, encoder asset hashes, feature center, prototype bank, class mapping, and rejection parameters.

No existing DINOv3 Registry or feedback database is attached to a DINOv2 model fingerprint.

Human confirmation remains the only path that can formally update learned prototype evidence. Inference by itself may route observations but must not promote an observation into a formal prototype.

When legacy human-reviewed source media are intentionally imported, Neri re-encodes the original crop/image with the current DINOv2 encoder before adding evidence. If the original media are unavailable, the old DINOv3 vector is not convertible and is skipped.

## Runtime package rename

The DINO subsystem is renamed from `system/dinov3/` to `system/dinov2/`.

The DINOv2 package owns equivalents of the existing runtime responsibilities, including encoder, checkpoint, classifier, rejection, registry, feedback, runtime construction, component installation, public distribution, and API helpers.

Public/internal symbols using `DinoV3*`, `dinov3_*`, or `DINOv3` are removed or renamed to DINOv2 equivalents unless they exist solely to produce a deliberate unsupported-legacy error.

## ImageProcessor integration

`ImageProcessor` replaces:

- `dinov3_classifier` with `dinov2_classifier`;
- `load_dinov3_classifier()` with `load_dinov2_classifier()`;
- `_dinov3_observations` with `_dinov2_observations`;
- DINOv3-specific persistence hooks with DINOv2 equivalents.

The detector -> expanded crop -> RGB crop -> DINO classifier flow remains otherwise unchanged.

No DINOv3 alias is retained after the migration.

## Manifest contract

The canonical manifest uses:

```json
{
  "schema_version": 2,
  "backend": "dinov2",
  "architecture": "dinov2_vitb14",
  "feature_dim": 768,
  "preprocessing": "letterbox224_imagenet",
  "event_aggregation": "mean_l2_normalized_crop_embeddings",
  "requires_detector": true,
  "supports_full_image": false,
  "supports_video_fast": true,
  "supports_video_all": false
}
```

The manifest also identifies the checkpoint and encoder assets/hashes. The checkpoint validator independently confirms the same contract; manifest metadata alone is not trusted.

`backend: "dinov3"` produces a specific unsupported-backend error rather than falling through to YOLO model loading.

## Model catalog and saved settings

`ModelBackend` changes from `Literal["yolo", "dinov3"]` to `Literal["yolo", "dinov2"]`.

Model discovery recognizes DINOv2 manifests and the installed DINOv2 component. DINOv3 manifests are not returned as selectable classification models, except that previously saved paths are recognized sufficiently to produce the explicit migration error when encountered.

Saved DINOv3 model paths are not silently mapped to a DINOv2 model.

## Backend API and maintenance endpoints

DINO-specific backend names become DINOv2 names, including component status, install, remove, Registry, feedback, and environment routes.

Canonical environment routes are:

- `GET /api/environment/dinov2-status`;
- `POST /api/environment/install-dinov2`;
- `POST /api/environment/remove-dinov2`.

The existing DINOv3 maintenance routes are removed after Flutter callers are migrated in the same change set.

## Flutter UI

Flutter model metadata, startup checks, environment panel text, Registry/feedback model names, and tests are renamed from DINOv3 to DINOv2.

The UI must not imply that DINOv3 models remain supported. If settings still reference a removed DINOv3 path, the user receives a clear message to install/select the DINOv2 component.

No UX redesign is required; this is a backend/model-family replacement using the existing model-selection and environment-management surfaces.

## Model distribution

Server-side model distribution capabilities and manifests change their DINO architecture/backend declarations to DINOv2.

A DINOv2 package published by Neri includes all runtime model assets required for offline inference and their integrity metadata. DINOv3 packages remain historical artifacts and are not offered to current DINOv2-only clients as compatible updates.

## Deletion policy

After DINOv2 tests are green, the migration removes:

- `system/dinov3/`;
- DINOv3-only backend maintenance modules;
- DINOv3-only Flutter models/widgets/helpers where the same responsibility has moved to DINOv2;
- DINOv3 component source-mirror installation logic;
- bundled DINOv3 example manifests;
- DINOv3-specific CI workflow names/configuration that no longer test supported behavior.

Historical design documents remain in `docs/` as project history and are not rewritten to pretend they described DINOv2.

## Testing strategy

The replacement is test-driven. Required coverage includes:

1. DINOv2 checkpoint validator accepts a valid frozen checkpoint and rejects DINOv3/mismatched feature-space checkpoints.
2. Encoder fixture verifies `(N, 768)`, finite values, L2 normalization, deterministic CPU inference, and asset hash checking.
3. A golden embedding fixture compares the Neri DINOv2 runtime against the benchmark extraction path within a declared numeric tolerance.
4. Preprocessing golden tests ensure the existing letterbox/ImageNet path remains unchanged.
5. Adaptive prototype classification tests cover variable prototypes per class and nearest-prototype class selection.
6. Fusion-rejection tests cover `linear_type=none` safe fallback and tensorized Ridge fusion.
7. Rejection golden tests compare runtime accept/reject decisions against frozen benchmark examples.
8. Registry/feedback tests prove DINOv3 embeddings are never attached to DINOv2 state and human-confirmed DINOv2 evidence still regenerates prototypes correctly.
9. ImageProcessor tests cover detector crops -> DINOv2 predictions -> persisted observations.
10. Model catalog/API tests verify DINOv2 discovery and explicit DINOv3 unsupported errors.
11. Environment/component tests verify offline local loading, complete inventory hashing, install/remove health, and no runtime network dependency.
12. Flutter tests verify DINOv2 status/install/remove/model-selection flows and removal of DINOv3 selectable state.
13. Packaging/release contract tests verify required DINOv2 runtime dependencies and assets are represented correctly.

## Performance acceptance

Correctness is defined first by parity with the frozen benchmark inference path, not by reproducing an aggregate benchmark number in unit tests.

Before release, an integration benchmark must run the deployment checkpoint through the Neri runtime and confirm that per-sample class decisions, rejection decisions, and summary metrics match the standalone benchmark within the expected floating-point tolerance.

No release is accepted if Neri's preprocessing, feature extraction, CL2N transform, prototype geometry, linear fusion, or thresholds differ from the frozen benchmark without an intentional new calibration experiment.

## Rollout sequence

1. Introduce DINOv2 checkpoint/encoder/classifier tests and fixtures while the old DINOv3 implementation still exists.
2. Implement the DINOv2 runtime and benchmark-parity rejection path.
3. Switch ImageProcessor/backend services/catalog to DINOv2.
4. Switch Registry/feedback persistence and model fingerprints to DINOv2.
5. Switch component installation/distribution/environment APIs to DINOv2.
6. Switch Flutter callers and tests to DINOv2.
7. Run Python, Flutter, packaging, and benchmark-parity verification.
8. Delete the DINOv3 runtime/component/UI implementation after all DINOv2 paths are verified.

This sequence prevents a half-migrated state where UI/backend names say DINOv2 while inference still executes DINOv3 code.

## Safety invariants

- DINOv2 and DINOv3 feature vectors are never treated as interchangeable.
- A checkpoint must cryptographically identify the encoder assets it was trained against.
- The deployed preprocessing and embedding extraction path must match training/benchmark behavior.
- Inference never retrains or recalibrates thresholds from production observations.
- Human confirmation remains required before formal learned prototype updates.
- Safe fusion fallback is represented explicitly and does not fabricate a linear fusion when none was selected.
- Runtime classification remains offline after component installation.
- A failed component/checkpoint integrity check prevents inference rather than falling back to an unverified model.
