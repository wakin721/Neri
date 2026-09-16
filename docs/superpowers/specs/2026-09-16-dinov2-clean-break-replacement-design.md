# DINOv2 Clean-Break Replacement Design

## Goal

Replace Neri's DINOv3 classification subsystem with a DINOv2-only runtime based on `facebook/dinov2-base` (ViT-B/14, 768-D CLS embeddings). The production classifier must reproduce the validation-selected frozen DINOv2 deployment checkpoint rather than a hypothetical linear-fusion variant.

This is an intentional breaking change. DINOv3 manifests, checkpoints, prototype vectors, feedback embeddings, component assets, runtime APIs, and active UI paths are no longer supported after migration.

## Frozen deployment authority

The authoritative deployment artifact is the benchmark-selected DINOv2 checkpoint (`deployment_selection.json` -> `repeat1_method1_frr5.pt`, also copied as `fusion_candidate.pt`). Although the benchmark method label is `Fusion-rejection`, validation selected the safe prototype fallback for the deployment champion. The frozen checkpoint is therefore a `multi_prototype` head with `rejection.mode = "multi_dual_margin"`; it contains no Ridge/SVM coefficients.

Production Neri must implement that exact frozen path:

1. DINOv2 crop embedding;
2. event aggregation when applicable;
3. CL2N transform (subtract frozen feature center, then L2 normalize);
4. nearest adaptive prototype classification;
5. selected-prototype cosine gate;
6. margin-adjusted distance-score gate;
7. class-distance scale division when present.

Linear prototype/Ridge/SVM fusion is out of scope for this replacement. It may be added later only with a separately validated checkpoint schema and parity tests.

## Canonical model and preprocessing

- backbone: `facebook/dinov2-base`;
- architecture id used by Neri manifests: `dinov2_vitb14`;
- feature dimension: `768`;
- benchmark model revision: `f9e44c814b77203eaa57a6bdbbd535f21ede1415`;
- feature source: `last_hidden_state[:, 0]`;
- encoder output: L2-normalized per crop;
- preprocessing: `letterbox224_imagenet`;
- image size: `224 x 224`;
- letterbox RGB: `(124, 116, 104)`;
- ImageNet mean/std: `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`;
- event aggregation: mean of crop embeddings followed by L2 normalization.

Neri must not use `AutoImageProcessor` resize/crop behavior. Runtime preprocessing must match the benchmark implementation.

## Encoder assets

The component root becomes `res/model/DINOv2/`. A complete local Hugging Face model directory is installed under `res/model/DINOv2/model/`. Runtime loading uses `AutoModel.from_pretrained(..., local_files_only=True, trust_remote_code=False)` and never fetches model assets during inference.

The frozen benchmark checkpoint identifies `models/dinov2-base/model.safetensors` with SHA-256 `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`. Neri component integrity continues to use an `install.json` inventory so all deployed model files can also be checked.

## Checkpoint contract

Neri accepts the frozen benchmark structure directly. Required fields include:

- `backbone == "facebook/dinov2-base"`;
- `feature_dim == 768`;
- `preprocessing == "letterbox224_imagenet"`;
- `event_aggregation == "mean_l2_normalized_crop_embeddings"`;
- `encoder_weights` and a 64-character `encoder_sha256`;
- `classes`;
- `feature_center` shaped `(768,)`;
- `head_type == "multi_prototype"`;
- `prototypes` shaped `(M, 768)`;
- `prototype_class_indices` shaped `(M,)`;
- `prototypes_per_class` matching the class mapping;
- adaptive-k metadata when present (`adaptive_k`, `k_by_class`, `adaptive_k_selection`, `adaptive_k_folds`, `prototype_norm_power`);
- `rejection.mode == "multi_dual_margin"`;
- finite `cosine_threshold`;
- finite `adjusted_distance_score_threshold`;
- finite nonnegative `margin_weight`;
- optional positive finite `class_distance_scales` of length `C`;
- frozen calibration/selection metadata may be retained for provenance but is never re-fit at inference time.

A DINOv3 checkpoint is rejected even though its feature dimension can also be 768.

## Exact comparison geometry

Given raw encoder/event feature `x_raw`:

```text
x = x_raw - feature_center
z = x / ||x||
```

`z` must be finite and non-zero. Prototype rows need not be unit vectors, so deployed distance and cosine calculations must preserve the benchmark formulas.

For every sample and prototype `p`:

```text
d2 = max(||z||^2 + ||p||^2 - 2 z·p, 0)
cos = (z·p) / (||z|| ||p||)
```

The winning prototype is the globally smallest `d2`. Its mapped class is the closed-set prediction.

## Margin rejection

For every class, compute its nearest-prototype squared distance. Let `d1` be the winner-class nearest distance and `d2_other` the nearest distance to a different class. The class margin is:

```text
class_margin = d2_other - d1
```

The base distance score is:

```text
distance_score = -winner_squared_distance
```

The adjusted score is:

```text
adjusted = distance_score + margin_weight * class_margin
```

If `class_distance_scales` exists:

```text
adjusted = adjusted / class_distance_scales[winner_class]
```

Accept the prediction only when both are true:

```text
winner_cosine >= cosine_threshold
adjusted >= adjusted_distance_score_threshold
```

Otherwise return `Unknown` while preserving `best_known_species`, winner prototype, cosine, squared distance, class margin, adjusted score, and thresholds for explanations/feedback.

## Registry and feedback

Registry and feedback remain fingerprint-scoped and human-confirmation driven. DINOv3 vectors are never converted to DINOv2 vectors. Only source media and semantic labels may be reused by re-encoding the original crop/image with DINOv2.

Inference may route observations but cannot create or promote formal learned prototypes without explicit human confirmation. Existing transaction, WAL/busy-timeout, provisional-candidate, and independent-event safeguards are preserved.

## Runtime/API/UI clean break

The active package becomes `system/dinov2/`; DINOv3-only source injection and component code are deleted after DINOv2 verification. `ImageProcessor`, backend services, model catalog, environment endpoints, Flutter models/widgets, and active copy are renamed to DINOv2 without compatibility aliases.

Canonical environment routes are:

- `GET /api/environment/dinov2-status`;
- `POST /api/environment/install-dinov2`;
- `POST /api/environment/remove-dinov2`.

`ModelBackend` becomes `Literal["yolo", "dinov2"]`. Old `backend: "dinov3"` manifests are not selectable and produce an explicit unsupported-legacy error if referenced by saved settings.

## Manifest

The Neri deployment manifest is schema version 2 and declares:

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

Manifest and checkpoint metadata are cross-validated; the manifest cannot override the checkpoint feature-space identity.

## Testing and release acceptance

Implementation is test-driven. Required gates are:

1. checkpoint validation accepts the real `multi_dual_margin` schema and rejects DINOv3/malformed state;
2. preprocessing golden test matches the benchmark letterbox implementation;
3. encoder test proves CLS extraction + L2 normalization and local-only loading;
4. classifier golden tests reproduce nearest-prototype, different-class margin, class-scale division, and dual acceptance gates;
5. parity fixtures exported from the frozen benchmark reproduce per-sample species, cosine, squared distance, margin score, and accepted/Unknown decisions within floating-point tolerance;
6. Registry/feedback isolation prevents DINOv3 vector reuse;
7. ImageProcessor/backend/catalog/component/environment/Flutter paths operate as DINOv2 only;
8. Python, Flutter, packaging, and CI suites are green before DINOv3 implementation files are removed.

No release is accepted if Neri's preprocessing, CLS extraction, CL2N transform, prototype geometry, margin formula, class-scale division, or frozen thresholds differ from the benchmark without a new calibration experiment.
