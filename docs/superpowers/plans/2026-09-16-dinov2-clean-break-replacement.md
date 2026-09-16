# DINOv2 Clean-Break Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Neri's DINOv3 subsystem with a DINOv2-only `facebook/dinov2-base` runtime that reproduces the validated Adaptive Multi-prototype + Fusion-rejection benchmark path.

**Architecture:** Keep the existing detector -> crop -> embedding-classifier -> Registry/feedback data flow, but replace the entire `system/dinov3` implementation and all active DINOv3 API/UI identities with DINOv2 equivalents. The DINOv2 runtime owns local offline encoder loading, CL2N feature geometry, adaptive per-class prototypes, and an explicit Fusion-rejection schema that represents either tensorized Ridge fusion or validation-selected `linear_type=none` fallback.

**Tech Stack:** Python 3, PyTorch, Hugging Face Transformers, safetensors, NumPy, FastAPI/Pydantic, SQLite, pytest, Flutter/Dart, flutter_test.

**Spec:** `docs/superpowers/specs/2026-09-16-dinov2-clean-break-replacement-design.md`

## Global Constraints

- Supported DINO architecture is exactly `dinov2_vitb14` / `facebook/dinov2-base`, feature dimension `768`.
- Encoder feature is `last_hidden_state[:, 0]`, followed by per-image L2 normalization.
- Preprocessing remains exactly `letterbox224_imagenet`; do not substitute Hugging Face resizing/cropping.
- Runtime comparison geometry is CL2N: encoder L2 -> subtract frozen feature center -> L2 normalize.
- Prototype class score is negative nearest-prototype squared distance in CL2N space.
- DINOv2 and DINOv3 768-D features are never treated as interchangeable.
- Production inference must be offline after component installation; encoder loading uses local assets only.
- `class_alpha=0` for the canonical Fusion-rejection policy so prototype geometry alone chooses the closed-set species.
- Frozen linear heads are numeric tensors, not pickled scikit-learn estimators.
- Registry/feedback remains model-fingerprint scoped; inference never promotes formal prototype evidence without human confirmation.
- Old `backend: "dinov3"` manifests must produce an explicit unsupported-legacy error, not YOLO fallback.
- DINOv3 code is deleted only after DINOv2 backend, UI, packaging, and parity tests are green.

---

## File Structure

New canonical runtime files:

- `system/dinov2/__init__.py` — exported DINOv2 constants and component API.
- `system/dinov2/preprocess.py` — Neri letterbox/ImageNet preprocessing contract under the new package.
- `system/dinov2/checkpoint.py` — DINOv2-only checkpoint schema, validation, fingerprinting, fusion tensors.
- `system/dinov2/encoder.py` — local `facebook/dinov2-base` loader and normalized CLS extraction.
- `system/dinov2/prototype_bank.py` — prototype records/banks moved from the old package.
- `system/dinov2/classifier.py` — Adaptive Multi-prototype + Fusion-rejection classifier and prediction DTOs.
- `system/dinov2/rejection.py` — immutable Fusion-rejection configuration and score/gate calculations.
- `system/dinov2/runtime.py` — manifest resolution, checkpoint construction, Registry/feedback wiring.
- `system/dinov2/registry.py`, `registry_impl.py`, `registry_examples.py`, `feedback.py`, `feedback_index.py`, `events.py` — fingerprint-isolated learning state.
- `system/dinov2/component.py`, `public_distribution.py`, `api.py`, `state.py` — local component lifecycle and public helpers.

Migration touchpoints:

- `system/model_sync/layout.py`, `catalog.py` — `DINOv2` component root and backend discovery.
- `system/image_processor.py` — attach/use/drain DINOv2 classifier observations.
- `system/backend/model_services.py`, `services.py` — DINOv2 manifest/classifier loading and job option validation.
- `system/backend/dinov2_environment.py`, `dinov2_maintenance.py`, `dinov2_feedback_api.py`, `dinov2_feedback_service.py`, `dinov2_registry_service.py`, `dinov2_persistence_fast.py` — backend-facing services.
- `frontend/lib/src/api_client.dart`, `api_client_core.dart`, DINO models/widgets/screens/startup helpers — DINOv2 API and copy.
- `server/model_distribution/capabilities.py`, `manifest.py`, `service.py` — DINOv2 capability/distribution contract.
- `.github/workflows/`, Windows packaging scripts, Python tests, and Flutter tests — verification and release contract.

Historical DINOv3 design documents under `docs/superpowers/` remain untouched.

---

### Task 1: Introduce the DINOv2 Checkpoint Contract and Test Fixtures

**Files:**
- Create: `tests/dinov2_fixtures.py`
- Create: `tests/test_dinov2_checkpoint.py`
- Create: `system/dinov2/checkpoint.py`
- Create: `system/dinov2/__init__.py`
- Reference: `system/dinov3/checkpoint.py`
- Reference: `tests/dinov3_multi_prototype_fixtures.py`

**Interfaces:**
- Produces: `DINO_BACKBONE = "dinov2_vitb14"`, `DINO_FEATURE_DIM = 768`, `DinoV2Checkpoint`, `FusionRejectionConfig`, `validate_checkpoint(payload) -> DinoV2Checkpoint`, `load_checkpoint(path) -> DinoV2Checkpoint`.
- Consumed by: Tasks 2-12.

- [ ] **Step 1: Write the failing checkpoint tests**

Create `make_dinov2_payload()` with two classes, four prototypes, counts `(1, 3)`, zero center, valid SHA-256-like encoder asset fingerprint, and configurable `linear_type`.

```python
def test_accepts_adaptive_prototypes_and_safe_fallback():
    checkpoint = validate_checkpoint(make_dinov2_payload(linear_type="none"))
    assert checkpoint.backbone == "dinov2_vitb14"
    assert checkpoint.feature_dim == 768
    assert checkpoint.prototypes_per_class == (1, 3)
    assert checkpoint.fusion.linear_type == "none"
    assert checkpoint.fusion.class_alpha == 0.0


def test_accepts_tensorized_ridge_head():
    checkpoint = validate_checkpoint(make_dinov2_payload(linear_type="ridge"))
    assert tuple(checkpoint.fusion.coef.shape) == (2, 768)
    assert tuple(checkpoint.fusion.intercept.shape) == (2,)


def test_rejects_dinov3_feature_space_even_when_dimension_is_768():
    payload = make_dinov2_payload()
    payload["backbone"] = "dinov3_vitb16"
    with pytest.raises(CheckpointValidationError, match="dinov2_vitb14"):
        validate_checkpoint(payload)


def test_rejects_class_alpha_that_changes_closed_class():
    payload = make_dinov2_payload()
    payload["fusion"]["class_alpha"] = 0.5
    with pytest.raises(CheckpointValidationError, match="class_alpha"):
        validate_checkpoint(payload)
```

Add separate tests for invalid prototype class indices, mismatched prototype counts, non-finite center/prototype tensors, missing feature center, invalid asset fingerprint length, and Ridge coefficient/intercept shape mismatch.

- [ ] **Step 2: Run the tests and verify RED**

```bash
pytest tests/test_dinov2_checkpoint.py -v
```

Expected: collection fails because `system.dinov2.checkpoint` does not exist.

- [ ] **Step 3: Implement the immutable checkpoint schema**

Use these public dataclasses:

```python
@dataclass(frozen=True)
class FusionRejectionConfig:
    linear_type: str
    prototype_scale: float
    linear_scale: float | None
    class_alpha: float
    score_alpha: float
    margin_weight: float
    cosine_threshold: float
    score_threshold: float
    coef: Any | None = None
    intercept: Any | None = None


@dataclass(frozen=True)
class DinoV2Checkpoint:
    path: Path | None
    backbone: str
    feature_dim: int
    classes: tuple[str, ...]
    feature_center: Any
    prototypes: Any
    prototype_class_indices: Any
    prototypes_per_class: tuple[int, ...]
    encoder_dir: str
    encoder_asset_fingerprint: str
    preprocessing: str
    event_aggregation: str
    fusion: FusionRejectionConfig
    fingerprint: str
```

`encoder_asset_fingerprint` is a SHA-256 digest over the sorted relative encoder file paths and each file's SHA-256, not a hash of only one file. Fingerprinting of the classifier covers backbone, feature dimension, classes, center, prototypes, prototype indices/counts, encoder asset fingerprint, preprocessing, aggregation, fusion scalars, and optional Ridge tensors.

- [ ] **Step 4: Run checkpoint tests GREEN**

```bash
pytest tests/test_dinov2_checkpoint.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add system/dinov2/__init__.py system/dinov2/checkpoint.py tests/dinov2_fixtures.py tests/test_dinov2_checkpoint.py
git commit -m "feat: add DINOv2 checkpoint contract"
```

---

### Task 2: Implement Benchmark-Exact Preprocessing and DINOv2 Encoder

**Files:**
- Create: `system/dinov2/preprocess.py`
- Create: `system/dinov2/encoder.py`
- Create: `tests/test_dinov2_preprocess.py`
- Create: `tests/test_dinov2_encoder.py`
- Reference: `system/dinov3/preprocess.py`
- Reference: `system/dinov3/encoder.py`

**Interfaces:**
- Consumes: `DinoV2Checkpoint` from Task 1.
- Produces: `preprocess_image(value, array_color="rgb") -> torch.Tensor`, `DinoV2Encoder.encode(crops, array_color="rgb") -> np.ndarray`, `resolve_encoder_dir(checkpoint, encoder_dir=None, search_roots=()) -> Path`, `fingerprint_encoder_assets(model_dir) -> str`.

- [ ] **Step 1: Write RED preprocessing and encoder tests**

```python
class FakeDinoV2Model:
    def eval(self):
        return self

    def requires_grad_(self, value):
        return self

    def to(self, device):
        return self

    def __call__(self, *, pixel_values):
        batch = pixel_values.shape[0]
        hidden = torch.zeros((batch, 2, 768), dtype=torch.float32)
        hidden[:, 0, 0] = 3.0
        hidden[:, 0, 1] = 4.0
        return SimpleNamespace(last_hidden_state=hidden)


def test_encoder_uses_cls_token_and_l2_normalizes(tmp_path, monkeypatch):
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    checkpoint = make_checkpoint_for_encoder(tmp_path)
    model = FakeDinoV2Model()
    encoder = DinoV2Encoder(checkpoint, model_factory=lambda model_dir: model, device="cpu")
    features = encoder.encode([image])
    assert features.shape == (1, 768)
    assert features[0, 0] == pytest.approx(0.6)
    assert features[0, 1] == pytest.approx(0.8)
    assert np.linalg.norm(features[0]) == pytest.approx(1.0)
```

Add a factory-capture test proving production loading passes `local_files_only=True` and `trust_remote_code=False`, a stale-checkpoint-path resolution test, an asset-fingerprint mismatch test, a deterministic CPU test, and a CUDA-request failure test when CUDA is unavailable.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_dinov2_preprocess.py tests/test_dinov2_encoder.py -v
```

Expected: missing module/functions.

- [ ] **Step 3: Copy preprocessing contract without semantic changes**

Move the existing letterbox implementation into `system/dinov2/preprocess.py`. Preserve `IMAGE_SIZE=224`, `LETTERBOX_RGB=(124, 116, 104)`, ImageNet mean/std, bicubic resize, RGB conversion, and CHW float output.

- [ ] **Step 4: Implement local Transformers encoder**

Production construction is exactly:

```python
model = AutoModel.from_pretrained(
    str(model_dir),
    local_files_only=True,
    trust_remote_code=False,
)
outputs = model(pixel_values=batch)
features = outputs.last_hidden_state[:, 0]
features = F.normalize(features.float(), dim=1)
```

Do not instantiate `AutoImageProcessor`; Neri preprocessing is authoritative.

- [ ] **Step 5: Run encoder/preprocessing tests GREEN**

```bash
pytest tests/test_dinov2_preprocess.py tests/test_dinov2_encoder.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add system/dinov2/preprocess.py system/dinov2/encoder.py tests/test_dinov2_preprocess.py tests/test_dinov2_encoder.py
git commit -m "feat: add offline DINOv2 encoder"
```

---

### Task 3: Implement Adaptive Multi-prototype and Fusion-rejection Inference

**Files:**
- Create: `system/dinov2/prototype_bank.py`
- Create: `system/dinov2/rejection.py`
- Create: `system/dinov2/classifier.py`
- Create: `tests/test_dinov2_classifier.py`
- Create: `tests/test_dinov2_fusion_rejection.py`
- Reference: `system/dinov3/classifier.py`
- Reference: `system/dinov3/multi_dual_classifier.py`

**Interfaces:**
- Consumes: `DinoV2Checkpoint`, `FusionRejectionConfig`, encoder features.
- Produces: `DinoV2Prediction`, `DinoV2Observation`, `DinoV2Classifier.classify_features()`, `classify_crops()`, `classify_event()`, `explain_feature()`.

- [ ] **Step 1: Write RED CL2N/adaptive-prototype tests**

```python
def test_closed_class_uses_nearest_adaptive_prototype_in_cl2n_space():
    checkpoint = validate_checkpoint(make_dinov2_payload(linear_type="none"))
    classifier = DinoV2Classifier(checkpoint)
    feature = np.zeros(768, dtype=np.float32)
    feature[0] = 1.0
    prediction = classifier.classify_features(feature[None, :])[0]
    assert prediction.best_known_species == "A"
    assert prediction.nearest_prototype_index == 0
```

Add a test that non-unit input features are rejected before CL2N, and a test that zero-length centered features raise a deterministic `ValueError`.

- [ ] **Step 2: Write RED Fusion-rejection tests**

Use the benchmark formula explicitly:

```python
prototype_scaled = prototype_logits / prototype_scale
linear_scaled = linear_logits / linear_scale
fused_logits = (1.0 - score_alpha) * prototype_scaled + score_alpha * linear_scaled
gap = fused_logits[winner] - np.max(np.delete(fused_logits, winner))
rejection_score = fused_logits[winner] + margin_weight * gap
accepted = cosine_to_selected_prototype >= cosine_threshold and rejection_score >= score_threshold
```

The winner is calculated from prototype logits because `class_alpha == 0`.

```python
def test_linear_none_uses_prototype_score_only():
    checkpoint = validate_checkpoint(make_dinov2_payload(linear_type="none"))
    classifier = DinoV2Classifier(checkpoint)
    assert classifier.checkpoint.fusion.linear_type == "none"
    assert classifier.checkpoint.fusion.score_alpha == 0.0


def test_ridge_changes_rejection_score_without_changing_species():
    checkpoint = validate_checkpoint(make_dinov2_payload(linear_type="ridge"))
    classifier = DinoV2Classifier(checkpoint)
    feature = unit_feature(0)
    prediction = classifier.classify_features(feature[None, :])[0]
    assert prediction.best_known_species == "A"
    assert classifier.checkpoint.fusion.class_alpha == 0.0
    assert math.isfinite(prediction.rejection_score)
```

- [ ] **Step 3: Verify RED**

```bash
pytest tests/test_dinov2_classifier.py tests/test_dinov2_fusion_rejection.py -v
```

Expected: missing classifier/rejection implementation.

- [ ] **Step 4: Implement CL2N and prototype logits**

```python
centered = features - feature_center[None, :]
z = centered / np.linalg.norm(centered, axis=1, keepdims=True)
squared_distances = ((z[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
prototype_logits = -nearest_distance_per_class
```

For each class, retain the index and cosine of its nearest prototype. The predicted class is `argmax(prototype_logits)`.

- [ ] **Step 5: Implement optional Ridge logits and fused rejection**

```python
linear_logits = z @ coef.T + intercept
```

For `linear_type="none"`, require `score_alpha == 0.0` and use prototype-scaled logits directly. For `linear_type="ridge"`, apply the frozen `linear_scale`, `score_alpha`, and margin formula from Step 2. Acceptance requires both gates.

- [ ] **Step 6: Preserve formal/provisional prototype semantics**

Port `PrototypeRecord` and `PrototypeBank`. Formal matches may be accepted. Provisional matches remain `accepted=False`, `assistive_match=True`, and carry Registry metadata without becoming a formal species prediction.

- [ ] **Step 7: Run classifier tests GREEN**

```bash
pytest tests/test_dinov2_classifier.py tests/test_dinov2_fusion_rejection.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add system/dinov2/prototype_bank.py system/dinov2/rejection.py system/dinov2/classifier.py tests/test_dinov2_classifier.py tests/test_dinov2_fusion_rejection.py
git commit -m "feat: add DINOv2 fusion rejection classifier"
```

---

### Task 4: Build the DINOv2 Manifest Runtime and Explicit DINOv3 Legacy Error

**Files:**
- Create: `system/dinov2/runtime.py`
- Create: `system/dinov2/state.py`
- Create: `tests/test_dinov2_runtime.py`
- Reference: `system/dinov3/runtime.py`

**Interfaces:**
- Consumes: Tasks 1-3 classifier/encoder/checkpoint interfaces.
- Produces: `DinoV2Runtime`, `DinoV2ManifestError`, `resolve_dinov2_manifest(model_path) -> Path`, `load_dinov2_model(model_path, registry=None, feedback=None, state_root=None, encoder_factory=None, device=None, use_fp16=False) -> DinoV2Runtime`.

- [ ] **Step 1: Write RED runtime tests**

```python
def test_runtime_loads_dinov2_manifest(tmp_path):
    manifest = write_dinov2_runtime_fixture(tmp_path)
    runtime = load_dinov2_model(
        manifest,
        encoder_factory=FakeEncoder,
        state_root=tmp_path / "state",
    )
    try:
        assert runtime.checkpoint.backbone == "dinov2_vitb14"
        assert isinstance(runtime.classifier, DinoV2Classifier)
    finally:
        runtime.close()


def test_runtime_rejects_dinov3_manifest_explicitly(tmp_path):
    manifest = tmp_path / "legacy.neri.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "backend": "dinov3",
            "checkpoint": "legacy.pt",
            "architecture": "dinov3_vitb16",
        }),
        encoding="utf-8",
    )
    with pytest.raises(DinoV2ManifestError, match="DINOv3.*不再支持"):
        resolve_dinov2_manifest(manifest)
```

Add independent tests for schema version, architecture, feature dimension, encoder asset fingerprint, preprocessing, and event aggregation mismatch.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_dinov2_runtime.py -v
```

- [ ] **Step 3: Implement manifest resolution and validation**

Require `schema_version == 2`, `backend == "dinov2"`, `architecture == "dinov2_vitb14"`, `feature_dim == 768`, matching `encoder_asset_fingerprint`, `letterbox224_imagenet`, and `mean_l2_normalized_crop_embeddings`.

- [ ] **Step 4: Construct classifier, encoder, and owned state stores**

Follow the current lifecycle contract: create stores when absent, verify fingerprint equality for injected stores, close only owned stores, and close partially created stores on construction failure.

- [ ] **Step 5: Run runtime tests GREEN**

```bash
pytest tests/test_dinov2_runtime.py -v
```

- [ ] **Step 6: Commit**

```bash
git add system/dinov2/runtime.py system/dinov2/state.py tests/test_dinov2_runtime.py
git commit -m "feat: add DINOv2 manifest runtime"
```

---

### Task 5: Port Registry, Feedback, Observation APIs, and Persistence to DINOv2 Fingerprints

**Files:**
- Create: `system/dinov2/registry.py`
- Create: `system/dinov2/registry_impl.py`
- Create: `system/dinov2/registry_examples.py`
- Create: `system/dinov2/feedback.py`
- Create: `system/dinov2/feedback_index.py`
- Create: `system/dinov2/events.py`
- Create: `system/dinov2/api.py`
- Create: `tests/test_dinov2_registry.py`
- Create: `tests/test_dinov2_feedback.py`
- Create: `tests/test_dinov2_state_isolation.py`
- Reference: `system/dinov3/registry.py`, `registry_impl.py`, `registry_examples.py`, `feedback.py`, `feedback_index.py`, `events.py`, `api.py`.

**Interfaces:**
- Consumes: DINOv2 fingerprint and CL2N feature center.
- Produces: DINOv2 Registry/feedback stores with `prototype_bank(feature_center) -> PrototypeBank`, observation APIs, and human-confirmation lifecycle.

- [ ] **Step 1: Write RED state-isolation and human-confirmation tests**

```python
def test_dinov3_state_is_not_opened_as_dinov2_state(tmp_path):
    legacy = tmp_path / "dinov3" / "registry.sqlite3"
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    path = registry_path_for_fingerprint(tmp_path, "a" * 64)
    assert path != legacy
    assert "a" * 64 in path.name


def test_inference_observation_does_not_formalize_prototype(store, observation):
    store.record_candidate(observation)
    bank = store.prototype_bank(np.zeros(768, dtype=np.float32))
    assert all(record.source != "confirmed" for record in bank.formal)
```

Add a confirmation test that the same observation becomes eligible for learned prototype regeneration only after the user confirms the species.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_dinov2_registry.py tests/test_dinov2_feedback.py tests/test_dinov2_state_isolation.py -v
```

- [ ] **Step 3: Port persistence code with DINOv2 names**

Preserve SQLite transaction boundaries, WAL, busy timeout, candidate clustering, independent-event grouping, same-name merge semantics, and explicit human-confirmation invariants. Do not rewrite DINOv3 SQLite files in place.

- [ ] **Step 4: Validate DINOv2 evidence before prototype regeneration**

Require `(768,)` finite embeddings and exact model fingerprint match before a confirmed observation can contribute to learned prototypes.

- [ ] **Step 5: Port the existing semantic regression coverage**

Create DINOv2 equivalents of the active Registry/feedback tests covering management, concurrency, examples, same-name merge, named-candidate migration, candidate actions, clear-candidates, cluster view, human species linking, persistence, runtime feedback, and validation bridge.

- [ ] **Step 6: Run state tests GREEN**

```bash
pytest tests/test_dinov2_registry*.py tests/test_dinov2_feedback*.py tests/test_dinov2_state_isolation.py -v
```

- [ ] **Step 7: Commit**

```bash
git add system/dinov2/registry.py system/dinov2/registry_impl.py system/dinov2/registry_examples.py system/dinov2/feedback.py system/dinov2/feedback_index.py system/dinov2/events.py system/dinov2/api.py tests/test_dinov2_registry*.py tests/test_dinov2_feedback*.py tests/test_dinov2_state_isolation.py
git commit -m "feat: port open world state to DINOv2"
```

---

### Task 6: Switch ImageProcessor and Backend Inference Wiring to DINOv2

**Files:**
- Modify: `system/image_processor.py` — `ImageProcessor.__init__`, DINO classifier attachment/drain methods, batched crop classification branch.
- Modify: `system/backend/services.py` — DINO manifest detection, job option validation, runtime construction, observation persistence hooks.
- Modify: `system/backend/model_services.py` — DINO manifest class loading.
- Create: `system/backend/dinov2_feedback_service.py`
- Create: `system/backend/dinov2_registry_service.py`
- Create: `system/backend/dinov2_persistence_fast.py`
- Create: `system/backend/dinov2_feedback_api.py`
- Create: `tests/test_dinov2_image_processor.py`
- Create: `tests/test_dinov2_backend_integration.py`

**Interfaces:**
- Consumes: `load_dinov2_model()` and `DinoV2Classifier`.
- Produces: `ImageProcessor.load_dinov2_classifier(classifier)`, `ImageProcessor.drain_dinov2_observations()`, backend persistence using DINOv2 observations.

- [ ] **Step 1: Write RED ImageProcessor integration test**

Attach a fake DINOv2 classifier, feed one detector crop, assert `classify_crops()` receives RGB crops, and assert `drain_dinov2_observations()` returns DINOv2 observation records once then empties.

- [ ] **Step 2: Write RED backend runtime-selection test**

A classification manifest with `backend: "dinov2"` must call `load_dinov2_model`. A `backend: "dinov3"` selection must raise the DINOv3 unsupported message before any call to `YOLO(classification_model_path)`.

- [ ] **Step 3: Verify RED**

```bash
pytest tests/test_dinov2_image_processor.py tests/test_dinov2_backend_integration.py -v
```

- [ ] **Step 4: Rename ImageProcessor attachment state**

Replace `dinov3_classifier`, `load_dinov3_classifier`, `_dinov3_observations`, and `drain_dinov3_observations` with DINOv2-only names. Do not retain aliases.

- [ ] **Step 5: Switch backend services and persistence imports**

Rename `_dinov3_manifest_payload` to `_dino_manifest_payload` returning either the DINOv2 payload or an explicit legacy DINOv3 marker. Load only DINOv2 through `system.dinov2.runtime`. Route observation persistence through the new DINOv2 feedback service.

- [ ] **Step 6: Run integration regressions GREEN**

```bash
pytest tests/test_dinov2_image_processor.py tests/test_dinov2_backend_integration.py tests/test_preview_fast.py tests/test_preview_single_fast.py tests/test_validation_cache_wiring.py -v
```

- [ ] **Step 7: Commit**

```bash
git add system/image_processor.py system/backend/services.py system/backend/model_services.py system/backend/dinov2_feedback_service.py system/backend/dinov2_registry_service.py system/backend/dinov2_persistence_fast.py system/backend/dinov2_feedback_api.py tests/test_dinov2_image_processor.py tests/test_dinov2_backend_integration.py
git commit -m "refactor: route classification through DINOv2"
```

---

### Task 7: Replace Model Layout and Catalog Discovery with DINOv2

**Files:**
- Modify: `system/model_sync/layout.py` — `ModelLayout`, `_build_layout`.
- Modify: `system/model_sync/catalog.py` — `ModelBackend`, manifest parser, installed component discovery, saved-path resolution.
- Modify: `system/backend/models.py` — backend metadata schema used in API responses.
- Create: `tests/test_dinov2_model_catalog.py`
- Modify: `tests/test_model_resource_paths.py`
- Modify: `tests/test_model_catalog.py`

**Interfaces:**
- Produces: `ModelLayout.dinov2_root`, `ModelBackend = Literal["yolo", "dinov2"]`, DINOv2 manifest discovery and saved-path resolution.

- [ ] **Step 1: Write RED catalog tests**

```python
def test_manifest_exposes_dinov2_capabilities(tmp_path):
    layout = get_model_layout(tmp_path)
    head = layout.cls_user / "head.pt"
    head.write_bytes(b"x")
    manifest = layout.cls_user / "head.neri.json"
    manifest.write_text(json.dumps({
        "schema_version": 2,
        "backend": "dinov2",
        "display_name": "DINOv2 reviewed",
        "checkpoint": "head.pt",
        "architecture": "dinov2_vitb14",
        "feature_dim": 768,
        "requires_detector": True,
        "supports_video_fast": True,
        "supports_video_all": False,
    }), encoding="utf-8")
    model = discover_models(layout, "cls")[0]
    assert model.backend == "dinov2"
    assert model.architecture == "dinov2_vitb14"
```

Port the healthy installed-component test to `layout.dinov2_root` and verify only the manifest named by `install.json` is surfaced. Port the unhealthy-component test and expect no installed model entry.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_dinov2_model_catalog.py tests/test_model_resource_paths.py tests/test_model_catalog.py -v
```

- [ ] **Step 3: Change canonical component root and backend literal**

Replace `ModelLayout.dinov3_root` with `dinov2_root = root / "DINOv2"`. Replace `Literal["yolo", "dinov3"]` with `Literal["yolo", "dinov2"]`.

- [ ] **Step 4: Change catalog manifest parsing**

Discover `backend: "dinov2"` manifests. Do not return DINOv3 manifests as selectable models. Preserve recognition of a saved DINOv3 path only long enough for backend selection to emit the explicit legacy error.

- [ ] **Step 5: Run catalog tests GREEN**

```bash
pytest tests/test_dinov2_model_catalog.py tests/test_model_resource_paths.py tests/test_model_catalog.py -v
```

- [ ] **Step 6: Commit**

```bash
git add system/model_sync/layout.py system/model_sync/catalog.py system/backend/models.py tests/test_dinov2_model_catalog.py tests/test_model_resource_paths.py tests/test_model_catalog.py
git commit -m "refactor: make model catalog DINOv2 only"
```

---

### Task 8: Replace Component Installation and Distribution with Offline DINOv2 Assets

**Files:**
- Create: `system/dinov2/component.py`
- Create: `system/dinov2/public_distribution.py`
- Modify: `system/dinov2/__init__.py` — export component functions/constants.
- Modify: `server/model_distribution/capabilities.py` — DINOv2 capability declarations.
- Modify: `server/model_distribution/manifest.py` — DINOv2 package manifest validation.
- Modify: `server/model_distribution/service.py` — publish DINOv2 package metadata.
- Create: `tests/test_dinov2_component.py`
- Create: `tests/test_dinov2_public_distribution.py`
- Modify: `tests/test_model_distribution.py`

**Interfaces:**
- Produces: `DinoV2ComponentPaths`, `dinov2_component_paths()`, `dinov2_component_status()`, `install_dinov2_component()`, `remove_dinov2_component()`.

- [ ] **Step 1: Write RED component health tests**

A healthy fixture contains exactly these required payload files plus any Transformer tokenizer-independent metadata explicitly listed by `install.json`:

```text
model/config.json
model/model.safetensors
model/preprocessor_config.json
classifier.pt
classifier.neri.json
LICENSE_DINOv2.md
install.json
```

Test complete inventory hashing, an undeclared extra file, a missing required file, declared-size mismatch, SHA-256 mismatch, and the healthy status response.

- [ ] **Step 2: Write RED distribution tests**

Assert server capabilities advertise backend `dinov2`, architecture `dinov2_vitb14`, cloud root `/Neri_Data/Model/DINOv2`, and no compatibility declaration for DINOv3.

- [ ] **Step 3: Verify RED**

```bash
pytest tests/test_dinov2_component.py tests/test_dinov2_public_distribution.py tests/test_model_distribution.py -v
```

- [ ] **Step 4: Implement component lifecycle**

Reuse atomic staging, safe path validation, complete inventory validation, and hash verification from the current component. Do not install a source-code mirror. `DinoV2ComponentPaths.model_dir` must point to the directory passed to `AutoModel.from_pretrained`.

- [ ] **Step 5: Update model-distribution metadata**

Publish DINOv2 package identity, inventory, classifier manifest, and model asset fingerprint through the existing distribution service.

- [ ] **Step 6: Run component/distribution tests GREEN**

```bash
pytest tests/test_dinov2_component.py tests/test_dinov2_public_distribution.py tests/test_model_distribution.py -v
```

- [ ] **Step 7: Commit**

```bash
git add system/dinov2/__init__.py system/dinov2/component.py system/dinov2/public_distribution.py server/model_distribution/capabilities.py server/model_distribution/manifest.py server/model_distribution/service.py tests/test_dinov2_component.py tests/test_dinov2_public_distribution.py tests/test_model_distribution.py
git commit -m "feat: distribute offline DINOv2 component"
```

---

### Task 9: Replace Backend Environment Maintenance APIs with DINOv2

**Files:**
- Create: `system/backend/dinov2_environment.py`
- Create: `system/backend/dinov2_maintenance.py`
- Modify: `system/backend/main_core.py` — router registration.
- Modify: `system/backend/maintenance.py` — optional DINOv2 dependency install helper.
- Create: `tests/test_dinov2_environment_api.py`
- Create: `tests/test_dinov2_environment_maintenance.py`

**Interfaces:**
- Produces: `GET /api/environment/dinov2-status`, `POST /api/environment/install-dinov2`, `POST /api/environment/remove-dinov2`.
- Status fields: `installed`, `healthy`, `architecture`, `component_version`, `model_id`, `classifier_filename`, `classifier_fingerprint`, `classifier_head_type`, `message`.

- [ ] **Step 1: Write RED API tests**

```python
def test_dinov2_status_route(client, monkeypatch):
    response = client.get("/api/environment/dinov2-status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["architecture"] == "DINOv2 ViT-B/14"
    assert payload["model_id"] == "facebook/dinov2-base"
```

Add install/remove tests asserting operation names `install_dinov2` and `remove_dinov2`.

- [ ] **Step 2: Write RED dependency-maintenance test**

Patch module discovery so PyTorch exists but `transformers` and `safetensors` do not. Assert the DINOv2 maintenance worker requests those two packages before `install_dinov2_component()` is called.

- [ ] **Step 3: Verify RED**

```bash
pytest tests/test_dinov2_environment_api.py tests/test_dinov2_environment_maintenance.py -v
```

- [ ] **Step 4: Implement DINOv2 environment router and worker**

Mirror current shutdown/restart/progress semantics. Use DINOv2 operation names/messages. Remove source-code mirror health checks and source commit fields.

- [ ] **Step 5: Register router and optional dependency installation**

Keep the current package-source selection behavior. Preserve the shared PyTorch environment and install `transformers` plus `safetensors` through the selected package source.

- [ ] **Step 6: Run environment regressions GREEN**

```bash
pytest tests/test_dinov2_environment_api.py tests/test_dinov2_environment_maintenance.py tests/test_runtime_patches.py -v
```

- [ ] **Step 7: Commit**

```bash
git add system/backend/dinov2_environment.py system/backend/dinov2_maintenance.py system/backend/main_core.py system/backend/maintenance.py tests/test_dinov2_environment_api.py tests/test_dinov2_environment_maintenance.py
git commit -m "feat: add DINOv2 environment maintenance"
```

---

### Task 10: Migrate Flutter API, Models, Startup Checks, Registry UI, and Copy

**Files:**
- Modify: `frontend/lib/src/api_client.dart`
- Modify: `frontend/lib/src/api_client_core.dart`
- Create: `frontend/lib/src/dinov2_startup_check.dart`
- Create: `frontend/lib/src/models/dinov2_explanation.dart`
- Create: `frontend/lib/src/models/dinov2_feedback.dart`
- Create: `frontend/lib/src/models/dinov2_registry.dart`
- Create: `frontend/lib/src/widgets/dinov2_feature_scatter.dart`
- Create: `frontend/lib/src/widgets/dinov2_registry_dialog.dart`
- Modify: `frontend/lib/src/screens/settings_screen.dart`
- Modify: `frontend/lib/src/screens/species_validation_screen.dart`
- Modify: `frontend/lib/src/main_window.dart`
- Create DINOv2-named replacements for active `frontend/test/dinov3_*` tests.

**Interfaces:**
- Produces: `fetchDinoV2ComponentStatus()`, `installDinoV2(envChoice:, packageSource:)`, `removeDinoV2()`, DINOv2 Registry/feedback DTOs and DINOv2-only UI copy.

- [ ] **Step 1: Port environment API test to RED DINOv2 expectations**

```dart
test('parses DINOv2 environment status', () async {
  final client = NeriApiClient(
    httpClient: MockClient((request) async {
      expect(request.url.path, '/api/environment/dinov2-status');
      return http.Response(jsonEncode({
        'installed': true,
        'healthy': true,
        'architecture': 'DINOv2 ViT-B/14',
        'component_version': 1,
        'model_id': 'facebook/dinov2-base',
        'classifier_filename': 'classifier.pt',
        'classifier_fingerprint': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        'classifier_head_type': 'adaptive_multi_fusion_rejection',
        'message': 'DINOv2 ViT-B/14 已安装。',
      }), 200);
    }),
  );
  final status = await client.fetchDinoV2ComponentStatus();
  expect(status.architecture, 'DINOv2 ViT-B/14');
  expect(status.modelId, 'facebook/dinov2-base');
});
```

Port install/remove expectations to `/api/environment/install-dinov2` and `/api/environment/remove-dinov2`.

- [ ] **Step 2: Port Registry/feedback/startup tests to DINOv2 names**

Rename imports and DTOs while preserving behavior assertions for candidate actions, clear candidates, cluster view, same-name handling, selected-box learning, feature scatter, startup Registry features, video mode restrictions, and feedback APIs.

- [ ] **Step 3: Verify RED Flutter tests**

Run the newly created DINOv2 test files with `flutter test`; expected result is compile failure for missing DINOv2 methods/types.

- [ ] **Step 4: Implement DINOv2 API methods and DTOs**

Replace public DINOv3 method/type names with DINOv2 names. Do not add deprecated aliases.

- [ ] **Step 5: Implement DINOv2 UI helpers/widgets and copy**

Switch active imports to the DINOv2 files and replace visible DINOv3 strings on current UI paths. Keep screen layout and user workflow unchanged.

- [ ] **Step 6: Run Flutter suite GREEN**

```bash
cd frontend
flutter test
```

Expected: complete Flutter suite passes.

- [ ] **Step 7: Commit**

```bash
git add frontend/lib frontend/test
git commit -m "refactor: migrate Flutter DINO UI to DINOv2"
```

---

### Task 11: Update Packaging, CI, and Release Contracts

**Files:**
- Create: `.github/workflows/dinov2-implementation-verification.yml`
- Delete after replacement is validated: `.github/workflows/dinov3-implementation-verification.yml`
- Modify: `.github/workflows/build-windows-x64.yml`
- Modify: `scripts/package_windows.ps1`
- Modify: `scripts/prepare_windows.ps1`
- Modify: `scripts/prune_lite_python.ps1`
- Modify: `tests/test_lite_package_contract.py`
- Modify: `tests/test_release_contract.py`
- Create: `tests/test_dinov2_offline_contract.py`

**Interfaces:**
- Produces: packaged application that loads an installed DINOv2 component without runtime network access and CI that verifies DINOv2 parity/component health.

- [ ] **Step 1: Write RED offline/release contract tests**

Assert the active runtime has no import of `system.dinov3`, contains no `NERI_DINOV3_SOURCE`, `DinoV2Encoder` calls `AutoModel.from_pretrained` with `local_files_only=True`, and package scripts do not copy a DINOv3 source mirror.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_dinov2_offline_contract.py tests/test_lite_package_contract.py tests/test_release_contract.py -v
```

- [ ] **Step 3: Create the DINOv2 verification workflow**

Run DINOv2 checkpoint, encoder, classifier, runtime, component, backend integration, and benchmark-parity tests. Keep the Windows build workflow responsible for packaged application verification.

- [ ] **Step 4: Update Windows packaging scripts**

Preserve the optional `res/model/DINOv2` location and remove DINOv3 source-mirror assumptions. Keep `transformers`/`safetensors` in the optional maintained environment rather than the base `requirements.txt` unless current packaging explicitly vendors optional ML dependencies.

- [ ] **Step 5: Run release tests GREEN**

```bash
pytest tests/test_dinov2_offline_contract.py tests/test_lite_package_contract.py tests/test_release_contract.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows scripts tests/test_dinov2_offline_contract.py tests/test_lite_package_contract.py tests/test_release_contract.py
git commit -m "build: switch release contracts to DINOv2"
```

---

### Task 12: Add Benchmark-Parity Gate, Delete DINOv3, and Run Full Verification

**Files:**
- Create: `tests/test_dinov2_benchmark_parity.py`
- Create: `tests/fixtures/dinov2_benchmark_parity.json`
- Delete: `system/dinov3/`
- Delete: `system/backend/dinov3_environment.py`
- Delete: `system/backend/dinov3_maintenance.py`
- Delete: `system/backend/dinov3_feedback_api.py`
- Delete: `system/backend/dinov3_feedback_service.py`
- Delete: `system/backend/dinov3_registry_service.py`
- Delete: `system/backend/dinov3_persistence_fast.py`
- Delete: active `tests/test_dinov3_*.py` files superseded by the DINOv2 tests.
- Delete: active `frontend/test/dinov3_*.dart` files superseded by DINOv2 tests.
- Delete: active DINOv3 Flutter source files superseded in Task 10.
- Delete: `scripts/calibrate_dinov3_multi_dual.py`.
- Delete: `res/model/cls/user/README_DINOv3.md`.
- Delete: `res/model/cls/user/dinov3_classifier_merged_reviewed_20260908.neri.json`.
- Modify: `README.md` and current deployment docs that describe the supported runtime; historical design/plan documents remain unchanged.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: DINOv2-only repository state and frozen parity gate.

- [ ] **Step 1: Add frozen benchmark-parity fixture/test before deletion**

Export a small fixed set from the already frozen standalone DINOv2 benchmark. Each JSON case contains a 768-value encoder embedding plus expected `best_known_species`, selected-prototype cosine, fused rejection score, and accepted flag.

```python
@pytest.mark.parametrize("case", load_cases())
def test_runtime_matches_frozen_benchmark_decisions(case):
    feature = np.asarray(case["embedding"], dtype=np.float32)[None, :]
    prediction = classifier.classify_features(feature)[0]
    assert prediction.best_known_species == case["best_known_species"]
    assert prediction.known_score == pytest.approx(case["cosine"], abs=1e-6)
    assert prediction.rejection_score == pytest.approx(case["rejection_score"], abs=1e-6)
    assert prediction.accepted is case["accepted"]
```

Include at least one accepted Known, one rejected Known-near-threshold, one rejected Unknown, and one case whose checkpoint uses a real Ridge fusion tensor.

- [ ] **Step 2: Verify parity test GREEN before removing DINOv3**

```bash
pytest tests/test_dinov2_benchmark_parity.py -v
```

Expected: all parity cases pass.

- [ ] **Step 3: Delete active DINOv3 implementation and superseded tests**

Remove every file listed in this task. Do not delete historical documents under `docs/superpowers/specs` or `docs/superpowers/plans`.

- [ ] **Step 4: Scan active code for forbidden DINOv3 runtime references**

```bash
git grep -n -E 'system\.dinov3|DinoV3|dinov3_classifier|NERI_DINOV3_SOURCE|install-dinov3|remove-dinov3|dinov3-status' -- ':!docs/superpowers/**'
```

Expected: only the narrow legacy-error detection/message code that recognizes old `backend: "dinov3"` manifests. There must be no DINOv3 import, executable runtime path, selectable model, or installation route.

- [ ] **Step 5: Run complete Python suite**

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Run complete Flutter suite**

```bash
cd frontend
flutter test
```

Expected: all tests pass.

- [ ] **Step 7: Verify CI/release contract**

Confirm `.github/workflows/dinov2-implementation-verification.yml` includes checkpoint, encoder, classifier, runtime, component, integration, and parity tests. Confirm the Windows build workflow still exercises packaged startup/release contracts.

- [ ] **Step 8: Review final semantic invariants**

The final diff must satisfy all of these literal conditions:

```text
backend=dinov2
architecture=dinov2_vitb14
feature_dim=768
preprocessing=letterbox224_imagenet
class_alpha=0
no DINOv3 runtime imports
no DINOv3 prototype migration
runtime encoder loading is local_files_only
```

- [ ] **Step 9: Commit the clean break**

```bash
git add -A
git commit -m "refactor: complete DINOv2 clean break"
```

---

## Final Verification Matrix

Before claiming completion, all commands below must pass:

```bash
pytest tests/test_dinov2_checkpoint.py -v
pytest tests/test_dinov2_preprocess.py tests/test_dinov2_encoder.py -v
pytest tests/test_dinov2_classifier.py tests/test_dinov2_fusion_rejection.py -v
pytest tests/test_dinov2_runtime.py -v
pytest tests/test_dinov2_registry*.py tests/test_dinov2_feedback*.py tests/test_dinov2_state_isolation.py -v
pytest tests/test_dinov2_image_processor.py tests/test_dinov2_backend_integration.py -v
pytest tests/test_dinov2_model_catalog.py -v
pytest tests/test_dinov2_component.py tests/test_dinov2_public_distribution.py -v
pytest tests/test_dinov2_environment_api.py tests/test_dinov2_environment_maintenance.py -v
pytest tests/test_dinov2_benchmark_parity.py -v
pytest -q
cd frontend && flutter test
```

A successful migration means Neri no longer executes DINOv3 code, DINOv2 per-sample class/rejection behavior matches the frozen benchmark path, and current Python/Flutter/release tests all pass.
