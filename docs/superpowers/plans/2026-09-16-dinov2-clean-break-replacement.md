# DINOv2 Clean-Break Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Neri's DINOv3 subsystem with a DINOv2-only `facebook/dinov2-base` runtime that reproduces the validated Adaptive Multi-prototype + Fusion-rejection benchmark path.

**Architecture:** Keep the existing detector -> crop -> embedding-classifier -> Registry/feedback data flow, but replace the entire `system/dinov3` implementation and all DINOv3 API/UI identities with DINOv2 equivalents. The DINOv2 runtime owns local offline encoder loading, CL2N feature geometry, adaptive per-class prototypes, and an explicit Fusion-rejection schema that can represent either tensorized Ridge fusion or validation-selected `linear_type=none` fallback.

**Tech Stack:** Python 3, PyTorch, Hugging Face Transformers, safetensors, NumPy, FastAPI/Pydantic, SQLite, pytest, Flutter/Dart, flutter_test.

**Spec:** `docs/superpowers/specs/2026-09-16-dinov2-clean-break-replacement-design.md`

## Global Constraints

- Supported DINO architecture is exactly `dinov2_vitb14` / `facebook/dinov2-base`, feature dimension `768`.
- Encoder feature is `last_hidden_state[:, 0]`, followed by per-image L2 normalization.
- Preprocessing remains exactly `letterbox224_imagenet`; do not substitute Hugging Face resizing/cropping.
- Runtime comparison geometry is CL2N: encoder L2 -> subtract frozen feature center -> L2 normalize.
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
- `system/dinov2/preprocess.py` — unchanged Neri letterbox/ImageNet preprocessing contract under the new package.
- `system/dinov2/checkpoint.py` — DINOv2-only checkpoint schema, validation, fingerprinting, fusion tensors.
- `system/dinov2/encoder.py` — local `facebook/dinov2-base` loader and normalized CLS extraction.
- `system/dinov2/prototype_bank.py` — prototype records/banks moved from the old package.
- `system/dinov2/classifier.py` — adaptive Multi-prototype + Fusion-rejection classifier and prediction DTOs.
- `system/dinov2/rejection.py` — immutable Fusion-rejection configuration and score/gate calculations.
- `system/dinov2/runtime.py` — manifest resolution, checkpoint construction, Registry/feedback wiring.
- `system/dinov2/registry.py`, `registry_impl.py`, `registry_examples.py`, `feedback.py`, `feedback_index.py` — renamed fingerprint-isolated learning state.
- `system/dinov2/component.py`, `public_distribution.py`, `api.py`, `state.py`, `events.py` — local component lifecycle and API helpers.

Migration touchpoints:

- `system/model_sync/layout.py`, `catalog.py` — `DINOv2` component root and backend discovery.
- `system/image_processor.py` — attach/use/drain DINOv2 classifier observations.
- `system/backend/model_services.py`, `services.py` — DINOv2 manifest/classifier loading and job option validation.
- `system/backend/dinov2_environment.py`, `dinov2_maintenance.py`, `dinov2_feedback_api.py`, `dinov2_feedback_service.py`, `dinov2_registry_service.py`, `dinov2_persistence_fast.py` — backend-facing renamed services.
- `frontend/lib/src/api_client*.dart`, DINO models/widgets/screens/startup helpers — DINOv2 API and copy.
- `server/model_distribution/*` — DINOv2 capability metadata/distribution contract.
- `.github/workflows/*`, packaging scripts, tests — verification and release contract.

Historical DINOv3 design docs remain untouched.

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

- [ ] **Step 1: Write failing checkpoint tests**

Create fixtures with variable per-class prototype counts and both fallback and Ridge fusion metadata. Include these assertions:

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
```

Also test invalid class indices, mismatched prototype counts, non-finite tensors, `class_alpha != 0`, missing feature center, invalid encoder hash, and Ridge shapes.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest tests/test_dinov2_checkpoint.py -v
```

Expected: import/collection failure because `system.dinov2.checkpoint` does not exist.

- [ ] **Step 3: Implement the minimal immutable checkpoint schema**

Implement dataclasses equivalent to:

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
    encoder_weights: str
    encoder_sha256: str
    preprocessing: str
    event_aggregation: str
    fusion: FusionRejectionConfig
    fingerprint: str
```

Validate all tensor shapes and require `class_alpha == 0.0` for this schema. Fingerprinting must cover encoder identity, classes, center, prototypes, prototype indices/counts, fusion scalars, and optional Ridge tensors.

- [ ] **Step 4: Run checkpoint tests GREEN**

Run:

```bash
pytest tests/test_dinov2_checkpoint.py -v
```

Expected: all pass.

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
- Produces: `preprocess_image(value, array_color="rgb") -> torch.Tensor`, `DinoV2Encoder.encode(crops, array_color="rgb") -> np.ndarray`, `resolve_encoder_dir(...) -> Path`.

- [ ] **Step 1: Write RED preprocessing and encoder tests**

Preserve a golden preprocessing test for the existing 224 x 224 letterbox result and define a fake Transformers model:

```python
class FakeDinoV2Model:
    def eval(self): return self
    def requires_grad_(self, _value): return self
    def to(self, _device): return self
    def __call__(self, pixel_values):
        batch = pixel_values.shape[0]
        hidden = torch.zeros((batch, 2, 768), dtype=torch.float32)
        hidden[:, 0, 0] = 3.0
        hidden[:, 0, 1] = 4.0
        return SimpleNamespace(last_hidden_state=hidden)


def test_encoder_uses_cls_token_and_l2_normalizes(...):
    features = encoder.encode([image])
    assert features.shape == (1, 768)
    assert features[0, 0] == pytest.approx(0.6)
    assert features[0, 1] == pytest.approx(0.8)
    assert np.linalg.norm(features[0]) == pytest.approx(1.0)
```

Test `local_files_only=True`, local directory resolution, hash/inventory failure, CPU determinism, and CUDA request failure on systems without CUDA.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_dinov2_preprocess.py tests/test_dinov2_encoder.py -v
```

Expected: missing module/functions.

- [ ] **Step 3: Copy preprocessing contract without semantic changes**

Move the existing Neri letterbox implementation into `system/dinov2/preprocess.py`; keep `IMAGE_SIZE=224`, current fill RGB, ImageNet mean/std, bicubic resize, CHW float output.

- [ ] **Step 4: Implement local Transformers encoder**

Use dependency injection for tests, but production loading follows:

```python
model = AutoModel.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=False,
)
outputs = model(pixel_values=batch)
features = outputs.last_hidden_state[:, 0]
features = F.normalize(features.float(), dim=1)
```

Do not instantiate `AutoImageProcessor`; Neri's preprocessing remains authoritative.

- [ ] **Step 5: Run encoder/preprocessing tests GREEN**

Run:

```bash
pytest tests/test_dinov2_preprocess.py tests/test_dinov2_encoder.py -v
```

Expected: all pass.

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
- Produces: `DinoV2Prediction`, `DinoV2Observation`, `DinoV2Classifier.classify_features()`, `.classify_crops()`, `.classify_event()`, `.explain_feature()`.

- [ ] **Step 1: Write RED CL2N/adaptive-prototype tests**

Use deliberately uneven counts `(1, 3)` and verify nearest-prototype species selection after center subtraction and re-normalization:

```python
def test_closed_class_uses_nearest_adaptive_prototype_in_cl2n_space():
    classifier = DinoV2Classifier(validate_checkpoint(make_dinov2_payload()))
    prediction = classifier.classify_features(unit_feature(0)[None, :])[0]
    assert prediction.best_known_species == "A"
    assert prediction.nearest_prototype_index == 0
```

- [ ] **Step 2: Write RED Fusion-rejection tests**

Cover both safe fallback and Ridge:

```python
def test_linear_none_never_requires_linear_logits(): ...

def test_ridge_can_change_rejection_score_but_not_closed_class():
    prediction = classifier.classify_features(feature[None, :])[0]
    assert prediction.best_known_species == prototype_winner
    assert prediction.class_alpha == 0.0
```

Verify the score calculation uses frozen scales, `score_alpha`, and `margin_weight`; acceptance requires both selected-prototype cosine gate and fused score gate.

- [ ] **Step 3: Run RED tests**

Run:

```bash
pytest tests/test_dinov2_classifier.py tests/test_dinov2_fusion_rejection.py -v
```

Expected: missing classifier/rejection implementation.

- [ ] **Step 4: Implement CL2N and nearest-prototype class scores**

The core transform is:

```python
centered = features - feature_center[None, :]
z = centered / np.linalg.norm(centered, axis=1, keepdims=True)
```

Compute per-prototype squared distance and cosine, then reduce to the nearest prototype for each class.

- [ ] **Step 5: Implement frozen linear logits and fused score**

For Ridge:

```python
linear_logits = z @ coef.T + intercept
```

Scale prototype/linear scores using checkpoint scalars. Keep class selection prototype-only because `class_alpha=0`. Calculate the configured winner-vs-runner-up margin term and final rejection score, then apply cosine + score thresholds.

- [ ] **Step 6: Preserve Registry provisional semantics**

Port `PrototypeRecord` / `PrototypeBank` behavior so formal prototypes can be accepted, provisional prototypes remain assistive (`accepted=False`, `assistive_match=True`), and inference only emits routing metadata.

- [ ] **Step 7: Run classifier tests GREEN**

Run:

```bash
pytest tests/test_dinov2_classifier.py tests/test_dinov2_fusion_rejection.py -v
```

Expected: all pass.

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
- Consumes: Task 1-3 classifier/encoder/checkpoint interfaces.
- Produces: `DinoV2Runtime`, `DinoV2ManifestError`, `resolve_dinov2_manifest()`, `load_dinov2_model()`.

- [ ] **Step 1: Write RED runtime tests**

Create a schema-v2 manifest fixture and assert:

```python
def test_runtime_loads_dinov2_manifest(...):
    runtime = load_dinov2_model(manifest, encoder_factory=FakeEncoder, state_root=tmp_path / "state")
    assert runtime.checkpoint.backbone == "dinov2_vitb14"
    assert isinstance(runtime.classifier, DinoV2Classifier)


def test_runtime_rejects_dinov3_manifest_explicitly(...):
    manifest.write_text(json.dumps({"backend": "dinov3", ...}))
    with pytest.raises(DinoV2ManifestError, match="DINOv3.*不再支持"):
        resolve_dinov2_manifest(manifest)
```

Also test manifest/checkpoint architecture, hash, preprocessing, and aggregation mismatches.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_dinov2_runtime.py -v
```

- [ ] **Step 3: Implement DINOv2 manifest resolution/validation**

Require `schema_version == 2`, `backend == "dinov2"`, `architecture == "dinov2_vitb14"`, feature dimension `768`, and matching encoder/preprocessing metadata.

- [ ] **Step 4: Construct classifier + encoder + state stores**

Follow the existing lifecycle pattern: create stores when not injected, validate model fingerprints when injected, close only owned stores on `runtime.close()`, and clean up partially created stores on exceptions.

- [ ] **Step 5: Run runtime tests GREEN**

Run:

```bash
pytest tests/test_dinov2_runtime.py -v
```

- [ ] **Step 6: Commit**

```bash
git add system/dinov2/runtime.py system/dinov2/state.py tests/test_dinov2_runtime.py
git commit -m "feat: add DINOv2 manifest runtime"
```

---

### Task 5: Port Registry, Feedback, and Observation Persistence to DINOv2 Fingerprints

**Files:**
- Create/port: `system/dinov2/registry.py`
- Create/port: `system/dinov2/registry_impl.py`
- Create/port: `system/dinov2/registry_examples.py`
- Create/port: `system/dinov2/feedback.py`
- Create/port: `system/dinov2/feedback_index.py`
- Create/port: `system/dinov2/events.py`
- Create: `tests/test_dinov2_registry.py`
- Create: `tests/test_dinov2_feedback.py`
- Create: `tests/test_dinov2_state_isolation.py`
- Reference: corresponding `system/dinov3/*` files and existing `tests/test_dinov3_registry*.py`, `tests/test_dinov3_feedback*.py`.

**Interfaces:**
- Consumes: DINOv2 fingerprint and CL2N feature center.
- Produces: DINOv2 Registry/feedback stores with existing human-confirmation lifecycle and `prototype_bank(feature_center)` provider contract.

- [ ] **Step 1: Write RED state-isolation test**

```python
def test_dinov3_state_is_not_opened_as_dinov2_state(tmp_path):
    legacy = tmp_path / "dinov3" / "registry.sqlite3"
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    path = registry_path_for_fingerprint(tmp_path, "a" * 64)
    assert path != legacy
    assert "a" * 64 in path.name
```

Add tests that formal prototypes change only after explicit human confirmation and that provisional evidence remains provisional.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_dinov2_registry.py tests/test_dinov2_feedback.py tests/test_dinov2_state_isolation.py -v
```

- [ ] **Step 3: Port persistence code with DINOv2 naming only**

Preserve SQLite transaction, WAL/busy-timeout, candidate clustering, independent-event grouping, and explicit human-confirmation invariants from the current runtime. Rename tables/metadata only where the identifier is externally visible; do not rewrite historical DBs in place.

- [ ] **Step 4: Ensure prototype regeneration uses DINOv2 vectors only**

Validate feature shape `(768,)`, finiteness, model fingerprint, and source observation ownership before learned prototype regeneration.

- [ ] **Step 5: Run state tests GREEN plus existing semantic regressions**

Run the new DINOv2 state tests and port the existing registry/feedback semantic tests under DINOv2 names.

- [ ] **Step 6: Commit**

```bash
git add system/dinov2/registry.py system/dinov2/registry_impl.py system/dinov2/registry_examples.py system/dinov2/feedback.py system/dinov2/feedback_index.py system/dinov2/events.py tests/test_dinov2_registry.py tests/test_dinov2_feedback.py tests/test_dinov2_state_isolation.py
git commit -m "feat: port open world state to DINOv2"
```

---

### Task 6: Switch ImageProcessor and Backend Inference Wiring to DINOv2

**Files:**
- Modify: `system/image_processor.py` DINO classifier fields, crop-classification branch, observation draining.
- Modify: `system/backend/services.py` DINO manifest detection, job validation, runtime construction, persistence calls.
- Modify: `system/backend/model_services.py` DINO model-class loading.
- Create/rename: `system/backend/dinov2_feedback_service.py`
- Create/rename: `system/backend/dinov2_registry_service.py`
- Create/rename: `system/backend/dinov2_persistence_fast.py`
- Create/rename: `system/backend/dinov2_feedback_api.py`
- Test: `tests/test_dinov2_image_processor.py`
- Test: `tests/test_dinov2_backend_integration.py`

**Interfaces:**
- Consumes: `load_dinov2_model()` and `DinoV2Classifier`.
- Produces: `ImageProcessor.load_dinov2_classifier(classifier)`, `drain_dinov2_observations()`, backend persistence using DINOv2 observations.

- [ ] **Step 1: Write RED ImageProcessor integration test**

Assert the existing crop batching calls the attached classifier and emits `DinoV2Observation` records, not DINOv3 records.

- [ ] **Step 2: Write RED backend runtime-selection test**

A classification manifest with `backend: "dinov2"` must call `load_dinov2_model`; a `backend: "dinov3"` selection must raise the explicit unsupported error before Ultralytics loading.

- [ ] **Step 3: Verify RED**

Run:

```bash
pytest tests/test_dinov2_image_processor.py tests/test_dinov2_backend_integration.py -v
```

- [ ] **Step 4: Rename ImageProcessor attachment state**

Replace `dinov3_classifier`, `load_dinov3_classifier`, `_dinov3_observations`, and `drain_dinov3_observations` with DINOv2-only names. Do not retain aliases.

- [ ] **Step 5: Switch backend services to DINOv2 runtime**

Update manifest detection to return DINOv2 payloads; explicitly detect DINOv3 payloads to raise the migration error. Persist DINOv2 observations via renamed feedback services.

- [ ] **Step 6: Run integration tests GREEN**

Run the two new test modules plus the existing preview/validation tests that exercise `ProcessingJobManager`.

- [ ] **Step 7: Commit**

```bash
git add system/image_processor.py system/backend/services.py system/backend/model_services.py system/backend/dinov2_feedback_service.py system/backend/dinov2_registry_service.py system/backend/dinov2_persistence_fast.py system/backend/dinov2_feedback_api.py tests/test_dinov2_image_processor.py tests/test_dinov2_backend_integration.py
git commit -m "refactor: route classification through DINOv2"
```

---

### Task 7: Replace Model Layout and Catalog Discovery with DINOv2

**Files:**
- Modify: `system/model_sync/layout.py`
- Modify: `system/model_sync/catalog.py`
- Modify: `system/backend/models.py` if backend literals are duplicated there.
- Create: `tests/test_dinov2_model_catalog.py`
- Modify: `tests/test_model_resource_paths.py`
- Modify: `tests/test_model_catalog.py`

**Interfaces:**
- Produces: `ModelLayout.dinov2_root`, `ModelBackend = Literal["yolo", "dinov2"]`, DINOv2 manifest discovery and saved-path resolution.

- [ ] **Step 1: Write RED layout/catalog tests**

Port the current installed-component test so a healthy `res/model/DINOv2` install exposes only the manifest declared by `install.json`, while an unhealthy component is hidden.

```python
def test_manifest_exposes_dinov2_capabilities(tmp_path):
    ...
    assert model.backend == "dinov2"
    assert model.architecture == "dinov2_vitb14"
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_dinov2_model_catalog.py tests/test_model_resource_paths.py tests/test_model_catalog.py -v
```

- [ ] **Step 3: Change canonical component root**

Replace `ModelLayout.dinov3_root` with `dinov2_root = root / "DINOv2"` and update callers.

- [ ] **Step 4: Change catalog backend and manifest parsing**

Recognize only `backend: "dinov2"` as an active DINO model. Keep a small legacy-manifest detector outside the selectable-model path so saved DINOv3 settings can fail explicitly instead of being interpreted as YOLO.

- [ ] **Step 5: Run catalog tests GREEN**

Run the same command from Step 2.

- [ ] **Step 6: Commit**

```bash
git add system/model_sync/layout.py system/model_sync/catalog.py system/backend/models.py tests/test_dinov2_model_catalog.py tests/test_model_resource_paths.py tests/test_model_catalog.py
git commit -m "refactor: make model catalog DINOv2 only"
```

---

### Task 8: Replace DINOv3 Component Installation and Distribution with Offline DINOv2 Assets

**Files:**
- Create: `system/dinov2/component.py`
- Create: `system/dinov2/public_distribution.py`
- Modify: `server/model_distribution/capabilities.py`
- Modify: `server/model_distribution/manifest.py`
- Modify: `server/model_distribution/service.py` as needed for DINOv2 package metadata.
- Create: `tests/test_dinov2_component.py`
- Create: `tests/test_dinov2_public_distribution.py`
- Modify: `tests/test_model_distribution.py`

**Interfaces:**
- Produces: `DinoV2ComponentPaths`, `dinov2_component_paths()`, `dinov2_component_status()`, `install_dinov2_component()`, `remove_dinov2_component()`.

- [ ] **Step 1: Write RED component health tests**

Require all of:

```text
model/config.json
model/model.safetensors
model/preprocessor_config.json
classifier.pt
classifier.neri.json
LICENSE_DINOv2.md
install.json
```

Test complete inventory hashing, undeclared extra files, missing files, size mismatch, SHA mismatch, and a healthy install.

- [ ] **Step 2: Write RED distribution contract tests**

Server capability response must advertise `dinov2_vitb14`, backend `dinov2`, and the DINOv2 package manifest; no current package may claim DINOv3 compatibility.

- [ ] **Step 3: Verify RED**

Run:

```bash
pytest tests/test_dinov2_component.py tests/test_dinov2_public_distribution.py tests/test_model_distribution.py -v
```

- [ ] **Step 4: Implement component layout and install/remove lifecycle**

Adapt the existing atomic staging/inventory validation pattern, but remove Meta source-tree installation entirely. The installed model directory must be sufficient for `AutoModel.from_pretrained(..., local_files_only=True)`.

- [ ] **Step 5: Update model-distribution metadata**

Return DINOv2 package metadata and hashes through the existing anonymous distribution service contract.

- [ ] **Step 6: Run component/distribution tests GREEN**

Run the same command from Step 3.

- [ ] **Step 7: Commit**

```bash
git add system/dinov2/component.py system/dinov2/public_distribution.py server/model_distribution/capabilities.py server/model_distribution/manifest.py server/model_distribution/service.py tests/test_dinov2_component.py tests/test_dinov2_public_distribution.py tests/test_model_distribution.py
git commit -m "feat: distribute offline DINOv2 component"
```

---

### Task 9: Replace Backend Environment Maintenance APIs with DINOv2

**Files:**
- Create: `system/backend/dinov2_environment.py`
- Create: `system/backend/dinov2_maintenance.py`
- Modify: `system/backend/main_core.py` router registration.
- Modify: optional dependency installation helpers in `system/backend/maintenance.py`.
- Create: `tests/test_dinov2_environment_api.py`
- Create: `tests/test_dinov2_environment_maintenance.py`

**Interfaces:**
- Produces routes: `GET /api/environment/dinov2-status`, `POST /api/environment/install-dinov2`, `POST /api/environment/remove-dinov2`.

- [ ] **Step 1: Write RED API tests**

Assert exact route names, response architecture `DINOv2 ViT-B/14`, classifier fingerprint, and operation names `install_dinov2` / `remove_dinov2`.

- [ ] **Step 2: Write RED dependency-maintenance test**

The install worker must preserve shared PyTorch and ensure `transformers` and `safetensors` are present before component health verification.

- [ ] **Step 3: Verify RED**

Run:

```bash
pytest tests/test_dinov2_environment_api.py tests/test_dinov2_environment_maintenance.py -v
```

- [ ] **Step 4: Implement DINOv2 environment router and worker**

Mirror current shutdown/restart/progress semantics. Replace all operation names and messages with DINOv2. Remove source-code mirror checks.

- [ ] **Step 5: Register DINOv2 router and dependency installer**

Keep package-source selection behavior. Install only the extra runtime packages needed beyond the shared PyTorch environment.

- [ ] **Step 6: Run environment tests GREEN**

Run the same command from Step 3 plus `tests/test_runtime_patches.py` if maintenance imports affect startup.

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
- Rename/create DINOv2 equivalents for `frontend/lib/src/dinov3_startup_check.dart`, `models/dinov3_explanation.dart`, `models/dinov3_feedback.dart`, `models/dinov3_registry.dart`, `widgets/dinov3_feature_scatter.dart`, `widgets/dinov3_registry_dialog.dart`.
- Modify: `frontend/lib/src/screens/settings_screen.dart`
- Modify: `frontend/lib/src/screens/species_validation_screen.dart`
- Modify: `frontend/lib/src/main_window.dart`
- Create/rename DINOv2 Flutter tests corresponding to current `frontend/test/dinov3_*` tests.

**Interfaces:**
- Produces: `fetchDinoV2ComponentStatus()`, `installDinoV2(...)`, `removeDinoV2()`, DINOv2 registry/feedback DTOs and DINOv2-only UI copy.

- [ ] **Step 1: Port environment API test to RED DINOv2 expectations**

Use exact paths:

```dart
expect(request.url.path, '/api/environment/dinov2-status');
...
final status = await client.fetchDinoV2ComponentStatus();
expect(status.architecture, 'DINOv2 ViT-B/14');
```

Also port install/remove expectations to `/install-dinov2` and `/remove-dinov2`.

- [ ] **Step 2: Port Registry/feedback/startup tests to DINOv2 names**

Keep behavior assertions unchanged where the semantic lifecycle is unchanged; change backend/model-family identities only.

- [ ] **Step 3: Verify RED Flutter tests**

Run the DINOv2-named subset with `flutter test` and expect compile failures for missing DINOv2 APIs/types.

- [ ] **Step 4: Rename API methods and DTOs**

Replace all public DINOv3 API method/type names with DINOv2 names. Do not add deprecated aliases.

- [ ] **Step 5: Rename UI helpers/widgets and copy**

Replace visible `DINOv3` strings with `DINOv2` on active UI paths. Preserve historical docs only; current UI must not advertise DINOv3 support.

- [ ] **Step 6: Run Flutter DINOv2 tests GREEN**

Run all DINOv2-specific Flutter tests, then:

```bash
cd frontend && flutter test
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
- Modify: `.github/workflows/dinov3-implementation-verification.yml` -> replace with DINOv2-named workflow.
- Modify: `.github/workflows/build-windows-x64.yml` if runtime dependency/package assertions reference DINOv3.
- Modify: `scripts/package_windows.ps1`
- Modify: `scripts/prepare_windows.ps1`
- Modify: `scripts/prune_lite_python.ps1`
- Modify: `requirements.txt` only if these packages belong in the base environment; otherwise keep them in maintenance installation logic.
- Modify: `tests/test_lite_package_contract.py`
- Modify: `tests/test_release_contract.py`
- Create: `tests/test_dinov2_offline_contract.py`

**Interfaces:**
- Produces: packaged application that can load an already installed DINOv2 component without network access and can install required optional Python packages through the maintenance path.

- [ ] **Step 1: Write RED offline/release contract tests**

Assert no active runtime import references `dinov3`, no `NERI_DINOV3_SOURCE`, and no network model fetch is required by `system.dinov2.encoder`.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_dinov2_offline_contract.py tests/test_lite_package_contract.py tests/test_release_contract.py -v
```

- [ ] **Step 3: Update packaging and workflow contracts**

Rename DINO verification workflow and its test targets. Ensure packaging preserves the optional DINOv2 component location and does not bundle obsolete DINOv3 source mirrors.

- [ ] **Step 4: Run packaging/release tests GREEN**

Run the same tests and validate PowerShell scripts syntactically on Windows CI.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows scripts requirements.txt tests/test_dinov2_offline_contract.py tests/test_lite_package_contract.py tests/test_release_contract.py
git commit -m "build: switch release contracts to DINOv2"
```

---

### Task 12: Delete DINOv3 Runtime, Add Benchmark-Parity Gate, and Run Full Verification

**Files:**
- Delete: `system/dinov3/` runtime implementation.
- Delete: DINOv3-only backend implementation files after their DINOv2 replacements are green.
- Delete/rename: active `tests/test_dinov3_*.py` tests superseded by DINOv2 tests.
- Delete/rename: active `frontend/test/dinov3_*.dart` tests superseded by DINOv2 tests.
- Delete: bundled active DINOv3 user manifest under `res/model/cls/user/`.
- Create: `tests/test_dinov2_benchmark_parity.py`
- Create: `tests/fixtures/dinov2_benchmark_parity.json` or an equivalent small non-sensitive fixture containing frozen embeddings/expected scores and decisions.
- Modify: `README.md` / current deployment docs only where they describe the currently supported runtime; leave historical specs/plans unchanged.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: DINOv2-only repository state and a parity gate for deployment semantics.

- [ ] **Step 1: Add a frozen benchmark-parity test before deletion**

Use a small fixture exported from the standalone benchmark containing precomputed encoder embeddings plus expected prototype winner, selected-prototype cosine, fused score, acceptance decision, and expected species.

```python
@pytest.mark.parametrize("case", load_cases())
def test_runtime_matches_frozen_benchmark_decisions(case):
    prediction = classifier.classify_features(np.asarray(case["embedding"], dtype=np.float32)[None, :])[0]
    assert prediction.best_known_species == case["best_known_species"]
    assert prediction.known_score == pytest.approx(case["cosine"], abs=1e-6)
    assert prediction.rejection_score == pytest.approx(case["rejection_score"], abs=1e-6)
    assert prediction.accepted is case["accepted"]
```

- [ ] **Step 2: Verify parity test GREEN before removing DINOv3**

Run:

```bash
pytest tests/test_dinov2_benchmark_parity.py -v
```

Expected: pass against the completed DINOv2 runtime.

- [ ] **Step 3: Delete active DINOv3 code and superseded tests**

Remove the old runtime/backend/UI implementation only now. Do not delete historical documents under `docs/superpowers/specs` or `docs/superpowers/plans`.

- [ ] **Step 4: Scan active code for forbidden DINOv3 references**

Run:

```bash
git grep -n -E 'system\.dinov3|backend.?[=: ]+["'"']dinov3["'"']|DinoV3|dinov3_classifier|NERI_DINOV3_SOURCE' -- ':!docs/superpowers/**'
```

Expected: no active runtime/UI/test matches. Any intentional legacy-error string must be narrow, reviewed, and not import or execute DINOv3 code.

- [ ] **Step 5: Run the complete Python suite**

Run:

```bash
pytest -q
```

Expected: all pass.

- [ ] **Step 6: Run the complete Flutter suite**

Run:

```bash
cd frontend && flutter test
```

Expected: all pass.

- [ ] **Step 7: Run release/build verification**

Run the DINOv2 implementation verification workflow locally where possible and rely on GitHub Actions for Windows packaging/build checks. Verify the DINOv2 component health test and benchmark-parity test are included in CI.

- [ ] **Step 8: Review the final diff for semantic leftovers**

Confirm:

```text
backend=dinov2
architecture=dinov2_vitb14
feature_dim=768
preprocessing=letterbox224_imagenet
class_alpha=0
no DINOv3 runtime imports
no DINOv3 prototype migration
```

- [ ] **Step 9: Commit the clean break**

```bash
git add -A
git commit -m "refactor: complete DINOv2 clean break"
```

---

## Final Verification Matrix

Before claiming completion, all of the following must be green:

```bash
pytest tests/test_dinov2_checkpoint.py -v
pytest tests/test_dinov2_preprocess.py tests/test_dinov2_encoder.py -v
pytest tests/test_dinov2_classifier.py tests/test_dinov2_fusion_rejection.py -v
pytest tests/test_dinov2_runtime.py -v
pytest tests/test_dinov2_registry.py tests/test_dinov2_feedback.py tests/test_dinov2_state_isolation.py -v
pytest tests/test_dinov2_image_processor.py tests/test_dinov2_backend_integration.py -v
pytest tests/test_dinov2_model_catalog.py -v
pytest tests/test_dinov2_component.py tests/test_dinov2_public_distribution.py -v
pytest tests/test_dinov2_environment_api.py tests/test_dinov2_environment_maintenance.py -v
pytest tests/test_dinov2_benchmark_parity.py -v
pytest -q
cd frontend && flutter test
```

A successful migration means Neri no longer executes DINOv3 code, DINOv2 per-sample classifier/rejection behavior matches the frozen benchmark path, and current Python/Flutter/release tests all pass.
