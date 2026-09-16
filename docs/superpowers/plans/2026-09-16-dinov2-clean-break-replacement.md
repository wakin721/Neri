# DINOv2 Clean-Break Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Neri's DINOv3 subsystem with a DINOv2-only `facebook/dinov2-base` runtime that reproduces the frozen validation-selected Adaptive Multi-prototype + `multi_dual_margin` checkpoint.

**Architecture:** Keep detector -> crop -> embedding classifier -> Registry/feedback flow. Replace DINOv3 implementation/API/UI identities with DINOv2, use benchmark-exact 224 letterbox preprocessing and CLS embeddings, then CL2N + nearest adaptive prototype + cosine and margin-adjusted distance rejection. Linear Ridge/SVM fusion is not part of the production checkpoint and is out of scope for this migration.

**Tech Stack:** Python 3, PyTorch, Transformers, safetensors, NumPy, FastAPI/Pydantic, SQLite, pytest, Flutter/Dart.

**Spec:** `docs/superpowers/specs/2026-09-16-dinov2-clean-break-replacement-design.md`

## Global Constraints

- Checkpoint backbone is `facebook/dinov2-base`; Neri manifest architecture is `dinov2_vitb14`; feature dimension is `768`.
- Benchmark model revision is `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.
- Crop embedding is `last_hidden_state[:, 0]` followed by L2 normalization.
- Preprocessing is exactly 224x224 letterbox, fill `(124,116,104)`, bicubic, ImageNet mean/std.
- Comparison geometry is CL2N: subtract frozen center, then L2 normalize.
- Closed class is the globally nearest prototype by squared Euclidean distance.
- Rejection mode is exactly `multi_dual_margin` for the deployment checkpoint.
- `adjusted = (-winner_d2 + margin_weight * different_class_margin) / winner_class_scale` when scales exist.
- Accept iff winner cosine and adjusted score both pass frozen thresholds.
- DINOv2 and DINOv3 vectors are never interchangeable.
- Runtime is offline after component installation.
- Human confirmation remains required for formal learned prototype updates.
- DINOv3 code is deleted only after DINOv2 parity and full tests are green.

---

### Task 1: DINOv2 Checkpoint Contract

**Files:**
- Create: `system/dinov2/__init__.py`
- Create: `system/dinov2/checkpoint.py`
- Create: `tests/dinov2_fixtures.py`
- Create: `tests/test_dinov2_checkpoint.py`

**Interfaces:** `DinoV2Checkpoint`, `DinoV2Rejection`, `validate_checkpoint(payload)`, `load_checkpoint(path)`.

- [ ] Write tests first for a two-class adaptive prototype fixture with counts `(1,3)`, `head_type="multi_prototype"`, backbone `facebook/dinov2-base`, center `(768,)`, prototype mapping, and `rejection.mode="multi_dual_margin"`.
- [ ] Add failing tests for DINOv3 backbone, wrong feature dimension, missing/non-finite center, invalid prototype mapping/counts, invalid SHA-256, invalid cosine threshold, invalid adjusted score threshold, negative margin weight, and invalid class-distance scales.
- [ ] Run `pytest tests/test_dinov2_checkpoint.py -v` and verify RED because `system.dinov2.checkpoint` is absent.
- [ ] Implement immutable dataclasses and validation. Preserve adaptive-k/provenance metadata as read-only values. Classifier fingerprint must cover feature-space identity, center, prototypes, mapping/counts, encoder hash, preprocessing/aggregation, and rejection fields.
- [ ] Run the test module GREEN.
- [ ] Commit `feat: add DINOv2 checkpoint contract`.

### Task 2: Benchmark-Exact Preprocessing and Encoder

**Files:**
- Create: `system/dinov2/preprocess.py`
- Create: `system/dinov2/encoder.py`
- Create: `tests/test_dinov2_preprocess.py`
- Create: `tests/test_dinov2_encoder.py`

**Interfaces:** `preprocess_image(...) -> torch.Tensor`, `DinoV2Encoder.encode(...) -> np.ndarray`, `resolve_encoder_dir(...)`.

- [ ] Write a preprocessing golden test matching benchmark `dinov2_common.preprocess`: EXIF transpose, RGB, `ImageOps.contain(...,224,224,BICUBIC)`, fill `(124,116,104)`, ImageNet normalize.
- [ ] Write a fake-model RED test where CLS token `[3,4,0...]` becomes `[0.6,0.8,0...]`; verify `(N,768)` and unit norm.
- [ ] Write RED tests proving production model load passes `local_files_only=True`, `trust_remote_code=False`, and that asset hash mismatch prevents inference.
- [ ] Run the two test modules and verify RED.
- [ ] Implement preprocessing and local Transformers loading without `AutoImageProcessor`.
- [ ] Run GREEN and commit `feat: add offline DINOv2 encoder`.

### Task 3: Adaptive Prototype + Multi-Dual-Margin Classifier

**Files:**
- Create: `system/dinov2/prototype_bank.py`
- Create: `system/dinov2/rejection.py`
- Create: `system/dinov2/classifier.py`
- Create: `tests/test_dinov2_classifier.py`
- Create: `tests/test_dinov2_margin_rejection.py`

**Interfaces:** `DinoV2Prediction`, `DinoV2Observation`, `DinoV2Classifier.classify_features()`, `classify_crops()`, `classify_event()`, `explain_feature()`.

- [ ] Write RED test for CL2N and global nearest-prototype winner with unequal per-class prototype counts.
- [ ] Write RED test proving different-class margin ignores additional prototypes from the winning class.
- [ ] Write RED test for exact benchmark math: `distance_score=-winner_d2`, `class_margin=second_nearest_class_d2-winner_class_d2`, optional class-scale division, two simultaneous acceptance gates.
- [ ] Write RED tests for cosine-pass/score-fail, score-pass/cosine-fail, both-pass, and both-fail.
- [ ] Run classifier tests RED.
- [ ] Implement benchmark formulas with float32 geometry and float64 threshold comparisons matching training code operation order.
- [ ] Port provisional Registry prototype semantics: provisional matches are assistive and never formally accepted by inference alone.
- [ ] Run GREEN and commit `feat: add DINOv2 margin rejection classifier`.

### Task 4: DINOv2 Manifest Runtime

**Files:** `system/dinov2/runtime.py`, `system/dinov2/state.py`, `tests/test_dinov2_runtime.py`.

- [ ] Write RED tests for schema-v2 `backend="dinov2"`, architecture/dimension/hash/preprocessing cross-validation, checkpoint-path-to-manifest resolution, and explicit unsupported DINOv3 error.
- [ ] Implement `DinoV2Runtime`, `DinoV2ManifestError`, `resolve_dinov2_manifest()`, `load_dinov2_model()` with owned-store cleanup semantics copied from the current runtime.
- [ ] Run GREEN and commit `feat: add DINOv2 manifest runtime`.

### Task 5: Registry and Feedback State

**Files:** port DINOv3 registry/feedback/event modules to `system/dinov2/`; create `tests/test_dinov2_registry.py`, `test_dinov2_feedback.py`, `test_dinov2_state_isolation.py`.

- [ ] Write RED tests proving DINOv3 DB/vector state is not opened as DINOv2 state, only human confirmation updates formal evidence, and regenerated vectors are finite `(768,)` DINOv2 features.
- [ ] Port transaction/WAL/busy-timeout, candidate clustering, independent-event grouping, provisional routing, and fingerprint scoping without legacy vector migration.
- [ ] Run GREEN and commit `feat: port open world state to DINOv2`.

### Task 6: ImageProcessor and Backend Inference Wiring

**Files:** `system/image_processor.py`, `system/backend/services.py`, `model_services.py`, DINOv2 feedback/registry/persistence services, integration tests.

- [ ] Write RED tests that a `backend="dinov2"` model routes through `load_dinov2_model`, crop batches call the DINOv2 classifier, observations persist through DINOv2 services, and DINOv3 manifests raise before Ultralytics fallback.
- [ ] Rename attachment state/methods to DINOv2 with no aliases and switch backend persistence wiring.
- [ ] Run GREEN and commit `refactor: route classification through DINOv2`.

### Task 7: Model Layout and Catalog

**Files:** `system/model_sync/layout.py`, `catalog.py`, backend model DTOs, catalog tests.

- [ ] Write RED tests for `res/model/DINOv2`, `ModelBackend = Literal["yolo","dinov2"]`, healthy installed manifest discovery, unhealthy component hiding, and saved-path resolution.
- [ ] Implement DINOv2-only discovery; keep only a narrow legacy detector needed for a clear unsupported message.
- [ ] Run GREEN and commit `refactor: make model catalog DINOv2 only`.

### Task 8: Component Installation and Distribution

**Files:** `system/dinov2/component.py`, `public_distribution.py`, server model-distribution files, component/distribution tests.

- [ ] Write RED health tests requiring local model config/weights, classifier, manifest, license, and complete `install.json`; cover missing/extra/hash/size failures.
- [ ] Write RED server capability test advertising DINOv2 only.
- [ ] Implement atomic install/remove/inventory verification without DINOv3 source-tree mirroring.
- [ ] Run GREEN and commit `feat: distribute offline DINOv2 component`.

### Task 9: Environment Maintenance API

**Files:** DINOv2 environment/maintenance routers, main router registration, optional dependency installer, tests.

- [ ] Write RED tests for `/api/environment/dinov2-status`, `/install-dinov2`, `/remove-dinov2`, operation names, progress behavior, and `transformers`/`safetensors` dependency installation while preserving shared PyTorch.
- [ ] Implement and register DINOv2 maintenance paths; delete source-mirror checks.
- [ ] Run GREEN and commit `feat: add DINOv2 environment maintenance`.

### Task 10: Flutter Migration

**Files:** Flutter API client, startup check, DINO DTOs/widgets/dialogs/screens, DINOv2-named tests.

- [ ] Port environment API tests to exact DINOv2 routes and types, then verify RED compile failures.
- [ ] Port Registry/feedback/startup tests and active UI strings to DINOv2 names with no aliases.
- [ ] Run targeted tests then `cd frontend && flutter test` GREEN.
- [ ] Commit `refactor: migrate Flutter DINO UI to DINOv2`.

### Task 11: Packaging and CI

**Files:** DINO verification workflow, Windows packaging scripts, release-contract tests.

- [ ] Write RED contract tests forbidding active `system.dinov3`, `DinoV3`, `NERI_DINOV3_SOURCE`, and runtime network model fetches outside historical docs/legacy error text.
- [ ] Rename CI targets and package contracts to DINOv2; preserve optional component location.
- [ ] Run release tests GREEN and commit `build: switch release contracts to DINOv2`.

### Task 12: Benchmark Parity, Delete DINOv3, Full Verification

**Files:** parity fixture/test, deletion of active DINOv3 code/tests/UI manifest, current docs.

- [ ] Export a small frozen fixture from the benchmark containing raw normalized feature, expected best-known species, winner cosine, winner squared distance, class margin, adjusted score, and accepted/Unknown decision.
- [ ] Write and run `tests/test_dinov2_benchmark_parity.py` GREEN before deletion.
- [ ] Delete active DINOv3 runtime/backend/UI implementation and superseded tests; keep historical design docs.
- [ ] Run a repository scan for active DINOv3 imports/symbols and review intentional legacy error strings.
- [ ] Run `pytest -q` and `cd frontend && flutter test`.
- [ ] Run release/build CI checks and confirm the DINOv2 parity test is in CI.
- [ ] Commit `refactor: complete DINOv2 clean break`.

## Final Verification Matrix

```bash
pytest tests/test_dinov2_checkpoint.py -v
pytest tests/test_dinov2_preprocess.py tests/test_dinov2_encoder.py -v
pytest tests/test_dinov2_classifier.py tests/test_dinov2_margin_rejection.py -v
pytest tests/test_dinov2_runtime.py -v
pytest tests/test_dinov2_registry.py tests/test_dinov2_feedback.py tests/test_dinov2_state_isolation.py -v
pytest tests/test_dinov2_benchmark_parity.py -v
pytest -q
cd frontend && flutter test
```

Completion means Neri executes no DINOv3 runtime code and its DINOv2 deployment decisions match the frozen benchmark checkpoint.