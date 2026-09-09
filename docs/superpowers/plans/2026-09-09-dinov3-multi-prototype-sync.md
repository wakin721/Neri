# DINOv3 Multi-prototype Strict Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DINOv3 Linear Head inference with the supplied K=3 Multi-prototype format, add training-free fingerprint-scoped prototype overlays, and make `res/model/DINOv3` a validated strict mirror of `NeriCloud/Neri_Data/Model/DINOv3` whenever DINOv3 is installed or repaired.

**Architecture:** The base component is immutable mirrored state under `res/model/DINOv3`; `install.json` names and hashes the active Multi-prototype checkpoint and manifest. Runtime constructs an effective prototype bank from the base checkpoint plus the existing fingerprint-scoped SQLite registry, while Candidate/Provisional/Confirmed/Mature status determines whether local prototypes are excluded, assistive-only, or formal competitors. NeriCloud sync downloads the entire component tree to staging, validates it without rewriting mirrored files, then swaps it atomically with rollback.

**Tech Stack:** Python 3.12+/PyTorch/NumPy/FastAPI/SQLite, OpenList-compatible NeriCloud API, Flutter/Dart Material 3, pytest, flutter_test, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-09-dinov3-environment-installer-design.md`

## Global Constraints

- Base architecture is `dinov3_vitb16`, feature dimension 768, patch size 16.
- Encoder source commit is `6876159a11b4df116f30f667f8c9888617df0751`.
- Backbone filename is `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` with SHA-256 `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`.
- Base classifier filename is `multi_prototype.pt` with SHA-256 `4bb63f224a11e318c9a3586006146cad94a4df95f3dd5fe6157963aeafe0ab43`.
- Base classifier has 17 classes, 51 prototypes, `head_type=multi_prototype`, `selection_k=3`, threshold `0.31004515290260315`.
- Decision rule is squared Euclidean distance to nearest prototype after applying `feature_center`; rejection score is cosine similarity to that winning prototype.
- Linear Head logits must not participate in production DINOv3 inference.
- Overlay maximum is K=3; Candidate=not in bank, Provisional=assistive only, Confirmed/Mature=formal.
- Provisional requires >=4 human-confirmed independent events; Confirmed requires >=10 events and >=2 cameras; Mature requires >=20 events and >=3 cameras.
- Additional overlay clusters require >=4 events per cluster and pairwise centroid cosine <=0.98.
- Duplicate registration is blocked when at least 4 confirmed events exist and >=80% are formally accepted as the same existing species.
- `res/model/DINOv3` must contain only NeriCloud-mirrored base component files; local registry/overlay data stays outside this root.
- Install/repair is full mirror sync: cloud additions/modifications/deletions all appear locally; local extras disappear.
- Removing DINOv3 removes only the mirrored component and preserves PyTorch, YOLO dependencies, and registry overlays.
- A changed base checkpoint SHA/fingerprint starts a separate registry namespace; no silent migration.

---

### Task 1: Multi-prototype checkpoint schema and immutable fixture helpers

**Files:**
- Modify: `system/dinov3/checkpoint.py`
- Create: `tests/dinov3_multi_prototype_fixtures.py`
- Modify: `tests/test_dinov3_backend_integration.py`
- Create: `tests/test_dinov3_multi_prototype_checkpoint.py`

**Interfaces:**
- Produces `DinoV3Checkpoint.feature_center`, `.prototypes`, `.prototype_class_indices`, `.prototypes_per_class`, `.head_type`, `.selection_k`, `.decision`, `.rejection_score`.
- Removes production dependence on `.head_weight` / `.head_bias`.
- Keeps `load_checkpoint(path) -> DinoV3Checkpoint` and `validate_checkpoint(saved, *, path=None, fingerprint=None) -> DinoV3Checkpoint`.

- [ ] **Step 1: Add a synthetic Multi-prototype fixture builder**

```python
# tests/dinov3_multi_prototype_fixtures.py
from __future__ import annotations
import torch


def make_multi_prototype_payload(*, classes=("A", "B"), threshold=0.31):
    feature_dim = 768
    feature_center = torch.zeros(feature_dim, dtype=torch.float32)
    prototypes = []
    indices = []
    for class_index in range(len(classes)):
        for prototype_index in range(3):
            vector = torch.zeros(feature_dim, dtype=torch.float32)
            vector[class_index * 3 + prototype_index] = 1.0
            prototypes.append(vector)
            indices.append(class_index)
    return {
        "schema_version": 1,
        "backbone": "dinov3_vitb16",
        "feature_dim": feature_dim,
        "classes": list(classes),
        "feature_center": feature_center,
        "head_type": "multi_prototype",
        "selection_k": 3,
        "prototypes": torch.stack(prototypes),
        "prototype_class_indices": torch.tensor(indices, dtype=torch.int64),
        "prototypes_per_class": [3] * len(classes),
        "threshold": threshold,
        "decision": "squared_euclidean_to_nearest_prototype",
        "rejection_score": "cosine_similarity_to_winning_prototype",
        "encoder_weights": "dinov3-vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "encoder_sha256": "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c",
        "encoder_code_commit": "6876159a11b4df116f30f667f8c9888617df0751",
        "preprocessing": "letterbox224_imagenet",
        "event_aggregation": "mean_l2_normalized_crop_embeddings",
    }
```

- [ ] **Step 2: Write failing schema tests**

```python
def test_multi_prototype_checkpoint_does_not_require_linear_head():
    checkpoint = validate_checkpoint(make_multi_prototype_payload())
    assert checkpoint.head_type == "multi_prototype"
    assert tuple(checkpoint.prototypes.shape) == (6, 768)
    assert tuple(checkpoint.feature_center.shape) == (768,)
    assert checkpoint.selection_k == 3


def test_multi_prototype_checkpoint_rejects_invalid_class_indices():
    payload = make_multi_prototype_payload()
    payload["prototype_class_indices"][0] = 99
    with pytest.raises(CheckpointValidationError):
        validate_checkpoint(payload)


def test_multi_prototype_checkpoint_rejects_non_k3_contract():
    payload = make_multi_prototype_payload()
    payload["selection_k"] = 5
    with pytest.raises(CheckpointValidationError):
        validate_checkpoint(payload)
```

- [ ] **Step 3: Run RED checkpoint tests**

Run: `python -m pytest -q tests/test_dinov3_multi_prototype_checkpoint.py`
Expected: FAIL because the current loader requires `head_state.weight/bias` and one prototype per class.

- [ ] **Step 4: Replace the Linear Head schema with Multi-prototype validation**

Implement `DinoV3Checkpoint` with the new fields. Normalize tensor dtypes on CPU, validate `(M, 768)` prototypes, `(M,)` int64 class indices, `(768,)` feature center, class-index range, per-class counts, K=3, decision/rejection strings, encoder commit/SHA, and finite threshold. `_payload_fingerprint` must include feature center, prototypes, class indices, threshold, and encoder identity; `load_checkpoint()` continues to override it with the real file SHA-256.

- [ ] **Step 5: Run GREEN checkpoint and existing DINOv3 tests**

Run: `python -m pytest -q tests/test_dinov3_multi_prototype_checkpoint.py tests/test_dinov3_backend_integration.py tests/test_dinov3_lazy_import.py`
Expected: PASS after updating old test fixtures to the Multi-prototype builder.

- [ ] **Step 6: Commit**

Commit message: `feat: validate DINOv3 multi-prototype checkpoints`

### Task 2: Pure Multi-prototype classifier and effective-bank primitives

**Files:**
- Modify: `system/dinov3/classifier.py`
- Create: `tests/test_dinov3_multi_prototype_classifier.py`
- Modify: `tests/test_dinov3_image_processor.py`

**Interfaces:**
- Produces `PrototypeBank` and `PrototypeRecord` value objects in `classifier.py`.
- Produces `DinoV3Classifier.classify_features(features)` using centered Euclidean selection and winning-prototype cosine rejection.
- `DinoV3Prediction` gains `assistive_match`, `nearest_prototype_index`, `squared_distance`, and retains `source`, `registry_id`, `registration_status`.

- [ ] **Step 1: Write failing nearest-prototype tests**

```python
def test_classifier_uses_centered_squared_euclidean_not_linear_logits():
    payload = make_multi_prototype_payload(classes=("A", "B"), threshold=-1.0)
    payload["feature_center"][0] = 0.25
    checkpoint = validate_checkpoint(payload)
    classifier = DinoV3Classifier(checkpoint)
    feature = torch.zeros(768).numpy().astype("float32")
    feature[0] = 1.25
    feature /= np.linalg.norm(feature)
    prediction = classifier.classify_features(feature[None, :])[0]
    assert prediction.species == "A"
    assert prediction.nearest_prototype_index == 0
    assert prediction.squared_distance == pytest.approx(0.0, abs=1e-6)


def test_candidate_ranking_collapses_prototypes_by_species():
    checkpoint = validate_checkpoint(make_multi_prototype_payload(threshold=-1.0))
    prediction = DinoV3Classifier(checkpoint).classify_features(_feature_near_a())[0]
    assert [item["name"] for item in prediction.candidates] == ["A", "B"]
    assert len({item["name"] for item in prediction.candidates}) == 2
```

- [ ] **Step 2: Run RED classifier tests**

Run: `python -m pytest -q tests/test_dinov3_multi_prototype_classifier.py`
Expected: FAIL because the current classifier calculates Linear Head logits and single-prototype consistency.

- [ ] **Step 3: Implement centered Multi-prototype classification**

For each normalized input row `x`, compute `centered = x - checkpoint.feature_center.numpy()`, vectorized squared distances to every prototype, winning prototype index, mapped class index, and cosine score using normalized `centered` and normalized winning prototype only. Build top-3 species candidates by the minimum prototype distance per species. Do not compute or reference Linear Head state.

- [ ] **Step 4: Add effective-bank merge primitives**

```python
@dataclass(frozen=True)
class PrototypeRecord:
    species: str
    embedding: np.ndarray
    source: str
    registry_id: int | None = None
    registration_status: str | None = None


@dataclass(frozen=True)
class PrototypeBank:
    formal: tuple[PrototypeRecord, ...]
    provisional: tuple[PrototypeRecord, ...]
```

`DinoV3Classifier` constructs base formal records from checkpoint prototypes and asks an optional registry adapter for overlay records. Formal records compete first; Provisional records are evaluated only if no formal result reaches threshold and return `accepted=False, assistive_match=True`.

- [ ] **Step 5: Run GREEN classifier/image processor tests**

Run: `python -m pytest -q tests/test_dinov3_multi_prototype_classifier.py tests/test_dinov3_image_processor.py`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: classify DINOv3 with multi-prototype banks`

### Task 3: K=3 fingerprint-scoped overlay lifecycle and duplicate guard

**Files:**
- Modify: `system/dinov3/registry.py`
- Modify: `system/dinov3/simple_shot.py`
- Modify: `system/dinov3/api.py`
- Modify: `tests/test_dinov3_registry.py`
- Create: `tests/test_dinov3_overlay_bank.py`

**Interfaces:**
- Produces `SpeciesRegistry.prototype_bank(feature_center, *, threshold, formal_matcher) -> PrototypeBank`.
- Produces deterministic `deterministic_k_means(embeddings, *, max_k, min_cluster_size=4, max_centroid_cosine=0.98)` operating in centered feature space.
- Registry state transitions become Candidate <4, Provisional >=4, Confirmed >=10 + >=2 cameras, Mature >=20 + >=3 cameras.
- Produces duplicate-registration failure before promotion when >=80% of >=4 confirmed events resolve to the same formal species.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_four_events_promote_to_provisional_but_not_formal(registry):
    _record_events(registry, count=4, cameras=1)
    entry = registry.get(1)
    assert entry.status == "provisional"
    bank = registry.prototype_bank(np.zeros(768, dtype=np.float32), threshold=0.31)
    assert len(bank.formal) == 0
    assert len(bank.provisional) == 1


def test_confirmed_requires_ten_events_and_two_cameras(registry):
    _record_events(registry, count=10, cameras=1)
    assert registry.get(1).status == "provisional"
    _record_one_event_from_new_camera(registry)
    assert registry.get(1).status == "confirmed"
```

- [ ] **Step 2: Write failing K=3 and duplicate tests**

```python
def test_mature_overlay_never_exceeds_three_supported_clusters():
    prototypes = deterministic_k_means(_three_supported_clusters(), max_k=3)
    assert prototypes.shape == (3, 768)


def test_single_anomaly_cannot_create_extra_cluster():
    prototypes = deterministic_k_means(_two_clusters_plus_one_outlier(), max_k=3)
    assert prototypes.shape[0] == 2


def test_registration_blocked_when_eighty_percent_match_same_formal_species(registry):
    _record_events(registry, count=5, cameras=2)
    matcher = lambda event: "Existing" if event.index < 4 else None
    with pytest.raises(RegistrationConditionError, match="already matches Existing"):
        registry.register(1, formal_matcher=matcher)
```

- [ ] **Step 3: Run RED overlay tests**

Run: `python -m pytest -q tests/test_dinov3_registry.py tests/test_dinov3_overlay_bank.py`
Expected: FAIL because current registry starts prototypes only at four events, promotes Provisional only through explicit register, supports at most deterministic two-means, and has no formal-bank duplicate guard.

- [ ] **Step 4: Implement deterministic 1..K cluster selection**

Use deterministic farthest-first initialization, assignment by squared Euclidean distance, arithmetic centroid updates, and fixed iteration count. Evaluate k from `min(max_k, floor(n/4))` down to 1; accept the largest k where every cluster has >=4 events and every centroid pair has cosine <=0.98. Store centered arithmetic centroids for overlay classification; preserve raw normalized event embeddings for unknown clustering/join behavior.

- [ ] **Step 5: Implement lifecycle and effective bank**

`_refresh()` derives status from event count/camera count; `prototype_bank()` transforms events with `event - feature_center` and exposes Provisional separately from Confirmed/Mature. Candidate entries are excluded. Preserve identity and display-name behavior.

- [ ] **Step 6: Implement duplicate guard and API error propagation**

Before Candidate->Provisional registration, require at least 4 confirmed events. Evaluate each against the formal matcher; if one species owns >=80% of all evaluated events, raise `RegistrationConditionError` with that species name. Keep the existing FastAPI 4xx error response convention.

- [ ] **Step 7: Run GREEN registry/API tests**

Run: `python -m pytest -q tests/test_dinov3_registry.py tests/test_dinov3_overlay_bank.py tests/test_dinov3_api.py`
Expected: PASS.

- [ ] **Step 8: Commit**

Commit message: `feat: add K3 DINOv3 prototype overlays`

### Task 4: Strict NeriCloud mirror component and local integrity contract

**Files:**
- Modify: `system/dinov3/component.py`
- Modify: `tests/test_dinov3_component.py`
- Modify: `tests/test_dinov3_model_catalog.py`

**Interfaces:**
- `DINO_CLASSIFIER_FILENAME = "multi_prototype.pt"` and expected production SHA is `4bb63f...ab43`.
- `install_dinov3_component(...)` downloads the full remote root via `DinoV3CloudClient.download_tree("", stage)` and never rewrites mirrored files.
- Produces `load_component_manifest(paths) -> dict[str, Any]`, `validate_component_tree(paths) -> ComponentValidationResult`.
- `dinov3_component_status()` adds classifier filename/fingerprint/head type/K.

- [ ] **Step 1: Replace partial-download tests with strict-mirror RED tests**

```python
def test_install_mirrors_complete_cloud_tree_and_removes_local_extras(tmp_path):
    target = tmp_path / "model" / "DINOv3"
    target.mkdir(parents=True)
    (target / "local-extra.txt").write_text("must disappear")
    cloud = FakeCloudTree(valid_multi_prototype_component())
    install_dinov3_component(root=target, cloud_client=cloud)
    assert not (target / "local-extra.txt").exists()
    assert tree_signature(target) == tree_signature(cloud.fixture_root)


def test_install_does_not_rewrite_cloud_manifest(tmp_path):
    cloud = FakeCloudTree(valid_multi_prototype_component())
    original = cloud.read_bytes("classifier.neri.json")
    install_dinov3_component(root=tmp_path / "DINOv3", cloud_client=cloud)
    assert (tmp_path / "DINOv3" / "classifier.neri.json").read_bytes() == original
```

- [ ] **Step 2: Add `install.json files[]` integrity RED tests**

```python
def test_health_rejects_declared_file_hash_mismatch(component_root):
    make_valid_component(component_root)
    (component_root / "source" / "dinov3" / "__init__.py").write_text("tampered")
    status = dinov3_component_status(root=component_root)
    assert status["healthy"] is False
    assert "SHA-256" in status["message"]
```

- [ ] **Step 3: Run RED component tests**

Run: `python -m pytest -q tests/test_dinov3_component.py tests/test_dinov3_model_catalog.py`
Expected: FAIL because current installer cherry-picks files, copies a local Linear Head seed, and rewrites the model manifest.

- [ ] **Step 4: Implement strict mirror download**

Remove `_default_seed_dir`, `_copy_seed_asset`, `_normalize_model_manifest`, old Linear Head filename/hash constants, and per-file hand-coded install download sequence. Recursively download the authoritative DINOv3 root into staging. Read `install.json` only after staging is complete. Reject unsafe relative paths, duplicate `files[]` entries, missing files, paths outside staging, invalid SHA-256, optional size mismatch, wrong classifier filename/hash/head type/K, wrong backbone identity, wrong source commit, or missing license.

- [ ] **Step 5: Validate Multi-prototype checkpoint and source before activation**

Call `load_checkpoint()` on the staged classifier and verify its file fingerprint equals `install.json.classifier.sha256`; verify source files exist and source commit metadata matches; verify the backbone hash. Import the staged source and instantiate `dinov3_vitb16(pretrained=False)` without downloading pretrained weights. Unit tests mock the large model/state-dict path where necessary.

- [ ] **Step 6: Make activation rollback-safe including post-activation validation**

Keep staging on the same filesystem, rename old target to backup, rename stage to target, rerun local tree validation against the active target, then delete backup. On failure after the swap, remove the failed target and restore backup.

- [ ] **Step 7: Run GREEN component/catalog tests**

Run: `python -m pytest -q tests/test_dinov3_component.py tests/test_dinov3_model_catalog.py tests/test_dinov3_encoder_component.py`
Expected: PASS.

- [ ] **Step 8: Commit**

Commit message: `feat: mirror DINOv3 component from NeriCloud`

### Task 5: Runtime discovery, maintenance API, and status migration

**Files:**
- Modify: `system/model_sync/catalog.py`
- Modify: `system/dinov3/encoder.py`
- Modify: `system/backend/dinov3_environment.py`
- Modify: `system/backend/dinov3_maintenance.py`
- Modify: `tests/test_dinov3_environment_api.py`
- Modify: `tests/test_dinov3_encoder_component.py`

**Interfaces:**
- Discovery reads active manifest filename from `install.json.classifier.manifest`, not a hardcoded reviewed Linear Head manifest.
- Environment status exposes `classifier_filename`, `classifier_fingerprint`, `classifier_head_type`, `selection_k`.
- Install operation means strict sync; removal retains overlay registry.

- [ ] **Step 1: Write RED dynamic-discovery/status tests**

```python
def test_catalog_uses_manifest_declared_by_install_json(layout, monkeypatch):
    make_healthy_component(layout.dinov3_root, classifier_manifest="multi_prototype.neri.json")
    models = discover_models(layout, "cls")
    assert [item.backend for item in models].count("dinov3") == 1
    assert models[-1].path.name == "multi_prototype.neri.json"


def test_environment_status_reports_multi_prototype_identity(client, monkeypatch):
    response = client.get("/api/environment/dinov3-status")
    assert response.json()["classifier_head_type"] == "multi_prototype"
    assert response.json()["selection_k"] == 3
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest -q tests/test_dinov3_environment_api.py tests/test_dinov3_model_catalog.py tests/test_dinov3_encoder_component.py`

- [ ] **Step 3: Implement dynamic discovery and enriched status**

Keep installed source/backbone precedence already added on the feature branch. Replace hardcoded classifier manifest assumptions with validated `install.json` metadata and map component validation data into the API response.

- [ ] **Step 4: Verify maintenance semantics**

`start_dinov3_install()` still reuses current PyTorch choice/source if torch is unavailable, then calls strict mirror install. `start_dinov3_remove()` removes only component root and schedules backend restart; it does not delete registry directories.

- [ ] **Step 5: Run GREEN backend tests**

Run: `python -m pytest -q tests/test_dinov3_environment_api.py tests/test_dinov3_model_catalog.py tests/test_dinov3_encoder_component.py`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: expose multi-prototype DINOv3 component status`

### Task 6: Flutter DINOv3 environment maintenance panel

**Files:**
- Modify: `frontend/lib/src/api_client.dart`
- Modify: `frontend/lib/src/screens/settings_screen.dart`
- Modify/Create: `frontend/test/dinov3_environment_api_test.dart`
- Create: `frontend/test/dinov3_environment_maintenance_test.dart`

**Interfaces:**
- `DinoV3ComponentStatus` parses installed/healthy/classifier/head/K/source fields.
- `NeriApiClient.fetchDinoV3ComponentStatus()`, `.installDinoV3(envChoice, {packageSource})`, `.removeDinoV3()`.
- Settings panel is directly below PyTorch and shares maintenance polling/busy state.

- [ ] **Step 1: Finish/repair API client tests and verify RED where methods/status fields are incomplete**

```dart
test('parses multi-prototype DINOv3 status', () async {
  final status = await client.fetchDinoV3ComponentStatus();
  expect(status.classifierHeadType, 'multi_prototype');
  expect(status.selectionK, 3);
  expect(status.healthy, isTrue);
});
```

- [ ] **Step 2: Implement API client model/methods**

Use the wrapper `frontend/lib/src/api_client.dart` and its injected `_modelSyncHttpClient`; do not enlarge `api_client_core.dart` merely for these endpoints. POST install JSON with `env_choice` and `package_source`; POST remove with no body.

- [ ] **Step 3: Write widget RED tests**

Test the visible labels and behavior:

```dart
expect(find.text('DINOv3 ViT-B/16'), findsOneWidget);
expect(find.text('安装'), findsOneWidget); // absent
expect(find.text('安装/修复'), findsOneWidget); // unhealthy
expect(find.text('删除'), findsOneWidget); // healthy
```

Removal confirmation must state that PyTorch and locally learned species records are retained.

- [ ] **Step 4: Implement panel and maintenance state integration**

Add `_dinoV3Status`, `_installingDinoV3`, `_removingDinoV3`; include `install_dinov3` and `remove_dinov3` in resume/polling mapping and in the global environment-maintenance busy predicate. After task completion, refresh DINOv3 status and classification-model availability. Subtitle: `安装或同步 DINOv3 Multi-prototype 模型、官方 source 与 ViT-B/16 backbone`.

- [ ] **Step 5: Run Flutter GREEN tests**

Run: `cd frontend && flutter test test/dinov3_environment_api_test.dart test/dinov3_environment_maintenance_test.dart test/model_selection_test.dart test/dinov3_registry_test.dart`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add DINOv3 multi-prototype maintenance UI`

### Task 7: Remove obsolete Linear Head assumptions and run full verification

**Files:**
- Modify as required by test failures: DINOv3 tests/manifests/docs only
- Do not modify unrelated YOLO or model-sync behavior.

**Interfaces:**
- No production DINOv3 path depends on `head_state.weight`, `head_state.bias`, `head_species`, or head/prototype consistency.
- No production install path copies/normalizes a local DINOv3 classifier seed into the mirrored root.

- [ ] **Step 1: Search the implementation branch for obsolete Linear Head production assumptions**

Run:

```bash
grep -R "head_state\|head_weight\|head_bias\|head_prototype_consistent\|dinov3_classifier_merged_reviewed_20260908" system frontend/lib tests res/install -n
```

Expected: any remaining matches are explicitly compatibility-test/history fixtures only; remove production matches.

- [ ] **Step 2: Run complete Python suite**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 3: Run complete Flutter suite and static analysis**

Run: `cd frontend && flutter analyze && flutter test`
Expected: PASS.

- [ ] **Step 4: Run Windows package workflow on the implementation branch**

Expected required steps: Python integration tests PASS, Flutter analysis/tests PASS, Windows release build PASS, portable package assembly PASS. Fresh package must not require `res/model/DINOv3` to exist before optional installation.

- [ ] **Step 5: Compare feature branch against `nightly`**

Confirm only intended DINOv3/environment/docs/tests/workflow changes remain; specifically verify no accidental deletions from `main_core.py` or unrelated model roots.

- [ ] **Step 6: Invoke verification-before-completion and re-run the final evidence commands it requires**

Do not claim completion before fresh verification output is available.

- [ ] **Step 7: Integrate the verified implementation into `nightly`**

Use a fast-forward/merge only after all required checks pass; then verify the resulting `nightly` commit's workflow status.
