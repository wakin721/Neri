# DINOv3 Environment Installer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an install/remove workflow under Environment Maintenance that provisions the reviewed DINOv3 ViT-B/16 classifier, official source, license, and ViT-B/16 backbone under `res/model/DINOv3`, using NeriCloud for component assets.

**Architecture:** Add a focused `system/dinov3/component.py` module that owns component layout, health checks, NeriCloud synchronization, integrity validation, and removal. Reuse the existing maintenance worker for long-running install/remove operations, expose three environment APIs, extend classification discovery/runtime path resolution, and add a Flutter panel that reflects component health and maintenance progress.

**Tech Stack:** Python 3.13, FastAPI/Pydantic, PyTorch, urllib/OpenList-compatible NeriCloud API, Flutter/Dart Material 3, pytest, flutter_test.

**Spec:** `docs/superpowers/specs/2026-09-09-dinov3-environment-installer-design.md`

## Global Constraints

- Runtime assets live under `res/model/DINOv3`.
- Architecture remains `dinov3_vitb16`, feature dimension 768, patch size 16.
- Backbone filename is `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
- Backbone SHA-256 is `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`.
- Meta source identity remains `facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751`.
- NeriCloud component root is `Neri_Data/Model/DINOv3`; default OpenList origin is `https://cloud.myneri.top`, overridable with `NERI_DINOV3_CLOUD_ORIGIN`.
- `source/`, `LICENSE_DINOv3.md`, `install.json`, and `dinov3-vitb16/...pth` are downloaded from NeriCloud.
- Reviewed classifier seed is shipped by Neri outside `res/model/DINOv3` and copied into the installed component.
- Removing DINOv3 must never uninstall PyTorch/torchvision or shared YOLO dependencies.
- Failed install must preserve a previous healthy install.

---

### Task 1: Component layout, status, NeriCloud client, and atomic installer

**Files:**
- Create: `system/dinov3/component.py`
- Create: `tests/test_dinov3_component.py`
- Modify: `system/model_sync/layout.py`

**Interfaces:**
- Produces: `DinoV3ComponentPaths`, `dinov3_component_paths()`, `dinov3_component_status() -> dict[str, object]`, `install_dinov3_component(...) -> Path`, `remove_dinov3_component() -> None`.
- Produces: `ModelLayout.dinov3_root: Path`.

- [ ] **Step 1: Write failing layout/status tests**

```python
def test_model_layout_exposes_dinov3_root(tmp_path):
    layout = get_model_layout(tmp_path)
    assert layout.dinov3_root == tmp_path / "model" / "DINOv3"


def test_component_status_is_not_installed_when_root_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(component, "resource_root", lambda: tmp_path)
    status = component.dinov3_component_status()
    assert status["installed"] is False
    assert status["healthy"] is False
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/test_dinov3_component.py tests/test_model_sync_layout.py -q`
Expected: FAIL because the component module/layout field do not exist.

- [ ] **Step 3: Implement component constants, paths, status validation, OpenList-compatible NeriCloud downloads, recursive `source/` sync, SHA-256 validation, staging, atomic replace, and idempotent removal**

Key constants/signatures:

```python
DINO_COMPONENT_ROOT = "Neri_Data/Model/DINOv3"
DINO_CLOUD_ORIGIN_ENV = "NERI_DINOV3_CLOUD_ORIGIN"
DINO_DEFAULT_CLOUD_ORIGIN = "https://cloud.myneri.top"
DINO_BACKBONE_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
DINO_SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"


def dinov3_component_status(*, root: Path | None = None) -> dict[str, object]: ...
def install_dinov3_component(*, on_progress=None, opener=None, root: Path | None = None) -> Path: ...
def remove_dinov3_component(*, root: Path | None = None) -> None: ...
```

Download into `temp/dinov3-install-*`, copy classifier/manifest seed from `res/install/dinov3`, validate `install.json`, validate source package/license/backbone/classifier, then replace `res/model/DINOv3` only after validation passes.

- [ ] **Step 4: Add corruption/rollback/removal tests and run GREEN**

Run: `pytest tests/test_dinov3_component.py tests/test_model_sync_layout.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: add DINOv3 component installer core`

### Task 2: Maintenance operations and FastAPI surface

**Files:**
- Modify: `system/backend/maintenance.py`
- Modify: `system/backend/models.py`
- Modify: `system/backend/main_core.py`
- Create: `tests/test_dinov3_environment_api.py`

**Interfaces:**
- Produces: `start_dinov3_install()`, `start_dinov3_remove()`.
- Produces API response schema `DinoV3ComponentStatusResponse`.
- Produces routes `GET /api/environment/dinov3-status`, `POST /api/environment/install-dinov3`, `POST /api/environment/remove-dinov3`.

- [ ] **Step 1: Write failing API tests**

```python
def test_dinov3_status_endpoint(client, monkeypatch):
    monkeypatch.setattr(main_core, "dinov3_component_status", lambda: {"installed": False, "healthy": False, "architecture": "DINOv3 ViT-B/16", "component_version": 1, "source_commit": DINO_SOURCE_COMMIT, "message": ""})
    response = client.get("/api/environment/dinov3-status")
    assert response.status_code == 200
    assert response.json()["healthy"] is False


def test_install_endpoint_returns_202(client, monkeypatch):
    monkeypatch.setattr(main_core, "start_dinov3_install", lambda: {"operation": "install_dinov3", "message": "started", "progress": 0})
    response = client.post("/api/environment/install-dinov3")
    assert response.status_code == 202
```

- [ ] **Step 2: Run focused API tests and verify RED**

Run: `pytest tests/test_dinov3_environment_api.py -q`
Expected: FAIL because routes/functions are missing.

- [ ] **Step 3: Extend maintenance worker**

Add operations `install_dinov3` and `remove_dinov3` to CLI choices and dispatch. Map component progress callbacks into `_write_status`. Keep the existing single-operation lock and backend restart flow. Do not call `_run_pytorch_install` unless importing `torch` fails; if torch is absent, reuse the existing resolved PyTorch install flow before component installation.

- [ ] **Step 4: Add API schemas/routes and run GREEN**

Run: `pytest tests/test_dinov3_environment_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: expose DINOv3 environment maintenance API`

### Task 3: Runtime source/backbone resolution and classification discovery

**Files:**
- Modify: `system/dinov3/encoder.py`
- Modify: `system/model_sync/catalog.py`
- Modify: `tests/test_dinov3_model_catalog.py`
- Modify: `tests/test_dinov3_lazy_import.py`

**Interfaces:**
- Encoder source precedence: explicit `source_root` -> installed `res/model/DINOv3/source` -> `NERI_DINOV3_SOURCE`.
- Encoder weight precedence includes installed `res/model/DINOv3/dinov3-vitb16/...pth`.
- `discover_models(layout, "cls")` includes the installed DINOv3 manifest only when component assets are healthy enough to run.

- [ ] **Step 1: Add failing resolution/discovery tests**
- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_dinov3_model_catalog.py tests/test_dinov3_lazy_import.py -q`

- [ ] **Step 3: Implement installed-component resolution and discovery**
- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest tests/test_dinov3_model_catalog.py tests/test_dinov3_lazy_import.py tests/test_model_catalog.py -q`

- [ ] **Step 5: Commit**

Commit message: `feat: discover installed DINOv3 component`

### Task 4: Flutter API client and Environment Maintenance panel

**Files:**
- Modify: `frontend/lib/src/api_client.dart`
- Modify: `frontend/lib/src/screens/settings_screen.dart`
- Create: `frontend/test/dinov3_environment_maintenance_test.dart`

**Interfaces:**
- Produces Dart model `DinoV3ComponentStatus`.
- Produces client methods `fetchDinoV3ComponentStatus()`, `installDinoV3()`, `removeDinoV3()`.
- Adds DINOv3 panel immediately after the PyTorch panel.

- [ ] **Step 1: Write failing widget/API parsing tests**

Assert button labels `安装`, `安装/修复`, `删除`, and removal confirmation text that PyTorch is retained.

- [ ] **Step 2: Run Flutter test and verify RED**

Run: `cd frontend && flutter test test/dinov3_environment_maintenance_test.dart`

- [ ] **Step 3: Implement API model/methods and panel state machine**

Add `_installingDinoV3`, `_removingDinoV3`, `_dinoV3Status`; include both operations in `_maintenanceInProgress`, resume/watch mapping, and status refresh after maintenance completion.

- [ ] **Step 4: Run Flutter tests and verify GREEN**

Run: `cd frontend && flutter test test/dinov3_environment_maintenance_test.dart test/model_selection_test.dart`

- [ ] **Step 5: Commit**

Commit message: `feat: add DINOv3 environment maintenance UI`

### Task 5: Reviewed classifier seed and packaging contract

**Files:**
- Create binary: `res/install/dinov3/dinov3_classifier_merged_reviewed_20260908.pt`
- Create: `res/install/dinov3/dinov3_classifier_merged_reviewed_20260908.neri.json`
- Modify: `scripts/package_windows.ps1` only if current `res` copy excludes the seed path.
- Create/modify: `tests/test_dinov3_component.py`

**Interfaces:**
- Seed checkpoint SHA-256 must be `b2f334da61c9feee51cff51bcded0af16878dcd5d72ea4f2e3e47b7f2adab76a`.
- Manifest points to installed checkpoint filename and `res/model/DINOv3/dinov3-vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` contract.

- [ ] **Step 1: Add seed-integrity failing test**
- [ ] **Step 2: Extract `classifier.pt` from the supplied 7z, verify SHA-256, and add it as the canonical seed asset**
- [ ] **Step 3: Add/generate the canonical manifest seed**
- [ ] **Step 4: Run seed/component tests**

Run: `pytest tests/test_dinov3_component.py tests/test_dinov3_backend_integration.py -q`

- [ ] **Step 5: Commit**

Commit message: `feat: ship reviewed DINOv3 classifier seed`

### Task 6: Full verification and integration into nightly

**Files:**
- No feature files unless verification exposes a defect.

**Interfaces:**
- Uses the existing Windows workflow to run backend tests, Flutter tests, build, package, and artifact upload.

- [ ] **Step 1: Run complete Python suite in CI**

Expected workflow step: `Run Python backend tests` PASS.

- [ ] **Step 2: Run complete Flutter suite in CI**

Expected workflow step: `Run Flutter tests` PASS.

- [ ] **Step 3: Verify Windows release/package assembly**

Expected workflow steps: `Build Flutter Windows release`, `Assemble portable application`, and artifact upload PASS.

- [ ] **Step 4: Compare implementation branch against `nightly` and confirm only intended files changed**

- [ ] **Step 5: Fast-forward `nightly` to the verified implementation commit and verify the resulting nightly workflow**
