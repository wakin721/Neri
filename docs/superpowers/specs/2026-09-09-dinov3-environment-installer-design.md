# DINOv3 environment installer design

Date: 2026-09-09
Branch: `nightly`
Status: approved in chat; implementation pending user review of this written spec

## Goal

Make DINOv3 ViT-B/16 an optional environment component that users can install and remove from **Advanced Settings -> Environment Maintenance**, directly below the existing PyTorch maintenance panel.

The installer must provision everything required for the current Neri DINOv3 classifier to run on a clean Neri installation without requiring the user to clone Meta's repository, set `NERI_DINOV3_SOURCE`, or manually place the ViT-B/16 backbone.

The runtime model assets must live under `res/model/DINOv3`.

## Non-goals

- Do not bundle the ViT-B/16 backbone in every Windows artifact.
- Do not switch the classifier to ViT-7B/16, ViT-L/16, or another feature space.
- Do not uninstall PyTorch when DINOv3 is removed, because PyTorch is shared with YOLO.
- Do not reuse the normal `detect/` and `cls/` model-sync logical path contract for DINOv3 environment components.
- Do not download an unpinned `main` version of Meta's DINOv3 source.

## Fixed model contract

Neri's reviewed DINOv3 classifier remains tied to:

- architecture: `dinov3_vitb16`
- patch size: 16
- feature dimension: 768
- source commit: `facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751`
- backbone filename: `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
- backbone SHA-256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`
- classifier filename: `dinov3_classifier_merged_reviewed_20260908.pt`
- classifier source: the reviewed `classifier.pt` supplied in `dinov3_classifier_merged_reviewed_20260908.7z`, renamed to the canonical filename above
- manifest filename: `dinov3_classifier_merged_reviewed_20260908.neri.json`

The existing 17-species classifier, prototypes, threshold, and event aggregation remain unchanged.

## Runtime layout

After a successful install, the component root is:

```text
res/model/DINOv3/
├── dinov3_classifier_merged_reviewed_20260908.pt
├── dinov3_classifier_merged_reviewed_20260908.neri.json
├── dinov3-vitb16/
│   └── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── source/
│   └── dinov3/
│       ├── __init__.py
│       ├── hub/
│       ├── layers/
│       ├── models/
│       └── ...
├── LICENSE_DINOv3.md
└── install.json
```

`ModelLayout` gains a canonical `dinov3_root` pointing to `res/model/DINOv3`. The existing lowercase `res/model` canonical root remains unchanged; only the child directory is named `DINOv3` as requested.

`install.json` records at least:

- schema version
- installed architecture
- source repository and pinned commit
- classifier filename and SHA-256
- backbone filename and SHA-256
- install timestamp
- installer/component version

`install.json` is evidence, not the sole source of truth. Runtime status is derived by validating the real files.

## Asset sources

### Backbone

The canonical NeriCloud logical path is:

```text
NeriCloud/Neri_Data/Model/DINO/dinov3-vitb16/
  dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

The installer uses a dedicated NeriCloud download adapter to resolve that logical path to a concrete download URL. URL construction is centralized in one backend helper; the UI never contains the NeriCloud URL.

The concrete NeriCloud origin remains configurable so mirrors can be changed without touching the DINOv3 runtime. If the configured origin cannot resolve the canonical logical path, installation fails closed with an actionable error. It must not silently fall back to a Meta signed URL.

Every downloaded backbone is written to a temporary file first, SHA-256 verified, then atomically moved into `res/model/DINOv3/dinov3-vitb16/`.

### Official DINOv3 source

Download a GitHub archive for exactly:

`facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751`

The user does not need Git installed. Extract only the runtime source needed to provide the `dinov3` package plus the DINOv3 license file. Do not track upstream `main` at install time.

### Reviewed classifier head

The reviewed classifier is small enough to ship as an installer seed asset with Neri. During DINOv3 installation it is copied into `res/model/DINOv3/dinov3_classifier_merged_reviewed_20260908.pt` and validated with the existing DINOv3 checkpoint validator.

This seed copy must remain outside `res/model/DINOv3` so that deleting the installed component does not destroy the only local reinstall source. The seed is an installer resource, not a discoverable model asset.

The manifest is likewise supplied from a trusted Neri installer resource or generated from the fixed model contract, then placed under `res/model/DINOv3` during installation.

## Dependency policy

DINOv3 installation reuses the existing PyTorch environment selected by Neri's PyTorch maintenance flow.

If `torch` is missing or unusable, DINOv3 installation invokes the same resolved PyTorch target and package-source policy used by the existing PyTorch installer before continuing.

Do not install Meta's complete training/evaluation `requirements.txt`. Install only dependencies proven necessary by Neri's inference import path. The implementation must test the minimal dependency set against:

```python
from dinov3.hub.backbones import dinov3_vitb16
model = dinov3_vitb16(pretrained=False)
```

Training/evaluation-only packages such as `submitit`, `scikit-learn`, and `torchmetrics` are not installed unless a runtime import test demonstrates they are required by the actual Neri inference path.

## Backend maintenance operations

Extend the existing environment-maintenance worker instead of creating a second task framework.

New operations:

- `install_dinov3`
- `remove_dinov3`

Both operations participate in the existing single-maintenance-operation lock, status file, progress reporting, log file, and backend restart behavior.

### Install phases

Recommended progress stages:

1. validate maintenance preconditions
2. validate/install shared PyTorch runtime if required
3. create a staging directory under `temp`
4. obtain and validate the reviewed classifier seed
5. download pinned official DINOv3 source archive
6. extract the `dinov3` runtime source and license
7. download the ViT-B/16 backbone from NeriCloud
8. SHA-256 validate the backbone
9. construct and validate the final manifest
10. import the pinned `dinov3` package from staging
11. instantiate `dinov3_vitb16(pretrained=False)`
12. load the backbone state dict strictly
13. load and validate the reviewed Neri classifier checkpoint
14. atomically replace/install `res/model/DINOv3`
15. write `install.json`
16. restart the Python backend

A failed install must leave the previous valid installation intact. New assets are assembled in staging and only replace the installed root after all validation succeeds.

### Remove phases

Removal deletes the DINOv3 component root after closing any active DINOv3 runtime/registry handles:

```text
res/model/DINOv3/
```

Removal must not uninstall:

- `torch`
- `torchvision`
- any shared YOLO dependency

If DINOv3 is currently selected as the classification model, the backend must clear or safely invalidate that selection before/while removing the component so the next settings refresh cannot point at a deleted manifest.

After successful removal the backend restarts and the UI returns to the install state.

## Installation status

Add a backend status function that returns at minimum:

```json
{
  "installed": true,
  "healthy": true,
  "architecture": "DINOv3 ViT-B/16",
  "component_version": 1,
  "source_commit": "6876159a11b4df116f30f667f8c9888617df0751",
  "message": ""
}
```

`installed=true` and `healthy=true` require all of the following:

- component root exists
- classifier checkpoint exists and validates
- manifest exists and matches the fixed architecture and classifier
- pinned official source exists
- license file exists
- backbone exists
- backbone SHA-256 matches exactly
- source package can be imported

A missing or corrupted asset returns `healthy=false`. The UI presents **Install/Repair** rather than **Delete** when the installation is incomplete or corrupt.

## FastAPI surface

Add:

- `GET /api/environment/dinov3-status`
- `POST /api/environment/install-dinov3`
- `POST /api/environment/remove-dinov3`

Install/remove endpoints return the same maintenance-start response shape used by current environment operations and HTTP 202 when a task is accepted.

Conflicting maintenance operations are rejected by the existing maintenance lock.

## Flutter UI

Under **Environment Maintenance**, immediately below **Install PyTorch**, add a panel:

Title:

`DINOv3 ViT-B/16`

Subtitle:

`安装 DINOv3 推理依赖、官方 source 与 ViT-B/16 backbone`

Button state:

| Backend state | Button |
| --- | --- |
| not installed | `安装` |
| installed but unhealthy | `安装/修复` |
| install running | progress indicator / disabled |
| healthy | `删除` |
| remove running | progress indicator / disabled |

The panel shares the existing environment-maintenance busy state, progress display, polling, and error presentation.

The Environment Maintenance section subtitle is updated to include DINOv3.

Before removal, show a confirmation dialog that states that DINOv3 model assets/source will be deleted while the shared PyTorch environment is retained.

## Runtime source resolution

Change the DINOv3 encoder source lookup order to:

1. explicit `source_root` passed by caller
2. installed `res/model/DINOv3/source`
3. legacy `NERI_DINOV3_SOURCE` environment variable for developer compatibility

A production user should not need `NERI_DINOV3_SOURCE` after installing the component.

Backbone resolution must treat the installed canonical path under `res/model/DINOv3/dinov3-vitb16` as the primary path. Legacy manifest-relative lookup may remain as a compatibility fallback.

## Classification-model discovery

Extend classification discovery so a healthy DINOv3 install contributes the canonical manifest under `res/model/DINOv3` as a classification model.

The DINOv3 model must not appear merely because a manifest file exists. Discovery requires a healthy component or, at minimum, all runtime assets required by the manifest.

After installation the model list contains:

`DINOv3 ViT-B/16 · 17 species · reviewed 2026-09-08`

After removal it disappears without affecting YOLO classification models under `res/model/cls/user` and `res/model/cls/sync`.

## Error handling and recovery

- Downloads use temporary `.download`/staging files.
- SHA mismatch deletes the temporary backbone and reports a verification failure.
- Interrupted installation leaves the prior valid installation untouched.
- A stale staging directory is safe to delete on the next install attempt.
- Source archive extraction rejects path traversal entries.
- HTTP redirects must remain HTTPS and pass a host policy appropriate to the configured NeriCloud/GitHub source.
- The installer logs the logical source and final hashes, but does not log credentials, signed query parameters, or secret headers.
- Removal is idempotent: removing an already absent component returns a clean not-installed result.

## License handling

The installed component includes Meta's `LICENSE.md` renamed to `LICENSE_DINOv3.md` without modifying its license text. Neri documentation must identify DINOv3 as third-party Meta software/model material and preserve the redistribution terms required by the DINOv3 License.

## Tests

### Python unit tests

Add tests for:

- canonical `dinov3_root` layout
- DINOv3 status: absent, complete, corrupted backbone, missing source, invalid classifier
- NeriCloud backbone download to staging
- SHA-256 rejection
- safe source archive extraction
- pinned source commit metadata
- install atomicity and rollback
- remove idempotency
- remove does not remove PyTorch/shared dependencies
- selected DINOv3 model is invalidated on removal
- installed-source path resolution precedence
- classification discovery before install / after install / after remove
- API 202 and maintenance conflict behavior

Network operations are mocked in unit tests.

### Integration verification

A controlled integration test using local fixture assets must verify:

1. import installed `dinov3`
2. construct `dinov3_vitb16(pretrained=False)`
3. strictly load a compatible state dict fixture
4. load/validate the reviewed classifier checkpoint contract
5. discover the DINOv3 manifest as a classification model

The real 300+ MB backbone is not required for normal CI unit tests. A separate/manual verification path may validate the production NeriCloud backbone hash.

### Flutter tests

Add widget tests for:

- panel appears directly below PyTorch panel
- absent -> Install
- unhealthy -> Install/Repair
- healthy -> Delete
- running -> disabled/progress state
- removal confirmation text mentions PyTorch is retained
- maintenance busy state blocks conflicting environment actions

## Packaging changes

The base Windows artifact continues to omit the large DINOv3 backbone and official source.

It must include only the small trusted installer seed assets required for reinstalling the reviewed classifier/manifest and the backend/UI installer implementation.

`package_windows.ps1` must not require `res/model/DINOv3` to exist in a fresh package, because DINOv3 is optional.

## Acceptance criteria

The feature is complete when all of the following are true on a clean Windows Neri package:

1. Environment Maintenance shows a DINOv3 ViT-B/16 install panel directly below PyTorch.
2. Clicking Install can provision required shared Python runtime pieces, pinned official DINOv3 source, the reviewed classifier, license, and the NeriCloud ViT-B/16 backbone without Git or manual environment variables.
3. The installed assets reside under `res/model/DINOv3`.
4. Backbone SHA-256 is validated before activation.
5. DINOv3 appears in classification model selection only after the component is healthy.
6. The current 17-species DINOv3 classifier can be constructed using the installed ViT-B/16 backbone.
7. Clicking Delete removes DINOv3-specific runtime assets and makes the model disappear.
8. Deleting DINOv3 does not remove or break PyTorch/YOLO.
9. Interrupted or failed installation does not destroy a previous valid install.
10. Python and Flutter test suites remain passing.
