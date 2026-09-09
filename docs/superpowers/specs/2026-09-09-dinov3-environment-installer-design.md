# DINOv3 Multi-prototype Environment and Incremental Learning Design

Date: 2026-09-09
Target branch: `nightly`
Implementation branch: `chore/dinov3-environment-installer`
Status: approved in chat; written-spec review pending

This specification supersedes the earlier Linear Head classifier and split asset-source portions of this document. The final design uses a Multi-prototype classifier as the sole primary DINOv3 classification head and treats NeriCloud as the authoritative source for the entire base DINOv3 component directory.

## 1. Goals

1. Add an optional DINOv3 ViT-B/16 environment component under **Advanced Settings -> Environment Maintenance**, directly below PyTorch.
2. When the user installs or repairs DINOv3, make local `res/model/DINOv3` an exact validated mirror of `NeriCloud/Neri_Data/Model/DINOv3` at the completion of that operation.
3. Replace the current Linear Head + prototype-consistency classifier with pure Multi-prototype inference.
4. Use the supplied `multi_prototype.pt` as the new base classifier contract.
5. Support training-free incremental species learning with a local K=3 prototype overlay.
6. Preserve learned local species when the base DINOv3 component is repaired, resynchronized, removed, or later reinstalled with the same base-model fingerprint.
7. Keep PyTorch and YOLO shared dependencies independent from DINOv3 removal.

## 2. Non-goals

- Do not fine-tune the DINOv3 backbone when adding species.
- Do not train or update a Linear Head during incremental learning.
- Do not write user-learned prototypes into `res/model/DINOv3`.
- Do not automatically upload local learned species to NeriCloud.
- Do not silently migrate a local registry to a different base-model fingerprint.
- Do not use GitHub as the normal runtime source for DINOv3 source code or model files.
- Do not make DINOv3 installation uninstall or replace shared PyTorch/YOLO components except through the existing PyTorch maintenance policy when PyTorch itself is missing or unusable.

## 3. Authoritative storage model

### 3.1 NeriCloud base component

The authoritative cloud component root is:

```text
NeriCloud/Neri_Data/Model/DINOv3
```

For the OpenList-compatible client this maps to:

```text
/Neri_Data/Model/DINOv3
```

The corresponding local mirror root is:

```text
res/model/DINOv3
```

`res/model/DINOv3` is base-model state only. It is treated as read-only by normal inference and incremental-learning code.

### 3.2 Local incremental overlay

User-learned species remain outside the mirrored model directory in the existing fingerprint-scoped registry:

```text
<local application data>/dinov3/<base-model-fingerprint>/registry.sqlite3
```

The base-model fingerprint is the SHA-256 fingerprint of the active Multi-prototype checkpoint. A new checkpoint fingerprint gets a separate registry. Existing registries are retained but are not automatically merged across fingerprints.

This separation is mandatory: strict cloud mirroring must never erase local learned species.

## 4. Base DINOv3 component contract

The component remains tied to:

- backbone: `dinov3_vitb16`
- feature dimension: 768
- patch size: 16
- encoder source commit: `facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751`
- backbone filename: `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
- backbone SHA-256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`
- preprocessing: `letterbox224_imagenet`
- event aggregation: `mean_l2_normalized_crop_embeddings`

The new base classifier file is `multi_prototype.pt` with SHA-256:

```text
4bb63f224a11e318c9a3586006146cad94a4df95f3dd5fe6157963aeafe0ab43
```

The supplied checkpoint contains:

- 17 classes
- 51 prototypes
- exactly 3 base prototypes per class
- prototype dimension 768
- `head_type = multi_prototype`
- `selection_k = 3`
- `decision = squared_euclidean_to_nearest_prototype`
- `rejection_score = cosine_similarity_to_winning_prototype`
- global threshold `0.31004515290260315`
- a 768-dimensional `feature_center`

Linear Head `weight` and `bias` are not part of the new primary classifier contract.

## 5. Multi-prototype checkpoint schema

The DINOv3 checkpoint validator must accept the Multi-prototype schema as the production format and must no longer require `head_state.weight` or `head_state.bias` for this format.

Required fields:

```text
schema_version
backbone
feature_dim
classes
feature_center
preprocessing
event_aggregation
encoder_weights
encoder_sha256
encoder_code_commit
head_type
prototypes
prototype_class_indices
prototypes_per_class
threshold
selection_k
decision
rejection_score
```

Validation requirements:

- `backbone == dinov3_vitb16`
- `feature_dim == 768`
- `head_type == multi_prototype`
- `selection_k == 3`
- `decision == squared_euclidean_to_nearest_prototype`
- `rejection_score == cosine_similarity_to_winning_prototype`
- `feature_center.shape == (768,)`
- `prototypes.shape == (M, 768)` with `M > 0`
- `prototype_class_indices.shape == (M,)`
- each prototype class index is within `[0, len(classes))`
- `prototypes_per_class` agrees with `prototype_class_indices`
- base production checkpoint has three prototypes for each of the 17 supplied classes
- threshold is finite
- tensors contain only finite values
- encoder SHA-256 and source commit match the installed encoder contract

The checkpoint fingerprint remains bound to the checkpoint file SHA-256 and scopes the local overlay registry.

## 6. Base Multi-prototype inference

For each event embedding `x` produced by the existing DINOv3 encoder/event aggregation pipeline:

```text
z = x - feature_center
```

For every prototype `p_j`, compute:

```text
d_j = ||z - p_j||^2
```

The winning prototype is:

```text
j* = argmin_j d_j
```

The predicted class is mapped through:

```text
prototype_class_indices[j*]
```

The rejection score is the cosine similarity between centered feature `z` and the winning prototype:

```text
score = cosine(z, p_j*)
```

For a formal known species, the prediction is accepted only when:

```text
score >= threshold
```

The base global threshold is the value stored in the checkpoint. The initial production value is `0.31004515290260315`.

No Linear Head logits or head/prototype consistency rule participate in the new primary classification path.

## 7. Species-level candidate ranking

UI candidates and exported candidate metadata must be species-level, not raw prototype-level.

For each species `c`:

```text
species_distance[c] = min_j d_j, for prototype j belonging to species c
```

Candidate species are ranked by ascending `species_distance`.

Each candidate may expose:

```text
species
nearest_prototype_index
squared_distance
cosine_score
source = base | overlay
registration_status
```

The default UI candidate list remains top 3 species.

## 8. Effective prototype bank

At runtime, Neri constructs an effective bank from two sources:

```text
Base bank from NeriCloud-synchronized multi_prototype.pt
    +
Fingerprint-scoped local overlay bank
```

### 8.1 Formal bank

The formal competitive bank contains:

- all base checkpoint species
- all local `Confirmed` species
- all local `Mature` species

All formal prototypes compete globally using the same centered 768-dimensional feature space and the same nearest-prototype rule.

### 8.2 Provisional bank

`Provisional` species are kept in a separate assistive bank.

They do not override a formal accepted result. They are evaluated only when no base/Confirmed/Mature result is formally accepted. If a Provisional species reaches the same global rejection threshold, it can be returned as an assistive suggestion with:

```text
accepted = false
assistive_match = true
registration_status = provisional
```

### 8.3 Candidate entries

`Candidate` registry entries do not enter the classifier bank. They participate only in unknown clustering, human review, and registration readiness checks.

## 9. Training-free incremental learning lifecycle

Incremental learning operates on independent event embeddings and never performs gradient training.

### Candidate

- fewer than 4 independent confirmed events, or registration conditions not yet met
- does not enter the classifier bank
- remains available for clustering and human review

### Provisional

- at least 4 independent human-confirmed events
- produces one overlay prototype
- may provide assistive recognition
- never returns `accepted = true`

### Confirmed

- at least 10 diverse independent events
- requires at least 2 cameras or equivalent diversity evidence already represented by the registry's registration policy
- becomes a formal Known species
- may maintain 1 or 2 stable prototypes
- returns `accepted = true` when its winning prototype meets threshold

### Mature

- at least 20 independent events
- requires at least 3 cameras or equivalent diversity evidence represented by the registry policy
- remains a formal Known species
- may maintain up to 3 stable prototypes

The number of prototypes is a maximum, not a mandatory count. A Mature species may remain at one or two prototypes if additional stable subclusters are not supported.

## 10. K=3 overlay prototype maintenance

The maximum overlay prototype count is fixed at:

```text
K = 3
```

Prototype updates use only human-confirmed independent event embeddings.

Unknown clustering continues to use the registry's existing raw L2-normalized event embedding/cosine behavior so the discovery process is not destabilized by the classifier migration.

When an entry is promoted into the classifier overlay:

1. load the confirmed event embeddings
2. transform each event embedding into the base classifier space with `z = x - feature_center`
3. choose a deterministic cluster count from 1 to the status-specific maximum
4. compute prototype centroids as arithmetic means in centered feature space
5. do not L2-normalize the stored centroid solely for Euclidean classification
6. use cosine normalization only when computing the rejection score

Cluster count limits:

- Provisional: maximum 1
- Confirmed: maximum 2
- Mature: maximum 3

The clustering algorithm must be deterministic. It may use deterministic K-means or deterministic farthest-first initialization followed by deterministic assignment/update. A new cluster must have sufficient independent-event support; unsupported tiny clusters are rejected and the entry falls back to fewer prototypes.

Exact support criteria must be deterministic and covered by tests; they must not permit a single anomalous event to create a new prototype.

## 11. Registration duplicate protection

Before a Candidate is registered as a new species, Neri must check its human-confirmed events against the current formal bank.

If a substantial and stable share of those events is already formally accepted as an existing base/Confirmed/Mature species, registration is blocked and the UI instructs the user to verify whether the candidate is actually an existing species.

This guard prevents duplicate species creation without introducing a new arbitrary runtime distance margin.

## 12. Conflict resolution

Conflict rules are deterministic:

1. Candidate entries never compete with formal classes.
2. Base, Confirmed, and Mature prototypes compete in one global nearest-prototype search.
3. The global winning formal species is accepted only if its winning prototype cosine score reaches the checkpoint threshold.
4. A Provisional match is considered only when no formal result is accepted.
5. A Provisional match never overrides an accepted formal species.
6. If no formal or Provisional match reaches threshold, return `Unknown`.

The classifier returns source and registry metadata so downstream UI/export code can distinguish base, overlay, provisional, and unknown results.

## 13. Strict NeriCloud mirror synchronization

A successful DINOv3 Install or Install/Repair operation must make local:

```text
res/model/DINOv3
```

match the authoritative NeriCloud tree:

```text
NeriCloud/Neri_Data/Model/DINOv3
```

The operation is a full mirror, not additive synchronization.

Consequences after a successful sync:

- cloud-added files exist locally
- cloud-modified files replace local versions
- cloud-deleted files are absent locally
- local extra files not present in NeriCloud are absent locally

No learned registry data may be stored inside the mirrored root.

## 14. Sync transaction

The install/repair flow is:

```text
resolve/check shared PyTorch runtime
    -> recursively enumerate NeriCloud DINOv3 tree
    -> create temp/dinov3-sync-* staging tree
    -> download every authoritative file to staging
    -> validate install.json
    -> validate complete declared file set and hashes
    -> validate Multi-prototype checkpoint schema and classifier SHA
    -> validate backbone SHA
    -> validate source package structure and encoder commit metadata
    -> validate license presence
    -> smoke-test source import and backbone construction
    -> move existing res/model/DINOv3 to rollback backup
    -> atomically activate staging as res/model/DINOv3
    -> smoke-test active component
    -> delete backup on success
```

If any pre-activation step fails, the existing installation is untouched.

If activation or post-activation smoke testing fails, the backup is restored.

Temporary download/staging files live outside `res/model/DINOv3`.

## 15. install.json component manifest

`install.json` is authoritative metadata inside the NeriCloud mirror. It must name the active classifier rather than requiring Neri to guess which `.pt` to load.

Minimum structure:

```text
schema_version
component
component_version
architecture
feature_dim
source.repository
source.commit
backbone.filename
backbone.sha256
classifier.filename
classifier.sha256
classifier.head_type = multi_prototype
classifier.selection_k = 3
classifier.manifest
files[]
```

`files[]` lists every integrity-controlled component file other than `install.json` itself, with at least relative path, SHA-256, and optional size. This permits local health checks without needing live NeriCloud access.

The classifier section for the supplied base model must identify `multi_prototype.pt` and SHA-256 `4bb63f224a11e318c9a3586006146cad94a4df95f3dd5fe6157963aeafe0ab43`.

## 16. Local health and cloud synchronization state

Local component health is independent from current network availability.

`healthy = true` requires the installed mirror to satisfy its local `install.json` integrity contract, including valid Multi-prototype checkpoint, source, license, backbone, and declared hashes.

Failure to reach NeriCloud does not make an already-valid local installation unhealthy.

Cloud freshness is checked when the user starts Install/Repair. A failed cloud synchronization reports the failure but preserves the last healthy local component.

## 17. Environment Maintenance UI

Under **Environment Maintenance**, directly below PyTorch, show:

Title:

```text
DINOv3 ViT-B/16
```

Subtitle:

```text
安装或同步 DINOv3 Multi-prototype 模型、官方 source 与 ViT-B/16 backbone
```

Button states:

| State | Action |
| --- | --- |
| component absent | `安装` |
| local component incomplete/corrupt | `安装/修复` |
| install/sync running | progress, disabled |
| local component healthy | `删除` |
| remove running | progress, disabled |

Install and repair both perform full NeriCloud mirror synchronization.

Removing DINOv3 deletes `res/model/DINOv3` but does not delete:

- PyTorch/torchvision
- YOLO dependencies
- fingerprint-scoped local DINOv3 registry/overlay data

A separate explicit future action would be required to delete learned local registry data.

## 18. Backend API

Retain the environment API surface:

- `GET /api/environment/dinov3-status`
- `POST /api/environment/install-dinov3`
- `POST /api/environment/remove-dinov3`

Install/remove use the existing maintenance worker, lock, progress, status-file, log, and backend restart mechanisms.

Status should expose at least:

```text
installed
healthy
architecture
component_version
classifier_filename
classifier_fingerprint
classifier_head_type
selection_k
source_commit
message
```

## 19. Runtime source and backbone resolution

Production resolution order for source:

1. explicit developer/test `source_root`
2. installed `res/model/DINOv3/source`
3. legacy `NERI_DINOV3_SOURCE` compatibility fallback

Production resolution for the encoder backbone includes the installed path under:

```text
res/model/DINOv3/dinov3-vitb16/
```

A normal user must not need Git or `NERI_DINOV3_SOURCE` after installation.

## 20. Classification model discovery

Classification discovery obtains the active DINOv3 classifier from `install.json` and its declared `.neri.json` manifest.

The DINOv3 model is discoverable only when the component is locally healthy.

After removal, it disappears from classification model selection without changing YOLO classification/detection model roots.

Model display metadata should identify it as Multi-prototype, for example:

```text
DINOv3 ViT-B/16 · Multi-prototype · 17 base species
```

Local Confirmed/Mature learned species are runtime overlay state and do not require rewriting the base model manifest.

## 21. Registry compatibility and fingerprint behavior

The registry remains fingerprint-scoped.

For the supplied `multi_prototype.pt`, the fingerprint is its checkpoint file SHA-256.

If NeriCloud updates unrelated source/license files while `multi_prototype.pt` remains byte-identical, the same overlay registry continues to load.

If the base checkpoint changes and therefore its SHA-256 changes:

- a new registry namespace is used
- the old registry remains on disk
- it is not silently merged into the new model
- UI may later offer an explicit migration/revalidation workflow, but that is outside this implementation

## 22. Error handling

- Every download lands in staging first.
- Downloaded files are verified against `install.json` before activation.
- Backbone and classifier SHA mismatches abort installation.
- Multi-prototype schema mismatch aborts installation.
- Source structure/commit mismatch aborts installation.
- Missing license aborts installation.
- Existing healthy local component is retained on failed synchronization.
- Atomic replacement rollback restores the previous component on activation failure.
- Removal is idempotent.
- Credentials, authorization headers, signed query parameters, and secrets are never written to maintenance logs.

## 23. Testing strategy

### Checkpoint tests

- supplied Multi-prototype schema loads successfully
- Linear Head fields are not required for Multi-prototype format
- invalid `feature_center`, prototype dimensions, indices, per-class counts, K, decision, rejection-score, or hashes are rejected
- classifier fingerprint equals file SHA-256

### Classifier tests

- centered feature transformation is applied exactly once
- squared Euclidean nearest-prototype determines winning prototype
- class mapping uses `prototype_class_indices`
- rejection score is cosine to the winning prototype
- threshold rejects/accepts correctly
- candidate ranking aggregates minimum distance per species
- no Linear Head logits affect decisions

### Overlay lifecycle tests

- Candidate is excluded from classifier bank
- four qualifying events produce Provisional with one prototype
- Provisional is assistive only
- ten diverse events can promote to Confirmed with at most two prototypes
- twenty diverse events can promote to Mature with at most three prototypes
- deterministic clustering returns repeatable prototypes
- tiny/anomalous subcluster does not create a new prototype
- Confirmed/Mature prototypes join the formal bank
- Provisional never overrides an accepted formal match
- duplicate-registration guard blocks a candidate consistently matching an existing formal species

### Synchronization tests

- recursive cloud tree is mirrored
- cloud-added files appear locally
- cloud-modified files replace local versions
- cloud-deleted/local-extra files disappear after successful sync
- `files[]` integrity is enforced
- failed download/hash/schema/import leaves prior local component untouched
- post-activation failure restores backup
- local registry remains outside the mirrored directory and survives install/repair/remove

### API/UI tests

- absent -> Install
- unhealthy -> Install/Repair
- healthy -> Delete
- install/repair starts strict sync
- progress blocks conflicting maintenance operations
- remove confirmation states that PyTorch and learned local registry data are retained
- installed healthy Multi-prototype model appears in classification selection
- removed model disappears

### Full regression

- complete Python test suite passes
- complete Flutter test suite passes
- Flutter Windows release builds
- portable Windows package assembles without requiring DINOv3 to be preinstalled
- YOLO remains functional after DINOv3 install/remove

## 24. Acceptance criteria

The feature is complete when:

1. A clean Neri package can install DINOv3 from NeriCloud without Git or manual source placement.
2. Successful install/repair leaves `res/model/DINOv3` matching the authoritative NeriCloud DINOv3 tree and containing no user overlay data.
3. `multi_prototype.pt` is validated by filename/manifest and SHA-256 and becomes the only primary DINOv3 classifier head.
4. Linear Head logits no longer participate in production DINOv3 classification.
5. Base inference uses centered features, squared Euclidean nearest-prototype selection, and cosine rejection exactly as declared by the checkpoint.
6. The supplied 17 base species expose three base prototypes each.
7. New species can progress Candidate -> Provisional -> Confirmed -> Mature without gradient training.
8. Overlay species maintain at most K=3 prototypes.
9. Provisional matches are assistive only; Confirmed/Mature matches can be formally accepted.
10. Local learned species survive DINOv3 repair, removal, and reinstall when the same checkpoint fingerprint is active.
11. A changed checkpoint fingerprint never silently consumes the previous overlay registry.
12. Failed cloud synchronization cannot destroy a previously healthy local component.
13. Removing DINOv3 does not remove PyTorch, YOLO dependencies, or learned registry data.
14. Python, Flutter, and Windows packaging regression checks pass.
