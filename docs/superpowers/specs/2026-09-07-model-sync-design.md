# Neri model synchronization design

Approved scope: Neri automatically synchronizes detection models, classification models, and `tracker.yaml` from `NeriCloud/Neri_Data/Model`, preferring direct OneDrive downloads and falling back to a Neri server proxy only when direct download is unavailable or fails. Local user-provided models and synchronized models are separated physically so synchronization can safely update/delete only managed content. Synchronization must not block application startup or prevent use of already-installed models when the network is unavailable.

## 1. Canonical directory layout

Remote NeriCloud layout:

```text
NeriCloud/Neri_Data/Model/
├─ detect/
├─ cls/
└─ tracker.yaml
```

Canonical local layout:

```text
res/model/
├─ detect/
│  ├─ user/
│  └─ sync/
├─ cls/
│  ├─ user/
│  └─ sync/
└─ tracker.yaml
```

Mappings:

- `NeriCloud/Neri_Data/Model/detect/*` -> `res/model/detect/sync/*`
- `NeriCloud/Neri_Data/Model/cls/*` -> `res/model/cls/sync/*`
- `NeriCloud/Neri_Data/Model/tracker.yaml` -> `res/model/tracker.yaml`

`user` directories are never modified by model synchronization. `sync` directories are fully managed by NeriCloud synchronization. `tracker.yaml` is a separately managed synchronized file and does not appear in the model selector.

Allowed synchronized extensions are intentionally restricted:

- detect: `.pt`
- cls: `.pt`, `.onnx`, `.engine`
- tracker: exactly `tracker.yaml`

Other remote files are ignored/rejected by the model distribution service rather than copied to clients.

## 2. One-time migration

On first run after this layout is introduced, Neri performs an idempotent migration before normal model enumeration:

- model files directly under legacy `res/model/` move to `res/model/detect/user/`;
- classification model files directly under legacy `res/model_cls/` move to `res/model/cls/user/`;
- legacy `res/model_cls/tracker.yaml` moves to `res/model/tracker.yaml` when the canonical target does not already contain a file;
- non-model auxiliary files are not blindly migrated into model directories;
- pre-existing files in the canonical `user` directories are preserved; filename collisions must not overwrite user data. The migration should retain both files using a deterministic non-destructive rename or leave the legacy file in place and report the conflict;
- the migration is safe to run repeatedly and never moves anything out of `sync` into `user`.

After migration, all application model paths, defaults, inference paths, model enumeration, tests, and UI helper text use `res/model/...` rather than `res/model` or `res/model_cls`.

## 3. Local model enumeration and selection

Detection model enumeration scans both:

- `res/model/detect/user`
- `res/model/detect/sync`

Classification model enumeration scans both:

- `res/model/cls/user`
- `res/model/cls/sync`

The backend returns a unified list while preserving source metadata. `ModelInfo` is extended with at least:

- `source`: `user` or `sync`
- `kind`: `detect` or `cls`

The model selector groups entries by source, displaying user models separately from `NeriCloud` synchronized models. Duplicate filenames across `user` and `sync` are valid and must both be selectable.

Persisted model settings use the full resolved/normalized path rather than relying only on a filename. Backward compatibility should resolve legacy saved filenames against migrated user models where possible. If a previously selected synchronized model disappears from the cloud, the selection becomes unavailable gracefully and Neri chooses no replacement silently unless the existing application behavior already defines a safe default.

`tracker.yaml` is resolved from `res/model/tracker.yaml`, with the existing built-in/default tracker fallback retained if that file does not exist.

## 4. Synchronization lifecycle

Neri startup sequence:

1. run the idempotent legacy directory migration;
2. enumerate local `user + sync` models immediately;
3. expose the main UI and allow normal use;
4. start one background model synchronization check;
5. fetch and validate the remote manifest;
6. compare local synchronized content with the manifest;
7. download new/changed files;
8. validate downloads and atomically replace synchronized files;
9. remove stale files from `sync` only after a complete valid manifest has been obtained;
10. refresh model lists and synchronization status.

Automatic synchronization runs once per application startup. A manual `立即同步` action is also exposed. Periodic polling is intentionally out of scope for the first version.

Only one synchronization run may execute at a time. A manual request while synchronization is already active should return the active status instead of starting another concurrent run.

Synchronization never blocks the main UI. Existing local models remain usable throughout network checks and download failures. A model that is currently in use by an already-running processing job is not hot-swapped inside that job; the newly synchronized file becomes effective for subsequent jobs.

## 5. Manifest and change detection

The Neri model distribution service publishes a manifest for the approved remote tree. Each file entry contains at least:

```json
{
  "path": "detect/bird_v3.pt",
  "size": 48372912,
  "sha256": "..."
}
```

The manifest also has a stable `manifest_id` or digest representing the complete manifest. An optional source update timestamp may be included for display/debugging, but synchronization decisions do not depend on timestamps.

Client change detection uses file size and SHA-256. SHA-256 is authoritative. The local synchronization state may cache previously verified size/hash values to avoid repeatedly hashing unchanged very large files, but cached values must only be trusted when the local file identity/size/mtime metadata still matches the verified state. A cache miss or suspicious metadata change causes a fresh hash.

State is stored under:

```text
res/model/.sync-state.json
```

The state records at least:

- last successful synchronization time;
- last successful `manifest_id`;
- verified synchronized file hashes/sizes;
- last error/status metadata needed by the UI.

The state file is an optimization and status record, not the authority defining whether a file may be deleted. Directory ownership defines that boundary: only files under `detect/sync`, `cls/sync`, and managed `tracker.yaml` participate in synchronization.

## 6. Managed deletion rules

Deletion is fail-safe:

- user files under `detect/user` and `cls/user` are never deleted or overwritten by synchronization;
- stale files under `detect/sync` and `cls/sync` may be deleted only after the client has successfully obtained and validated a complete current manifest;
- if manifest retrieval fails, times out, returns invalid JSON, fails signature/digest validation if such validation is added, or contains invalid paths, no local synchronized model is deleted;
- a file omitted by a valid complete manifest is removed from the corresponding `sync` directory;
- `tracker.yaml` is retained when the cloud or download path is temporarily unavailable. A missing tracker entry in an otherwise valid manifest is treated conservatively: do not delete the last known-good local tracker unless the protocol explicitly marks tracker removal in a future version.

Temporary `.part` files are not treated as synchronized models and may be cleaned on startup when stale.

## 7. Download integrity and atomic replacement

For every new or changed synchronized file:

1. download to a temporary sibling or synchronization temp path ending in `.part`;
2. stream bytes to disk rather than buffering large model files in memory;
3. verify expected byte size;
4. calculate SHA-256 and compare with the manifest;
5. only after successful verification atomically replace/move the completed file into its final target;
6. preserve the previous valid file when download or verification fails.

A failed checksum never overwrites an existing model. Partial files may be retained only when the chosen direct-download implementation can safely resume them; otherwise they are discarded before retry. The first implementation may choose a simpler restart-from-zero behavior while still reporting byte progress.

## 8. Server-side model distribution service

Model distribution is separate from the training upload broker so upload capabilities, download capabilities, quotas, and path policies remain isolated.

Public API shape:

```text
GET  /api/models/v1/manifest
POST /api/models/v1/direct
GET  /api/models/v1/proxy/{token}
```

### `GET /api/models/v1/manifest`

Returns only the allow-listed logical model tree. It does not expose OneDrive/OpenList credentials or arbitrary storage paths. The service reads the remote `NeriCloud/Neri_Data/Model` tree through server-side OpenList/NeriCloud access and computes or retrieves trusted size + SHA-256 metadata.

The manifest must reject/omit:

- absolute paths;
- `..` traversal;
- backslash-based traversal;
- unexpected nested locations outside `detect`, `cls`, and `tracker.yaml`;
- disallowed file extensions;
- duplicate paths after Unicode/case normalization where that would be ambiguous on Windows.

### `POST /api/models/v1/direct`

The client submits a requested logical `path`, expected `sha256`, and `manifest_id`. The server verifies that this exact file exists in the current manifest before issuing any capability.

If valid, the server asks OpenList/NeriCloud for a short-lived direct download URL and returns it. The returned URL must be HTTPS and restricted to approved Microsoft/OneDrive/SharePoint hosts. Credentials remain server-side. Direct URLs are treated as bearer capabilities and are never logged.

### `GET /api/models/v1/proxy/{token}`

This endpoint is the fallback path, not the default data path. A short-lived opaque token binds exactly one validated manifest file and expected digest. The token cannot be used to choose or concatenate another path.

The proxy streams the requested file from NeriCloud/OpenList to the desktop without loading the whole model in memory. It should support ordinary HTTP streaming semantics needed by the client and enforce practical limits/rate controls so the fallback endpoint cannot become an unrestricted public file relay.

The server never accepts an arbitrary storage path from the proxy caller.

## 9. Download source priority and fallback

For each required file, the client attempts sources in this order:

1. obtain a OneDrive direct capability from `/api/models/v1/direct`;
2. download directly from OneDrive/Microsoft storage;
3. if direct capability creation fails, the direct URL expires, the HTTP transfer fails after bounded retry, or integrity validation fails, request/use the Neri proxy fallback;
4. stream through `/api/models/v1/proxy/{token}`;
5. perform the same size and SHA-256 verification;
6. on fallback failure, retain the last known-good local file and record a non-fatal synchronization error.

A checksum mismatch from a direct download is sufficient reason to discard that candidate and retry via the trusted fallback path. If the proxy result also fails checksum verification, the synchronization is failed for that file and the existing local file is preserved.

## 10. Local synchronization API

The local Python backend owns synchronization logic, migration, hashing, file replacement, deletion, and status. Flutter does not implement the synchronization state machine.

Local API:

```text
GET  /api/model-sync/status
POST /api/model-sync/run
```

The status model includes at least:

- state: `idle`, `checking`, `downloading`, `completed`, or `failed`;
- current logical file, if any;
- total file count;
- completed file count;
- current received bytes;
- current total bytes, when known;
- last successful synchronization time;
- latest human-readable error;
- counts of cloud detect/cls models when known.

`POST /api/model-sync/run` starts synchronization when idle and otherwise returns the current active status. Status mutations are thread-safe. Errors are non-fatal to the local backend.

## 11. Flutter UI

Settings receives a `模型同步` section containing:

- current state;
- last successful synchronization time;
- remote detect/cls counts when known;
- progress while downloading;
- last synchronization error when failed;
- `立即同步` / `重试` action.

The UI must clearly state that a failed sync does not disable already-installed local models.

Model selectors group both model types by source:

```text
用户模型
  26s.pt
  custom_bird.pt

NeriCloud
  wildlife_v4.pt
  bird_global.pt
```

The display source is metadata only; selection values are unique full paths so same-named files in `user` and `sync` do not collide.

When synchronization completes, Flutter refreshes settings/model lists without forcing a running task to switch models.

## 12. Failure behavior

Failures must degrade to the last known-good local state:

- cannot reach manifest service -> keep all synchronized files, show failed/checking error;
- malformed/incomplete manifest -> keep all synchronized files and perform no deletion;
- direct link generation failure -> attempt proxy;
- direct transfer timeout/network error -> bounded retry, then proxy;
- direct checksum mismatch -> discard candidate, then proxy;
- proxy failure/checksum mismatch -> retain existing file and report failure;
- one file fails while others succeed -> do not claim the overall run fully successful; retain successful verified replacements, preserve old failed targets, and do not perform unsafe stale deletion unless the implementation can prove manifest reconciliation is still safe;
- local permission/disk-space failure -> abort affected write, keep previous file, report a readable local error;
- migration collision -> never overwrite user data; report/log the conflict and continue with other migration items.

Model synchronization errors are operational errors, not application-startup errors.

## 13. Security requirements

- OpenList and OneDrive master credentials remain server-side only.
- Manifest and direct/proxy APIs expose only allow-listed model paths.
- Reject traversal, absolute paths, invalid separators, unexpected extensions, and normalized path collisions.
- Validate direct URLs against approved HTTPS Microsoft storage hosts before returning/using them.
- Do not log bearer direct URLs or proxy capability tokens.
- Proxy tokens are random, short-lived, single-file capabilities and should be bounded by expiry and optionally single-use/download-count controls.
- Proxy and manifest endpoints should have request-rate and byte-rate controls appropriate for public clients.
- Downloaded content is never activated before SHA-256 validation.
- Synchronization never writes into user-owned model directories.

## 14. Testing requirements

Backend/local tests:

- legacy migration into canonical `user` folders;
- migration idempotency and collision preservation;
- enumeration of `user + sync` detect/cls models;
- duplicate filenames remain independently selectable by full path;
- old saved filename resolves after migration when unambiguous;
- tracker resolves from `res/model/tracker.yaml` with fallback when absent;
- valid manifest diff produces create/update/delete plan only inside managed paths;
- invalid/incomplete manifest produces no deletion plan;
- checksum mismatch never replaces a valid old model;
- `.part` download is atomically promoted only after size/hash validation;
- direct-download failure invokes proxy fallback;
- direct checksum mismatch invokes proxy fallback;
- proxy failure preserves old local model;
- status transitions and single-active-sync concurrency behavior;
- startup synchronization does not block local settings/model enumeration.

Server tests:

- manifest exposes only allow-listed paths/extensions;
- path traversal and normalized duplicate paths are rejected;
- direct capability can only be issued for current manifest entries;
- invalid/stale manifest ID or checksum request is rejected;
- direct URL host/scheme validation;
- proxy token is bound to exactly one model file and expires;
- proxy cannot read arbitrary NeriCloud/OpenList paths;
- streaming does not buffer a whole model in memory;
- fallback rate/size limits behave as configured.

Flutter tests:

- new canonical directory defaults;
- model selectors display both user and NeriCloud groups;
- same filename from both sources produces two distinct selectable entries;
- model sync status/progress/error states render correctly;
- `立即同步`/`重试` invokes the local API;
- completed synchronization refreshes settings/model lists without changing a running job.

CI should run both Python and Flutter test suites before Windows packaging, preserving the current Windows build verification.

## 15. Non-goals for the first implementation

- periodic hourly/background polling while Neri remains open;
- delta/binary patching of model files;
- user authentication for choosing private per-user model catalogs;
- upload of user models back to NeriCloud;
- synchronizing arbitrary files outside the approved Model tree;
- hot-swapping the model instance of a processing job that is already running.

The first implementation optimizes for correctness, integrity, and server-bandwidth avoidance: use local models immediately, synchronize once per startup, download directly from OneDrive whenever possible, fall back safely through Neri only when required, and never risk user-owned model files.