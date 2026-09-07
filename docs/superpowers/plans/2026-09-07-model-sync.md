# NeriCloud Model Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe automatic synchronization of Neri detection models, classification models, and `tracker.yaml` from NeriCloud, using direct OneDrive downloads first and a bounded Neri proxy fallback while preserving user-owned models and last-known-good synchronized files.

**Architecture:** Introduce a dedicated local `system/model_sync` subsystem for canonical layout migration, cataloging, remote protocol handling, durable synchronization state, verified downloads, and status. Add a separate `server/model_distribution` FastAPI service that exposes an allow-listed manifest plus short-lived OneDrive direct-download/proxy capabilities without exposing OpenList credentials. Flutter consumes only the local backend API and groups user/NeriCloud models by source.

**Tech Stack:** Python 3.13, stdlib `urllib`/`hashlib`/`sqlite3`/`threading`, FastAPI/Pydantic, OpenList + OneDrive, Flutter 3.44.6/Dart, package:http, GitHub Actions Windows x64.

**Spec:** `../specs/2026-09-07-model-sync-design.md`

## Global Constraints

- Canonical remote tree is exactly `NeriCloud/Neri_Data/Model/{detect,cls}` plus `tracker.yaml`; server-side OpenList root is fixed to `/Neri_Data/Model`.
- Canonical local tree is exactly `res/model/detect/{user,sync}`, `res/model/cls/{user,sync}`, and `res/model/tracker.yaml`.
- Synchronization may write/delete only `detect/sync`, `cls/sync`, and the managed `tracker.yaml`; it must never alter files under `detect/user` or `cls/user`.
- Legacy files are migrated non-destructively. A collision never overwrites user data.
- Detection sync accepts only `.pt`; classification sync accepts only `.pt`, `.onnx`, `.engine`; tracker is exactly `tracker.yaml`.
- Manifest file size and SHA-256 determine content identity. Modification timestamps are cache hints only.
- Every client download goes to a `.part` file, is size-checked and SHA-256 checked, then enters service through `os.replace`. A bad download never replaces an existing model.
- Source order is OneDrive direct capability first, then Neri proxy. If both fail, keep the old local file and report a non-fatal synchronization failure.
- Stale synchronized model deletion is allowed only after a complete valid manifest and a run in which all required creates/updates succeeded. If any required file fails, skip all stale deletion for that run.
- A missing `tracker.yaml` manifest entry never deletes the last-known-good local tracker in this protocol version.
- Automatic synchronization runs once at local-backend startup and must return immediately to the UI; there is also a manual `立即同步` action. No periodic cloud polling is added.
- A model already loaded by a running processing job is not hot-swapped. Updated files apply to later jobs.
- OpenList/Microsoft credentials, upstream Authorization/Cookie headers, direct URLs, and proxy bearer tokens must never be committed or logged.
- External model distribution is separate from `server/training_broker`; do not reuse training ownership, upload, retention, or path code.
- Keep new code modular. Do not add model-network logic to the existing large `system/backend/services.py` or Flutter `settings_screen.dart` beyond small integration calls/rendering.
- Every implementation task follows red-green-refactor: add focused failing tests, run them and observe the intended failure, implement the minimum behavior, re-run focused tests, then commit.

---

## Task 1: Canonical Model Layout, Runtime Migration, and Repository Resource Move

**Files:**
- Create: `system/model_sync/__init__.py`
- Create: `system/model_sync/layout.py`
- Create: `tests/test_model_sync_layout.py`
- Modify: `.gitignore`
- Modify: `.gitattributes`
- Move: `res/model/11s_p2_0319.pt` -> `res/model/detect/user/11s_p2_0319.pt`
- Move: `res/model/26m_p2_0224.pt` -> `res/model/detect/user/26m_p2_0224.pt`
- Move: `res/model/md_v1000.0.0-larch.pt` -> `res/model/detect/user/md_v1000.0.0-larch.pt`
- Move: `res/model/md_v1000.0.0-sorrel.pt` -> `res/model/detect/user/md_v1000.0.0-sorrel.pt`
- Move: `res/model_cls/26s-cls_0115.pt` -> `res/model/cls/user/26s-cls_0115.pt`
- Move: `res/model_cls/tracker.yaml` -> `res/model/tracker.yaml`
- Delete from repository resources: `res/model/tracker.yaml`
- Create: `res/model/detect/sync/.gitkeep`
- Create: `res/model/cls/sync/.gitkeep`

### Step 1: Write failing migration tests

- [ ] Add `.gitignore` exceptions for `tests/test_model_sync_layout.py` so the new test is tracked.
- [ ] Create `tests/test_model_sync_layout.py` with temporary-directory tests that express the migration contract before `system.model_sync.layout` exists.

Use cases to encode:

```python
class ModelLayoutMigrationTests(unittest.TestCase):
    def test_migrates_legacy_models_and_tracker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_root = Path(temp_dir) / "res"
            (resource_root / "model").mkdir(parents=True)
            (resource_root / "model_cls").mkdir(parents=True)
            (resource_root / "model" / "detect.pt").write_bytes(b"det")
            (resource_root / "model_cls" / "classify.onnx").write_bytes(b"cls")
            (resource_root / "model_cls" / "tracker.yaml").write_text("tracker_type: botsort\n")

            report = migrate_legacy_layout(resource_root)

            self.assertTrue((resource_root / "Model/detect/user/detect.pt").is_file())
            self.assertTrue((resource_root / "Model/cls/user/classify.onnx").is_file())
            self.assertTrue((resource_root / "Model/tracker.yaml").is_file())
            self.assertEqual(report.moved, 3)

    def test_migration_is_idempotent_and_preserves_collisions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_root = Path(temp_dir) / "res"
            legacy = resource_root / "model"
            target = resource_root / "Model/detect/user"
            legacy.mkdir(parents=True)
            target.mkdir(parents=True)
            (legacy / "same.pt").write_bytes(b"legacy")
            (target / "same.pt").write_bytes(b"user")

            first = migrate_legacy_layout(resource_root)
            second = migrate_legacy_layout(resource_root)

            self.assertEqual((target / "same.pt").read_bytes(), b"user")
            retained = sorted(target.glob("same.legacy-*.pt"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].read_bytes(), b"legacy")
            self.assertEqual(second.moved, 0)
```

Also test:
- only approved model extensions migrate;
- a legacy `res/model/tracker.yaml` is used only as fallback when neither canonical tracker nor `res/model_cls/tracker.yaml` exists;
- a pre-existing canonical tracker wins and is never overwritten;
- all four canonical model directories are created.

### Step 2: Run the focused tests and observe red

- [ ] Run:

```bash
python -m unittest tests.test_model_sync_layout -v
```

Expected red reason: `ModuleNotFoundError` for `system.model_sync.layout` or missing migration symbols.

### Step 3: Implement the layout module

- [ ] Add the following public contracts in `system/model_sync/layout.py`:

```python
@dataclass(frozen=True)
class ModelLayout:
    resource_root: Path
    root: Path
    detect_user: Path
    detect_sync: Path
    cls_user: Path
    cls_sync: Path
    tracker: Path
    state_file: Path

@dataclass(frozen=True)
class MigrationReport:
    moved: int
    collisions: tuple[str, ...]

def get_model_layout(resource_root: Path | None = None) -> ModelLayout: ...
def migrate_legacy_layout(resource_root: Path | None = None) -> MigrationReport: ...
def resolve_tracker_config(resource_root: Path | None = None) -> str: ...
```

Implementation rules:
- default `resource_root` is `Path(resource_path("res")).resolve()`;
- always create `Model/detect/user`, `Model/detect/sync`, `Model/cls/user`, and `Model/cls/sync`;
- legacy detect extensions: `.pt` only;
- legacy cls extensions: `.pt`, `.onnx`, `.engine`;
- collision names are deterministic `stem.legacy-1.suffix`, then `legacy-2`, and so on;
- prefer canonical tracker; otherwise migrate `model_cls/tracker.yaml`; otherwise migrate `model/tracker.yaml`; otherwise leave tracker absent;
- `resolve_tracker_config()` returns the canonical tracker path when present and literal `"botsort.yaml"` when absent.

### Step 4: Move tracked resources to the canonical repository layout

- [ ] In a normal git workspace, use `git mv` for the five tracked model files and the runtime tracker listed above.
- [ ] Remove the obsolete tracked `res/model/tracker.yaml` from the new-install resource tree.
- [ ] Add `.gitkeep` files in both empty `sync` directories.
- [ ] Change `.gitattributes` to:

```text
res/model/cls/user/26s-cls_0115.pt filter=lfs diff=lfs merge=lfs -text
```

Do not add `.sync-state.json` to source control; it is created only at runtime.

### Step 5: Verify layout tests and source tree

- [ ] Run:

```bash
python -m unittest tests.test_model_sync_layout -v
```

- [ ] Verify no tracked runtime model remains under legacy roots:

```bash
git ls-files res/model res/model_cls
```

Expected: no model/tracker files from those legacy roots.

### Step 6: Commit

- [ ] Commit with:

```bash
git add .gitignore .gitattributes system/model_sync tests/test_model_sync_layout.py res/model
git commit -m "refactor: establish managed model layout"
```

---

## Task 2: Unified User/Sync Model Catalog and Saved-Selection Compatibility

**Files:**
- Create: `system/model_sync/catalog.py`
- Create: `tests/test_model_catalog.py`
- Modify: `.gitignore`
- Modify: `system/backend/models.py`
- Modify: `system/backend/services.py`
- Modify: `system/backend/main.py`
- Modify: `system/image_processor.py`

### Step 1: Write failing catalog tests

- [ ] Track `tests/test_model_catalog.py` in `.gitignore`.
- [ ] Test a temporary canonical tree containing the same filename in `user` and `sync`.

Required assertions:

```python
models = discover_models(layout, "detect")
self.assertEqual(
    [(item.name, item.source, item.kind) for item in models],
    [("bird.pt", "user", "detect"), ("bird.pt", "sync", "detect")],
)
self.assertNotEqual(models[0].path, models[1].path)
```

Also test:
- detect ignores `.onnx`;
- cls accepts `.pt`, `.onnx`, `.engine`;
- source order is `user` then `sync`, with case-insensitive filename ordering inside each source;
- `resolve_saved_model_path("bird.pt", models)` chooses the unique matching legacy filename when unambiguous;
- if the same legacy filename exists in both sources, legacy filename resolution prefers `user` to preserve old local selections;
- an exact full path always wins;
- a missing full path resolves to `None`;
- `resolve_tracker_config()` uses canonical tracker and then `botsort.yaml` fallback.

### Step 2: Run red

- [ ] Run:

```bash
python -m unittest tests.test_model_catalog -v
```

Expected red: `system.model_sync.catalog` is missing.

### Step 3: Implement a pure catalog layer

- [ ] Add:

```python
@dataclass(frozen=True)
class DiscoveredModel:
    name: str
    path: str
    size_bytes: int | None
    source: Literal["user", "sync"]
    kind: Literal["detect", "cls"]

def discover_models(layout: ModelLayout, kind: Literal["detect", "cls"]) -> list[DiscoveredModel]: ...
def resolve_saved_model_path(saved: object, models: Sequence[DiscoveredModel]) -> str | None: ...
```

Do not make `catalog.py` import FastAPI/Pydantic backend models.

### Step 4: Adapt backend model schema and enumeration

- [ ] Extend `system/backend/models.py::ModelInfo`:

```python
class ModelInfo(BaseModel):
    name: str
    path: str
    size_bytes: int | None = None
    source: Literal["user", "sync"]
    kind: Literal["detect", "cls"]
```

- [ ] Change `model_directory()` to return `res/model/detect` and `classification_model_directory()` to return `res/model/cls`.
- [ ] Make `list_available_models()` and `list_available_classification_models()` translate `DiscoveredModel` objects to `ModelInfo`; do not recursively scan arbitrary directories.
- [ ] Replace filename-or-path selection logic in `/api/settings` with `resolve_saved_model_path` so full paths are canonical and old filename settings remain compatible.
- [ ] Preserve current first-detection-model default only when there is no saved `selected_model` key at all. Never silently replace a saved-but-missing synchronized model.
- [ ] Preserve the legacy `selected_cls_model` alias read path.

### Step 5: Point runtime tracking at the canonical tracker

- [ ] In `system/image_processor.py`, replace the hardcoded `res/model_cls/tracker.yaml` block with `resolve_tracker_config()` from `system.model_sync.layout`.
- [ ] Do not change Ultralytics tracking behavior beyond the path source and existing `botsort.yaml` fallback.

### Step 6: Verify focused and backend regressions

- [ ] Run:

```bash
python -m unittest tests.test_model_sync_layout tests.test_model_catalog -v
python -m unittest discover -s tests
```

Expected: all current backend tests pass.

### Step 7: Commit

- [ ] Commit with:

```bash
git add .gitignore system/model_sync/catalog.py system/backend/models.py system/backend/services.py system/backend/main.py system/image_processor.py tests/test_model_catalog.py
git commit -m "feat: catalog user and synced models"
```

---

## Task 3: Server-Side Manifest Builder and OpenList Read Adapter

**Files:**
- Create: `server/model_distribution/__init__.py`
- Create: `server/model_distribution/config.py`
- Create: `server/model_distribution/storage.py`
- Create: `server/model_distribution/manifest.py`
- Create: `tests/test_model_distribution.py`
- Modify: `.gitignore`

### Step 1: Write failing server manifest tests

- [ ] Track `tests/test_model_distribution.py`.
- [ ] Implement a `FakeModelStore` in the test that exposes only the server storage interface and returns synthetic bytes/metadata.
- [ ] Test that a manifest containing `detect/a.pt`, `cls/b.onnx`, and `tracker.yaml` returns sorted entries with exact `size` and SHA-256.
- [ ] Test rejection/omission of:
  - `detect/a.onnx`;
  - `cls/readme.txt`;
  - nested `detect/sub/a.pt`;
  - `../secret.pt`;
  - absolute paths;
  - backslash paths;
  - case/Unicode-normalized duplicate logical paths.
- [ ] Test hash-cache behavior: unchanged `(path, size, modified)` does not re-read bytes; changed metadata causes a new hash; missing `modified` causes a fresh hash instead of trusting cache.
- [ ] Test `manifest_id` is stable for identical sorted entries and changes when content hash/size/path changes.

### Step 2: Run red

- [ ] Run:

```bash
python -m unittest tests.test_model_distribution -v
```

Expected red: `server.model_distribution` modules do not exist.

### Step 3: Implement immutable configuration and storage contracts

- [ ] Add `DistributionConfig` with these defaults:

```python
@dataclass(frozen=True)
class DistributionConfig:
    state_dir: Path
    root_path: str = "/Neri_Data/Model"
    openlist_url: str = "http://127.0.0.1:5244"
    openlist_token: str = ""
    public_url: str = "https://myneri.top/api/models"
    capability_ttl_seconds: int = 600
    requests_per_minute: int = 120
    daily_proxy_ip_bytes: int = 2 * 1024**3
    daily_proxy_total_bytes: int = 20 * 1024**3
```

`from_env()` reads `NERI_MODEL_STATE`, `NERI_OPENLIST_TOKEN`, and optional model-specific limit variables. Never provide a secret default.

- [ ] In `storage.py`, define:

```python
@dataclass(frozen=True)
class RemoteEntry:
    name: str
    size: int
    modified: str | None
    is_dir: bool

@dataclass(frozen=True)
class UpstreamLink:
    url: str
    headers: dict[str, str]

class ModelStore(Protocol):
    def list_dir(self, path: str) -> list[RemoteEntry]: ...
    def stat(self, path: str) -> RemoteEntry | None: ...
    def resolve_link(self, path: str) -> UpstreamLink: ...
    def iter_bytes(self, link: UpstreamLink, range_header: str | None = None) -> Iterator[bytes]: ...
```

- [ ] Implement `OpenListModelStore` with bounded `urllib` timeouts against `/api/fs/list`, `/api/fs/get`, and `/api/fs/link`. It always joins requests below the fixed configured root and never accepts a client-provided OpenList root.

### Step 4: Implement manifest hashing and canonicalization

- [ ] Add:

```python
@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size: int
    sha256: str

@dataclass(frozen=True)
class ManifestSnapshot:
    manifest_id: str
    files: tuple[ManifestEntry, ...]
```

- [ ] Use SQLite at `state_dir/model_distribution.sqlite3` for hash cache records keyed by logical path. Store `size`, `modified`, `sha256`, and verification time.
- [ ] Build only from the shallow fixed folders `detect` and `cls`, plus a direct `tracker.yaml` stat. Do not recursively walk NeriCloud.
- [ ] Compute hashes by streaming through `ModelStore.iter_bytes()` in chunks. Cache a hash only when a non-empty `modified` value is available and both size/modified match on later reads.
- [ ] Calculate `manifest_id` from compact canonical JSON of sorted `[path,size,sha256]` tuples:

```python
payload = json.dumps(
    [[f.path, f.size, f.sha256] for f in files],
    ensure_ascii=False,
    separators=(",", ":"),
).encode("utf-8")
manifest_id = hashlib.sha256(payload).hexdigest()
```

### Step 5: Verify manifest behavior

- [ ] Run:

```bash
python -m unittest tests.test_model_distribution -v
```

### Step 6: Commit

- [ ] Commit with:

```bash
git add .gitignore server/model_distribution tests/test_model_distribution.py
git commit -m "feat: build secure model manifest"
```

---

## Task 4: Short-Lived Direct Download and Proxy Capabilities

**Files:**
- Create: `server/model_distribution/capabilities.py`
- Create: `server/model_distribution/app.py`
- Modify: `server/model_distribution/storage.py`
- Modify: `tests/test_model_distribution.py`

### Step 1: Extend tests for direct/proxy security before implementation

- [ ] Add tests covering:
  - exact `manifest_id + path + sha256` is required;
  - stale manifest ID returns a conflict/error rather than a capability;
  - a checksum not matching the current manifest entry is rejected;
  - direct URL only accepts HTTPS, no username/password, port 443/default, and Microsoft hosts ending `.1drv.com`, `.sharepoint.com`, or `.microsoftpersonalcontent.com`;
  - `http://`, non-Microsoft hosts, embedded credentials, and unexpected ports are rejected;
  - direct response never exposes `Authorization`, `Cookie`, or `Set-Cookie` upstream headers;
  - capability token is random, expires after 600 seconds, and is bound to one exact logical path/hash/size;
  - a token cannot be used for a different file;
  - use count is bounded to four proxy opens so retry/range requests are possible without unlimited reuse;
  - proxy forwards a valid `Range` header and streams multiple chunks rather than joining the whole file;
  - proxy budget rejects excess per-IP or total bytes.

### Step 2: Run red

- [ ] Run:

```bash
python -m unittest tests.test_model_distribution -v
```

Expected red: capability and route symbols are absent.

### Step 3: Implement capability and budget persistence

- [ ] In `capabilities.py`, create `CapabilityStore` using the same state SQLite file, storing only SHA-256 hashes of bearer tokens, never the raw token.
- [ ] Persist `logical_path`, `sha256`, `size`, `expires_at`, and `uses_remaining=4`.
- [ ] Add a small SQLite-backed `BudgetStore` keyed by HMAC-hashed client IP, with configured request/minute and proxy-byte/day limits. Reuse the training broker pattern conceptually but keep tables and code model-specific.

Public service objects:

```python
@dataclass(frozen=True)
class BoundCapability:
    path: str
    sha256: str
    size: int
    expires_at: float

class CapabilityStore:
    def issue(self, entry: ManifestEntry) -> tuple[str, float]: ...
    def consume(self, token: str) -> BoundCapability: ...
```

### Step 4: Implement direct-link safety policy

- [ ] Add:

```python
SAFE_DIRECT_HEADER_NAMES = frozenset({"accept", "referer", "user-agent"})

def valid_direct_download_url(url: str) -> bool: ...
def safe_direct_headers(headers: Mapping[str, str]) -> dict[str, str]: ...
```

If OpenList returns a direct URL that fails host validation, return no direct URL and leave the proxy token usable. If OpenList requires unsafe headers, omit them from the desktop response; direct may fail and the desktop will fall back to proxy.

### Step 5: Implement exact FastAPI contract

- [ ] `GET /health` returns `{ "status": "ok", "schema_version": 1 }`.
- [ ] `GET /v1/manifest` returns:

```json
{
  "schema_version": 1,
  "manifest_id": "<64 hex>",
  "files": [
    {"path": "detect/a.pt", "size": 123, "sha256": "<64 hex>"}
  ]
}
```

- [ ] `POST /v1/direct` accepts:

```json
{
  "manifest_id": "<64 hex>",
  "path": "detect/a.pt",
  "sha256": "<64 hex>"
}
```

and returns:

```json
{
  "path": "detect/a.pt",
  "size": 123,
  "sha256": "<64 hex>",
  "direct_url": "https://... or null",
  "direct_headers": {},
  "proxy_token": "<opaque token>",
  "expires_at": 1780000000.0
}
```

The proxy token is issued after manifest validation even if direct link creation fails, so direct-unavailable cleanly degrades to proxy.

- [ ] `GET /v1/proxy/{token}` consumes/bounds the token, requests a fresh OpenList upstream link server-side, forwards only a syntactically valid single HTTP byte range, and returns `StreamingResponse`. Copy only safe response headers required for file transfer (`Content-Length`, `Content-Range`, `Accept-Ranges`, `Content-Type`, `ETag`, `Last-Modified`) and never propagate cookies/auth headers.
- [ ] Add a sanitized exception handler. Error bodies must not contain OpenList URLs, upstream URLs, tokens, or credential values.
- [ ] Disable route/access logging of capability-bearing URLs in the deployment config later; application code must not log request bodies or tokens.

### Step 6: Verify server behavior

- [ ] Run:

```bash
python -m unittest tests.test_model_distribution -v
python -m unittest discover -s tests
```

### Step 7: Commit

- [ ] Commit with:

```bash
git add server/model_distribution tests/test_model_distribution.py
git commit -m "feat: issue model download capabilities"
```

---

## Task 5: Desktop Manifest Protocol and Distribution Client

**Files:**
- Create: `system/model_sync/protocol.py`
- Create: `system/model_sync/client.py`
- Create: `tests/test_model_sync_client.py`
- Modify: `.gitignore`

### Step 1: Write failing protocol/client tests

- [ ] Track `tests/test_model_sync_client.py`.
- [ ] Test strict manifest parsing:
  - `schema_version == 1`;
  - manifest ID is 64 lowercase hex;
  - duplicate normalized paths fail the entire manifest;
  - only allowed path shapes/extensions are accepted;
  - negative/oversized-invalid sizes are rejected;
  - SHA-256 must be 64 hex.
- [ ] Test capability parsing and client behavior:
  - invalid/non-Microsoft direct URL is discarded rather than used;
  - proxy token must match bounded URL-safe token syntax;
  - direct headers are filtered again client-side;
  - JSON responses are size-bounded;
  - non-2xx response produces a sanitized `ModelDistributionError`;
  - stream progress callback receives monotonically increasing byte counts;
  - default base URL is `https://myneri.top/api/models/v1` and `NERI_MODEL_SERVICE_URL` can override it for controlled deployment/testing.

Use injected opener functions/fake response objects; do not make real network calls in unit tests.

### Step 2: Run red

- [ ] Run:

```bash
python -m unittest tests.test_model_sync_client -v
```

### Step 3: Implement typed protocol objects

- [ ] Add:

```python
@dataclass(frozen=True)
class ModelManifestEntry:
    path: str
    size: int
    sha256: str

@dataclass(frozen=True)
class ModelManifest:
    manifest_id: str
    files: tuple[ModelManifestEntry, ...]

@dataclass(frozen=True)
class DownloadCapability:
    path: str
    size: int
    sha256: str
    direct_url: str | None
    direct_headers: dict[str, str]
    proxy_token: str
    expires_at: float
```

Implement one shared local logical-path validator matching server rules. Treat any invalid entry as an invalid manifest rather than partially accepting an unsafe/ambiguous manifest.

### Step 4: Implement `ModelDistributionClient`

- [ ] Public interface:

```python
class ModelDistributionClient:
    def fetch_manifest(self) -> ModelManifest: ...
    def request_download(
        self,
        manifest_id: str,
        entry: ModelManifestEntry,
    ) -> DownloadCapability: ...
    def download_to(
        self,
        url: str,
        headers: Mapping[str, str],
        target: Path,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> None: ...
    def proxy_url(self, token: str) -> str: ...
```

Implementation:
- stdlib `urllib.request` only; do not add `requests`;
- connect/request timeout around 20 seconds and stream-idle behavior bounded by socket response reads;
- JSON bodies capped to 1 MiB;
- file chunks 1 MiB;
- no bearer capability values in exception messages;
- direct URL redirects must be revalidated at every redirect hop or handled with a redirect handler that refuses transitions outside the allowed Microsoft HTTPS hosts;
- proxy URL is constructed from the configured Neri service base plus validated opaque token, never accepted from remote JSON.

### Step 5: Verify

- [ ] Run:

```bash
python -m unittest tests.test_model_sync_client -v
```

### Step 6: Commit

- [ ] Commit with:

```bash
git add .gitignore system/model_sync/protocol.py system/model_sync/client.py tests/test_model_sync_client.py
git commit -m "feat: add model distribution client"
```

---

## Task 6: Durable Desktop Synchronization State Machine

**Files:**
- Create: `system/model_sync/state.py`
- Create: `system/model_sync/manager.py`
- Create: `tests/test_model_sync_manager.py`
- Modify: `.gitignore`
- Modify: `system/model_sync/__init__.py`

### Step 1: Write failing manager tests with fake distribution client

- [ ] Track `tests/test_model_sync_manager.py`.
- [ ] Use real temporary files and a fake `ModelDistributionClient`; fake only the network boundary.
- [ ] Cover these required behaviors:
  1. matching local size/hash produces no download;
  2. state-cache `(size, mtime_ns, sha256)` skips rehash only when the file still matches cached identity;
  3. changed local mtime/size forces rehash;
  4. direct success writes `.part`, verifies size/hash, then atomically replaces target;
  5. direct HTTP failure deletes/reinitializes the candidate and uses proxy;
  6. direct checksum mismatch uses proxy;
  7. proxy checksum mismatch preserves old file and marks run `failed`;
  8. a completely new failed file remains absent, not half-written;
  9. a stale `sync` file is removed only after valid manifest plus all required writes succeed;
  10. if one required file fails, no stale file is removed anywhere in that run;
  11. files under `user` are never inspected for deletion or overwritten;
  12. missing tracker entry preserves existing tracker;
  13. two simultaneous `start()` calls create only one worker and return the same active `run_id`;
  14. state transitions are `idle -> checking -> downloading -> completed` or `failed`;
  15. successful state file is atomically persisted and reloadable;
  16. stale `.part` files are cleaned before a fresh run;
  17. synced detect/cls counts reflect manifest content.

### Step 2: Run red

- [ ] Run:

```bash
python -m unittest tests.test_model_sync_manager -v
```

### Step 3: Implement atomic state persistence

- [ ] `state.py` defines:

```python
@dataclass(frozen=True)
class VerifiedFileState:
    size: int
    mtime_ns: int
    sha256: str

@dataclass(frozen=True)
class SyncState:
    last_successful_sync: str | None
    manifest_id: str | None
    files: dict[str, VerifiedFileState]
```

- [ ] Read malformed/missing state as empty state, never as permission to delete files.
- [ ] Write JSON to `.sync-state.json.tmp`, flush/fsync where practical, then `os.replace` to `.sync-state.json`.

### Step 4: Implement thread-safe status model

- [ ] `manager.py` defines:

```python
@dataclass(frozen=True)
class ModelSyncStatus:
    state: Literal["idle", "checking", "downloading", "completed", "failed"]
    run_id: str | None = None
    current_file: str | None = None
    total_files: int = 0
    completed_files: int = 0
    received_bytes: int = 0
    total_bytes: int | None = None
    last_successful_sync: str | None = None
    manifest_id: str | None = None
    error: str | None = None
    cloud_detect_count: int | None = None
    cloud_cls_count: int | None = None

class ModelSyncManager:
    def status(self) -> ModelSyncStatus: ...
    def start(self) -> ModelSyncStatus: ...
    def sync_once_for_test(self) -> ModelSyncStatus: ...
```

`start()` acquires a lock, refuses duplicate workers, creates a daemon thread, sets a random run ID, and returns immediately.

### Step 5: Implement deterministic sync algorithm

- [ ] Sequence inside one run:

```text
clean stale .part files
fetch + validate manifest
calculate manifest counts
plan creates/updates by verified local identity/hash
for each required file, sequentially:
    request direct+proxy capability
    try direct to .part
    verify expected size + sha256
    if direct failed: restart .part and try proxy
    verify again
    os.replace(.part, final)
if any required file failed:
    skip every stale deletion
    persist verified successes but do not advance last_successful_sync/manifest_id
    state = failed
else:
    delete stale files only below detect/sync and cls/sync
    retain tracker if absent from manifest
    persist full verified state + manifest_id + UTC success time
    state = completed
```

- [ ] Remote logical-to-local mapping is fixed:
  - `detect/x.pt` -> `layout.detect_sync / "x.pt"`;
  - `cls/x.ext` -> `layout.cls_sync / "x.ext"`;
  - `tracker.yaml` -> `layout.tracker`.
- [ ] Reject any mapping whose resolved target escapes its expected managed parent even though protocol validation already ran.
- [ ] Hash local files in 1 MiB chunks.
- [ ] Use `os.replace` only after successful verification.
- [ ] Progress reports current-file received bytes and expected total. It does not claim a completed file until atomic promotion succeeds.
- [ ] Error strings shown to UI are concise and exclude direct URLs/tokens.

### Step 6: Verify manager and all Python tests

- [ ] Run:

```bash
python -m unittest tests.test_model_sync_manager -v
python -m unittest discover -s tests
```

### Step 7: Commit

- [ ] Commit with:

```bash
git add .gitignore system/model_sync/state.py system/model_sync/manager.py system/model_sync/__init__.py tests/test_model_sync_manager.py
git commit -m "feat: synchronize verified cloud models"
```

---

## Task 7: Local FastAPI Status/Run Endpoints and Non-Blocking Startup Hook

**Files:**
- Create: `system/model_sync/api.py`
- Create: `tests/test_model_sync_api.py`
- Modify: `.gitignore`
- Modify: `system/model_sync/__init__.py`
- Modify: `system/backend/models.py`
- Modify: `system/backend/main.py`

### Step 1: Write failing API/lifecycle tests

- [ ] Track `tests/test_model_sync_api.py`.
- [ ] Test response serialization for every status field.
- [ ] Test `GET` handler returns manager status without starting a new run.
- [ ] Test `POST` handler calls `start()` once and returns an already-active status unchanged when manager is active.
- [ ] Test startup wiring calls legacy migration before background `start()`.
- [ ] Test the startup function does not wait for `sync_once`/network completion: fake `start()` returns immediately while a blocking fake worker remains blocked.

Keep these tests at the function/router boundary; do not add a runtime `httpx` dependency solely for TestClient.

### Step 2: Run red

- [ ] Run:

```bash
python -m unittest tests.test_model_sync_api -v
```

### Step 3: Add response schema and router

- [ ] Add Pydantic `ModelSyncStatusResponse` with all `ModelSyncStatus` fields.
- [ ] Add router endpoints exactly:

```python
@router.get("/api/model-sync/status", response_model=ModelSyncStatusResponse)
def model_sync_status() -> ModelSyncStatusResponse: ...

@router.post("/api/model-sync/run", response_model=ModelSyncStatusResponse, status_code=202)
def run_model_sync() -> ModelSyncStatusResponse: ...
```

- [ ] Add a process singleton factory in `system/model_sync/__init__.py`:

```python
def get_model_sync_manager() -> ModelSyncManager: ...
```

Default client base URL comes from `NERI_MODEL_SERVICE_URL` or `https://myneri.top/api/models/v1`.

### Step 4: Wire startup in correct order

- [ ] In `system/backend/main.py::app_lifespan`:
  1. call `migrate_legacy_layout()` synchronously;
  2. initialize/start the existing training queue as today;
  3. call `get_model_sync_manager().start()`; this only creates a background thread;
  4. `yield` immediately;
  5. training queue still stops on shutdown; model sync thread is daemon/non-blocking and does not delay process exit.
- [ ] Include the model-sync router once alongside the privacy router.
- [ ] Migration or model-sync startup exceptions are logged as warnings and must not prevent the local API from starting.

### Step 5: Verify API and regression suite

- [ ] Run:

```bash
python -m unittest tests.test_model_sync_api -v
python -m unittest discover -s tests
```

### Step 6: Commit

- [ ] Commit with:

```bash
git add .gitignore system/model_sync/api.py system/model_sync/__init__.py system/backend/models.py system/backend/main.py tests/test_model_sync_api.py
git commit -m "feat: expose local model sync status"
```

---

## Task 8: Flutter Model Metadata and Source-Grouped Selectors

**Files:**
- Modify: `frontend/lib/src/models/settings.dart`
- Modify: `frontend/lib/src/screens/start_screen.dart`
- Modify: `frontend/lib/src/screens/settings_screen.dart`
- Modify: `frontend/test/model_selection_test.dart`

### Step 1: Update tests first

- [ ] Change existing hardcoded model roots in fixtures from `res/model` / `res/model_cls` to `res/model/detect` / `res/model/cls`.
- [ ] Add `source` and `kind` to `ModelInfo` test fixtures.
- [ ] Add widget tests with duplicate filenames:

```dart
const ModelInfo(
  name: 'bird.pt',
  path: r'C:\Neri\res\Model\detect\user\bird.pt',
  source: 'user',
  kind: 'detect',
),
const ModelInfo(
  name: 'bird.pt',
  path: r'C:\Neri\res\Model\detect\sync\bird.pt',
  source: 'sync',
  kind: 'detect',
),
```

Required assertions:
- both paths remain distinct selector values;
- menus visibly contain `用户模型` and `NeriCloud` group labels;
- each duplicate `bird.pt` remains selectable;
- default directory helper text uses canonical paths;
- settings-page selector receives the same grouping behavior as the start-page selector.

### Step 2: Run red

- [ ] Run:

```bash
cd frontend
flutter test test/model_selection_test.dart
```

Expected red: `ModelInfo` lacks new fields and group labels do not exist.

### Step 3: Extend Flutter model metadata

- [ ] Change defaults in `NeriSettings.fromJson` to:

```dart
modelDirectory: json['model_directory'] as String? ?? 'res/model/detect',
classificationModelDirectory:
    json['classification_model_directory'] as String? ?? 'res/model/cls',
```

- [ ] Extend `ModelInfo`:

```dart
class ModelInfo {
  const ModelInfo({
    required this.name,
    required this.path,
    required this.source,
    required this.kind,
    this.sizeBytes,
  });

  final String name;
  final String path;
  final String source;
  final String kind;

  String get sourceLabel => source == 'sync' ? 'NeriCloud' : '用户模型';
}
```

Parsing must reject no data catastrophically: unknown/missing source defaults to `user`, kind defaults from the collection in backend-normal cases but the JSON factory may use `detect` as a compatibility fallback.

### Step 4: Group start-screen `DropdownMenu` entries

- [ ] Update constants to canonical directories.
- [ ] Keep `不使用` first.
- [ ] Add disabled group header entries for each non-empty source, then the model entries. Header values use reserved strings that cannot equal a filesystem path, for example `__neri_header_user__` and `__neri_header_sync__`.
- [ ] Preserve actual `model.path` as the only selectable value for model rows.
- [ ] Do not merge or deduplicate same-named files.

### Step 5: Group settings-screen model options

- [ ] Extend the private `_SettingsOption<T>` / menu rendering just enough to support disabled section headers or separators.
- [ ] Render the same `用户模型` then `NeriCloud` ordering for both detection and classification menus.
- [ ] Keep draft/save values as exact full paths.
- [ ] Do not change unrelated settings behavior or auto-save timing.

### Step 6: Verify focused Flutter tests

- [ ] Run:

```bash
cd frontend
flutter test test/model_selection_test.dart
flutter analyze
```

### Step 7: Commit

- [ ] Commit with:

```bash
git add frontend/lib/src/models/settings.dart frontend/lib/src/screens/start_screen.dart frontend/lib/src/screens/settings_screen.dart frontend/test/model_selection_test.dart
git commit -m "feat: group user and cloud models"
```

---

## Task 9: Flutter Sync Status Controller, Manual Sync, and Settings Card

**Files:**
- Create: `frontend/lib/src/models/model_sync_status.dart`
- Create: `frontend/lib/src/model_sync_controller.dart`
- Create: `frontend/test/model_sync_test.dart`
- Modify: `.gitignore`
- Modify: `frontend/lib/src/api_client.dart`
- Modify: `frontend/lib/src/main_window.dart`
- Modify: `frontend/lib/src/screens/settings_screen.dart`

### Step 1: Write failing typed API/controller/UI tests

- [ ] Allowlist `frontend/test/model_sync_test.dart` in `.gitignore`.
- [ ] Test `ModelSyncStatus.fromJson` for all fields and active-state helpers.
- [ ] Test `NeriApiClient.fetchModelSyncStatus()` calls `GET /api/model-sync/status`.
- [ ] Test `runModelSync()` calls `POST /api/model-sync/run`.
- [ ] Test controller behavior with a fake/MockClient sequence:
  - first `checking`, then `downloading`, then `completed`;
  - polls only while active;
  - calls `onCatalogChanged` once when an observed active run reaches `completed`;
  - also calls it once when an observed active run reaches `failed`, because partial verified updates may have succeeded;
  - does not continuously poll a terminal state;
  - manual start begins/restarts active polling;
  - dispose cancels timers.
- [ ] Widget-test settings card states:
  - idle/completed: `立即同步` enabled;
  - checking/downloading: progress shown, button disabled;
  - failed: message says local models remain usable and shows `重试`;
  - cloud detect/cls counts and last-success time render when present.

### Step 2: Run red

- [ ] Run:

```bash
cd frontend
flutter test test/model_sync_test.dart
```

### Step 3: Implement typed status and API methods

- [ ] `ModelSyncStatus` fields mirror local response exactly, including `runId` and `manifestId`.
- [ ] Add:

```dart
Future<ModelSyncStatus> fetchModelSyncStatus();
Future<ModelSyncStatus> runModelSync();
```

to `NeriApiClient` with existing `_ensureSuccess` behavior.

### Step 4: Isolate polling in `ModelSyncController`

- [ ] Implement a `ChangeNotifier` controller rather than adding another state machine to `main_window.dart`:

```dart
class ModelSyncController extends ChangeNotifier {
  ModelSyncController(
    this.apiClient, {
    required this.onCatalogChanged,
    this.pollInterval = const Duration(seconds: 1),
  });

  final NeriApiClient apiClient;
  final Future<void> Function() onCatalogChanged;
  final Duration pollInterval;
  ModelSyncStatus? get status;

  Future<void> refresh();
  Future<void> runNow();
  @override
  void dispose();
}
```

Rules:
- initial terminal status does not cause an unnecessary catalog refresh;
- transition from an observed active `run_id` to terminal triggers exactly one refresh;
- polling errors surface as controller error state/message without crashing the app;
- no polling occurs after terminal state until another manual run/startup active state is observed.

### Step 5: Integrate controller with `MainWindow`

- [ ] Construct the controller once with the existing API client.
- [ ] After backend readiness/initial settings fetch, call `controller.refresh()`; this observes the backend-started automatic sync without blocking UI.
- [ ] `onCatalogChanged` re-fetches `/api/settings` and updates `_settings`/available model lists.
- [ ] Preserve `_selectedModelPath` and `_selectedClassificationModelPath` when those exact paths remain in the refreshed catalog. If a selected path is gone, accept backend-resolved `selected_model`/`selected_classification_model` rather than guessing another synchronized file.
- [ ] Do not alter any `ProcessingJob` request already submitted/running.
- [ ] Dispose the controller with the window.

### Step 6: Render model synchronization settings section

- [ ] Pass current status and `onRunModelSync` callback into `SettingsScreen`.
- [ ] Add a focused `模型同步` panel in the existing software/environment area, showing:
  - state label;
  - current logical filename;
  - determinate bytes when total known, otherwise indeterminate progress;
  - completed/total file counts;
  - last successful sync formatted in local time;
  - NeriCloud detection/classification counts;
  - sanitized error text;
  - `立即同步` or `重试`.
- [ ] Failure copy must explicitly say existing local models remain available.

### Step 7: Verify Flutter tests/analyzer

- [ ] Run:

```bash
cd frontend
flutter test test/model_sync_test.dart test/model_selection_test.dart
flutter analyze
```

### Step 8: Commit

- [ ] Commit with:

```bash
git add .gitignore frontend/lib/src/models/model_sync_status.dart frontend/lib/src/model_sync_controller.dart frontend/lib/src/api_client.dart frontend/lib/src/main_window.dart frontend/lib/src/screens/settings_screen.dart frontend/test/model_sync_test.dart
git commit -m "feat: show and control model synchronization"
```

---

## Task 10: Model Distribution Deployment Files, Packaging Guards, and Full Verification

**Files:**
- Create: `server/model_distribution/requirements.txt`
- Create: `server/model_distribution/README.md`
- Create: `server/model_distribution/neri-models.service`
- Create: `server/model_distribution/nginx-location.conf`
- Modify: `.github/workflows/build-windows-x64.yml`
- Verify: all files changed in Tasks 1-9

### Step 1: Add deployment documentation/config with no secrets

- [ ] `requirements.txt` pins the same server-compatible FastAPI/uvicorn versions used by the training broker unless current project constraints require a compatible patch version.
- [ ] `neri-models.service` uses:
  - dedicated `neri-models` user/group;
  - `/opt/neri-models` working directory;
  - `/etc/neri-models.env` environment file;
  - loopback `127.0.0.1:8002`;
  - one uvicorn worker;
  - `--no-access-log --no-proxy-headers`;
  - writable state only under `/var/lib/neri-models`;
  - systemd hardening comparable to `neri-training.service`.
- [ ] `nginx-location.conf` maps public `/api/models/` to `http://127.0.0.1:8002/`, disables proxy buffering for streamed fallback, disables access logging on bearer-token proxy paths, uses long enough read timeout for large models, and does not enable arbitrary request bodies.
- [ ] README documents only secret variable names, never values:

```text
NERI_MODEL_STATE=/var/lib/neri-models
NERI_OPENLIST_TOKEN=<set securely on server>
NERI_MODEL_PROXY_IP_BYTES=2147483648
NERI_MODEL_PROXY_TOTAL_BYTES=21474836480
```

Document OpenList fixed root `/Neri_Data/Model`, public health URL, rollback, and the fact that server traffic is normally limited to manifest hashing/control while model bytes go directly to OneDrive whenever possible.

### Step 2: Add CI/package assertions

- [ ] Keep existing `flutter test`, Windows release build, and `python -m unittest discover -s tests` steps.
- [ ] After copying `res` into the portable package, assert canonical directories exist and legacy directories are absent:

```powershell
$requiredModelPaths = @(
  "res\Model\detect\user",
  "res\Model\detect\sync",
  "res\Model\cls\user",
  "res\Model\cls\sync"
)
foreach ($relativePath in $requiredModelPaths) {
  if (-not (Test-Path (Join-Path $package $relativePath))) {
    throw "Missing canonical model path: $relativePath"
  }
}
if (Test-Path (Join-Path $package "res\model")) {
  throw "Legacy res/model directory is still packaged."
}
if (Test-Path (Join-Path $package "res\model_cls")) {
  throw "Legacy res/model_cls directory is still packaged."
}
```

- [ ] Extend the portable Python import check with `system.model_sync.manager` to catch missing packaged modules without adding a new dependency.

### Step 3: Run focused server verification

- [ ] Run:

```bash
python -m unittest tests.test_model_distribution -v
```

- [ ] With a fake/local test store only, verify manifest -> direct capability -> proxy-token flow. Do not place any real token, direct URL, or OneDrive file in tests.

### Step 4: Run complete local verification before claiming completion

- [ ] Run:

```bash
python -m unittest discover -s tests
cd frontend
flutter test
flutter analyze
```

- [ ] Confirm `git grep` has no runtime references to the old model roots except intentional migration code/tests/docs:

```bash
git grep -n -E 'res/model_cls|res/model' -- system frontend/lib frontend/test .github
```

Every remaining match must be either legacy migration compatibility or a test explicitly exercising that migration.

- [ ] Confirm no secret/capability values are tracked:

```bash
git grep -n -E 'NERI_OPENLIST_TOKEN=.+|Authorization: Bearer|proxy_token.*[A-Za-z0-9_-]{20,}|https://[^ ]*1drv\.com/' -- . ':!docs/superpowers'
```

Expected: no real credential, token, or concrete direct OneDrive capability.

### Step 5: Push and verify Windows CI

- [ ] Push the implementation branch/`dev` according to the chosen execution workflow.
- [ ] Wait only through the current tool call/session for the corresponding `Build Windows x64` GitHub Actions run to finish; do not claim success from a queued/in-progress run.
- [ ] Verify these steps are green from the completed run:
  - Flutter tests;
  - Flutter Windows release build;
  - Python backend tests;
  - portable app assembly including canonical model layout checks;
  - artifact upload.

If CI exposes an unexpected failure, invoke `superpowers:systematic-debugging` before changing implementation.

### Step 6: Commit deployment/CI integration

- [ ] Commit with:

```bash
git add server/model_distribution .github/workflows/build-windows-x64.yml
git commit -m "build: package and serve synchronized models"
```

---

## Final Acceptance Checklist

Before reporting the feature complete, invoke `superpowers:verification-before-completion` and collect fresh evidence for every item below:

- [ ] An upgraded installation migrates legacy detection/classification models to `user` without overwriting collisions.
- [ ] New installations contain only the canonical `res/model` model tree.
- [ ] Both `user` and `sync` models are returned with source/kind metadata and same-named files remain distinct by full path.
- [ ] Runtime tracking reads `res/model/tracker.yaml` and retains `botsort.yaml` fallback.
- [ ] Manifest exposes only allow-listed model files, size, SHA-256, and stable manifest ID.
- [ ] Server never accepts arbitrary OpenList paths and never exposes OpenList credentials.
- [ ] Direct OneDrive URL is attempted before proxy and is revalidated for Microsoft HTTPS hosts.
- [ ] Direct failure/checksum mismatch falls back to proxy.
- [ ] Dual failure preserves the last-known-good local model.
- [ ] `.part` data is never activated before size/SHA-256 verification.
- [ ] A failed required update suppresses all stale synchronized-model deletion for that run.
- [ ] `user` directories are never modified by synchronization.
- [ ] Missing tracker entry does not delete the local tracker.
- [ ] Startup sync is asynchronous/non-blocking and only one sync worker runs.
- [ ] Manual `立即同步` works and UI shows progress/failure without disabling local models.
- [ ] Terminal sync refreshes model catalog without mutating an already-running processing job.
- [ ] Python full test suite passes.
- [ ] Flutter full test suite and analyzer pass.
- [ ] Completed Windows x64 GitHub Actions build passes and uploads the portable artifact.

Operational deployment of `server/model_distribution` to the live Neri host is separate from repository implementation unless the execution environment has an authorized server-management connection. Do not claim the public endpoint is deployed until an actual health request to the deployed service succeeds.