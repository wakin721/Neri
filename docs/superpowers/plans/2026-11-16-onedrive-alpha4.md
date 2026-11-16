# Alpha4: CPU/GPU OneDrive direct downloads

**Goal:** Both Neri1.2preview variants prefer OneDrive links, preserving verified partial downloads and the existing server proxy fallback.

**Architecture:** Publish immutable OneDrive mirror folders identified by the current CPU/GPU manifest IDs. The server keeps its local manifests and local fallback bytes; a direct-link resolver maps each runtime to its mirror. This preserves manifest IDs and cache compatibility. Credentials remain server-side.

- [x] Add independent CPU/GPU mirror configuration and direct-link resolution without changing proxy routing.
- [x] Validate source files, deduplicate matching model assets through OneDrive copy, and prepare a background administrator publisher with checksum verification and rollback.
- [ ] Activate both cloud mirrors and verify actual OneDrive links and downloaded ranges.
- [ ] Prefer OneDrive even when a verified partial download exists; preserve prefix and fall back to proxy on errors.
- [x] Update the release to `3.1.1-alpha4`.
- [ ] Run relevant regression checks, push nightly, build alpha4 locally, and report precise deployment/CI status.

The previous proxy-based download remains usable during upload. Do not delete local releases, installed models, learning records, or existing download caches.
