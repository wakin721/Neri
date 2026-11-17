# Validation repair implementation plan

**Spec:** `docs/superpowers/specs/2026-11-17-validation-correctness-design.md`

- [x] Add detector-category provenance, isolation, target-wise photo classification and protected legacy normalization; test source screenshot scenarios.
- [x] Preserve species/file selection across regroup/refresh and last-group annotation (delegated).
- [x] Repair video session playback races, retry and error logging with deterministic tests (delegated).
- [x] Prepare guarded compiled-UI/backend deployment with backup/rollback, excluding unrelated changes (delegated).
- [x] Run focused regressions and independent review; build Windows release and verify engine hotfix.
- [x] Validate exact screenshot inputs and local-installed runtime, deploy backed-up update, verify checksums and protected data.
