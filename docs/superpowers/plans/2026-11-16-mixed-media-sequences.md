# Mixed Media Sequence Implementation Plan

**Goal:** Joint DINOv3 photo/video event classification with safe local deployment.

**Spec:** `docs/superpowers/specs/2026-11-16-mixed-media-sequences-design.md`

- [x] Timestamp metadata: trusted embedded capture time, provenance, invalid/missing fallback (delegated).
- [x] Evidence aggregation: optional positive weights in both model recipes; no-weights parity and replication invariance (delegated).
- [x] Video feature cache: transactional raw embeddings, source/settings validation, corrupt-cache recovery (delegated).
- [x] Shared finalization: proxy observations for video frames, real timestamps/camera identities, content digests, balanced video track weights, source/result mapping.
- [x] Job integration: defer mixed finalization, cache/restore video features, checkpoint partial mixed results, preserve file progress and errors.
- [x] Regression and real-model verification for mixed events, missing times, same-frame objects, independent cameras, chronological history and cancellation/resume.
- [x] Independent review, staged backed-up local installation update, installed-runtime smoke test.
