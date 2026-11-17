# Video trajectory and fallback time plan

**Spec:** `docs/superpowers/specs/2026-11-16-video-track-time-fallback-design.md`

- [x] Implement and test reliable video/burst/none reference resolver (delegated).
- [x] Separate video target grouping from photo events; classification uses video-specific tracks in both recipes.
- [x] Supply reference times only to strictly causal same-camera history, preserving provenance and real/approximate distinction.
- [x] Cache all video features, snapshot group settings, invalidate old policy decisions, and retain cancellation/source guards.
- [x] Update regression tests, installation verifier, documentation and explicit deployment file list.
- [x] Independent review, targeted regressions and installed CPU/GPU real-model smoke verification after backed-up update.
