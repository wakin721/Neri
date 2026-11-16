# DINOv3 Fast Video Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Enable efficient DINOv3 sampled-video inference on CPU and GPU.

**Architecture:** Reuse the bounded video pipeline; retain unique per-frame crops and summarize a complete video once. Adapt model discovery and mixed-job persistence without changing encoder precision.

**Tech Stack:** Python, NumPy, OpenCV, PyTorch / LiteRT, pytest; existing Flutter capability controls.

**Spec:** `docs/superpowers/specs/2026-11-16-dinov3-fast-video-design.md`

## Global Constraints

- Keep calibrated float32 encoder behavior and full tracking unsupported.
- Old installed manifests remain readable.
- Persist only finalized video results; preserve existing user changes.

## Review Focus

- Multi-frame batches and retries must preserve unique source identities.
- Same-frame targets and different videos must not share tracks.
- Rejected boxes must survive serialization as Unknown.
- Errors and cancellation must release crop embeddings and preserve completed videos.
- Mixed/resumed jobs must finalize/cache only photos.

### Task 1: Capabilities (delegated)

- [x] Test old/new manifests and discovery for PyTorch and LiteRT.
- [x] Allow old false and new true fast capability flags; advertise runtime support.
- [x] Run manifest/catalog tests.

### Task 2: Fast video inference

- [x] Add failing end-to-end sampled video tests for frame identity, one final classifier pass, tracks, Unknown, errors and cleanup.
- [x] Add optional deferred sequence classification to `system/dino/image_processor.py`.
- [x] Implement `system/dinov3/video.py` for per-video feature collection and final sequence classification, integrating with `_detect_video_fast_batch`.
- [x] Run video and DINO image/sequence regression tests.

### Task 3: Job integration (delegated)

- [x] Test mixed jobs, video-only jobs, cancellation and resumed completed videos.
- [x] Update DINO job validation, photo cache restore/finalization boundaries and completed video reuse.
- [x] Run related job/cache tests.

### Task 4: Verify and review

- [x] Run combined targeted regression checks, inspect diff, and request independent review.
- [x] Measure local CPU/GPU execution where assets are available; document limits and results.
