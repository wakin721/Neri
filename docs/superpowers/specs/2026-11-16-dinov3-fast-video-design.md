# DINOv3 fast video

Enable the existing fast video mode for DINOv3 on PyTorch CPU/GPU and LiteRT CPU. Reuse bounded in-memory decoding, prefetching, frame sampling and automatic inference batch reduction. Keep calibrated float32 encoder behavior and full tracking unsupported.

## Approaches

Use the existing fast pipeline with a per-video sequence summary. Reusing the full tracking path would bypass the DINOv3 adapter, while writing sampled JPEGs would add storage and decoding costs. A separate encoder runtime or reduced precision would change the calibrated model contract and is outside this change.

Each sampled frame gets a unique internal key. Encode crops during inference but defer the classification head until all sampled frames of that video are available. Link target tracks using the existing spatial/visual grouping, and classify each video's complete sampled sequence once. Different videos and same-frame objects must remain separate. No cross-video historical context is inferred from decode completion order. Persist only finalized video results and release frame embeddings after completion, error or cancellation.

Old installed manifests remain readable. Advertise fast video support through model discovery so the existing UI enables it. New exports advertise fast support, keeping full tracking disabled. Photo feature caching and sequence finalization remain photo-only; preserve successful completed videos when resuming mixed jobs.

Validation covers frame ordering and identity, rejected boxes, multi-object tracks, empty/corrupt videos, cancellation, OOM retries, mixed/resumed jobs, old/new manifest compatibility, and existing video/photo regressions. Measure actual local CPU/GPU throughput when model assets permit, distinguishing measurements from unit-test evidence.
