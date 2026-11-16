# Neri1.2preview cloud runtime implementation plan

**Goal:** Distribute Neri1.2preview CPU/GPU components through NeriCloud and move the download choice into environment maintenance, defaulting to automatic hardware detection.

**Architecture:** Keep the existing calibrated model, resumable verified downloads, and shared learning family. CPU uses LiteRT FP32 without PyTorch; GPU uses the existing PyTorch component. Separate cloud manifests prevent either download from pulling the other variant.

**Constraints:** Preserve alpha3 naming, MDV6 built-in detector, noise labels and skipped-video handling. Never push the classifier weights or their intermediate Git history to GitHub. Preserve local model files and learned samples.

- [x] Remove the two frontend runtime selectors; add automatic/CPU/GPU download choice under environment maintenance and forward `runtime_choice`.
- [x] Extend installer/status/health checks for LiteRT; choose a compatible GPU from hardware probing for automatic mode and retain shared learning through replacement. GPU capability probing runs in a short-lived process so Windows DLLs remain replaceable during repair.
- [x] Publish independent CPU manifest at `/dinov3/manifest?runtime=cpu`, with logical paths `DINOv3/cpu/**`; retain the existing default GPU manifest and capability checks.
- [x] Stop bundling classifier weights in portable builds and remove dependency on tracked chunks in CI; retain explicit local preparation tools.
- [x] Prepare verified CPU assets and service bundle on NeriCloud; administrator activated the service at 07:40:22 Asia/Taipei. Full production loopback download and public small-file/range checks passed.
- [ ] Rebuild unpushed Git commits without classifier resources, preserve local assets, run relevant tests, build alpha3 locally, and push nightly to trigger CI.

**Verification focus:** GPU/CPU manifest isolation, corrupt download rollback, CPU install without torch, shared learning preservation, hardware detection without torch, and no classifier weights in Git history or app package.
