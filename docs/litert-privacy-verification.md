# LiteRT privacy-upload experiment

Branch: `codex/litert-upload-privacy`. Date: 2026-10-25.

The existing uncommitted privacy-upload implementation was preserved before this
experiment at `temp/litert-baseline-20261025-013617/` (tracked binary patch plus
copies of untracked sources). Existing stashes were retained. No production
broker update is needed for LiteRT; the experiment uses the existing schema 3.

## Implementation

`system/training/litert_detector.py` runs the official
[YOLO26n w8a32 export](https://github.com/ultralytics/yolo-flutter-app/blob/main/doc/models.md)
using the `ai-edge-litert` CPU Interpreter API. The Flutter plugin is mobile-only;
its pre-exported model is reused in Neri's existing Python queue. There is no
frontend callback dependency and no `torch`, `ultralytics` or `tensorflow` import.

The detector validates the model's appended metadata and tensor format, performs
RGB/float32 preprocessing, letterboxing, confidence filtering and class-aware
NMS, then restores boxes into the upload-copy coordinates. The actual pinned
release has **NCHW `(1,3,640,640)` input**, despite the mobile documentation's
NHWC examples; both layouts are handled explicitly. Its output is normalized
xywh + 80 COCO class scores in `(1,84,8400)` format. The strongest class is chosen
before privacy-class filtering to avoid turning an animal into a lower-scoring
person candidate. Invalid/nonfinite/saturated outputs fail without uploading.

Existing whole-person/vehicle/screen/book redaction, metadata removal, consent,
device IDs, negative-label newline encoding, retries and owned deletion remain
in the queue. Runtime failure never falls back to uploading unprocessed pixels.
The model has the same general COCO privacy limits; it is not a dedicated
face/license-plate detector. Detection recall is not assumed identical to the
earlier YOLO11n/1280-pixel implementation.

`requirements.txt` includes `requirements-privacy.txt`, which pins LiteRT 2.3.0.
The lite packaging import check verifies the native runtime loads after PyTorch
is removed. The privacy UI/policy now explain LiteRT and do not direct users to
install PyTorch for upload anonymization.

## Model and runtime

- Model: `yolo26n_w8a32.tflite`, official `v0.6.6` asset, 2,875,553 bytes.
- SHA-256: `d9cef07ce652ccfa9ce58e4ac8a4df98ff037739a9dad20a8afcae21b545df73`.
- The downloaded hash matched the official release asset digest.
- Runtime: official `ai-edge-litert==2.3.0` Windows x64 Python 3.13 wheel.
- No runtime conversion, training dependencies, GPU drivers or PyTorch download.
- Platform verified here: Windows x64 / Python 3.13. Native Windows ARM64 and
  other desktop architectures have not been verified; no new desktop executable
  has been built or installed by this experiment.

## Validation

Created an isolated venv at `temp/litert-validation/venv` with base requirements,
pytest and httpx. `torch`, `torchvision`, `ultralytics` and `tensorflow` were absent.
The actual LiteRT/XNNPACK runtime loaded the official model and processed
Ultralytics' public `bus.jpg`: **five privacy regions** were redacted. The result
was visually inspected. The reference was used locally only and was not uploaded.

Six focused Python files passed **76 tests and 39 subtests**, including geometry,
RGB normalization, duplicate suppression, per-class filtering, invalid tensors,
queue failure handling, ID persistence and transport ownership. Both Flutter
privacy files passed **20 tests**. The changed Dart files were also analyzed.

The reproducible checker is `scripts/verify_litert_privacy.py`:

```powershell
python scripts/verify_litert_privacy.py --cache-dir temp/litert-validation --output-dir temp/litert-validation/offline-smoke
```

It requires a clean environment and defaults to memory-only transport. An
optional `--reference-image` is always processed locally, never uploaded.
`--live` explicitly enables uploads of only the generated synthetic fixtures,
followed by owned deletion. Its private test-state SQLite directory must remain
private and should be retained if cleanup fails.

The authorized live run passed with no forbidden packages installed or loaded:

- Two synthetic samples (one ordinary, one confirmed empty), **six file PUTs**.
- Broker completion accepted both samples with their device-ID receipts.
- Repeating completion caused no file retransfers.
- Source hashes were unchanged and prepared JPEGs had no EXIF.
- Both owned test-sample delete requests succeeded, without changing user consent
  or reading user photographs. This checker does not delete empty cloud folders.
- Public result: `temp/litert-validation/live-smoke/result.json`.

## Disk space measured on this machine

The LiteRT wheel download was about 19.7 MiB. Installed LiteRT distribution files
were about 56.1 MiB. Including its dedicated dependencies and the model, the
measured total was **67,309,329 bytes (64.2 MiB)**. This excludes NumPy, OpenCV and
Pillow already required by Neri, and does not claim to be the entire application
size or runtime RAM usage. File inventory measurements can differ slightly from
allocated filesystem space and later generated bytecode caches.

For comparison, the existing CPU PyTorch test installation's `torch` directory
alone occupied **527,493,593 bytes (503.1 MiB)**, excluding torchvision and other
dependencies. This particular comparison is approximately eightfold; installed
sizes vary by version and CPU/CUDA build.
