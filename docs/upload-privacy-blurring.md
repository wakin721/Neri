# Local privacy processing for training uploads

Agreement version: **2026-10-25**. Earlier choices do not authorize uploads under
this version; the existing consent gate asks for a new explicit choice.

Every eligible photo, including a confirmed empty photo, goes through
`prepare_sample` before the transport can request upload sessions:

1. Read the primary frame, apply EXIF orientation, convert to RGB and resize to
   a maximum long edge of 2048 pixels. Source files are never modified.
2. Run the official pre-exported YOLO26n w8a32 model with LiteRT on CPU
   (640-pixel letterbox, confidence 0.15, per-class NMS IoU 0.5). The pinned
   model has NCHW float32 input and normalized `(1,84,8400)` raw output;
   metadata and tensor shapes are checked before processing. Restore boxes
   using the exact resize/padding geometry. No source paths, tracking state,
   PyTorch, Ultralytics Python package or cloud inference are used.
3. Expand detected privacy regions by 10% (at least four pixels), clip to image
   bounds, reduce each to at most 8×8 mosaic cells and apply a 51×51 box blur.
   Detection targets are whole people, vehicles, screens and books. Animals
   are not targeted; overlapping regions may nevertheless affect animal pixels.
4. Encode the processed copy as metadata-free JPEG, quality 82. Keep existing
   normalized wildlife annotations; never upload privacy boxes, face embeddings
   or recognized text. The internal annotation package records processing
   version, region count and `visual_anonymity_guaranteed: false`.

This adapts the detection/ROI box-blur approach from the
[Ultralytics object-blurring guide](https://docs.ultralytics.com/guides/object-blurring/).
It uses independent LiteRT predictions rather than `ObjectBlurrer.process`,
which tracks objects between frames and draws bounding boxes/labels. Independent
training photographs should not share tracking state or contain overlays.

## Runtime and model

The base requirements include `ai-edge-litert==2.3.0` through
`requirements-privacy.txt`. The Windows x64 lite package retains this runtime
when pruning PyTorch; its import check loads the native LiteRT interpreter and
asserts PyTorch is absent. Developers can install the same requirements directly.
No model conversion or PyTorch download is performed on user devices. Missing
or broken LiteRT runtimes block contributions and show a repair/update hint in
upload details. Cache presence does not replace checksum validation. Windows
x64 / Python 3.13 is verified; native Windows ARM64 is not verified.

After explicit participation, the first actual sample downloads
[`yolo26n_w8a32.tflite` from the official Flutter project's v0.6.6 release](https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6/yolo26n_w8a32.tflite)
to `privacy-models` inside the existing local training-state directory. The
fixed SHA-256 is
`d9cef07ce652ccfa9ce58e4ac8a4df98ff037739a9dad20a8afcae21b545df73`.
Downloads are size-bounded, checked and atomically installed. Cached bytes are
checked before loading; the class map must match the expected COCO targets.
The detector is cached in memory within the worker after successful loading.
GitHub receives an ordinary model download request, never the photo or labels.
Ultralytics and its models retain their own license terms.

Missing dependencies, download/hash failures, malformed or saturated detections,
and inference/processing errors raise `PrivacyProcessingError`. The queue retains
the task as failed and retries with its existing backoff. No upload authorization
or bytes are sent for that attempt. Consent changes/source changes are still
checked after processing and during transfer. Failed samples can be cleared with
the existing queue controls. Already completed historical remote submissions are
not scanned or retroactively altered by this local change.

## Limits

The standard COCO model has people/vehicle classes, **not dedicated face or
license-plate classes**. Whole-object redaction covers identities inside detected
objects but can miss standalone faces/plates, small objects, occlusion and
infrared/night scenes. Other text, watermarks and buildings can remain. This is
an additional safeguard, not irreversible anonymization or an OCR service.
Known person/vehicle labels remain excluded by the existing contribution policy.
Users must still screen sensitive photographs before confirming them.

## Verification

Run the upload/privacy tests without real user data or remote storage writes:

```powershell
python -m pytest -q tests/test_training_upload.py tests/test_training_privacy_blurring.py tests/test_training_litert.py tests/test_training_privacy_api.py tests/test_training_transport.py tests/test_training_broker.py
cd frontend
flutter test test/privacy_test.dart test/training_upload_debug_test.dart
```

The tests use deterministic detections for pixel changes, unchanged source bytes,
empty-photo coverage, no-upload-on-failure, retry, invalid detections, checksum
handling, and old-consent gating. A real-model smoke check is separate from these
offline tests and cannot establish recall across camera-trap deployments.

Initial blurring verification on 2026-10-25: the five Python test files above passed
59 tests and 26 subtests. Both Flutter files passed 18 tests. A real CPU YOLO11n
run on Ultralytics' public `bus.jpg` redacted five regions, produced an
87,217-byte JPEG and preserved the source hash. The same detector completed
the queue path using an in-memory transport; no broker/storage requests or
real user photographs were used. The output was visually inspected. This
is a smoke check, not evidence of face/plate detection recall in deployment.

## Device ID and deletion requests

The privacy settings card shows a selectable **device anonymous ID** and a copy
button even while participation is off. Only the 32-character public ID goes to
the clipboard. It is derived from a private random installation secret in the
training journal, not a hardware fingerprint. Consent changes, restarts and queue
clearing keep it; losing or changing the training-state directory creates another
identity. Users should save old public IDs before clearing application data.

New remote paths are `<training-root>/<device_id>/<species>/<sample>...` for
JPEG, YOLO labels and class files. The broker validates the private device proof,
binds sample records to the ID and returns the ID in its receipt. The desktop
checks the receipt before transferring bytes. See the broker's
[protocol upgrade notes](../server/training_broker/README.md#device-directory-protocol-upgrade)
for schema version 3, coordinated client/server deployment and legacy journals.

For deletion requests, stop participation and send the public ID plus the desired
scope to `wakin721@outlook.com`. The operator verifies and locates the request;
the ID is not a bearer credential or a public bulk-delete API. Neither private
device nor sample secrets should be sent by email. Unassociated historical data
does not automatically move into a new device directory and may need additional
sample IDs or submission times to locate.

After the device-ID changes, the same five Python files passed **68 tests and
31 subtests**, including a full queue → HTTP transport → broker → memory drive
check that the displayed ID matches all three remote file paths. The two Flutter
files passed **20 tests**, including exact clipboard contents with participation
off and a narrow viewport. Legacy journal migration/deletion and public-ID
spoofing are covered without modifying the live service or remote files.

Subsequent authorized deployment and real storage testing are recorded in the
[2026-10-25 live upload report](live-upload-verification-2026-10-25.md). That test
found and fixed the 0-byte negative-label upload issue: whitespace-only labels
now contain one newline, which Ultralytics still parses as zero objects. After
the fix, 69 Python tests and 33 subtests passed, and both real synthetic uploads
were verified and cleaned up.

The `codex/litert-upload-privacy` experiment then replaced the client detector
with LiteRT. See [LiteRT verification](litert-privacy-verification.md) for its
clean runtime, model output checks, disk measurements and live upload results.
Earlier YOLO11/PyTorch test records above remain historical observations.
