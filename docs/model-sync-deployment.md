# Model synchronization deployment

Deployed on 2026-09-08 (Asia/Shanghai) to the authorized Neri server.

The desktop uses `https://myneri.top/api/models/v1`. The model service authenticates
to the loopback OpenList instance and exposes only `/Neri_Data/Model` through the
existing allowlisted manifest protocol. Users do not enter OpenList credentials.
Validated Microsoft HTTPS links are preferred; the server proxy is the fallback.

## Server configuration

- Code: `/opt/neri-models/server/model_distribution`
- Unit: `neri-models.service`, enabled on boot, dedicated `neri-models` user
- Listen address: `127.0.0.1:8002`
- Private environment: `/etc/neri-models.env`, root-owned, mode `0600`
- State and hash cache: `/var/lib/neri-models`, service-owned, mode `0700`
- OpenList: `http://127.0.0.1:5244`, using the existing server-held credential
- Public health: `https://myneri.top/api/models/health`
- Nginx site backup: `/var/backups/neri-models-20260907-164720/myneri.top`

The `/api/models/` location proxies to port 8002 with access logging disabled.
Nginx configuration validation passed before reload. Existing training upload,
OpenList, and website API services remained active.

## Verification

The initial manifest contained two detection models, no classification models,
and `tracker.yaml`. Cold hash generation took 84.32 seconds; the next manifest
request took 5.21 seconds. The client now gives manifest generation up to 900
seconds, matching nginx, while other requests retain their existing timeout.

Loopback verification downloaded `tracker.yaml` through both OneDrive and the
server proxy and matched the manifest byte size and SHA-256. A `bytes=10-19`
proxy request returned exactly 10 bytes with status 206 and the correct
`Content-Range`. Public model, training, and website API health checks returned
HTTP 200 after the route was enabled.

The Windows desktop client then completed a full synchronization into an isolated
temporary model directory: all three files used OneDrive directly, with no proxy
requests. A second synchronization downloaded no files. A separate run forced the
direct connection to fail for `tracker.yaml`; the manager successfully used the
public server proxy and verified the result. Forty automated distribution,
client, manager, and local API tests passed, including anonymous access without
credential disclosure, incomplete listings, short streams, and range handling.

## Rollback

Stop and disable `neri-models.service`, remove only the `/api/models/` location,
run `nginx -t`, then reload nginx. The backup captures the predeployment site;
compare it before restoring so later website changes are not overwritten.
Existing local models remain available when synchronization cannot reach the
service. No OpenList credential is stored in this repository or sent to clients.

## DINOv2 Seq publication — 2026-10-14

Published the complete seven-file DINOv2 ViT-B/14 component to
`/Neri_Data/Model/DINOv2`. The active classifier is Seq Memory Val90, displayed
as `DINOv2 Seq`, with 42 classes and 3,311 memory exemplars. Classifier SHA-256:
`b469920e78e5bae3c8751c9ae49cfbb2c1ea50e970f41de3aa23557d93e7a180`.
Encoder SHA-256:
`d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`.

All files were uploaded to a temporary cloud directory and downloaded back for
byte-size and SHA-256 verification before renaming it to `DINOv2`. The installed
model service previously returned 404 for the DINOv2 protocol. Backed up its
`app.py`, `manifest.py`, and `service.py` to
`/var/backups/neri-models-seq-20261014`, deployed the repository versions, and
restarted only `neri-models.service`. Existing nginx and credential configuration
were retained. Model, training, and NeriCloud health checks returned HTTP 200.

The public DINOv2 manifest is
`https://myneri.top/api/models/v1/dinov2/manifest`; direct capabilities use
`POST /api/models/v1/dinov2/direct`, with the existing proxy endpoint as fallback.
DINOv2 installation is separate from the ordinary detection/classification sync.

Fixed the desktop download progress contract: report cumulative percentages and
text messages to the maintenance worker, including when a direct attempt retries
through the proxy. Previously it received per-file byte counts and a numeric
message. The installed desktop downloader was backed up under
`D:/Application/Neri/repair-backups/seq-cloud-20261014` before updating that file.
The targeted suite passed 40 tests and 18 subtests.

The installed Windows Neri Python runtime then updated an isolated, healthy
76-class Memory installation to the published 42-class Seq model. All seven
files downloaded directly from OneDrive, matched the published inventory, and
were activated by the real component installer. A deliberately unavailable
service preserved the working old installation. A separate forced direct-transfer
failure automatically fell back to the public proxy and verified the full
classifier hash. The downloaded encoder and classifier loaded offline on CPU
and completed one synthetic-image inference; this was a runtime smoke check,
not an accuracy evaluation. The model catalog exposed the result as `DINOv2 Seq`.

Detailed evidence is saved locally in
`temp/seq-memory-cloud-20261014/verification.json`. The user requested backend
verification only. No GUI click test or frontend update was performed; a healthy
component currently exposes only the Delete action in Environment Maintenance.
