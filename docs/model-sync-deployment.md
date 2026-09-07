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
