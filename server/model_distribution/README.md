# Neri Model Distribution

This service publishes the fixed NeriCloud model tree at `/Neri_Data/Model` to Neri desktop clients without exposing OpenList credentials. It serves a content-addressed manifest, issues short-lived direct OneDrive capabilities when the upstream link is safe, and provides a bounded Neri proxy fallback when direct transfer is unavailable.

## Fixed model tree

The service only exposes these logical paths:

- `detect/*.pt`
- `cls/*.pt`, `cls/*.onnx`, `cls/*.engine`
- `tracker.yaml`

No client-supplied OpenList root or arbitrary remote path is accepted.

## Environment

Create `/etc/neri-models.env` with secure values. Do not commit the file or any token values.

```text
NERI_MODEL_STATE=/var/lib/neri-models
NERI_OPENLIST_URL=http://127.0.0.1:5244
NERI_OPENLIST_TOKEN=<set securely on server>
NERI_MODEL_PUBLIC_URL=https://myneri.top/api/models
NERI_MODEL_CAPABILITY_TTL=600
NERI_MODEL_REQUESTS_PER_MINUTE=120
NERI_MODEL_PROXY_IP_BYTES=2147483648
NERI_MODEL_PROXY_TOTAL_BYTES=21474836480
```

`NERI_OPENLIST_TOKEN` is required. The remaining variables have defaults matching the values above.

## Deployment

A typical deployment keeps the repository package at `/opt/neri-models`, creates a virtual environment there, installs `server/model_distribution/requirements.txt`, and runs the supplied `neri-models.service`. The service listens only on `127.0.0.1:8002` and is exposed through the supplied nginx location.

```bash
python3 -m venv /opt/neri-models/venv
/opt/neri-models/venv/bin/pip install -r /opt/neri-models/server/model_distribution/requirements.txt
sudo install -m 0644 server/model_distribution/neri-models.service /etc/systemd/system/neri-models.service
sudo systemctl daemon-reload
sudo systemctl enable --now neri-models.service
```

Install `nginx-location.conf` inside the existing HTTPS `server` block for `myneri.top`, then run `nginx -t` before reloading nginx. The nginx route passes the real client IP through `X-Real-IP`; the application trusts that header only when the immediate peer is loopback.

## Endpoints

With the supplied nginx prefix, the public health endpoint is:

```text
https://myneri.top/api/models/health
```

The desktop protocol uses:

```text
GET  /api/models/v1/manifest
POST /api/models/v1/direct
GET  /api/models/v1/proxy/<opaque capability>
```

The manifest contains only logical path, byte size, SHA-256 and a stable manifest identifier. Direct links are returned only for validated Microsoft HTTPS hosts. OpenList authorization headers and cookies are never returned to the client.

Proxy capabilities expire, have bounded use counts, and are subject to per-minute request limits plus per-IP and global daily byte budgets. Access logging is disabled for the model distribution nginx location so bearer capabilities do not enter standard access logs.

## Traffic model

Normal downloads should go directly from OneDrive to the Neri desktop. The Neri server normally handles manifest construction, hashing/cache metadata and capability issuance. Proxy traffic is the fallback only when a usable OneDrive direct transfer is unavailable or fails client-side.

## Rollback

To stop model distribution without affecting the desktop application's existing local models:

```bash
sudo systemctl disable --now neri-models.service
```

Remove or disable the `/api/models/` nginx location and reload nginx. Neri desktop synchronization is non-fatal: if both direct and proxy downloads are unavailable, last-known-good local synchronized models remain usable.
