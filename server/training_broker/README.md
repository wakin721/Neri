# Neri OneDrive training uploads

Photos, YOLO `.txt` labels and `.classes.txt` class files travel from the desktop directly to Microsoft OneDrive. This service receives only small authorization and status requests. Videos, original file paths, EXIF and free-text notes are excluded by the desktop contribution pipeline. Storage is fixed to OpenList `/Neri_Data/Training`, which maps to OneDrive `/NeriCloud/Neri_Data/Training` on this deployment. New files are stored under `<root>/<device_id>/<species>/<sample>-r<revision>-<attempt>.*`.

Each installation generates a private random device secret. Its SHA-256 prefix (32 lowercase hexadecimal characters) is the public device ID shown and copied by the privacy settings UI; no hardware identifier is used. Each manually confirmed sample also has a random sample ID and separate private per-sample secret. The service verifies the device secret against its public ID and binds each sample to one device, creates a device directory and species directories beneath it, durably records the three file destinations, then returns Microsoft upload sessions. Public IDs alone cannot create capabilities for another device. Never include the OpenList token in desktop builds. Never log request bodies, device/sample secrets, session URLs or cancellation tokens.

The public ID helps the operator locate a deletion request. It does not authenticate the requester or permit a public bulk-delete operation. The broker journals `samples.device_id` with an index, so authorized operators can find associated samples, including incomplete attempts. Process verified deletion requests through the recorded targets and preserve journal consistency, rather than deleting only a folder while leaving live upload capabilities. Users should stop participation before requesting deletion. Requests must never require users to send their private device/sample secrets by email.

The identity is stored in the desktop's existing training-state SQLite database, independently of consent and queue clearing. Clearing that application data creates a different identity; users need to retain old public IDs when relevant. Old completed uploads that were never associated with an ID cannot automatically be attributed to a newly generated installation ID.

## Device-directory protocol upgrade

Health schema version is now **3**. New submission requests require `device_id` and `device_secret`; successful receipts echo the public ID, and the desktop refuses to send file bytes if it is missing or differs. Update the broker and desktop together; an older counterpart fails safely instead of uploading to a flat species directory. Existing completion, cancellation and per-sample deletion requests retain their prior format.

The broker adds `samples.device_id` non-destructively to an older journal. Existing target paths remain exactly as recorded, allowing old submissions to finish, expire or be deleted. A legacy sample can be associated with an installation only when a new submission request proves possession of its existing private sample secret. New transfers use the device directory; historical remote files are not automatically moved. Back up the broker database before deployment and preserve it for rollback/cleanup. This repository change does not itself deploy the service or create folders in the live drive.

OneDrive paths are case-insensitive, so case-folding and Unicode-normalization collisions within a sample are rejected. Old completed versions remain until the new triplet's actual sizes are confirmed. Retries revoke and remove incomplete attempts before making new paths; cancellation removes finished partial files too. An OS lock serializes operations across processes, including remote calls and explicit SQLite commits. Run one uvicorn worker.

## Direct-upload limits

OpenList v4.2.5 ignores the declared file size when it creates OneDrive upload sessions. Microsoft documents `fileSize` as a pre-upload quota check, not a hard upload ceiling. Consequently, the daily byte budgets here limit **declared session sizes**, and the server checks actual sizes after upload. They cannot prevent a modified client from temporarily consuming more OneDrive quota before verification or cleanup. This is a deliberate limitation of the requested direct path; no photo relay is enabled. Completed status requires exact recorded sizes; mismatches are removed. The service never gives clients access to arbitrary filenames or folders.

Incomplete attempts are revoked and removed after 20 minutes, checked every five minutes. Completed revisions are removed after 365 days, also checked every five minutes. Temporary Microsoft sessions created during a crash before their URL is recorded contain no client-uploaded bytes and expire at Microsoft. Deletions that fail remain in the journal and are retried; do not clear the journal to resolve a drive outage.

Default budgets: 60 control requests/minute/IP; declared 512 MiB/day/IP and 2 GiB/day total. These can be lowered via the environment. IPs are keyed by a server-local HMAC salt; request/body access logging is disabled. Protect the training folder through OpenList metadata ACLs including descendants; ordinary cloud users must not read it.

## Deployment

Use a dedicated `neri-training` account, code in `/opt/neri-training`, private state in `/var/lib/neri-training` (0700), and a root-readable `/etc/neri-training.env` (0600):

```
NERI_TRAINING_STATE=/var/lib/neri-training
NERI_OPENLIST_TOKEN=<insert the existing server-side OpenList token securely>
NERI_DAILY_IP_BYTES=536870912
NERI_DAILY_TOTAL_BYTES=2147483648
```

Create an isolated venv and install `requirements.txt`. Install `neri-training.service` and add the supplied nginx location inside the existing HTTPS server. Validate with `nginx -t` before reload. Keep port 8001 on loopback; `--no-proxy-headers` is intentional. Only the loopback nginx peer may supply X-Real-IP. Externally received X-Real-IP is overwritten by nginx.

Back up the specific nginx file and any existing Training metadata before changing them. Do not alter other website routes or OpenList mounts. The health endpoint is `https://myneri.top/api/training/health`. Test with a synthetic photo and JSON, then delete only the test sample using its private ownership secret. Deployment rollback stops/disables only `neri-training`, restores the nginx file and reloads nginx; preserve the journal and remote files for later cleanup.

The server does not inspect photo pixels because they bypass it. JPEG reencoding removes metadata but cannot guarantee removal of identities visible in the image. The agreement must explain this, name the operator/contact and disclose global OneDrive storage. Consent is default-off and independent from normal app settings.

## Local checks

```
python -m unittest tests.test_training_upload tests.test_training_privacy_api tests.test_training_broker tests.test_training_transport tests.test_empty_photo_deletion -q
```

The broker tests fake only the drive boundary. Live verification is additionally needed for deployed OpenList cache, ACL and Microsoft session behavior.
