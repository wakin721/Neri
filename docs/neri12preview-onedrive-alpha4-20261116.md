# Neri1.2preview OneDrive downloads in alpha4

Both CPU (LiteRT FP32) and GPU (PyTorch) use the same direct-download policy as ordinary NeriCloud models: prefer an allowlisted Microsoft HTTPS download URL, validate downloaded length and SHA-256, and fall back to the server proxy if needed. Validated 1 MiB cache checkpoints remain compatible with earlier installations and can resume over OneDrive.

The cloud service retains its existing CPU/GPU manifests and local release files. `NERI_DINOV3_GPU_DRIVE_ROOT` and `NERI_DINOV3_CPU_DRIVE_ROOT` map only direct-link resolution to immutable OneDrive mirror directories under `/Neri_Data/Model/DINOv3-mirrors`. Proxy capabilities continue reading local verified files, so OneDrive availability does not remove the fallback. Model weights remain outside GitHub and portable application packages.

`scripts/publish_dinov3_onedrive.py --start --bundle <verified-service-bundle>` runs as an administrator on the server. It verifies the existing sources, uploads or reuses matching OneDrive assets, verifies mirrored content, and activates both mappings after successful publication. Existing downloads remain supported while publication runs. Credentials stay on the server; signed URLs are excluded from progress logs. `--rollback /var/backups/neri-onedrive-<timestamp>` restores the previous service configuration.

Alpha4 also keeps a successfully activated component installed when Windows cannot remove an old backup, records deferred cleanup, and uses a fresh backup name on subsequent installation. Backend recovery captures startup output in `logs/backend_restart.log` and confirms `/api/health` before reporting completion. The frontend refreshes settings and component state when a backend reconnects, even if no jobs changed. Automatic runtime explanations are visible only in debug mode; the installed runtime label remains visible normally.

Shared learning records, model identities, calibration, and source checksums remain unchanged.
