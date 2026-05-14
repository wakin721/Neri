# Neri Python Backend

This directory contains a FastAPI backend for the Flutter Material 3 client. It wraps the existing Neri processing modules instead of replacing them:

- `system.config` supplies the app version and supported image/video extensions.
- `system.metadata_extractor` indexes EXIF and image dimensions for fast folder previews.
- `system.image_processor` is loaded lazily only when a job enables YOLO detection.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.neri_backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> to inspect the API.

## API overview

- `GET /api/health` checks backend readiness.
- `GET /api/settings` returns Neri version, supported formats, and saved settings.
- `POST /api/jobs` starts a folder or single-file processing job. By default it performs fast metadata indexing. Set `options.enable_detection` to `true` to load YOLO and run image inference. When `options.model_path` is omitted or points at `res/model`, the backend chooses the preferred `.pt` model from that folder.
- `GET /api/jobs` lists known jobs for the current process.
- `GET /api/jobs/{job_id}` returns progress and results for one job.
