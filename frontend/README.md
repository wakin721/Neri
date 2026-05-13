# Neri Flutter Frontend

Flutter Material 3 client for the Neri Python backend.

## Run

1. Start the backend from the repository root:

   ```bash
   python -m uvicorn backend.neri_backend.main:app --reload --host 127.0.0.1 --port 8000
   ```

2. Start Flutter:

   ```bash
   cd frontend
   flutter pub get
   flutter run
   ```

## What is implemented

- Material 3 light/dark themes generated from a wildlife-green seed color.
- Left-side `NavigationRail` for Start, Image Preview, Species Validation, Settings, and About sections.
- Backend settings and supported media formats display.
- Job creation form with input/output paths, model path, confidence, IOU, FP16, and detection toggles.
- Image Preview workspace with a left-side file list, image display, metadata panel, detection-result switch, species confidence filter, confidence slider, and a current-image detection button.
