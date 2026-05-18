# Neri Flutter Frontend

Flutter Material 3 client for the Neri Python backend.

## Run

1. Start the backend from the repository root:

   ```bash
   python -m uvicorn system.backend.main:app --reload --app-dir . --host 127.0.0.1 --port 721
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
- Job creation form with a remembered input path, automatic `res/model` model folder, confidence, IOU, FP16, and detection toggles.
- Responsive Image Preview workspace that switches between wide two-pane and stacked layouts, with a file list, image display, metadata panel, detection-result switch, species confidence filter, confidence slider, and a current-image detection button.
