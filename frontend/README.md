# Neri Flutter frontend

Neri's Windows desktop client uses a shared Material 3 Expressive (MD3E)
visual system in `lib/src/app_theme.dart`. Both the main application and the
standalone crash-report window use it. Theme mode, custom seed colors and
platform dynamic colors continue to use the existing saved preferences.

The Expressive adaptation uses Flutter's native Material widgets: emphasized
headings, tonal surfaces, rounded cards and selected list rows, press-state
button shapes, and the updated slider/progress geometry. The main processing
action adds spring press feedback while retaining native keyboard, focus and
accessibility behavior; it skips the spring when reduced motion is requested.
The fixed native navigation rail and IndexedStack are retained to preserve
Windows accessibility and page state. This is an application-level adaptation,
not an SDK-wide switch or a complete implementation of every M3E component.

Reference: [Material 3 Expressive](https://m3.material.io/).

Use the Flutter 3.44.x SDK used by the repository's CI:

```powershell
cd frontend
flutter pub get
flutter analyze
flutter test
flutter build windows --release
```
