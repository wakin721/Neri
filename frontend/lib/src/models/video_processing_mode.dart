const videoProcessingModeAll = 'all';
const videoProcessingModeFast = 'fast';
const videoProcessingModeSkip = 'skip';
const defaultVideoProcessingMode = videoProcessingModeFast;
const defaultVideoSampleCount = 3;

String normalizeVideoProcessingMode(String? value) {
  return switch (value) {
    videoProcessingModeAll => videoProcessingModeAll,
    videoProcessingModeFast => videoProcessingModeFast,
    videoProcessingModeSkip => videoProcessingModeSkip,
    _ => defaultVideoProcessingMode,
  };
}

bool videoProcessingEnabled(String? value) {
  return normalizeVideoProcessingMode(value) != videoProcessingModeSkip;
}
