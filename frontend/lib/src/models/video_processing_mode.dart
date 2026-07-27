const videoProcessingModeAll = 'all';
const videoProcessingModeFast = 'fast';
const videoProcessingModeSkip = 'skip';

String normalizeVideoProcessingMode(String? value) {
  return switch (value) {
    videoProcessingModeFast => videoProcessingModeFast,
    videoProcessingModeSkip => videoProcessingModeSkip,
    _ => videoProcessingModeAll,
  };
}

bool videoProcessingEnabled(String? value) {
  return normalizeVideoProcessingMode(value) != videoProcessingModeSkip;
}
