import 'dart:math' as math;

import 'package:flutter/painting.dart';

int previewDecodeWidth(Size media, Size viewport, double pixelRatio) {
  final fitted = applyBoxFit(BoxFit.contain, media, viewport).destination;
  final width = fitted.width * pixelRatio;
  // Reuse decoded images during small window resizes, without upscaling.
  return math.max(1, math.min(media.width.ceil(), (width / 64).ceil() * 64));
}
