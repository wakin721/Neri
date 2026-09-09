from pathlib import Path

path = Path('frontend/lib/src/widgets/detection_media_viewer.dart')
text = path.read_text(encoding='utf-8')

insert_anchor = """class _ValidationVideoPlayer extends StatefulWidget {\n"""
helper = r'''List<DetectionBox> currentVideoDetectionBoxes({
  required List<DetectionBox> boxes,
  required Duration position,
  required Duration duration,
  required Map<String, dynamic> detectionData,
}) {
  int? intData(String key) {
    final value = detectionData[key];
    if (value is int) return value;
    if (value is num) return value.round();
    return int.tryParse(value?.toString() ?? '');
  }

  int? totalFrameHint() {
    final processedFrames = intData('total_frames_processed');
    final stride = math.max(1, intData('vid_stride') ?? 1);
    final maxBoxFrame = boxes
        .map((box) => box.frameIndex)
        .whereType<int>()
        .fold<int?>(null, (maxFrame, frame) {
          if (maxFrame == null || frame > maxFrame) return frame;
          return maxFrame;
        });
    final processedHint = processedFrames == null
        ? null
        : math.max(1, processedFrames * stride);
    if (processedHint == null) {
      return maxBoxFrame == null ? null : maxBoxFrame + 1;
    }
    if (maxBoxFrame == null) return processedHint;
    return math.max(processedHint, maxBoxFrame + 1);
  }

  int? currentFrameIndexEstimate() {
    if (duration <= Duration.zero || position < Duration.zero) {
      return null;
    }
    final totalFrames = totalFrameHint();
    if (totalFrames == null || totalFrames <= 0) return null;
    final progress = (position.inMilliseconds / duration.inMilliseconds).clamp(
      0.0,
      1.0,
    );
    return (progress * totalFrames).round();
  }

  int frameTolerance() {
    final stride = math.max(1, intData('vid_stride') ?? 1);
    final totalFrames = totalFrameHint();
    if (duration > Duration.zero && totalFrames != null && totalFrames > 0) {
      final fps = totalFrames / duration.inMilliseconds * 1000;
      return math.max(stride, (fps * 0.25).ceil());
    }
    return math.max(stride, 6);
  }

  int trackSortIndex(DetectionBox box, int fallback) {
    final trackId = box.trackId;
    if (trackId == null || trackId.isEmpty) return fallback;
    return int.tryParse(trackId) ?? fallback;
  }

  _TimedBoxMatch? matchForBox(
    DetectionBox box,
    int index,
    int? currentFrame,
    int currentFrame25,
    int currentFrame30,
    int currentFrame60,
    double currentSeconds,
    int toleranceFrames,
    double toleranceSeconds,
  ) {
    if (box.frameIndex != null) {
      final distance = currentFrame == null
          ? [
              (box.frameIndex! - currentFrame25).abs(),
              (box.frameIndex! - currentFrame30).abs(),
              (box.frameIndex! - currentFrame60).abs(),
            ].reduce((a, b) => a < b ? a : b)
          : (box.frameIndex! - currentFrame).abs();
      if (distance <= toleranceFrames) {
        return _TimedBoxMatch(
          box: box,
          distance: distance.toDouble(),
          sortIndex: trackSortIndex(box, index),
        );
      }
      return null;
    }
    if (box.timestamp != null) {
      final distance = (box.timestamp! - currentSeconds).abs();
      if (distance <= toleranceSeconds) {
        return _TimedBoxMatch(
          box: box,
          distance: distance,
          sortIndex: trackSortIndex(box, index),
        );
      }
    }
    return null;
  }

  final currentMs = position.inMilliseconds;
  final currentSeconds = currentMs / 1000;
  final currentFrame = currentFrameIndexEstimate();
  final toleranceFrames = frameTolerance();
  const toleranceSeconds = 0.25;

  final currentFrame25 = (currentMs / 1000 * 25).round();
  final currentFrame30 = (currentMs / 1000 * 30).round();
  final currentFrame60 = (currentMs / 1000 * 60).round();

  final anyBoxHasTime = boxes.any(
    (box) => box.frameIndex != null || box.timestamp != null,
  );
  if (!anyBoxHasTime) return boxes;

  final selectedByTrack = <String, _TimedBoxMatch>{};
  final untrackedBoxes = <_TimedBoxMatch>[];
  for (var index = 0; index < boxes.length; index++) {
    final box = boxes[index];
    final match = matchForBox(
      box,
      index,
      currentFrame,
      currentFrame25,
      currentFrame30,
      currentFrame60,
      currentSeconds,
      toleranceFrames,
      toleranceSeconds,
    );
    if (match == null) continue;
    final trackId = box.trackId?.trim();
    if (trackId == null || trackId.isEmpty) {
      untrackedBoxes.add(match);
      continue;
    }
    final previous = selectedByTrack[trackId];
    if (previous == null || match.distance < previous.distance) {
      selectedByTrack[trackId] = match;
    }
  }

  final matches = <_TimedBoxMatch>[
    ...selectedByTrack.values,
    ...untrackedBoxes,
  ]..sort((a, b) => a.sortIndex.compareTo(b.sortIndex));
  return matches.map((match) => match.box).toList();
}

'''
if insert_anchor not in text:
    raise SystemExit('video player insert anchor not found')
text = text.replace(insert_anchor, helper + insert_anchor, 1)

start = text.find("  List<DetectionBox> _currentBoxes(Duration position) {\n")
end = text.find("  String _formatDuration(Duration d) {\n", start)
if start < 0 or end < 0:
    raise SystemExit('current-box method block not found')
replacement = r'''  List<DetectionBox> _currentBoxes(Duration position) {
    return currentVideoDetectionBoxes(
      boxes: widget.visibleBoxes,
      position: position,
      duration: _duration,
      detectionData: widget.detectionData,
    );
  }

'''
text = text[:start] + replacement + text[end:]
path.write_text(text, encoding='utf-8')
