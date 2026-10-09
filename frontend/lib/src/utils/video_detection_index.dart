import '../models/job.dart';

typedef IndexedDetectionBox = ({DetectionBox box, int originalIndex});

/// Indexed time-window lookup, retaining original order for track tie-breaking.
class VideoDetectionIndex {
  VideoDetectionIndex(this.boxes) {
    for (var i = 0; i < boxes.length; i++) {
      final box = boxes[i];
      final entry = (box: box, originalIndex: i);
      if (box.frameIndex != null) {
        _frames.add(entry);
      } else if (box.timestamp != null) {
        _times.add(entry);
      }
    }
    _frames.sort((a, b) => a.box.frameIndex!.compareTo(b.box.frameIndex!));
    _times.sort((a, b) => a.box.timestamp!.compareTo(b.box.timestamp!));
  }

  final List<DetectionBox> boxes;
  final List<IndexedDetectionBox> _frames = [];
  final List<IndexedDetectionBox> _times = [];

  int? get maxFrame => _frames.isEmpty ? null : _frames.last.box.frameIndex;
  bool get hasTime => _frames.isNotEmpty || _times.isNotEmpty;

  List<IndexedDetectionBox> candidates({
    required List<int> frames,
    required int frameTolerance,
    required double seconds,
    required double timeTolerance,
  }) {
    final found = <int, IndexedDetectionBox>{};
    void collect(
      List<IndexedDetectionBox> entries,
      num low,
      num high,
      num Function(DetectionBox) timeOf,
    ) {
      var start = 0;
      var end = entries.length;
      while (start < end) {
        final mid = (start + end) ~/ 2;
        if (timeOf(entries[mid].box) < low) {
          start = mid + 1;
        } else {
          end = mid;
        }
      }
      for (
        var i = start;
        i < entries.length && timeOf(entries[i].box) <= high;
        i++
      ) {
        final entry = entries[i];
        found[entry.originalIndex] = entry;
      }
    }

    for (final frame in frames) {
      collect(
        _frames,
        frame - frameTolerance,
        frame + frameTolerance,
        (box) => box.frameIndex!,
      );
    }
    collect(
      _times,
      seconds - timeTolerance,
      seconds + timeTolerance,
      (box) => box.timestamp!,
    );
    return found.values.toList()
      ..sort((a, b) => a.originalIndex.compareTo(b.originalIndex));
  }
}
