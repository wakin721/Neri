import 'package:flutter/foundation.dart';

import '../models/job.dart';

/// Large snapshots are ordered outside the UI isolate. Each comparison uses
/// precomputed keys instead of reparsing dates and filenames millions of times.
Future<List<DetectionItem>> sortMediaItemsForDisplayAsync(
  List<DetectionItem> items,
) async => items.length < 2048
    ? sortMediaItemsForDisplay(items)
    : compute(sortMediaItemsForDisplay, items);

List<DetectionItem> sortMediaItemsForDisplay(List<DetectionItem> items) {
  final keyed = [for (final item in items) _MediaSortEntry(item)];
  keyed.sort((a, b) => a.compareTo(b));
  return [for (final entry in keyed) entry.item];
}

/// The rest of the snapshot is already sorted. Reposition only the changed
/// photo, retaining the other objects and avoiding a full-root sort.
List<DetectionItem> replaceSortedMediaItem(
  List<DetectionItem> sorted,
  int index,
  DetectionItem updated,
) {
  if (sorted[index].path != updated.path) {
    throw ArgumentError('Metadata must belong to the same photo');
  }
  final result = List<DetectionItem>.from(sorted);
  result.removeAt(index);
  final key = _MediaSortEntry(updated);
  var low = 0;
  var high = result.length;
  while (low < high) {
    final mid = (low + high) ~/ 2;
    final comparison = _MediaSortEntry(result[mid]).compareTo(key);
    // Retain the original relative position among equal display keys.
    if (comparison < 0 || (comparison == 0 && mid < index)) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  result.insert(low, updated);
  return result;
}

List<DetectionItem> replaceSortedMediaItems(
  List<DetectionItem> currentItems,
  Iterable<DetectionItem> updates,
) {
  final updateByPath = <String, DetectionItem>{
    for (final item in updates)
      if (item.path.isNotEmpty) item.path: item,
  };
  if (updateByPath.isEmpty) return currentItems;

  var orderingChanged = false;
  for (final item in currentItems) {
    final replacement = updateByPath[item.path];
    if (replacement != null &&
        (mediaSortTimestamp(item) != mediaSortTimestamp(replacement) ||
            item.filename != replacement.filename)) {
      orderingChanged = true;
      break;
    }
  }
  final seen = currentItems.map((item) => item.path).toSet();
  final nextItems = <DetectionItem>[
    for (final item in currentItems) updateByPath[item.path] ?? item,
  ];
  for (final item in updateByPath.values) {
    if (seen.contains(item.path)) continue;
    nextItems.add(item);
    orderingChanged = true;
  }
  return orderingChanged ? sortMediaItemsForDisplay(nextItems) : nextItems;
}

DateTime? mediaSortTimestamp(DetectionItem item) {
  final date = item.dateTaken?.trim();
  if (date != null && date.isNotEmpty) {
    final time = item.detectionData['拍摄时间']?.toString().trim();
    if (date.contains(':') || date.contains('T')) {
      final timestamp = DateTime.tryParse(date);
      if (timestamp != null) return timestamp;
    }
    if (time != null && time.isNotEmpty) {
      final timestamp = DateTime.tryParse('$date $time');
      if (timestamp != null) return timestamp;
    }
  }
  final modified = item.modifiedAt?.trim();
  return modified == null || modified.isEmpty
      ? null
      : DateTime.tryParse(modified);
}

class _MediaSortEntry {
  _MediaSortEntry(this.item)
    : timestamp = mediaSortTimestamp(item),
      name = _naturalSegments(item.filename);

  final DetectionItem item;
  final DateTime? timestamp;
  final List<(String, int?)> name;
  List<(String, int?)>? _path;

  int compareTo(_MediaSortEntry other) {
    final left = timestamp;
    final right = other.timestamp;
    if (left != null && right != null) {
      final comparison = left.compareTo(right);
      if (comparison != 0) return comparison;
    } else if (left != null) {
      return -1;
    } else if (right != null) {
      return 1;
    }
    final comparison = _compareNatural(name, other.name);
    if (comparison != 0) return comparison;
    return _compareNatural(
      _path ??= _naturalSegments(item.path),
      other._path ??= _naturalSegments(other.item.path),
    );
  }
}

final _naturalPattern = RegExp(r'\d+|\D+');

List<(String, int?)> _naturalSegments(String value) => [
  for (final match in _naturalPattern.allMatches(value))
    (match[0]!.toLowerCase(), int.tryParse(match[0]!)),
];

int _compareNatural(List<(String, int?)> a, List<(String, int?)> b) {
  final length = a.length < b.length ? a.length : b.length;
  for (var i = 0; i < length; i++) {
    final left = a[i];
    final right = b[i];
    final comparison = left.$2 != null && right.$2 != null
        ? left.$2!.compareTo(right.$2!)
        : left.$1.compareTo(right.$1);
    if (comparison != 0) return comparison;
  }
  return a.length.compareTo(b.length);
}
