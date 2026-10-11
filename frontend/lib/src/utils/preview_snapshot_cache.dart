import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;

import '../crash_reporter.dart';
import '../models/job.dart';
import 'media_display_order.dart';

const previewCacheLimitSettingKey = 'preview_cache_max_mb';
const defaultPreviewCacheLimitMb = 128;
const maximumPreviewCacheLimitMb = 16384;
const previewCacheMegabyte = 1024 * 1024;

int normalizePreviewCacheLimitMb(Object? value) {
  final parsed = value is int
      ? value
      : value is String
      ? int.tryParse(value)
      : value is num && value.isFinite && value == value.roundToDouble()
      ? value.toInt()
      : null;
  return parsed != null && parsed >= 0 && parsed <= maximumPreviewCacheLimitMb
      ? parsed
      : defaultPreviewCacheLimitMb;
}

class PreviewSnapshot {
  const PreviewSnapshot(this.items, this.directories);

  final List<DetectionItem> items;
  final List<String> directories;
}

/// Disposable scan snapshots, separate from preferences and detection databases.
class PreviewSnapshotCache {
  PreviewSnapshotCache({
    Directory? directory,
    this.maxEntries = 16,
    int maxBytes = defaultPreviewCacheLimitMb * previewCacheMegabyte,
  }) : _directory = directory,
       _maxBytes = maxBytes;

  final Directory? _directory;
  final int maxEntries;
  int _maxBytes;
  bool _limitApplied = false;
  int get maxBytes => _maxBytes;
  Future<void> _pendingWrite = Future<void>.value();

  String get _directoryPath =>
      _directory?.path ??
      p.join(CrashReporter.logsDirectory.parent.path, 'temp', 'preview_cache');

  Future<PreviewSnapshot?> read(String scope) async {
    // Atomic replacement lets readers use the last completed snapshot while a
    // large background scan is being saved for another directory.
    if (_maxBytes == 0) return null;
    final snapshot = await compute(_readSnapshot, (
      _directoryPath,
      _scopeKey(scope),
      _maxBytes,
    ));
    return _maxBytes == 0 ? null : snapshot;
  }

  Future<void> write(
    String scope,
    List<DetectionItem> items, {
    List<String> directories = const [],
  }) {
    if (_maxBytes == 0) return _pendingWrite;
    final directory = _directoryPath;
    final key = _scopeKey(scope);
    // Keep writes ordered so an older scan cannot replace a newer snapshot.
    return _pendingWrite = _pendingWrite.then((_) async {
      // A newly selected limit also applies to scans already waiting to save.
      if (_maxBytes == 0) return;
      await compute(_writeSnapshot, (
        directory: directory,
        scope: key,
        items: items,
        directories: directories,
        maxEntries: maxEntries,
        maxBytes: _maxBytes,
      ));
    });
  }

  Future<void> setMaxBytes(int bytes) {
    if (bytes < 0) throw ArgumentError.value(bytes, 'bytes');
    if (_limitApplied && bytes == _maxBytes) return _pendingWrite;
    _limitApplied = true;
    _maxBytes = bytes;
    final directory = _directoryPath;
    return _pendingWrite = _pendingWrite.then(
      (_) => compute(_trimSnapshots, (directory, maxEntries, _maxBytes)),
    );
  }

  Future<void> flush() => _pendingWrite;
}

String _scopeKey(String scope) {
  final absolute = p.normalize(p.absolute(scope));
  return Platform.isWindows ? absolute.toLowerCase() : absolute;
}

File _snapshotFile(String directory, String scope) =>
    File(p.join(directory, '${sha256.convert(utf8.encode(scope))}.json'));

Future<PreviewSnapshot?> _readSnapshot((String, String, int) request) async {
  final (directory, scope, maxBytes) = request;
  try {
    final file = _snapshotFile(directory, scope);
    if (await file.length() > maxBytes) return null;
    final data = jsonDecode(await file.readAsString());
    if (data is! Map<String, dynamic> ||
        data['version'] != 1 ||
        data['scope'] != scope ||
        data['items'] is! List ||
        data['directories'] is! List) {
      return null;
    }
    final items = (data['items'] as List)
        .cast<Map<String, dynamic>>()
        .map(DetectionItem.fromJson)
        .toList();
    return PreviewSnapshot(
      sortMediaItemsForDisplay(items),
      (data['directories'] as List).cast<String>().toList(),
    );
  } on FileSystemException {
    return null;
  } on FormatException {
    return null;
  } on TypeError {
    return null;
  }
}

typedef _WriteRequest = ({
  String directory,
  String scope,
  List<DetectionItem> items,
  List<String> directories,
  int maxEntries,
  int maxBytes,
});

Future<void> _writeSnapshot(_WriteRequest request) async {
  File? temporary;
  try {
    final directory = Directory(request.directory);
    await directory.create(recursive: true);
    final target = _snapshotFile(request.directory, request.scope);
    temporary = File(
      '${target.path}.${DateTime.now().microsecondsSinceEpoch}.tmp',
    );
    await temporary.writeAsString(
      jsonEncode({
        'version': 1,
        'scope': request.scope,
        'directories': request.directories,
        'items': request.items.map(_itemJson).toList(),
      }),
      flush: true,
    );
    await temporary.rename(target.path);
    await _trimSnapshots((
      request.directory,
      request.maxEntries,
      request.maxBytes,
    ));
  } on FileSystemException {
    // Read-only storage or cleanup failures must not interrupt browsing.
  } on JsonUnsupportedObjectError {
    // An unserializable snapshot is still usable in memory.
  } finally {
    try {
      if (temporary != null && await temporary.exists()) {
        await temporary.delete();
      }
    } on FileSystemException {
      // Temporary files can be removed by the existing cache cleanup action.
    }
  }
}

Future<void> _trimSnapshots((String, int, int) request) async {
  final (path, maxEntries, maxBytes) = request;
  try {
    final directory = Directory(path);
    if (!await directory.exists()) return;
    // Retain recent snapshots within the current disk budget.
    final files = <({File file, FileStat stat})>[];
    await for (final entry in directory.list()) {
      if (entry is File && entry.path.endsWith('.json')) {
        files.add((file: entry, stat: await entry.stat()));
      }
    }
    files.sort((a, b) => b.stat.modified.compareTo(a.stat.modified));
    var bytes = 0;
    var retained = 0;
    for (final entry in files) {
      if (retained >= maxEntries || bytes + entry.stat.size > maxBytes) {
        await entry.file.delete();
      } else {
        bytes += entry.stat.size;
        retained++;
      }
    }
  } on FileSystemException {
    // Cache cleanup is best effort when files are locked or storage is read-only.
  }
}

Map<String, dynamic> _itemJson(DetectionItem item) => {
  'filename': item.filename,
  'path': item.path,
  'file_type': item.fileType,
  'date_taken': item.dateTaken,
  'modified_at': item.modifiedAt,
  'width': item.width,
  'height': item.height,
  'size_bytes': item.sizeBytes,
  'species': item.species,
  'confidence': item.confidence,
  'detection_boxes': [
    for (final box in item.detectionBoxes)
      {
        'species': box.species,
        'bbox': box.bbox,
        'confidence': box.confidence,
        'frame_index': box.frameIndex,
        'timestamp': box.timestamp,
        'track_id': box.trackId,
        'observation_id': box.observationId,
        'registry_id': box.registryId,
        'predicted_species': box.predictedSpecies,
        'feedback_status': box.feedbackStatus,
        'candidates': box.candidates,
      },
  ],
  'detection_data': item.detectionData,
  'error': item.error,
  'validated': item.validated,
};
