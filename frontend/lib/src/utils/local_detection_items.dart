import 'dart:io';

import '../models/job.dart';

/// Callers replace lists/sets when their contents change.
class ValidationItemsCache {
  List<DetectionItem>? _source;
  String? _inputPath;
  Set<String>? _paths;
  List<DetectionItem> _items = const <DetectionItem>[];

  List<DetectionItem> itemsFor(
    List<DetectionItem> source,
    String inputPath,
    Set<String> paths,
  ) {
    if (identical(source, _source) &&
        inputPath == _inputPath &&
        identical(paths, _paths)) {
      return _items;
    }
    final scoped = validationItemsInInputFolder(source, inputPath);
    _items = paths.isEmpty
        ? scoped
        : scoped
              .where((item) => paths.contains(_localPathKey(item.path)))
              .toList();
    _source = source;
    _inputPath = inputPath;
    _paths = paths;
    return _items;
  }
}

List<DetectionItem> validationItemsInInputFolder(
  Iterable<DetectionItem> items,
  String inputPath,
) {
  final trimmedInputPath = inputPath.trim();
  if (trimmedInputPath.isEmpty) return const <DetectionItem>[];

  final inputDirectory = _localPathKey(
    Directory(trimmedInputPath).absolute.path,
  );
  return <DetectionItem>[
    for (final item in items)
      if (item.path.trim().isNotEmpty &&
          _localPathKey(File(item.path).absolute.parent.path) == inputDirectory)
        item,
  ];
}

String _localPathKey(String path) {
  final normalized = path.replaceAll('\\', '/').replaceAll(RegExp(r'/+$'), '');
  return Platform.isWindows ? normalized.toLowerCase() : normalized;
}

Future<List<DetectionItem>> existingLocalDetectionItems(
  Iterable<DetectionItem> items,
) async {
  final candidates = [
    for (final item in items)
      if (item.path.trim().isNotEmpty) item,
  ];
  final existing = <DetectionItem>[];
  // Bound filesystem requests when a processing job has thousands of results.
  for (var start = 0; start < candidates.length; start += 32) {
    final end = (start + 32).clamp(0, candidates.length).toInt();
    final batch = candidates.sublist(start, end);
    final existence = await Future.wait(
      batch.map((item) => File(item.path).exists()),
    );
    for (var index = 0; index < batch.length; index++) {
      if (existence[index]) existing.add(batch[index]);
    }
  }
  return existing;
}
