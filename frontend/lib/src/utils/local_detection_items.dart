import 'dart:io';

import '../models/job.dart';

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
  final existence = await Future.wait(
    candidates.map((item) => File(item.path).exists()),
  );
  return [
    for (var index = 0; index < candidates.length; index++)
      if (existence[index]) candidates[index],
  ];
}
