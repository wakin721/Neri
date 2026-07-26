import 'dart:io';

import '../models/job.dart';

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
