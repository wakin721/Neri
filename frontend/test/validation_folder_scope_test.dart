import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/utils/local_detection_items.dart';

void main() {
  test('validation keeps only media directly inside the selected folder', () {
    final inputDirectory = Directory.systemTemp.createTempSync(
      'neri_validation_folder_scope_',
    );
    addTearDown(() => inputDirectory.deleteSync(recursive: true));

    final nestedDirectory = Directory(
      '${inputDirectory.path}${Platform.pathSeparator}nested',
    )..createSync();
    final siblingDirectory = Directory('${inputDirectory.path}-sibling')
      ..createSync();
    addTearDown(() => siblingDirectory.deleteSync(recursive: true));

    DetectionItem item(String filename, Directory directory) => DetectionItem(
      filename: filename,
      path: '${directory.path}${Platform.pathSeparator}$filename',
      fileType: 'jpg',
    );

    final directImage = item('direct.jpg', inputDirectory);
    final nestedImage = item('nested.jpg', nestedDirectory);
    final siblingImage = item('sibling.jpg', siblingDirectory);

    expect(
      validationItemsInInputFolder(<DetectionItem>[
        directImage,
        nestedImage,
        siblingImage,
      ], inputDirectory.path),
      equals(<DetectionItem>[directImage]),
    );
  });
}
