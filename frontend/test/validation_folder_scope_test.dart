import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/utils/local_detection_items.dart';

void main() {
  test('validation includes descendants but excludes sibling folders', () {
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

    final items = <DetectionItem>[directImage, nestedImage, siblingImage];
    expect(
      validationItemsInInputFolder(items, inputDirectory.path),
      equals(<DetectionItem>[directImage, nestedImage]),
    );
    expect(
      validationItemsInInputFolder(items, nestedDirectory.path),
      equals(<DetectionItem>[nestedImage]),
    );
  });
}
