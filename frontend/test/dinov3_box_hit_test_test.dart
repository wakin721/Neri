import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer.dart';

void main() {
  late Directory tempDir;
  late File imageFile;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('neri_dinov3_hit_test_');
    imageFile = File('${tempDir.path}/frame.png');
    await imageFile.writeAsBytes(
      base64Decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl3Z1sAAAAASUVORK5CYII=',
      ),
    );
  });

  tearDown(() async {
    if (await tempDir.exists()) {
      await tempDir.delete(recursive: true);
    }
  });

  testWidgets('clicking a fitted DINO box selects its observation id', (
    tester,
  ) async {
    DetectionBox? selected;
    final box = DetectionBox(
      species: '盘羊',
      bbox: const <double>[0.2, 0.2, 0.8, 0.8],
      observationId: 'obs-1',
    );
    final item = DetectionItem(
      filename: 'frame.png',
      path: imageFile.path,
      fileType: 'png',
      detectionBoxes: <DetectionBox>[box],
    );

    await tester.pumpWidget(
      MaterialApp(
        home: Center(
          child: SizedBox(
            width: 400,
            height: 300,
            child: DetectionMediaViewer(
              item: item,
              visibleBoxes: <DetectionBox>[box],
              showDetections: true,
              selectedObservationId: null,
              onOpenExternal: () {},
              onDetectionBoxSelected: (value) => selected = value,
            ),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    await tester.tapAt(tester.getCenter(find.byType(DetectionMediaViewer)));
    await tester.pump();

    expect(selected?.observationId, 'obs-1');
  });
}
