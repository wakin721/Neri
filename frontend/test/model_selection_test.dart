import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';
import 'package:neri_flutter/src/utils/detection_species.dart';
import 'package:neri_flutter/src/utils/local_detection_items.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer.dart';

void main() {
  testWidgets('探测模型可以选择不使用', (tester) async {
    final inputController = TextEditingController();
    addTearDown(inputController.dispose);
    String? selectedModelPath;

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: StartScreen(
            settings: const NeriSettings(
              appTitle: 'Neri',
              appVersion: 'test',
              supportedImageExtensions: <String>['.jpg'],
              supportedVideoExtensions: <String>['.mp4'],
              modelDirectory: 'res/model',
              classificationModelDirectory: 'res/model_cls',
              availableModels: <ModelInfo>[
                ModelInfo(name: 'detector.pt', path: 'res/model/detector.pt'),
              ],
              availableClassificationModels: <ModelInfo>[
                ModelInfo(
                  name: 'classifier.pt',
                  path: 'res/model_cls/classifier.pt',
                ),
              ],
              speciesTypes: <String, String>{},
              settings: <String, dynamic>{},
              gpuAvailable: false,
              missingYoloDependencies: <String>[],
            ),
            inputController: inputController,
            selectedModelPath: '',
            onModelChanged: (value) => selectedModelPath = value,
            selectedClassificationModelPath: 'res/model_cls/classifier.pt',
            onClassificationModelChanged: (_) {},
            videoMode: 'all',
            onVideoModeChanged: (_) {},
            vidStride: 1,
            onVidStrideChanged: (_) {},
            useFp16: false,
            onUseFp16Changed: (_) {},
            confidence: 0.25,
            onConfidenceChanged: (_) {},
            iou: 0.30,
            onIouChanged: (_) {},
            submitting: false,
            onCreateJob: () {},
            onCancelJob: (_) {},
            onResumeJob: (_) {},
            onDeleteJob: (_) {},
            onClearJobs: () {},
            pendingStartJobIds: const <String>{},
            pendingStopJobIds: const <String>{},
            jobs: const [],
          ),
        ),
      ),
    );

    final dropdowns = tester
        .widgetList<DropdownMenu<String>>(find.byType(DropdownMenu<String>))
        .toList();
    final detectionModelDropdown = dropdowns.first;

    expect(detectionModelDropdown.initialSelection, '');
    expect(detectionModelDropdown.dropdownMenuEntries.first.value, '');
    expect(detectionModelDropdown.dropdownMenuEntries.first.label, '不使用');

    detectionModelDropdown.onSelected?.call('');
    expect(selectedModelPath, '');
  });

  test('候选物种选项包含纯分类和检测框的 Top2/Top3', () {
    const item = DetectionItem(
      filename: 'image.jpg',
      path: 'image.jpg',
      fileType: 'jpg',
      species: <String>['top1'],
      detectionBoxes: <DetectionBox>[
        DetectionBox(
          species: 'box-top1',
          bbox: <double>[0, 0, 10, 10],
          candidates: <Map<String, dynamic>>[
            <String, dynamic>{'name': 'box-top1', 'conf': 0.9},
            <String, dynamic>{'name': 'box-top2', 'conf': 0.7},
          ],
        ),
      ],
      detectionData: <String, dynamic>{
        '分类候选项': <Map<String, dynamic>>[
          <String, dynamic>{'name': 'top1', 'conf': 0.91},
          <String, dynamic>{'name': 'top2', 'conf': 0.72},
          <String, dynamic>{'name': 'top3', 'conf': 0.40},
        ],
      },
    );

    expect(detectionSpeciesOptions(item, globalOption: 'global'), <String>[
      'global',
      'top1',
      'box-top1',
      'box-top2',
      'top2',
      'top3',
    ]);
  });

  test('任务结果合并前会过滤已删除的本地文件', () async {
    final directory = await Directory.systemTemp.createTemp(
      'neri-existing-items-',
    );
    addTearDown(() => directory.delete(recursive: true));
    final existingFile = File('${directory.path}/existing.jpg');
    await existingFile.writeAsBytes(<int>[1]);
    final deletedFile = File('${directory.path}/deleted.jpg');

    final items = await existingLocalDetectionItems(<DetectionItem>[
      DetectionItem(
        filename: 'existing.jpg',
        path: existingFile.path,
        fileType: 'jpg',
      ),
      DetectionItem(
        filename: 'deleted.jpg',
        path: deletedFile.path,
        fileType: 'jpg',
      ),
    ]);

    expect(items.map((item) => item.path), <String>[existingFile.path]);
  });

  testWidgets('图片在显示前被删除时使用占位状态且不抛异常', (tester) async {
    final missingPath =
        '${Directory.systemTemp.path}/neri-missing-image-test.jpg';

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SizedBox(
            width: 400,
            height: 300,
            child: DetectionMediaViewer(
              item: DetectionItem(
                filename: 'neri-missing-image-test.jpg',
                path: missingPath,
                fileType: 'jpg',
              ),
              visibleBoxes: const <DetectionBox>[],
              showDetections: true,
              onOpenExternal: () {},
            ),
          ),
        ),
      ),
    );
    await tester.pump();

    expect(find.text('图片已被删除或移动'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
