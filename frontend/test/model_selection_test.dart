import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/models/video_processing_mode.dart';
import 'package:neri_flutter/src/screens/preview_screen.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';
import 'package:neri_flutter/src/utils/detection_species.dart';
import 'package:neri_flutter/src/utils/local_detection_items.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer.dart';

void main() {
  test('视频处理模式保留跳过视频并兼容旧值', () {
    expect(defaultVideoProcessingMode, videoProcessingModeFast);
    expect(defaultVideoSampleCount, 3);
    expect(
      normalizeVideoProcessingMode(videoProcessingModeSkip),
      videoProcessingModeSkip,
    );
    expect(normalizeVideoProcessingMode('unknown'), defaultVideoProcessingMode);
  });

  test('默认主题色为调色板中的珊瑚红', () {
    const settings = ThemeSettings();
    final coralOption = kSeedColorOptions.singleWhere(
      (option) => option.label == '珊瑚红',
    );

    expect(settings.seedColor, kDefaultSeedColor);
    expect(settings.seedColor, coralOption.color);
  });

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

  testWidgets('跳过视频时主界面禁用视频跳帧', (tester) async {
    final inputController = TextEditingController();
    addTearDown(inputController.dispose);

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
              availableModels: <ModelInfo>[],
              availableClassificationModels: <ModelInfo>[],
              speciesTypes: <String, String>{},
              settings: <String, dynamic>{},
              gpuAvailable: false,
              missingYoloDependencies: <String>[],
            ),
            inputController: inputController,
            selectedModelPath: '',
            onModelChanged: (_) {},
            selectedClassificationModelPath: '',
            onClassificationModelChanged: (_) {},
            videoMode: videoProcessingModeSkip,
            onVideoModeChanged: (_) {},
            vidStride: 5,
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

    final videoModeDropdown = tester
        .widgetList<DropdownMenu<String>>(find.byType(DropdownMenu<String>))
        .singleWhere(
          (dropdown) => dropdown.dropdownMenuEntries.any(
            (entry) => entry.value == videoProcessingModeSkip,
          ),
        );
    final strideDropdown = tester.widget<DropdownMenu<int>>(
      find.byType(DropdownMenu<int>),
    );

    expect(videoModeDropdown.initialSelection, videoProcessingModeSkip);
    expect(
      videoModeDropdown.dropdownMenuEntries
          .map((entry) => entry.label)
          .toList(),
      contains('跳过视频'),
    );
    expect(strideDropdown.enabled, isFalse);
    expect(strideDropdown.onSelected, isNull);
  });

  testWidgets('检测设置使用弹出菜单选择视频模式并隐藏跳帧设置', (tester) async {
    await tester.binding.setSurfaceSize(const Size(1200, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final apiClient = NeriApiClient(
      httpClient: MockClient((_) async => http.Response('{}', 200)),
    );
    final themeNotifier = ValueNotifier(const ThemeSettings());
    addTearDown(apiClient.close);
    addTearDown(themeNotifier.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SettingsScreen(
            settings: const NeriSettings(
              appTitle: 'Neri',
              appVersion: 'test',
              supportedImageExtensions: <String>['.jpg'],
              supportedVideoExtensions: <String>['.mp4'],
              modelDirectory: 'res/model',
              classificationModelDirectory: 'res/model_cls',
              availableModels: <ModelInfo>[],
              availableClassificationModels: <ModelInfo>[],
              speciesTypes: <String, String>{},
              settings: <String, dynamic>{
                'video_mode': videoProcessingModeSkip,
              },
              gpuAvailable: false,
              missingYoloDependencies: <String>[],
            ),
            autoGroupInferredBurstSize: null,
            apiClient: apiClient,
            themeNotifier: themeNotifier,
            onUpdateTheme: (_) {},
            closeBehavior: 'ask',
            onCloseBehaviorChanged: (_) {},
            onSaveSettings: (_) async {},
            onCheckForUpdates:
                ({required channel, required downloadSource}) async {},
            onShowMessage: (_) {},
          ),
        ),
      ),
    );
    await tester.pump();

    expect(find.text('视频处理模式'), findsOneWidget);
    expect(find.text('跳帧设置'), findsNothing);
    expect(find.byType(SegmentedButton<String>), findsNothing);
    expect(find.text('CPU 模式下批处理数和线程数固定为 1'), findsOneWidget);
    final cpuLockedSliders = tester
        .widgetList<Slider>(find.byType(Slider))
        .where((slider) => slider.value == 1 && slider.onChanged == null);
    expect(cpuLockedSliders, hasLength(2));

    final modeButton = find.widgetWithText(TextButton, '跳过视频');
    expect(modeButton, findsOneWidget);
    await tester.ensureVisible(modeButton);
    await tester.pumpAndSettle();
    await tester.tap(modeButton);
    await tester.pumpAndSettle();

    expect(find.text('全部识别'), findsWidgets);
    expect(find.text('快速识别'), findsWidgets);
    expect(find.text('跳过视频'), findsWidgets);
  });

  testWidgets('设置页有未保存修改时仍同步开始页的模型选择', (tester) async {
    await tester.binding.setSurfaceSize(const Size(1200, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final apiClient = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/models/classes') {
          return http.Response('[]', 200);
        }
        return http.Response('{}', 200);
      }),
    );
    final themeNotifier = ValueNotifier(const ThemeSettings());
    addTearDown(apiClient.close);
    addTearDown(themeNotifier.dispose);
    final savedDrafts = <Map<String, dynamic>>[];
    late StateSetter updateHost;
    var currentSettings = const NeriSettings(
      appTitle: 'Neri',
      appVersion: 'test',
      supportedImageExtensions: <String>['.jpg'],
      supportedVideoExtensions: <String>['.mp4'],
      modelDirectory: 'res/model',
      classificationModelDirectory: 'res/model_cls',
      availableModels: <ModelInfo>[
        ModelInfo(name: 'detector-a.pt', path: 'res/model/detector-a.pt'),
        ModelInfo(name: 'detector-b.pt', path: 'res/model/detector-b.pt'),
      ],
      availableClassificationModels: <ModelInfo>[],
      selectedModel: 'res/model/detector-a.pt',
      speciesTypes: <String, String>{},
      settings: <String, dynamic>{'selected_model': 'res/model/detector-a.pt'},
      gpuAvailable: false,
      missingYoloDependencies: <String>[],
    );

    await tester.pumpWidget(
      MaterialApp(
        home: StatefulBuilder(
          builder: (context, setState) {
            updateHost = setState;
            return Scaffold(
              body: SettingsScreen(
                settings: currentSettings,
                autoGroupInferredBurstSize: null,
                apiClient: apiClient,
                themeNotifier: themeNotifier,
                onUpdateTheme: (_) {},
                closeBehavior: 'ask',
                onCloseBehaviorChanged: (_) {},
                onSaveSettings: (settings) async {
                  savedDrafts.add(Map<String, dynamic>.from(settings));
                },
                onCheckForUpdates:
                    ({required channel, required downloadSource}) async {},
                onShowMessage: (_) {},
              ),
            );
          },
        ),
      ),
    );
    await tester.pump();

    expect(find.widgetWithText(TextButton, 'detector-a.pt'), findsOneWidget);
    final enabledSwitch = tester
        .widgetList<Switch>(find.byType(Switch))
        .firstWhere((control) => control.onChanged != null);
    enabledSwitch.onChanged?.call(!enabledSwitch.value);
    await tester.pump();

    updateHost(() {
      currentSettings = currentSettings.copyWith(
        selectedModel: 'res/model/detector-b.pt',
        settings: <String, dynamic>{
          ...currentSettings.settings,
          'selected_model': 'res/model/detector-b.pt',
        },
      );
    });
    await tester.pump();

    expect(find.widgetWithText(TextButton, 'detector-b.pt'), findsOneWidget);
    await tester.pump(const Duration(milliseconds: 900));
    await tester.pump();
    expect(savedDrafts, isNotEmpty);
    expect(savedDrafts.last['selected_model'], 'res/model/detector-b.pt');
  });

  testWidgets('仅双模型启用时显示综合置信度', (tester) async {
    await tester.binding.setSurfaceSize(const Size(1200, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    const item = DetectionItem(
      filename: 'missing.jpg',
      path: 'missing.jpg',
      fileType: 'jpg',
      species: <String>['灰雁'],
      confidence: 0.85,
    );

    Future<void> pumpPreview({required bool useCombinedConfidence}) {
      return tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PreviewScreen(
              inputPath: '.',
              items: const <DetectionItem>[item],
              selectedIndex: 0,
              selectedItem: item,
              speciesTypes: const <String, String>{},
              useCombinedConfidence: useCombinedConfidence,
              showDetections: true,
              onShowDetectionsChanged: (_) {},
              selectedSpeciesFilter: previewAllSpeciesLabel,
              onSpeciesFilterChanged: (_) {},
              confidenceThreshold: 0.25,
              onConfidenceThresholdChanged: (_) {},
              detecting: false,
              loading: false,
              onDetectCurrentImage: (_) {},
              onSelected: (_, __) {},
              onLoadMetadata: (_) async {},
              onRefresh: () async {},
              onOpenExternal: (_) {},
            ),
          ),
        ),
      );
    }

    await pumpPreview(useCombinedConfidence: true);
    expect(find.textContaining('综合置信度: 0.85'), findsOneWidget);

    await pumpPreview(useCombinedConfidence: false);
    expect(find.textContaining('综合置信度'), findsNothing);
    expect(find.textContaining('置信度: 0.85'), findsOneWidget);
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
