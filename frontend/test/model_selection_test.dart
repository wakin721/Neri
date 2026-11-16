import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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
import 'package:neri_flutter/src/utils/classification_runtime_mode.dart';
import 'package:neri_flutter/src/utils/validation_selection_modifiers.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer.dart';
import 'package:neri_flutter/src/widgets/app_menu_style.dart';
import 'package:neri_flutter/src/widgets/input_folder_field.dart';
import 'package:neri_flutter/src/widgets/selectable_list_card.dart';

void main() {
  test('bundled LiteRT species classifier keeps the DINO event contract', () {
    final model = ModelInfo.fromJson({
      'name': 'Neri1.2preview·LiteRT',
      'path': 'res/model/cls/bundled/Neri1.2preview-litert.neri.json',
      'source': 'bundled',
      'kind': 'cls',
      'backend': 'dinov3',
      'encoder_backend': 'litert',
      'requires_detector': true,
      'supports_video_fast': false,
      'supports_video_all': false,
    });
    expect(model.isLiteRt, isTrue);
    expect(model.isDinoClassifier, isTrue);
    expect(model.sourceLabel, '内置模型');
    expect(model.displayName, '内置模型 / Neri1.2preview·LiteRT');
    expect(model.selectorLabel, 'Neri1.2preview·LiteRT');
    expect(model.supportsVideoFast, isFalse);
    expect(model.supportsVideoAll, isFalse);
  });
  test('bundled LiteRT detector has a separate model source', () {
    final model = ModelInfo.fromJson({
      'name': 'MDV6-yolov10-c·LiteRT',
      'path': 'res/model/detect/bundled/MDV6-yolov10-c.tflite',
      'source': 'bundled',
      'backend': 'litert',
    });
    expect(model.isLiteRt, isTrue);
    expect(model.sourceLabel, '内置模型');
    expect(model.displayName, '内置模型 / MDV6-yolov10-c·LiteRT');
    expect(model.selectorLabel, 'MDV6-yolov10-c·LiteRT');
  });
  test('ModelInfo.selectorLabel 统一格式化且不重复拼接 LiteRT', () {
    const bundledDetector = ModelInfo(
      name: 'MDV6-yolov10-c·LiteRT',
      path: 'res/model/detect/bundled/MDV6-yolov10-c.tflite',
      source: 'bundled',
      backend: 'litert',
    );
    const bundledClassifier = ModelInfo(
      name: 'Neri1.2preview·LiteRT',
      path: 'res/model/cls/bundled/Neri1.2preview-litert.neri.json',
      source: 'bundled',
      kind: 'cls',
      backend: 'dinov3',
      encoderBackend: 'litert',
    );
    const userLiteRt = ModelInfo(
      name: 'camera.tflite',
      path: 'res/model/detect/user/camera.tflite',
      source: 'user',
      backend: 'litert',
    );
    const pytorchClassifier = ModelInfo(
      name: 'Neri1.2preview',
      path: 'res/model/cls/sync/Neri1.2preview.neri.json',
      source: 'sync',
      kind: 'cls',
      backend: 'dinov3',
      encoderBackend: 'pytorch',
    );

    expect(bundledDetector.selectorLabel, 'MDV6-yolov10-c·LiteRT');
    expect(bundledClassifier.selectorLabel, 'Neri1.2preview·LiteRT');
    expect(userLiteRt.selectorLabel, 'camera.tflite·LiteRT');
    expect(pytorchClassifier.selectorLabel, 'Neri1.2preview');
  });
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

  test('LiteRT only detection settings remain runnable without PyTorch', () {
    final settings = NeriSettings.fromJson({
      'available_models': [
        {
          'name': 'camera.tflite',
          'path': 'res/model/detect/sync/camera.tflite',
          'backend': 'litert',
          'supports_video_fast': true,
          'supports_video_all': false,
        },
      ],
      'missing_yolo_dependencies': ['torch', 'torchvision', 'ultralytics'],
      'pytorch_available': false,
      'litert_available': true,
      'detection_ready': true,
    });

    expect(settings.pytorchAvailable, isFalse);
    expect(settings.litertAvailable, isTrue);
    expect(settings.detectionReady, isTrue);
    expect(settings.availableModels.single.isLiteRt, isTrue);
    expect(settings.availableModels.single.supportsVideoAll, isFalse);
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
              modelDirectory: 'res/model/detect',
              classificationModelDirectory: 'res/model/cls',
              availableModels: <ModelInfo>[
                ModelInfo(
                  name: 'detector.pt',
                  path: 'res/model/detect/detector.pt',
                ),
              ],
              availableClassificationModels: <ModelInfo>[
                ModelInfo(
                  name: 'classifier.pt',
                  path: 'res/model/cls/classifier.pt',
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
            selectedClassificationModelPath: 'res/model/cls/classifier.pt',
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
        .widgetList<AppFormMenu<String>>(find.byType(AppFormMenu<String>))
        .toList();
    final detectionModelDropdown = dropdowns.first;

    expect(detectionModelDropdown.value, '');
    expect(detectionModelDropdown.options.first.value, '');
    expect(detectionModelDropdown.options.first.label, '不使用');

    detectionModelDropdown.onSelected('');
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
              modelDirectory: 'res/model/detect',
              classificationModelDirectory: 'res/model/cls',
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
        .widgetList<AppFormMenu<String>>(find.byType(AppFormMenu<String>))
        .singleWhere(
          (dropdown) => dropdown.options.any(
            (entry) => entry.value == videoProcessingModeSkip,
          ),
        );
    final strideDropdown = tester.widget<AppFormMenu<int>>(
      find.byType(AppFormMenu<int>),
    );

    expect(videoModeDropdown.value, videoProcessingModeSkip);
    expect(
      videoModeDropdown.options.map((entry) => entry.label).toList(),
      contains('跳过视频'),
    );
    expect(strideDropdown.enabled, isFalse);
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
              modelDirectory: 'res/model/detect',
              classificationModelDirectory: 'res/model/cls',
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
    expect(find.text('CPU 模式下批处理数锁定为自动（=1）；处理线程由软件自动安排'), findsOneWidget);
    expect(find.text('线程数'), findsNothing);
    final cpuLockedSliders = tester
        .widgetList<Slider>(find.byType(Slider))
        .where((slider) => slider.value == 0 && slider.onChanged == null);
    expect(cpuLockedSliders, hasLength(1));

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

  testWidgets('设置页长模型菜单不遮挡选择按钮', (tester) async {
    await tester.binding.setSurfaceSize(const Size(900, 400));
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
            settings: NeriSettings(
              appTitle: 'Neri',
              appVersion: 'test',
              supportedImageExtensions: const <String>['.jpg'],
              supportedVideoExtensions: const <String>['.mp4'],
              modelDirectory: 'res/model/detect',
              classificationModelDirectory: 'res/model/cls',
              availableModels: [
                for (var index = 0; index < 20; index++)
                  ModelInfo(
                    name: 'model-$index.pt',
                    path: 'res/model/detect/model-$index.pt',
                  ),
              ],
              availableClassificationModels: const <ModelInfo>[],
              speciesTypes: const <String, String>{},
              settings: const <String, dynamic>{},
              gpuAvailable: false,
              missingYoloDependencies: const <String>[],
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
    final trigger = find.widgetWithText(TextButton, '不使用').first;
    await tester.ensureVisible(trigger);
    await tester.pumpAndSettle();
    await tester.tap(trigger);
    await tester.pumpAndSettle();

    final menuItem = find.widgetWithText(MenuItemButton, 'model-0.pt');
    final menuScrollView = find
        .ancestor(of: menuItem, matching: find.byType(SingleChildScrollView))
        .first;
    expect(
      tester.getRect(menuScrollView).overlaps(tester.getRect(trigger)),
      isFalse,
    );
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
      modelDirectory: 'res/model/detect',
      classificationModelDirectory: 'res/model/cls',
      availableModels: <ModelInfo>[
        ModelInfo(
          name: 'detector-a.pt',
          path: 'res/model/detect/detector-a.pt',
        ),
        ModelInfo(
          name: 'detector-b.pt',
          path: 'res/model/detect/detector-b.pt',
        ),
      ],
      availableClassificationModels: <ModelInfo>[],
      selectedModel: 'res/model/detect/detector-a.pt',
      speciesTypes: <String, String>{},
      settings: <String, dynamic>{
        'selected_model': 'res/model/detect/detector-a.pt',
      },
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
        selectedModel: 'res/model/detect/detector-b.pt',
        settings: <String, dynamic>{
          ...currentSettings.settings,
          'selected_model': 'res/model/detect/detector-b.pt',
        },
      );
    });
    await tester.pump();

    expect(find.widgetWithText(TextButton, 'detector-b.pt'), findsOneWidget);
    await tester.pump(const Duration(milliseconds: 900));
    await tester.pump();
    expect(savedDrafts, isNotEmpty);
    expect(
      savedDrafts.last['selected_model'],
      'res/model/detect/detector-b.pt',
    );
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
    // Image dimensions/errors now come from asynchronous native file I/O.
    for (var attempt = 0; attempt < 100; attempt++) {
      if (find.text('图片已被删除或移动').evaluate().isNotEmpty) break;
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 10)),
      );
      await tester.pump();
    }

    expect(find.text('图片已被删除或移动'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('文件夹选择器打开时不能重复进入', (tester) async {
    const channel = MethodChannel('neri/dialogs');
    final selection = Completer<String?>();
    var invocationCount = 0;
    tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(channel, (
      call,
    ) {
      invocationCount += 1;
      return selection.future;
    });
    addTearDown(
      () => tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
        channel,
        null,
      ),
    );

    final controller = TextEditingController();
    addTearDown(controller.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: InputFolderField(controller: controller)),
      ),
    );

    await tester.tap(find.byTooltip('选择文件夹'));
    await tester.pump();

    expect(invocationCount, 1);
    final folderButton = find.descendant(
      of: find.byType(InputFolderField),
      matching: find.byType(IconButton),
    );
    expect(tester.widget<IconButton>(folderButton).onPressed, isNull);

    selection.complete(r'C:\camera-data');
    await tester.pump();

    expect(controller.text, r'C:\camera-data');
    expect(tester.widget<IconButton>(folderButton).onPressed, isNotNull);
  });

  test('校验多选修饰键可在失焦时清空', () {
    final modifiers = ValidationSelectionModifiers();

    expect(modifiers.press(LogicalKeyboardKey.shiftLeft), isTrue);
    expect(modifiers.rangeSelection, isTrue);
    expect(modifiers.toggleSelection, isFalse);

    modifiers.clear();
    expect(modifiers.rangeSelection, isFalse);

    expect(modifiers.press(LogicalKeyboardKey.controlLeft), isTrue);
    expect(modifiers.toggleSelection, isTrue);
    expect(modifiers.release(LogicalKeyboardKey.controlLeft), isTrue);
    expect(modifiers.toggleSelection, isFalse);
  });

  testWidgets('桌面鼠标长按不触发多选', (tester) async {
    var longPressCount = 0;
    var selectionCount = 0;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SizedBox(
            width: 300,
            height: 200,
            child: SelectableListCard<String>(
              items: const <String>['photo.jpg'],
              selectedIndex: 0,
              titleBuilder: (item) => item,
              onSelected: (_, __) => selectionCount += 1,
              onLongPress: (_, __) => longPressCount += 1,
            ),
          ),
        ),
      ),
    );

    final center = tester.getCenter(find.text('photo.jpg'));
    final mouse = await tester.startGesture(
      center,
      kind: PointerDeviceKind.mouse,
    );
    await tester.pump(kLongPressTimeout + const Duration(milliseconds: 50));
    await mouse.up();
    await tester.pump();

    expect(longPressCount, 0);
    expect(selectionCount, 1);

    final touch = await tester.startGesture(
      center,
      kind: PointerDeviceKind.touch,
    );
    await tester.pump(kLongPressTimeout + const Duration(milliseconds: 50));
    await touch.up();
    await tester.pump();

    expect(longPressCount, 1);
    expect(selectionCount, 1);
  });

  test('ModelInfo 解析 learning_family_id 并过滤空白值', () {
    final withFamily = ModelInfo.fromJson({
      'name': 'Neri1.2preview',
      'path': 'res/model/cls/bundled/Neri1.2preview-litert.neri.json',
      'encoder_backend': 'litert',
      'learning_family_id': 'neri-1.2-preview',
    });
    final withBlankFamily = ModelInfo.fromJson({
      'name': 'Legacy',
      'path': 'res/model/cls/user/legacy.pt',
      'learning_family_id': '   ',
    });
    final withoutFamily = ModelInfo.fromJson({
      'name': 'Plain',
      'path': 'res/model/cls/user/plain.pt',
      'learning_family_id': null,
    });

    expect(withFamily.learningFamilyId, 'neri-1.2-preview');
    expect(withBlankFamily.learningFamilyId, isNull);
    expect(withoutFamily.learningFamilyId, isNull);
  });

  test('运行方式解析严格按同 family_id 匹配且不跨 family 选模型', () {
    const cpuModel = ModelInfo(
      name: 'Neri1.2preview CPU',
      path: 'res/model/cls/bundled/Neri1.2preview-litert.neri.json',
      source: 'bundled',
      kind: 'cls',
      backend: 'dinov3',
      encoderBackend: 'litert',
      learningFamilyId: 'neri-1.2-preview',
    );
    const otherFamilyGpu = ModelInfo(
      name: 'Other GPU',
      path: 'res/model/cls/user/other-pytorch.neri.json',
      source: 'user',
      kind: 'cls',
      backend: 'dinov3',
      encoderBackend: 'pytorch',
      learningFamilyId: 'other-family',
    );
    const sameFamilyGpu = ModelInfo(
      name: 'Neri1.2preview GPU',
      path: 'res/model/cls/sync/Neri1.2preview-pytorch.neri.json',
      source: 'sync',
      kind: 'cls',
      backend: 'dinov3',
      encoderBackend: 'pytorch',
      learningFamilyId: 'neri-1.2-preview',
    );
    const nonFamilyModel = ModelInfo(
      name: 'Standalone',
      path: 'res/model/cls/user/standalone.pt',
    );

    const settingsMissingPeer = NeriSettings(
      appTitle: 'Neri',
      appVersion: 'test',
      supportedImageExtensions: <String>['.jpg'],
      supportedVideoExtensions: <String>['.mp4'],
      modelDirectory: 'res/model/detect',
      classificationModelDirectory: 'res/model/cls',
      availableModels: <ModelInfo>[],
      availableClassificationModels: <ModelInfo>[
        cpuModel,
        otherFamilyGpu,
        nonFamilyModel,
      ],
      speciesTypes: <String, String>{},
      settings: <String, dynamic>{},
      gpuAvailable: true,
      pytorchAvailable: true,
      litertAvailable: true,
      missingYoloDependencies: <String>[],
    );

    expect(resolveClassificationRuntimeMode(settingsMissingPeer, ''), isNull);
    expect(
      resolveClassificationRuntimeMode(
        settingsMissingPeer,
        nonFamilyModel.path,
      ),
      isNull,
    );

    final missingPeerResolution = resolveClassificationRuntimeMode(
      settingsMissingPeer,
      cpuModel.path,
    )!;
    expect(missingPeerResolution.activeMode, ClassificationRuntimeMode.cpu);
    expect(missingPeerResolution.cpuModel?.path, cpuModel.path);
    expect(missingPeerResolution.gpuModel, isNull);
    expect(missingPeerResolution.gpuAvailable, isFalse);

    final settingsReady = settingsMissingPeer.copyWith(
      availableClassificationModels: const <ModelInfo>[
        cpuModel,
        otherFamilyGpu,
        sameFamilyGpu,
      ],
    );
    final readyResolution = resolveClassificationRuntimeMode(
      settingsReady,
      cpuModel.path,
    )!;
    expect(readyResolution.gpuModel?.path, sameFamilyGpu.path);
    expect(readyResolution.gpuAvailable, isTrue);

    final noPytorch = resolveClassificationRuntimeMode(
      settingsReady.copyWith(pytorchAvailable: false),
      cpuModel.path,
    )!;
    expect(noPytorch.gpuAvailable, isFalse);

    final noGpuHardware = resolveClassificationRuntimeMode(
      settingsReady.copyWith(gpuAvailable: false),
      cpuModel.path,
    )!;
    expect(noGpuHardware.gpuAvailable, isFalse);
  });

  testWidgets('开始页分类模型区域不再展示运行方式切换 UI', (tester) async {
    await tester.binding.setSurfaceSize(const Size(1200, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final inputController = TextEditingController();
    addTearDown(inputController.dispose);

    const cpuModel = ModelInfo(
      name: 'Neri1.2preview·LiteRT',
      path: 'res/model/cls/bundled/Neri1.2preview-litert.neri.json',
      source: 'bundled',
      kind: 'cls',
      backend: 'dinov3',
      encoderBackend: 'litert',
      learningFamilyId: 'neri-1.2-preview',
      requiresDetector: true,
    );
    const gpuModel = ModelInfo(
      name: 'Neri1.2preview（PyTorch GPU）',
      path: 'res/model/cls/DINOv3/Neri1.2preview.neri.json',
      source: 'sync',
      kind: 'cls',
      backend: 'dinov3',
      encoderBackend: 'pytorch',
      learningFamilyId: 'neri-1.2-preview',
      requiresDetector: true,
    );
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: StartScreen(
            settings: NeriSettings(
              appTitle: 'Neri',
              appVersion: 'test',
              supportedImageExtensions: const <String>['.jpg'],
              supportedVideoExtensions: const <String>['.mp4'],
              modelDirectory: 'res/model/detect',
              classificationModelDirectory: 'res/model/cls',
              availableModels: const <ModelInfo>[],
              availableClassificationModels: const <ModelInfo>[
                cpuModel,
                gpuModel,
              ],
              selectedClassificationModel: cpuModel.path,
              speciesTypes: const <String, String>{},
              settings: const <String, dynamic>{},
              gpuAvailable: true,
              pytorchAvailable: true,
              litertAvailable: true,
              detectionReady: true,
              missingYoloDependencies: const <String>[],
            ),
            inputController: inputController,
            selectedModelPath: '',
            onModelChanged: (_) {},
            selectedClassificationModelPath: cpuModel.path,
            onClassificationModelChanged: (_) {},
            videoMode: videoProcessingModeSkip,
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
    await tester.pump();

    expect(find.text('运行方式'), findsNothing);
    expect(find.byKey(const Key('classification-runtime-cpu')), findsNothing);
    expect(find.byKey(const Key('classification-runtime-gpu')), findsNothing);
  });

  testWidgets('设置页检测设置不再展示运行方式切换但保留同 family 物种选择', (tester) async {
    await tester.binding.setSurfaceSize(const Size(1200, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final apiClient = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/models/classes') {
          return http.Response('[]', 200);
        }
        if (request.url.path == '/api/maintenance/status') {
          return http.Response('{"running": false}', 200);
        }
        if (request.url.path == '/api/maintenance/package-source') {
          return http.Response(
            '{"source": "tuna", "label": "清华源", "index_url": "https://pypi.tuna.tsinghua.edu.cn/simple"}',
            200,
          );
        }
        return http.Response('{}', 200);
      }),
    );
    final themeNotifier = ValueNotifier(const ThemeSettings());
    addTearDown(apiClient.close);
    addTearDown(themeNotifier.dispose);

    const cpuModel = ModelInfo(
      name: 'Neri1.2preview·LiteRT',
      path: 'res/model/cls/bundled/Neri1.2preview-litert.neri.json',
      source: 'bundled',
      kind: 'cls',
      backend: 'dinov3',
      encoderBackend: 'litert',
      learningFamilyId: 'neri-1.2-preview',
      requiresDetector: true,
      supportsVideoAll: false,
      supportsVideoFast: false,
    );
    const gpuModel = ModelInfo(
      name: 'Neri1.2preview（PyTorch GPU）',
      path: 'res/model/cls/DINOv3/Neri1.2preview.neri.json',
      source: 'sync',
      kind: 'cls',
      backend: 'dinov3',
      encoderBackend: 'pytorch',
      learningFamilyId: 'neri-1.2-preview',
      requiresDetector: true,
      supportsVideoAll: false,
      supportsVideoFast: true,
    );
    const otherModel = ModelInfo(
      name: 'OtherClassifier',
      path: 'res/model/cls/user/other.neri.json',
      source: 'user',
      kind: 'cls',
      backend: 'dinov3',
      encoderBackend: 'pytorch',
      learningFamilyId: 'other-family',
      requiresDetector: true,
    );

    final savedDrafts = <Map<String, dynamic>>[];

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SettingsScreen(
            settings: NeriSettings(
              appTitle: 'Neri',
              appVersion: 'test',
              supportedImageExtensions: const <String>['.jpg'],
              supportedVideoExtensions: const <String>['.mp4'],
              modelDirectory: 'res/model/detect',
              classificationModelDirectory: 'res/model/cls',
              availableModels: const <ModelInfo>[],
              availableClassificationModels: const <ModelInfo>[
                cpuModel,
                gpuModel,
                otherModel,
              ],
              selectedClassificationModel: cpuModel.path,
              speciesTypes: const <String, String>{},
              settings: <String, dynamic>{
                'selected_classification_model': cpuModel.path,
                'selected_species_names': <String>['岩羊'],
              },
              gpuAvailable: true,
              pytorchAvailable: true,
              litertAvailable: true,
              detectionReady: true,
              missingYoloDependencies: const <String>[],
            ),
            autoGroupInferredBurstSize: null,
            apiClient: apiClient,
            themeNotifier: themeNotifier,
            onUpdateTheme: (_) {},
            closeBehavior: 'ask',
            onCloseBehaviorChanged: (_) {},
            onSaveSettings: (draft) async {
              savedDrafts.add(Map<String, dynamic>.from(draft));
            },
            onCheckForUpdates:
                ({required channel, required downloadSource}) async {},
            onShowMessage: (_) {},
          ),
        ),
      ),
    );
    await tester.pump();

    expect(find.text('运行方式'), findsNothing);
    expect(find.byKey(const Key('classification-runtime-cpu')), findsNothing);
    expect(find.byKey(const Key('classification-runtime-gpu')), findsNothing);

    final clsTrigger = find.widgetWithText(TextButton, cpuModel.rawName);
    await tester.ensureVisible(clsTrigger);
    await tester.pumpAndSettle();
    await tester.tap(clsTrigger);
    await tester.pumpAndSettle();
    final gpuMenuItem = find.widgetWithText(MenuItemButton, gpuModel.rawName);
    await tester.ensureVisible(gpuMenuItem);
    await tester.pumpAndSettle();
    await tester.tap(gpuMenuItem);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 900));
    await tester.pump();

    expect(savedDrafts.last['selected_classification_model'], gpuModel.path);
    expect(savedDrafts.last['selected_species_names'], <String>['岩羊']);

    final gpuTrigger = find.widgetWithText(TextButton, gpuModel.rawName);
    await tester.ensureVisible(gpuTrigger);
    await tester.pumpAndSettle();
    await tester.tap(gpuTrigger);
    await tester.pumpAndSettle();
    final otherMenuItem = find.widgetWithText(
      MenuItemButton,
      otherModel.rawName,
    );
    await tester.ensureVisible(otherMenuItem);
    await tester.pumpAndSettle();
    await tester.tap(otherMenuItem);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 900));
    await tester.pump();

    expect(savedDrafts.last['selected_classification_model'], otherModel.path);
    expect(savedDrafts.last['selected_species_names'], isEmpty);
  });

  testWidgets('环境维护 Neri1.2preview 卡片支持下载版本选择、提示与 runtime_choice 请求', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1400, 1100));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final installRequests = <Map<String, dynamic>>[];
    var statusPayload = <String, dynamic>{
      'installed': false,
      'healthy': false,
      'display_name': 'Neri1.2preview',
      'message': 'Neri1.2preview 未安装。',
    };
    final apiClient = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/environment/dinov3-status') {
          return http.Response.bytes(
            utf8.encode(jsonEncode(statusPayload)),
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        if (request.url.path == '/api/environment/install-dinov3') {
          installRequests.add(
            Map<String, dynamic>.from(
              jsonDecode(request.body) as Map<String, dynamic>,
            ),
          );
          return http.Response(
            jsonEncode({
              'operation': 'install_dinov3',
              'state': 'starting',
              'message': '已启动安装',
              'progress': 0,
            }),
            202,
          );
        }
        if (request.url.path == '/api/maintenance/status') {
          return http.Response('{"running": false}', 200);
        }
        if (request.url.path == '/api/maintenance/package-source') {
          return http.Response(
            '{"source": "tuna", "label": "清华源", "index_url": "https://pypi.tuna.tsinghua.edu.cn/simple"}',
            200,
          );
        }
        return http.Response('{}', 200);
      }),
    );
    final themeNotifier = ValueNotifier(const ThemeSettings());
    addTearDown(apiClient.close);
    addTearDown(themeNotifier.dispose);

    final savedDrafts = <Map<String, dynamic>>[];
    Widget buildScreen({
      Map<String, dynamic>? savedSettings,
      int backendGeneration = 1,
    }) {
      return MaterialApp(
        home: Scaffold(
          body: SettingsScreen(
            settings: NeriSettings(
              appTitle: 'Neri',
              appVersion: 'test',
              supportedImageExtensions: const <String>['.jpg'],
              supportedVideoExtensions: const <String>['.mp4'],
              modelDirectory: 'res/model/detect',
              classificationModelDirectory: 'res/model/cls',
              availableModels: const <ModelInfo>[],
              availableClassificationModels: const <ModelInfo>[],
              speciesTypes: const <String, String>{},
              settings: savedSettings ?? const <String, dynamic>{},
              gpuAvailable: false,
              pytorchAvailable: false,
              litertAvailable: true,
              detectionReady: true,
              missingYoloDependencies: const <String>[],
            ),
            backendGeneration: backendGeneration,
            autoGroupInferredBurstSize: null,
            apiClient: apiClient,
            themeNotifier: themeNotifier,
            onUpdateTheme: (_) {},
            closeBehavior: 'ask',
            onCloseBehaviorChanged: (_) {},
            onSaveSettings: (draft) async {
              savedDrafts.add(Map<String, dynamic>.from(draft));
            },
            onCheckForUpdates:
                ({required channel, required downloadSource}) async {},
            onShowMessage: (_) {},
          ),
        ),
      );
    }

    await tester.pumpWidget(buildScreen());
    await tester.pumpAndSettle();
    await tester.tap(find.text('环境维护').first);
    await tester.pumpAndSettle();

    expect(find.text('下载版本'), findsOneWidget);
    expect(find.text('自动检测（默认）'), findsOneWidget);
    expect(find.textContaining('仅 GPU 模式要求 PyTorch'), findsNothing);

    Finder runtimeChoiceTrigger() => find.descendant(
      of: find.byKey(const Key('dino-runtime-choice')),
      matching: find.byType(TextButton),
    );

    await tester.ensureVisible(runtimeChoiceTrigger());
    await tester.pumpAndSettle();
    await tester.tap(runtimeChoiceTrigger());
    await tester.pumpAndSettle();
    expect(
      find.widgetWithText(MenuItemButton, '自动检测（默认）'),
      findsOneWidget,
    );
    expect(
      find.widgetWithText(MenuItemButton, 'CPU（LiteRT FP32）'),
      findsOneWidget,
    );
    expect(
      find.widgetWithText(MenuItemButton, 'GPU（PyTorch）'),
      findsOneWidget,
    );

    final cpuChoiceItem = find.widgetWithText(
      MenuItemButton,
      'CPU（LiteRT FP32）',
    );
    await tester.ensureVisible(cpuChoiceItem);
    await tester.pumpAndSettle();
    await tester.tap(cpuChoiceItem);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 900));
    await tester.pump();

    expect(savedDrafts.last['dino_runtime_choice'], 'cpu');
    expect(find.textContaining('不要求安装 PyTorch'), findsNothing);

    await tester.pumpWidget(buildScreen(savedSettings: const <String, dynamic>{}));
    await tester.pump();
    expect(find.text('CPU（LiteRT FP32）'), findsOneWidget);

    await tester.tap(find.byKey(const Key('dino-component-action')));
    await tester.pumpAndSettle();
    expect(find.textContaining('无需安装 PyTorch'), findsOneWidget);
    expect(find.textContaining('NeriCloud'), findsOneWidget);
    expect(find.textContaining('共享学习记录会保留'), findsOneWidget);
    expect(find.textContaining('后端会自动重启'), findsOneWidget);
    expect(find.textContaining('res/model/DINOv3'), findsNothing);
    expect(find.textContaining('SHA-256'), findsNothing);
    expect(find.textContaining('Ridge'), findsNothing);
    await tester.tap(find.text('开始安装'));
    await tester.pumpAndSettle();
    expect(installRequests.last['runtime_choice'], 'cpu');

    statusPayload = <String, dynamic>{
      'installed': true,
      'healthy': true,
      'display_name': 'Neri1.2preview',
      'encoder_backend': 'litert',
      'runtime_choice': 'cpu',
      'resolved_runtime_choice': 'cpu',
      'message': 'Neri1.2preview 已安装。',
    };
    await tester.pumpWidget(
      buildScreen(
        savedSettings: const <String, dynamic>{'dino_runtime_choice': 'gpu'},
        backendGeneration: 2,
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('GPU（PyTorch）'), findsOneWidget);
    expect(find.text('当前已安装：CPU（LiteRT FP32）'), findsOneWidget);
    expect(find.textContaining('需要 PyTorch 环境'), findsNothing);
    expect(find.widgetWithText(OutlinedButton, '删除'), findsOneWidget);
    expect(find.byKey(const Key('dino-component-switch')), findsOneWidget);

    await tester.tap(find.byKey(const Key('dino-component-switch')));
    await tester.pumpAndSettle();
    expect(find.textContaining('NeriCloud'), findsOneWidget);
    expect(find.textContaining('自动补齐'), findsWidgets);
    expect(find.textContaining('共享学习记录会保留'), findsOneWidget);
    expect(find.textContaining('后端会自动重启'), findsOneWidget);
    await tester.tap(find.text('开始安装'));
    await tester.pumpAndSettle();
    expect(installRequests.last['runtime_choice'], 'gpu');

    await tester.ensureVisible(runtimeChoiceTrigger());
    await tester.pumpAndSettle();
    await tester.tap(runtimeChoiceTrigger());
    await tester.pumpAndSettle();
    final autoChoiceItem = find.widgetWithText(
      MenuItemButton,
      '自动检测（默认）',
    );
    await tester.ensureVisible(autoChoiceItem);
    await tester.pumpAndSettle();
    await tester.tap(autoChoiceItem);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 900));
    await tester.pumpAndSettle();
    expect(find.text('当前已安装：CPU（LiteRT FP32）'), findsOneWidget);
    expect(find.textContaining('仅 GPU 模式要求 PyTorch'), findsNothing);
    await tester.pumpWidget(
      buildScreen(
        savedSettings: const <String, dynamic>{
          'dino_runtime_choice': 'auto',
          'debug_mode': true,
        },
        backendGeneration: 3,
      ),
    );
    await tester.pumpAndSettle();
    expect(find.text('当前已安装：CPU（LiteRT FP32）'), findsOneWidget);
    expect(
      find.textContaining('仅 GPU 模式要求 PyTorch'),
      findsOneWidget,
    );
    statusPayload = <String, dynamic>{
      'installed': false,
      'healthy': false,
      'display_name': 'Neri1.2preview',
      'message': 'Neri1.2preview 未安装。',
    };
    await tester.pumpWidget(
      buildScreen(
        savedSettings: const <String, dynamic>{'dino_runtime_choice': 'auto'},
        backendGeneration: 4,
      ),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('dino-component-action')));
    await tester.pumpAndSettle();
    expect(find.textContaining('检测到支持 GPU 的硬件时'), findsOneWidget);
    expect(find.textContaining('否则安装 CPU（LiteRT FP32）版本'), findsOneWidget);
    expect(find.textContaining('共享学习记录会保留'), findsOneWidget);
    await tester.tap(find.text('取消'));
    await tester.pumpAndSettle();
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('开始页与设置页下拉菜单展示精确内置模型名且不重复拼 LiteRT', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1200, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final inputController = TextEditingController();
    addTearDown(inputController.dispose);
    final apiClient = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/models/classes') {
          return http.Response('[]', 200);
        }
        if (request.url.path == '/api/maintenance/status') {
          return http.Response('{"running": false}', 200);
        }
        if (request.url.path == '/api/maintenance/package-source') {
          return http.Response(
            '{"source": "tuna", "label": "清华源", "index_url": "https://pypi.tuna.tsinghua.edu.cn/simple"}',
            200,
          );
        }
        return http.Response('{}', 200);
      }),
    );
    final themeNotifier = ValueNotifier(const ThemeSettings());
    addTearDown(apiClient.close);
    addTearDown(themeNotifier.dispose);

    const detector = ModelInfo(
      name: 'MDV6-yolov10-c·LiteRT',
      path: 'res/model/detect/bundled/MDV6-yolov10-c.tflite',
      source: 'bundled',
      kind: 'detect',
      backend: 'litert',
    );
    const classifier = ModelInfo(
      name: 'Neri1.2preview·LiteRT',
      path: 'res/model/cls/bundled/Neri1.2preview-litert.neri.json',
      source: 'bundled',
      kind: 'cls',
      backend: 'dinov3',
      encoderBackend: 'litert',
      requiresDetector: true,
      supportsVideoFast: false,
      supportsVideoAll: false,
    );
    final settings = NeriSettings(
      appTitle: 'Neri',
      appVersion: 'test',
      supportedImageExtensions: const <String>['.jpg'],
      supportedVideoExtensions: const <String>['.mp4'],
      modelDirectory: 'res/model/detect',
      classificationModelDirectory: 'res/model/cls',
      availableModels: const <ModelInfo>[detector],
      availableClassificationModels: const <ModelInfo>[classifier],
      selectedModel: detector.path,
      selectedClassificationModel: classifier.path,
      speciesTypes: const <String, String>{},
      settings: <String, dynamic>{
        'selected_model': detector.path,
        'selected_classification_model': classifier.path,
      },
      gpuAvailable: false,
      pytorchAvailable: false,
      litertAvailable: true,
      detectionReady: true,
      missingYoloDependencies: const <String>['torch'],
    );

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: StartScreen(
            settings: settings,
            inputController: inputController,
            selectedModelPath: detector.path,
            onModelChanged: (_) {},
            selectedClassificationModelPath: classifier.path,
            onClassificationModelChanged: (_) {},
            videoMode: videoProcessingModeSkip,
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
    await tester.pump();

    final startMenus = tester
        .widgetList<AppFormMenu<String>>(find.byType(AppFormMenu<String>))
        .toList();
    expect(
      startMenus[0].options.map((entry) => entry.label).toList(),
      containsAllInOrder(<String>['不使用', '内置模型', 'MDV6-yolov10-c·LiteRT']),
    );
    expect(
      startMenus[1].options.map((entry) => entry.label).toList(),
      containsAllInOrder(<String>['不使用', '内置模型', 'Neri1.2preview·LiteRT']),
    );
    expect(find.text('MDV6-yolov10-c·LiteRT'), findsOneWidget);
    expect(find.text('Neri1.2preview·LiteRT'), findsOneWidget);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SettingsScreen(
            settings: settings,
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

    expect(find.text('MDV6-yolov10-c·LiteRT'), findsOneWidget);
    expect(find.text('Neri1.2preview·LiteRT'), findsOneWidget);
  });
}
