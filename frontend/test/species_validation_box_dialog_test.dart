import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/app_theme.dart';
import 'package:neri_flutter/src/main_window.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';
import 'package:neri_flutter/src/screens/preview_screen.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer.dart';
import 'package:neri_flutter/src/widgets/dino_feature_scatter.dart';
import 'package:neri_flutter/src/widgets/sequence_pending_notice.dart';

void main() {
  late Directory directory;
  late String imagePath;

  setUp(() async {
    directory = await Directory.systemTemp.createTemp('neri_box_dialog_');
    imagePath = '${directory.path}/bird.png';
    await File(imagePath).writeAsBytes(
      base64Decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
      ),
    );
  });

  tearDown(() async => directory.delete(recursive: true));

  DetectionItem rejectedDinoItem() => DetectionItem(
    filename: 'bird.png',
    path: imagePath,
    fileType: 'png',
    width: 10,
    height: 10,
    species: const ['Unknown'],
    confidence: 0.086,
    detectionData: const {
      '物种名称': 'Unknown',
      '物种数量': '3',
      'sequence_completed': true,
    },
    detectionBoxes: [
      for (final score in [0.171, 0.116, 0.086])
        DetectionBox(
          species: 'Unknown',
          confidence: score,
          bbox: const [0.1, 0.1, 0.7, 0.7],
          candidates: [
            {
              'name': 'Unknown',
              'conf': score,
              'accepted': false,
              'raw_det_conf': 0.94,
              'box_known_score': -1.399,
              'box_threshold': -0.993,
              'sequence_known_score': -1.911,
              'sequence_threshold': -0.332,
              'source': 'dinov3_ridge_within',
            },
            {'name': '备选物种', 'conf': 0.999},
          ],
        ),
    ],
  );

  testWidgets(
    'DINO validation retains all rejected boxes and replaces the confidence slider with readonly scores',
    (tester) async {
      _setSize(tester, 1280);
      final api = NeriApiClient(
        httpClient: MockClient((_) async => _jsonResponse({})),
      );
      addTearDown(api.close);
      await tester.pumpWidget(_screen(api, directory.path, rejectedDinoItem()));
      await _waitForImage(tester);
      final viewer = tester.widget<DetectionMediaViewer>(
        find.byType(DetectionMediaViewer),
      );
      expect(viewer.visibleBoxes.length, 3);
      expect(viewer.visibleBoxes.map((box) => box.species).toSet(), {
        'Unknown',
      });
      expect(find.byType(Slider), findsNothing);
      expect(find.text('显示物种'), findsNothing);
      expect(
        find.byKey(const ValueKey('dino-rejection-single-line')),
        findsOneWidget,
      );
      expect(find.byTooltip('单框 -1.399 < 门限 -0.993（未通过）'), findsNWidgets(3));
      expect(find.byTooltip('序列 -1.911 < 门限 -0.332（未通过）'), findsNWidgets(3));
      expect(find.textContaining('最终拒识 · Unknown'), findsNWidgets(3));
      expect(find.textContaining('物种: Unknown  |  数量: 3'), findsOneWidget);
      expect(
        tester
            .getCenter(find.byKey(const ValueKey('dino-rejection-single-line')))
            .dy,
        closeTo(tester.getCenter(find.byType(Switch)).dy, 1),
      );
      await tester.tap(find.text('其他'));
      await tester.pumpAndSettle();
      final text = tester
          .widgetList<TextField>(find.byType(TextField))
          .map((field) => field.controller?.text)
          .toList();
      expect(text, contains('Unknown'));
      expect(text, contains('3'));
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'DINO preview ignores a stale display confidence threshold and shows readonly gates',
    (tester) async {
      _setSize(tester, 1280);
      final selected = rejectedDinoItem();
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PreviewScreen(
              inputPath: directory.path,
              items: [selected],
              selectedIndex: 0,
              selectedItem: selected,
              speciesTypes: const {},
              useCombinedConfidence: true,
              showDetections: true,
              onShowDetectionsChanged: (_) {},
              selectedSpeciesFilter: previewAllSpeciesLabel,
              onSpeciesFilterChanged: (_) {},
              confidenceThreshold: 0.95,
              onConfidenceThresholdChanged: (_) =>
                  fail('DINO confidence is readonly'),
              detecting: false,
              loading: false,
              onDetectCurrentImage: (_) {},
              onSelected: (_, _) {},
              onLoadMetadata: (_) async {},
              onRefresh: () async {},
              onOpenExternal: (_) {},
            ),
          ),
        ),
      );
      await _waitForImage(tester);
      final viewer = tester.widget<DetectionMediaViewer>(
        find.byType(DetectionMediaViewer),
      );
      expect(viewer.visibleBoxes.length, 3);
      expect(
        viewer.visibleBoxes.every((box) => box.species == 'Unknown'),
        isTrue,
      );
      expect(find.byType(Slider), findsNothing);
      expect(find.text('显示物种'), findsNothing);
      expect(
        find.byKey(const ValueKey('dino-rejection-single-line')),
        findsOneWidget,
      );
      expect(find.byTooltip('单框 -1.399 < 门限 -0.993（未通过）'), findsNWidgets(3));
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'legacy YOLO validation keeps the confidence slider and filtering',
    (tester) async {
      _setSize(tester, 1280);
      final api = NeriApiClient(
        httpClient: MockClient((_) async => _jsonResponse({})),
      );
      addTearDown(api.close);
      final legacy = DetectionItem(
        filename: 'bird.png',
        path: imagePath,
        fileType: 'png',
        width: 10,
        height: 10,
        confidence: 0.1,
        species: const ['bird'],
        detectionBoxes: const [
          DetectionBox(
            species: 'bird',
            confidence: 0.1,
            bbox: [0.1, 0.1, 0.7, 0.7],
            candidates: [
              {'name': 'bird', 'conf': 0.1},
            ],
          ),
        ],
      );
      await tester.pumpWidget(_screen(api, directory.path, legacy));
      await _waitForImage(tester);
      expect(
        tester
            .widget<DetectionMediaViewer>(find.byType(DetectionMediaViewer))
            .visibleBoxes,
        isEmpty,
      );
      expect(find.byType(Slider), findsOneWidget);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets('legacy preview and validation scan detection boxes linearly', (
    tester,
  ) async {
    _setSize(tester, 1280);
    const belowThreshold = DetectionBox(
      species: 'bird',
      confidence: 0.1,
      bbox: [0.1, 0.1, 0.7, 0.7],
    );
    const withCandidate = DetectionBox(
      species: 'sparrow',
      confidence: 0.1,
      bbox: [0.2, 0.2, 0.8, 0.8],
      candidates: [
        {'name': 'sparrow', 'conf': 0.1},
        {'name': 'bird', 'conf': 0.8},
      ],
    );
    final boxes = _CountingList([
      for (var i = 0; i < 256; i++) ...[belowThreshold, withCandidate],
    ]);
    final rawBoxes = _CountingList([
      for (var i = 0; i < boxes.length; i++)
        {
          'bbox': [i, i, i + 1, i + 1],
        },
    ]);
    final item = DetectionItem(
      filename: 'bird.png',
      path: '${directory.path}/unavailable.png',
      fileType: 'png',
      width: 10,
      height: 10,
      detectionBoxes: boxes,
      detectionData: {'检测框': rawBoxes},
    );

    void expectFilteredBoxes() {
      final viewer = tester.widget<DetectionMediaViewer>(
        find.byType(DetectionMediaViewer),
      );
      expect(viewer.visibleBoxes, hasLength(256));
      expect(viewer.visibleBoxes.every((box) => box.species == 'bird'), isTrue);
      expect(boxes.reads, lessThan(boxes.length * 32));
      expect(rawBoxes.reads, lessThan(rawBoxes.length * 32));
    }

    await tester.pumpWidget(_preview(item, selectedSpecies: 'bird'));
    expectFilteredBoxes();

    final api = NeriApiClient(
      httpClient: MockClient((_) async => _jsonResponse({})),
    );
    addTearDown(api.close);
    boxes.reads = 0;
    rawBoxes.reads = 0;
    await tester.pumpWidget(_screen(api, directory.path, item));
    expectFilteredBoxes();
    await tester.pumpWidget(const SizedBox.shrink());
  });

  testWidgets('Neri preview keeps the legacy toolbar usable when resized', (
    tester,
  ) async {
    _setSize(tester, 1280);
    final item = DetectionItem(
      filename: 'bird.png',
      path: '${directory.path}/unavailable.png',
      fileType: 'png',
      species: const ['bird'],
    );
    double? changedConfidence;
    bool? showDetections;
    DetectionItem? redetected;
    await tester.pumpWidget(
      _preview(
        item,
        onConfidenceChanged: (value) => changedConfidence = value,
        onShowDetectionsChanged: (value) => showDetections = value,
        onDetect: (selected) => redetected = selected,
      ),
    );
    final toggle = find.byType(Switch);
    final slider = find.byType(Slider);
    final detect = find.text('检测当前图像');
    expect(
      tester.getCenter(slider).dy,
      closeTo(tester.getCenter(toggle).dy, 1),
    );
    expect(
      tester.getCenter(detect).dy,
      closeTo(tester.getCenter(toggle).dy, 1),
    );

    for (final width in [600.0, 480.0]) {
      tester.view.physicalSize = Size(width, 900);
      await tester.pump();
      expect(tester.takeException(), isNull);
      expect(tester.getSize(slider).width, greaterThan(100));
      expect(
        tester.getCenter(toggle).dy,
        lessThan(tester.getCenter(slider).dy),
      );
      expect(find.byType(DropdownMenu<String>), findsOneWidget);

      changedConfidence = null;
      await tester.tapAt(
        tester.getCenter(slider) + Offset(tester.getSize(slider).width / 4, 0),
      );
      expect(changedConfidence, greaterThan(0.25));
    }
    await tester.tap(toggle);
    expect(showDetections, isFalse);
    await tester.tap(detect);
    expect(redetected, same(item));
    await tester.pumpWidget(const SizedBox.shrink());
  });

  testWidgets('Neri preview fits large text near the toolbar breakpoint', (
    tester,
  ) async {
    _setSize(tester, 704);
    final item = DetectionItem(
      filename: 'bird.png',
      path: '${directory.path}/unavailable.png',
      fileType: 'png',
    );
    await tester.pumpWidget(_preview(item, textScale: 1.6));
    final slider = find.byType(Slider);
    for (final size in [
      const Size(704, 900),
      const Size(600, 900),
      const Size(480, 420),
      const Size(1024, 900),
    ]) {
      tester.view.physicalSize = size;
      await tester.pump();
      expect(tester.takeException(), isNull);
      expect(tester.getSize(slider).width, greaterThan(100));
      if (size.height == 420) {
        await tester.ensureVisible(slider);
        await tester.pump();
        expect(tester.getRect(slider).bottom, lessThan(size.height));
      }
    }
    await tester.pumpWidget(const SizedBox.shrink());
  });

  testWidgets('a mixed item keeps DINO treatment in preview and validation', (
    tester,
  ) async {
    _setSize(tester, 1280);
    const plainBox = DetectionBox(
      species: 'sparrow',
      confidence: 0.1,
      bbox: [0.1, 0.1, 0.7, 0.7],
    );
    final item = DetectionItem(
      filename: 'bird.png',
      path: '${directory.path}/unavailable.png',
      fileType: 'png',
      width: 10,
      height: 10,
      detectionBoxes: const [
        plainBox,
        DetectionBox(
          species: 'bird',
          confidence: 0.9,
          bbox: [0.2, 0.2, 0.8, 0.8],
          candidates: [
            {'name': 'bird', 'accepted': false, 'box_known_score': 0.1},
          ],
        ),
      ],
    );
    List<DetectionBox> visibleBoxes() => tester
        .widget<DetectionMediaViewer>(find.byType(DetectionMediaViewer))
        .visibleBoxes;
    final scores = find.byKey(const ValueKey('dino-rejection-single-line'));
    final toggle = find.byType(Switch);
    await tester.pumpWidget(
      _preview(item, selectedSpecies: 'bird', confidence: 0.95),
    );
    expect(visibleBoxes().map((box) => box.species), ['sparrow', 'Unknown']);
    expect(find.byType(Slider), findsNothing);
    expect(
      tester.getCenter(scores).dy,
      closeTo(tester.getCenter(toggle).dy, 1),
    );

    tester.view.physicalSize = const Size(480, 900);
    await tester.pump();
    expect(tester.takeException(), isNull);
    expect(tester.getCenter(toggle).dy, lessThan(tester.getCenter(scores).dy));
    expect(tester.getSize(scores).width, greaterThan(300));

    tester.view.physicalSize = const Size(1280, 900);
    final api = NeriApiClient(
      httpClient: MockClient((_) async => _jsonResponse({})),
    );
    addTearDown(api.close);
    await tester.pumpWidget(_screen(api, directory.path, item));
    final filtered = visibleBoxes();
    expect(filtered.map((box) => box.species), ['sparrow', 'Unknown']);
    expect(filtered.first, same(plainBox));
    expect(filtered.last.observationId, startsWith('dinov3:'));
    await tester.pumpWidget(const SizedBox.shrink());
  });

  for (final batch in [false, true]) {
    testWidgets('correct passes active filters to callback batch=$batch', (
      tester,
    ) async {
      _setSize(tester, 1280);
      final api = NeriApiClient(
        httpClient: MockClient((_) async => _jsonResponse({})),
      );
      addTearDown(api.close);
      Map<String, double>? captured;
      double? frameRatio;
      String? entrypoint;
      Future<DetectionItem> markItem(
        DetectionItem item,
        String action, {
        String? speciesName,
        String? speciesCount,
        String? speciesType,
        String? remark,
        String? feedbackOperationId,
        Map<String, double>? confidenceSettings,
        double? minFrameRatio,
      }) async {
        captured = confidenceSettings;
        frameRatio = minFrameRatio;
        entrypoint = 'single';
        return item;
      }

      Future<List<DetectionItem>> markItems(
        List<DetectionItem> items,
        String action, {
        String? speciesName,
        String? speciesCount,
        String? speciesType,
        String? remark,
        String? feedbackOperationId,
        Map<String, double>? confidenceSettings,
        double? minFrameRatio,
      }) async {
        captured = confidenceSettings;
        frameRatio = minFrameRatio;
        entrypoint = 'batch';
        return items;
      }

      final items = <DetectionItem>[];
      for (var index = 0; index < (batch ? 4 : 1); index++) {
        // This test exercises annotation requests, independently of image decoding.
        final path = '${directory.path}/missing-$index.jpg';
        items.add(
          DetectionItem(
            filename: 'IMG_$index.png',
            path: path,
            fileType: 'png',
            dateTaken: '2026-11-13T12:00:0$index',
            width: 10,
            height: 10,
            species: const ['bird'],
            detectionBoxes: const [
              DetectionBox(
                species: 'bird',
                confidence: 0.95,
                bbox: [0, 0, 8, 8],
              ),
            ],
          ),
        );
      }
      await tester.pumpWidget(
        _screen(
          api,
          directory.path,
          items.first,
          itemsOverride: items,
          autoGroup: batch,
          minFrameRatio: 0.3,
          onMarkItem: markItem,
          onMarkItems: markItems,
        ),
      );
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 50)),
      );
      await tester.pumpAndSettle();
      final previousConfidence = tester
          .widget<Slider>(find.byType(Slider))
          .value;
      tester.widget<Slider>(find.byType(Slider)).onChanged!(0.8);
      await tester.pumpAndSettle();
      await tester.tap(find.text('正确').first);
      await tester.pump();
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 50)),
      );
      await tester.pump();
      expect(entrypoint, batch ? 'batch' : 'single');
      expect(captured?['global'], 0.8);
      expect(frameRatio, 0.3);
      tester.widget<Slider>(find.byType(Slider)).onChanged!(previousConfidence);
      await tester.pump(const Duration(seconds: 5));
      await tester.pumpWidget(const SizedBox.shrink());
      await tester.runAsync(() async {
        imageCache.clear();
        imageCache.clearLiveImages();
        await Future<void>.delayed(const Duration(milliseconds: 100));
      });
    });
  }

  test('resolveDinoValidationModelPath accepts dinov3 only', () {
    ModelInfo model(String backend, String path) =>
        ModelInfo(name: 'model', path: path, backend: backend);

    expect(
      resolveDinoValidationModelPath(
        model('dinov3', 'model.dinov3'),
        'model.dinov3',
      ),
      'model.dinov3',
    );
    expect(
      resolveDinoValidationModelPath(model('yolo', 'model.pt'), 'model.pt'),
      isNull,
    );
    expect(
      resolveDinoValidationModelPath(model('dinov3', 'model.dinov3'), '   '),
      isNull,
    );
    expect(resolveDinoValidationModelPath(null, 'model.dinov3'), isNull);
  });

  DetectionItem item({String? observationId = 'bird-1'}) => DetectionItem(
    filename: 'bird.png',
    path: imagePath,
    fileType: 'png',
    width: 10,
    height: 10,
    species: const ['中亚兔'],
    confidence: 0.752,
    detectionBoxes: [
      DetectionBox(
        species: '中亚兔',
        predictedSpecies: '中亚兔',
        confidence: 0.752,
        bbox: const [0.2, 0.2, 0.8, 0.8],
        observationId: observationId,
        candidates: const [
          {'name': '中亚兔', 'conf': 0.752},
        ],
      ),
    ],
  );

  Future<void> tapExport(WidgetTester tester, {String label = '导出'}) async {
    final button = find.widgetWithText(FilledButton, label);
    await tester.ensureVisible(button);
    await tester.pumpAndSettle();
    await tester.tap(button);
    await tester.pump(const Duration(milliseconds: 300));
  }

  for (final width in [900.0, 1280.0]) {
    testWidgets('pending results remain visible at width $width', (
      tester,
    ) async {
      _setSize(tester, width);
      tester.view.physicalSize = Size(width, 790);
      final api = NeriApiClient(
        httpClient: MockClient((request) async => _jsonResponse({})),
      );
      addTearDown(api.close);
      final pending = DetectionItem(
        filename: 'bird.png',
        path: imagePath,
        fileType: 'png',
        width: 10,
        height: 10,
        species: const ['拒识/Unknown'],
        detectionBoxes: const [
          DetectionBox(species: '拒识/Unknown', bbox: [0.2, 0.2, 0.8, 0.8]),
        ],
        detectionData: const {'sequence_pending': true},
      );
      await tester.pumpWidget(_screen(api, directory.path, pending));
      await _waitForImage(tester);
      if (width < 1000) {
        final scroll = tester.state<ScrollableState>(
          find.byType(Scrollable).first,
        );
        scroll.position.jumpTo(scroll.position.maxScrollExtent);
        await tester.pumpAndSettle();
      }
      expect(find.byTooltip(SequencePendingNotice.message), findsOneWidget);
      expect(find.textContaining('临时结果 · 序列待汇总'), findsNothing);
      expect(find.textContaining('置信度: N/A'), findsOneWidget);
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox.shrink());
    });
  }

  testWidgets('selected child can export every media directory in root', (
    tester,
  ) async {
    _setSize(tester, 1280);
    Map<String, dynamic>? exportBody;
    final api = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/validation/export') {
          exportBody = jsonDecode(request.body) as Map<String, dynamic>;
          return _jsonResponse({
            'output_path': '${directory.path}/result.csv',
            'output_paths': [
              '${directory.path}/result.csv',
              '${directory.path}/other/result.csv',
            ],
            'file_format': 'csv',
            'exported_count': 1,
            'favorite_exported_count': 2,
            'favorite_output_dir': '${directory.path}/first_only',
          });
        }
        return _jsonResponse({});
      }),
    );
    addTearDown(api.close);
    await tester.pumpWidget(
      _screen(
        api,
        directory.path,
        item(),
        rootPath: directory.parent.path,
        exportMode: 'ask',
      ),
    );
    await _waitForImage(tester);
    await tapExport(tester);
    expect(find.text('选择导出范围'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '仅导出当前子文件夹'), findsOneWidget);
    expect(find.widgetWithText(TextButton, '导出全部目录'), findsOneWidget);
    await tester.tap(find.text('导出全部目录'));
    await tester.pump(const Duration(milliseconds: 300));
    expect(find.text('选择导出格式'), findsOneWidget);
    await tester.tap(find.text('CSV'));
    await tester.pump(const Duration(milliseconds: 300));
    expect(exportBody?['input_path'], directory.parent.path);
    expect(exportBody?['per_directory'], true);
    expect(exportBody?['file_format'], 'csv');
    expect(find.textContaining('2 个收藏媒体文件到对应表格目录'), findsOneWidget);
    await tester.pump(const Duration(seconds: 5));
  });

  testWidgets('current child export keeps selected input and one table', (
    tester,
  ) async {
    _setSize(tester, 1280);
    Map<String, dynamic>? exportBody;
    final api = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/validation/export') {
          exportBody = jsonDecode(request.body) as Map<String, dynamic>;
          return _jsonResponse({
            'output_path': '${directory.path}/result.csv',
            'file_format': 'csv',
            'exported_count': 1,
          });
        }
        return _jsonResponse({});
      }),
    );
    addTearDown(api.close);
    await tester.pumpWidget(
      _screen(
        api,
        directory.path,
        item(),
        rootPath: directory.parent.path,
        exportMode: 'csv',
      ),
    );
    await _waitForImage(tester);
    await tapExport(tester, label: '导出为csv');
    await tester.tap(find.text('仅导出当前子文件夹'));
    await tester.pump(const Duration(milliseconds: 300));
    expect(find.text('选择导出格式'), findsNothing);
    expect(exportBody?['input_path'], directory.path);
    expect(exportBody?['per_directory'], false);
    await tester.pump(const Duration(seconds: 5));
  });

  testWidgets('format choice can be remembered', (tester) async {
    _setSize(tester, 1280);
    String? savedMode;
    String? format;
    final api = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/validation/export') {
          format =
              (jsonDecode(request.body) as Map<String, dynamic>)['file_format']
                  as String;
          return _jsonResponse({
            'output_path': '${directory.path}/result.xlsx',
            'file_format': 'excel',
            'exported_count': 1,
          });
        }
        return _jsonResponse({});
      }),
    );
    addTearDown(api.close);
    await tester.pumpWidget(
      _screen(
        api,
        directory.path,
        item(),
        onFormatModeChanged: (mode) async => savedMode = mode,
      ),
    );
    await _waitForImage(tester);
    await tapExport(tester);
    expect(find.text('选择导出范围'), findsNothing);
    await tester.tap(find.text('不再询问'));
    await tester.pump(const Duration(milliseconds: 300));
    await tester.tap(find.text('Excel'));
    await tester.pump(const Duration(milliseconds: 300));
    expect(savedMode, 'excel');
    expect(format, 'excel');
    await tester.pump(const Duration(seconds: 5));
  });

  for (final width in [1280.0, 600.0]) {
    testWidgets('CSV is primary and remember aligns with cancel at $width', (
      tester,
    ) async {
      _setSize(tester, width);
      final api = NeriApiClient(
        httpClient: MockClient((_) async => _jsonResponse({})),
      );
      addTearDown(api.close);
      await tester.pumpWidget(_screen(api, directory.path, item()));
      await _waitForImage(tester);
      await tapExport(tester);

      expect(find.widgetWithText(FilledButton, 'CSV'), findsOneWidget);
      expect(find.widgetWithText(TextButton, 'Excel'), findsOneWidget);
      final rememberY = tester.getCenter(find.text('不再询问')).dy;
      final cancelY = tester.getCenter(find.text('取消')).dy;
      expect((rememberY - cancelY).abs(), lessThan(3));
    });
  }

  testWidgets('cancelled export choice sends no request', (tester) async {
    _setSize(tester, 1280);
    var exports = 0;
    final api = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/validation/export') exports++;
        return _jsonResponse({});
      }),
    );
    addTearDown(api.close);
    await tester.pumpWidget(_screen(api, directory.path, item()));
    await _waitForImage(tester);
    await tapExport(tester);
    await tester.tap(find.text('取消'));
    await tester.pump(const Duration(milliseconds: 300));
    expect(exports, 0);
  });

  testWidgets('cancelled child scope sends no request', (tester) async {
    _setSize(tester, 1280);
    var exports = 0;
    final api = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/validation/export') exports++;
        return _jsonResponse({});
      }),
    );
    addTearDown(api.close);
    await tester.pumpWidget(
      _screen(api, directory.path, item(), rootPath: directory.parent.path),
    );
    await _waitForImage(tester);
    await tapExport(tester);
    await tester.tap(find.text('取消'));
    await tester.pump(const Duration(milliseconds: 300));
    expect(exports, 0);
  });

  for (final width in [1280.0, 600.0]) {
    testWidgets('export button fills right actions without card at $width', (
      tester,
    ) async {
      _setSize(tester, width);
      final api = NeriApiClient(
        httpClient: MockClient((_) async => _jsonResponse({})),
      );
      addTearDown(api.close);
      await tester.pumpWidget(
        _screen(
          api,
          directory.path,
          item(),
          theme: buildNeriTheme(ColorScheme.fromSeed(seedColor: Colors.brown)),
        ),
      );
      await _waitForImage(tester);
      final exportButton = find.widgetWithText(FilledButton, '导出');
      expect(
        find.ancestor(of: exportButton, matching: find.byType(Card)),
        findsNothing,
      );
      final quickMarkCard = find
          .ancestor(of: find.text('正确'), matching: find.byType(Card))
          .first;
      final quantityCard = find
          .ancestor(of: find.text('1'), matching: find.byType(Card))
          .first;
      final exportSize = tester.getSize(exportButton);
      expect(
        exportSize.width,
        tester.getSize(quickMarkCard).width +
            8 +
            tester.getSize(quantityCard).width,
      );
      expect(exportSize.height, 64);
    });
  }

  testWidgets(
    'unrelated metadata preserves confirmed box feedback before parent refresh',
    (tester) async {
      _setSize(tester, 1280);
      final original = item();
      final other = DetectionItem(
        filename: 'other.png',
        path: '${directory.path}/missing.png',
        fileType: 'png',
        species: const ['中亚兔'],
      );
      final api = NeriApiClient(
        httpClient: MockClient((request) async {
          if (request.method == 'POST') {
            final body = jsonDecode(request.body) as Map;
            return _jsonResponse({
              'operation_id': body['feedback_operation_id'],
              'item': {
                'path': imagePath,
                'species': ['已确认物种'],
                'detection_boxes': [
                  {
                    'species': '已确认物种',
                    'bbox': [0.2, 0.2, 0.8, 0.8],
                    'confidence': 0.752,
                    'observation_id': 'bird-1',
                  },
                ],
              },
            });
          }
          return _jsonResponse({'species': '中亚兔', 'nearest_species': []});
        }),
      );
      addTearDown(api.close);
      await tester.pumpWidget(
        _screen(
          api,
          directory.path,
          original,
          itemsOverride: [original, other],
        ),
      );
      await _waitForImage(tester);
      await tester.tapAt(tester.getCenter(find.byType(Image)));
      await tester.pumpAndSettle();
      final dialog = find.byKey(const ValueKey('dino-detection-box-dialog'));
      await tester.tap(
        find.descendant(
          of: dialog,
          matching: find.widgetWithText(OutlinedButton, '正确'),
        ),
      );
      await tester.pumpAndSettle();
      expect(find.text('物种：已确认物种'), findsOneWidget);
      await tester.tap(
        find.descendant(
          of: dialog,
          matching: find.widgetWithText(TextButton, '关闭'),
        ),
      );
      await tester.pumpAndSettle();
      await tester.pumpWidget(
        _screen(
          api,
          directory.path,
          original,
          itemsOverride: [
            original,
            DetectionItem(
              filename: other.filename,
              path: other.path,
              fileType: other.fileType,
              species: other.species,
              width: 20,
            ),
          ],
        ),
      );
      await tester.pumpAndSettle();
      await tester.tapAt(tester.getCenter(find.byType(Image)));
      await tester.pumpAndSettle();
      expect(
        find.text('物种：已确认物种'),
        findsOneWidget,
        reason:
            'An unrelated metadata echo must not overwrite newer cache-only feedback',
      );
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox());
    },
  );

  for (final width in [1280.0, 600.0]) {
    testWidgets(
      'box click opens details and feature explanation at width $width',
      (tester) async {
        _setSize(tester, width);
        final requests = <http.Request>[];
        final feedback = Completer<http.Response>();
        final api = NeriApiClient(
          httpClient: MockClient((request) async {
            requests.add(request);
            if (request.method == 'POST') return feedback.future;
            return _jsonResponse({
              'species': '中亚兔',
              'nearest_species': [
                {'name': '中亚兔', 'squared_distance': 0.1},
              ],
            });
          }),
        );
        addTearDown(api.close);
        await tester.pumpWidget(_screen(api, directory.path, item()));
        await _waitForImage(tester);

        expect(
          find.byKey(const ValueKey('dino-feature-explanation-button')),
          findsNothing,
        );
        expect(
          find.byKey(const ValueKey('detection-box-details')),
          findsNothing,
        );
        expect(requests, isEmpty);
        final center = tester.getCenter(find.byType(Image));
        await tester.tapAt(
          tester.getTopLeft(find.byType(Image)) + const Offset(10, 10),
        );
        await tester.pump();
        expect(find.byType(AlertDialog), findsNothing);

        await tester.tapAt(center);
        await tester.pumpAndSettle();
        final dialog = find.byKey(const ValueKey('dino-detection-box-dialog'));
        expect(dialog, findsOneWidget);
        expect(
          find.descendant(of: dialog, matching: find.text('#1 中亚兔 · 检测框校验')),
          findsOneWidget,
        );
        expect(
          find.descendant(of: dialog, matching: find.text('置信度：0.752')),
          findsOneWidget,
        );
        expect(
          find.descendant(of: dialog, matching: find.text('候选：中亚兔 75.2%')),
          findsOneWidget,
        );
        expect(
          find.descendant(
            of: dialog,
            matching: find.byType(DinoFeatureExplanationPanel),
          ),
          findsOneWidget,
        );
        expect(find.text('最近类：中亚兔'), findsOneWidget);
        expect(requests.single.url.path, endsWith('/bird-1/explain'));
        expect(
          requests.single.url.queryParameters['classification_model_path'],
          'model.dino',
        );

        final correct = find.descendant(
          of: dialog,
          matching: find.widgetWithText(OutlinedButton, '正确'),
        );
        await tester.tap(correct);
        await tester.pump();
        final buttons = find.descendant(
          of: dialog,
          matching: find.byType(OutlinedButton),
        );
        expect(
          tester
              .widgetList<OutlinedButton>(buttons)
              .every((button) => button.onPressed == null),
          isTrue,
        );
        final body = jsonDecode(requests.last.body) as Map<String, dynamic>;
        expect(body['observation_id'], 'bird-1');
        expect(body['file_path'], imagePath);
        expect(body['action'], 'correct');
        feedback.complete(
          _jsonResponse({
            'item': {
              'path': imagePath,
              'species': ['已确认物种'],
              'detection_boxes': [
                {
                  'species': '已确认物种',
                  'bbox': [0.2, 0.2, 0.8, 0.8],
                  'confidence': 0.752,
                  'observation_id': 'bird-1',
                },
              ],
            },
            'operation_id': body['feedback_operation_id'],
          }),
        );
        await tester.pumpAndSettle();
        expect(find.text('物种：已确认物种'), findsOneWidget);
        expect(tester.widget<OutlinedButton>(correct).onPressed, isNotNull);
        await tester.tap(
          find.descendant(
            of: dialog,
            matching: find.widgetWithText(TextButton, '关闭'),
          ),
        );
        await tester.pumpAndSettle();
        expect(dialog, findsNothing);
        expect(
          find.byKey(const ValueKey('detection-box-details')),
          findsNothing,
        );
        await tester.pump(const Duration(seconds: 5));
        await tester.pumpAndSettle();
        expect(tester.takeException(), isNull);
        await tester.pumpWidget(const SizedBox.shrink());
      },
    );
  }

  testWidgets(
    'box without observation still opens details with disabled feedback',
    (tester) async {
      _setSize(tester, 1280);
      final requests = <http.Request>[];
      final api = NeriApiClient(
        httpClient: MockClient((request) async {
          requests.add(request);
          return _jsonResponse({});
        }),
      );
      addTearDown(api.close);
      await tester.pumpWidget(
        _screen(api, directory.path, item(observationId: null)),
      );
      await _waitForImage(tester);
      await tester.tapAt(tester.getCenter(find.byType(Image)));
      await tester.pumpAndSettle();
      final dialog = find.byKey(const ValueKey('dino-detection-box-dialog'));
      expect(dialog, findsOneWidget);
      expect(find.text('该检测框没有可用的特征空间数据'), findsOneWidget);
      expect(
        find.byKey(const ValueKey('detection-box-details')),
        findsOneWidget,
      );
      final buttons = find.descendant(
        of: dialog,
        matching: find.byType(OutlinedButton),
      );
      expect(
        tester
            .widgetList<OutlinedButton>(buttons)
            .every((button) => button.onPressed == null),
        isTrue,
      );
      expect(requests, isEmpty);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'dinov3 box details separate per-box recognition evidence and sequence gate score',
    (tester) async {
      _setSize(tester, 1280);
      final api = NeriApiClient(
        httpClient: MockClient((_) async => _jsonResponse({})),
      );
      addTearDown(api.close);
      final dinov3Item = DetectionItem(
        filename: 'bird.png',
        path: imagePath,
        fileType: 'png',
        width: 10,
        height: 10,
        species: const ['岩松鼠'],
        confidence: 0.785,
        detectionBoxes: const [
          DetectionBox(
            species: '岩松鼠',
            predictedSpecies: '岩松鼠',
            confidence: 0.785,
            bbox: [0.2, 0.2, 0.8, 0.8],
            trackId: 'event:cam1:2026-11-01T10:00:00:track_00',
            candidates: [
              {
                'name': '岩松鼠',
                'conf': 0.785,
                'raw_det_conf': 0.722,
                'box_known_score': 1.090,
                'box_threshold': -0.993,
                'sequence_known_score': 0.864,
                'sequence_threshold': -0.332,
                'within_score': 0.480,
                'ridge_score': 0.256,
                'raw_ridge_score': 0.241,
              },
              {'name': '猪獾', 'ridge_score': 0.142},
            ],
          ),
        ],
      );
      await tester.pumpWidget(_screen(api, directory.path, dinov3Item));
      await _waitForImage(tester);
      await tester.tapAt(tester.getCenter(find.byType(Image)));
      await tester.pumpAndSettle();

      final details = find.byKey(const ValueKey('detection-box-details'));
      expect(details, findsOneWidget);
      expect(
        find.descendant(of: details, matching: find.text('置信度：0.785')),
        findsOneWidget,
      );
      expect(
        find.descendant(of: details, matching: find.text('检测置信度：0.722')),
        findsOneWidget,
      );
      expect(
        find.descendant(
          of: details,
          matching: find.text('单框门控：1.090 (阈值 -0.993)'),
        ),
        findsOneWidget,
      );
      expect(
        find.descendant(
          of: details,
          matching: find.text('序列门控：0.864 (阈值 -0.332)'),
        ),
        findsOneWidget,
      );
      expect(
        find.descendant(of: details, matching: find.text('Within分数：0.480')),
        findsOneWidget,
      );
      expect(
        find.descendant(
          of: details,
          matching: find.text('Ridge分数：0.256 (单框 0.241)'),
        ),
        findsOneWidget,
      );
      expect(find.text('候选：岩松鼠 78.5% / 猪獾 Ridge 0.142'), findsOneWidget);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'dinov3 box without observation id synthesizes one and enables feedback',
    (tester) async {
      _setSize(tester, 1280);
      final expectedObservationId = synthesizeDinoV3ObservationId(
        imagePath,
        0,
        const [0.2, 0.2, 0.8, 0.8],
      );
      expect(expectedObservationId, startsWith('dinov3:'));
      Map<String, dynamic>? feedbackBody;
      final dinov3Item = DetectionItem(
        filename: 'bird.png',
        path: imagePath,
        fileType: 'png',
        width: 10,
        height: 10,
        species: const ['岩松鼠'],
        confidence: 0.785,
        detectionBoxes: const [
          DetectionBox(
            species: '岩松鼠',
            confidence: 0.785,
            bbox: [0.2, 0.2, 0.8, 0.8],
            candidates: [
              {
                'name': '岩松鼠',
                'conf': 0.785,
                'best_known_species': '岩松鼠',
                'ridge_score': 0.256,
                'within_score': 0.480,
                'box_known_score': 1.090,
                'sequence_known_score': 0.864,
              },
            ],
          ),
        ],
      );
      final api = NeriApiClient(
        httpClient: MockClient((request) async {
          if (request.method == 'GET' &&
              request.url.path.endsWith('/explain')) {
            return _jsonResponse({
              'species': '岩松鼠',
              'known_score': 0.864,
              'threshold': -0.332,
              'rejection': {'mode': 'ridge_within'},
              'nearest_species': [
                {
                  'name': '岩松鼠',
                  'squared_distance': 0.1,
                  'cosine_score': 0.9,
                  'source': 'memory',
                },
              ],
              'projection': {
                'method': 'pca',
                'species': ['岩松鼠'],
                'points': [
                  {'kind': 'current', 'species': '岩松鼠', 'x': 0, 'y': 0},
                ],
              },
              'current_example_available': false,
            });
          }
          if (request.method == 'POST' &&
              request.url.path.endsWith('/api/dino/feedback/box')) {
            feedbackBody = jsonDecode(request.body) as Map<String, dynamic>;
            return _jsonResponse({
              'item': {
                'filename': 'bird.png',
                'path': imagePath,
                'file_type': 'png',
                'species': ['岩松鼠'],
                'detection_boxes': [
                  {
                    'species': '岩松鼠',
                    'bbox': [0.2, 0.2, 0.8, 0.8],
                    'confidence': 0.785,
                    'observation_id': expectedObservationId,
                    'predicted_species': '岩松鼠',
                  },
                ],
              },
              'operation_id': feedbackBody?['feedback_operation_id'],
            });
          }
          return _jsonResponse({});
        }),
      );
      addTearDown(api.close);
      await tester.pumpWidget(_screen(api, directory.path, dinov3Item));
      await _waitForImage(tester);
      await tester.tapAt(tester.getCenter(find.byType(Image)));
      await tester.pumpAndSettle();

      final dialog = find.byKey(const ValueKey('dino-detection-box-dialog'));
      expect(dialog, findsOneWidget);
      expect(
        find.descendant(of: dialog, matching: find.text('#1 岩松鼠 · 检测框校验')),
        findsOneWidget,
      );
      final panel = tester.widget<DinoFeatureExplanationPanel>(
        find.byType(DinoFeatureExplanationPanel),
      );
      expect(panel.observationId, expectedObservationId);
      expect(find.text('序列门控 0.8640'), findsOneWidget);
      expect(
        find.text('二维图用于展示白化特征空间近邻；实际判定结合因果历史适配、Within 近邻与 Ridge 序列门控。'),
        findsOneWidget,
      );
      expect(find.textContaining('该检测框缺少检测框标识'), findsNothing);

      for (final label in ['正确', '修改物种', '空 / 误检', '不参与学习']) {
        final button = find.descendant(
          of: dialog,
          matching: find.widgetWithText(OutlinedButton, label),
        );
        expect(button, findsOneWidget);
        expect(
          tester.widget<OutlinedButton>(button).onPressed,
          isNotNull,
          reason:
              '$label should be enabled for a synthesized dinov3 observation',
        );
      }

      await tester.tap(
        find.descendant(
          of: dialog,
          matching: find.widgetWithText(OutlinedButton, '正确'),
        ),
      );
      await tester.pumpAndSettle();
      expect(feedbackBody?['observation_id'], expectedObservationId);
      expect(feedbackBody?['file_path'], imagePath);

      await tester.pump(const Duration(seconds: 5));
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );
}

class _CountingList<T> extends ListBase<T> {
  _CountingList(this._items);

  final List<T> _items;
  int reads = 0;

  @override
  int get length => _items.length;

  @override
  set length(int value) => _items.length = value;

  @override
  T operator [](int index) {
    reads++;
    return _items[index];
  }

  @override
  void operator []=(int index, T value) => _items[index] = value;
}

Widget _preview(
  DetectionItem item, {
  String selectedSpecies = previewAllSpeciesLabel,
  double confidence = 0.25,
  double textScale = 1,
  ValueChanged<double>? onConfidenceChanged,
  ValueChanged<bool>? onShowDetectionsChanged,
  ValueChanged<DetectionItem>? onDetect,
}) => MaterialApp(
  theme: buildNeriTheme(ColorScheme.fromSeed(seedColor: Colors.green)),
  builder: (context, child) => MediaQuery(
    data: MediaQuery.of(
      context,
    ).copyWith(textScaler: TextScaler.linear(textScale)),
    child: child!,
  ),
  home: Scaffold(
    body: PreviewScreen(
      inputPath: File(item.path).parent.path,
      items: [item],
      selectedIndex: 0,
      selectedItem: item,
      speciesTypes: const {},
      useCombinedConfidence: false,
      showDetections: true,
      onShowDetectionsChanged: onShowDetectionsChanged ?? (_) {},
      selectedSpeciesFilter: selectedSpecies,
      onSpeciesFilterChanged: (_) {},
      confidenceThreshold: confidence,
      onConfidenceThresholdChanged: onConfidenceChanged ?? (_) {},
      detecting: false,
      loading: false,
      onDetectCurrentImage: onDetect ?? (_) {},
      onSelected: (_, _) {},
      onLoadMetadata: (_) async {},
      onRefresh: () async {},
      onOpenExternal: (_) {},
    ),
  ),
);

void _setSize(WidgetTester tester, double width) {
  tester.view.physicalSize = Size(width, 900);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.resetPhysicalSize);
  addTearDown(tester.view.resetDevicePixelRatio);
}

Future<void> _waitForImage(WidgetTester tester) async {
  for (
    var attempt = 0;
    attempt < 100 && find.byType(Image).evaluate().isEmpty;
    attempt++
  ) {
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 10)),
    );
    await tester.pump();
  }
  expect(find.byType(Image), findsOneWidget);
  await tester.pumpAndSettle();
}

http.Response _jsonResponse(Map<String, dynamic> json) => http.Response(
  jsonEncode(json),
  200,
  headers: const {'content-type': 'application/json; charset=utf-8'},
);

Widget _screen(
  NeriApiClient api,
  String path,
  DetectionItem item, {
  String? rootPath,
  String exportMode = 'ask',
  Future<void> Function(String)? onFormatModeChanged,
  ThemeData? theme,
  List<DetectionItem>? itemsOverride,
  bool autoGroup = false,
  double minFrameRatio = 0,
  MarkValidationItem? onMarkItem,
  MarkValidationItems? onMarkItems,
}) => MaterialApp(
  theme: theme,
  home: Scaffold(
    body: SpeciesValidationScreen(
      apiClient: api,
      inputPath: path,
      rootInputPath: rootPath ?? path,
      rootItems: itemsOverride ?? [item],
      exportFormatMode: exportMode,
      classificationModelPath: 'model.dino',
      items: itemsOverride ?? [item],
      loading: false,
      refreshVersion: 0,
      speciesTypes: const {},
      useCombinedConfidence: false,
      minFrameRatio: minFrameRatio,
      autoGroup: autoGroup,
      collapseGroups: autoGroup,
      autoGroupDetectBurst: false,
      autoGroupBurstSize: 4,
      autoGroupGapSeconds: 1800,
      autoSortQuickMarks: false,
      undoSteps: 20,
      quickMarkSpecies: const [],
      quickMarkRecentHistory: const [],
      quickMarkUsageCounts: const {},
      quantityButtons: const ['1'],
      exportColumns: const [],
      favoritePhotoPaths: const [],
      favoritePhotoExportMode: favoritePhotoExportAsk,
      emptyPhotoDeleteMode: emptyPhotoDeleteAsk,
      onRefresh: () async {},
      onLoadMetadata: (_) async {},
      onOpenExternal: (_) {},
      onMarkItem:
          onMarkItem ??
          (
            candidate,
            action, {
            speciesName,
            speciesCount,
            speciesType,
            remark,
          }) async => candidate,
      onMarkItems:
          onMarkItems ??
          (
            candidates,
            action, {
            speciesName,
            speciesCount,
            speciesType,
            remark,
          }) async => candidates,
      onQuickMarkUsed: (_) async {},
      onQuickMarkReverted: (_) async {},
      onRedetectItems: (_, {required confidence}) async {},
      onFavoritePhotoPathsChanged: (_) async {},
      onFavoritePhotoExportModeChanged: (_) async {},
      onExportFormatModeChanged: onFormatModeChanged ?? (_) async {},
      onEmptyPhotoDeleteModeChanged: (_) async {},
      onAutoGroupInferredBurstSizeChanged: (_) {},
    ),
  ),
);
