import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/app_theme.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';
import 'package:neri_flutter/src/widgets/dinov2_feature_scatter.dart';

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

  Future<void> tapExport(WidgetTester tester) async {
    final button = find.widgetWithText(FilledButton, '导出');
    await tester.ensureVisible(button);
    await tester.pumpAndSettle();
    await tester.tap(button);
    await tester.pump(const Duration(milliseconds: 300));
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
    await tapExport(tester);
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
          find.byKey(const ValueKey('dinov2-feature-explanation-button')),
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
        final dialog = find.byKey(
          const ValueKey('dinov2-detection-box-dialog'),
        );
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
            matching: find.byType(DinoV2FeatureExplanationPanel),
          ),
          findsOneWidget,
        );
        expect(find.text('最近类：中亚兔'), findsOneWidget);
        expect(requests.single.url.path, endsWith('/bird-1/explain'));
        expect(
          requests.single.url.queryParameters['classification_model_path'],
          'model.dinov2',
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
      final dialog = find.byKey(const ValueKey('dinov2-detection-box-dialog'));
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
}

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
}) => MaterialApp(
  theme: theme,
  home: Scaffold(
    body: SpeciesValidationScreen(
      apiClient: api,
      inputPath: path,
      rootInputPath: rootPath ?? path,
      rootItems: [item],
      exportFormatMode: exportMode,
      classificationModelPath: 'model.dinov2',
      items: [item],
      loading: false,
      refreshVersion: 0,
      speciesTypes: const {},
      useCombinedConfidence: false,
      minFrameRatio: 0,
      autoGroup: false,
      collapseGroups: false,
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
          (
            candidate,
            action, {
            speciesName,
            speciesCount,
            speciesType,
            remark,
          }) async => candidate,
      onMarkItems:
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
