import 'dart:collection';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';
import 'package:neri_flutter/src/utils/validation_cache_delta.dart';

const _widgetItemCount = 60;

void main() {
  late Directory tempDir;
  late NeriApiClient apiClient;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp(
      'neri_validation_incremental_',
    );
    final pngBytes = base64Decode(
      'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
    );
    for (var index = 0; index < _widgetItemCount; index++) {
      await File('${tempDir.path}/image-$index.jpg').writeAsBytes(pngBytes);
    }
    apiClient = NeriApiClient(
      httpClient: MockClient(
        (_) async => http.Response(
          '{}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        ),
      ),
    );
  });

  tearDown(() async {
    apiClient.close();
    if (await tempDir.exists()) {
      await tempDir.delete(recursive: true);
    }
  });

  test('pending echo checks only pending item paths', () {
    final items = <_Item>[
      for (var index = 0; index < 10000; index++) _Item('path-$index'),
    ];
    final indexes = <String, int>{
      for (var index = 0; index < items.length; index++)
        items[index].path: index,
    };
    var pathReads = 0;

    final accepted = ValidationCacheDelta.canAdoptPendingEcho<_Item>(
      nextItems: items,
      cachedLength: items.length,
      cachedIndexByPath: indexes,
      pendingPaths: const <String>{'path-4321'},
      pathOf: (item) {
        pathReads += 1;
        return item.path;
      },
      groupingSettingsUnchanged: true,
      refreshVersionUnchanged: true,
    );

    expect(accepted, isTrue);
    expect(pathReads, 1);
  });

  test(
    'pending echo falls back when structural preconditions are uncertain',
    () {
      final items = <_Item>[_Item('a'), _Item('b')];
      final indexes = <String, int>{'a': 0, 'b': 1};

      bool evaluate({
        List<_Item>? nextItems,
        int? cachedLength,
        Map<String, int>? cachedIndexByPath,
        Set<String>? pendingPaths,
        bool groupingSettingsUnchanged = true,
        bool refreshVersionUnchanged = true,
      }) {
        return ValidationCacheDelta.canAdoptPendingEcho<_Item>(
          nextItems: nextItems ?? items,
          cachedLength: cachedLength ?? items.length,
          cachedIndexByPath: cachedIndexByPath ?? indexes,
          pendingPaths: pendingPaths ?? const <String>{'a'},
          pathOf: (item) => item.path,
          groupingSettingsUnchanged: groupingSettingsUnchanged,
          refreshVersionUnchanged: refreshVersionUnchanged,
        );
      }

      expect(evaluate(cachedLength: 3), isFalse);
      expect(evaluate(cachedIndexByPath: const <String, int>{'b': 1}), isFalse);
      expect(
        evaluate(nextItems: <_Item>[_Item('changed'), _Item('b')]),
        isFalse,
      );
      expect(evaluate(groupingSettingsUnchanged: false), isFalse);
      expect(evaluate(refreshVersionUnchanged: false), isFalse);
      expect(evaluate(pendingPaths: const <String>{}), isFalse);
    },
  );

  testWidgets('one parent echo does not rescan grouping data for every item', (
    tester,
  ) async {
    debugPrint('[validation-perf] widget:start');
    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final key = GlobalKey<_IncrementalHarnessState>();
    debugPrint('[validation-perf] pumpWidget:start');
    await tester.pumpWidget(
      _IncrementalHarness(key: key, apiClient: apiClient, tempDir: tempDir),
    );
    debugPrint('[validation-perf] pumpWidget:done');
    await tester.pump();
    debugPrint('[validation-perf] first-pump:done');
    await tester.pump(const Duration(milliseconds: 100));
    debugPrint('[validation-perf] settle-100ms:done');

    debugPrint('[validation-perf] tap:start');
    await tester.tap(find.text('正确').first);
    debugPrint('[validation-perf] tap:done');
    await tester.pump();
    debugPrint('[validation-perf] post-tap-pump:done');
    await tester.pump(const Duration(milliseconds: 150));
    debugPrint('[validation-perf] settle-150ms:done');

    debugPrint(
      '[validation-perf] echo-reads:${key.currentState!.detectionDataReads}',
    );
    expect(
      key.currentState!.detectionDataReads,
      lessThan(200),
      reason:
          'A parent echo should inspect only the affected auto-group, not '
          'recompute the grouping signature for all $_widgetItemCount items.',
    );
  });
}

class _Item {
  const _Item(this.path);

  final String path;
}

class _CountingMap extends MapBase<String, dynamic> {
  _CountingMap(Map<String, dynamic> values, this.onRead)
    : _values = Map<String, dynamic>.from(values);

  final Map<String, dynamic> _values;
  final VoidCallback onRead;

  @override
  dynamic operator [](Object? key) {
    onRead();
    return _values[key];
  }

  @override
  void operator []=(String key, dynamic value) {
    _values[key] = value;
  }

  @override
  void clear() => _values.clear();

  @override
  Iterable<String> get keys => _values.keys;

  @override
  dynamic remove(Object? key) => _values.remove(key);
}

class _IncrementalHarness extends StatefulWidget {
  const _IncrementalHarness({
    required this.apiClient,
    required this.tempDir,
    super.key,
  });

  final NeriApiClient apiClient;
  final Directory tempDir;

  @override
  State<_IncrementalHarness> createState() => _IncrementalHarnessState();
}

class _IncrementalHarnessState extends State<_IncrementalHarness> {
  late List<DetectionItem> _items;
  int detectionDataReads = 0;

  @override
  void initState() {
    super.initState();
    _items = <DetectionItem>[
      for (var index = 0; index < _widgetItemCount; index++)
        DetectionItem(
          filename: 'image-$index.jpg',
          path: '${widget.tempDir.path}/image-$index.jpg',
          fileType: 'jpg',
          modifiedAt: '2026-09-14T00:00:00Z',
          species: const <String>['豹猫'],
          confidence: 0.95,
          detectionData: _CountingMap(const <String, dynamic>{
            '物种名称': '豹猫',
            '物种数量': '1',
          }, () => detectionDataReads += 1),
        ),
    ];
  }

  void _beginParentEchoMeasurement() {
    debugPrint(
      '[validation-perf] parent-echo:start prior-reads:$detectionDataReads',
    );
    detectionDataReads = 0;
  }

  DetectionItem _markedCopy(DetectionItem item, String action) {
    final data = Map<String, dynamic>.from(item.detectionData);
    final validated = action == 'unverified' ? null : true;
    if (action == 'unverified') {
      data.remove('最低置信度');
    } else {
      data['最低置信度'] = '人工校验';
    }
    return DetectionItem(
      filename: item.filename,
      path: item.path,
      fileType: item.fileType,
      dateTaken: item.dateTaken,
      modifiedAt: item.modifiedAt,
      width: item.width,
      height: item.height,
      sizeBytes: item.sizeBytes,
      species: item.species,
      confidence: item.confidence,
      detectionBoxes: item.detectionBoxes,
      detectionData: data,
      error: item.error,
      validated: validated,
    );
  }

  Future<DetectionItem> _markItem(
    DetectionItem item,
    String action, {
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
  }) async {
    final updated = _markedCopy(item, action);
    _beginParentEchoMeasurement();
    setState(() {
      _items = [
        for (final current in _items)
          if (current.path == item.path) updated else current,
      ];
    });
    return updated;
  }

  Future<List<DetectionItem>> _markItems(
    List<DetectionItem> items,
    String action, {
    String? speciesName,
    String? speciesCount,
    String? speciesType,
    String? remark,
  }) async {
    final targets = items.map((item) => item.path).toSet();
    final updatedByPath = <String, DetectionItem>{
      for (final item in items) item.path: _markedCopy(item, action),
    };
    _beginParentEchoMeasurement();
    setState(() {
      _items = [
        for (final current in _items)
          if (targets.contains(current.path))
            updatedByPath[current.path]!
          else
            current,
      ];
    });
    return [for (final item in items) updatedByPath[item.path]!];
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: SpeciesValidationScreen(
          apiClient: widget.apiClient,
          inputPath: widget.tempDir.path,
          items: _items,
          loading: false,
          refreshVersion: 0,
          speciesTypes: const <String, String>{'豹猫': '兽类'},
          useCombinedConfidence: false,
          minFrameRatio: 0,
          autoGroup: true,
          collapseGroups: false,
          autoGroupDetectBurst: false,
          autoGroupBurstSize: 20,
          autoGroupGapSeconds: 1800,
          autoSortQuickMarks: false,
          undoSteps: 20,
          quickMarkSpecies: const <String>['豹猫'],
          quickMarkRecentHistory: const <String>[],
          quickMarkUsageCounts: const <String, int>{},
          quantityButtons: const <String>['1'],
          exportColumns: const <String>[],
          favoritePhotoPaths: const <String>[],
          favoritePhotoExportMode: favoritePhotoExportNever,
          emptyPhotoDeleteMode: emptyPhotoDeleteNever,
          onRefresh: () async {},
          onLoadMetadata: (_) async {},
          onOpenExternal: (_) {},
          onMarkItem: _markItem,
          onMarkItems: _markItems,
          onQuickMarkUsed: (_) async {},
          onQuickMarkReverted: (_) async {},
          onRedetectItems: (_, {required confidence}) async {},
          onFavoritePhotoPathsChanged: (_) async {},
          onFavoritePhotoExportModeChanged: (_) async {},
          onEmptyPhotoDeleteModeChanged: (_) async {},
          onAutoGroupInferredBurstSizeChanged: (_) {},
        ),
      ),
    );
  }
}
