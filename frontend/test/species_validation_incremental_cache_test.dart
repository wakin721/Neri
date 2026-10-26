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

const _widgetItemCount = 600;

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

  for (final afterMark in [false, true]) {
    testWidgets(
      'metadata preserving group boundaries keeps large validation interactive (afterMark=$afterMark)',
      (tester) async {
        tester.view.physicalSize = const Size(1280, 900);
        tester.view.devicePixelRatio = 1;
        addTearDown(tester.view.resetPhysicalSize);
        addTearDown(tester.view.resetDevicePixelRatio);
        final key = GlobalKey<_IncrementalHarnessState>();
        await tester.pumpWidget(
          _IncrementalHarness(
            key: key,
            apiClient: apiClient,
            tempDir: tempDir,
            itemCount: 23310,
          ),
        );
        for (var attempt = 0; attempt < 500; attempt++) {
          if (_blockedValidation().evaluate().isEmpty) break;
          await tester.runAsync(
            () => Future<void>.delayed(const Duration(milliseconds: 10)),
          );
          await tester.pump(const Duration(milliseconds: 16));
        }
        expect(_blockedValidation(), findsNothing);
        key.currentState!.resetReadCount();
        if (afterMark) {
          key.currentState!.loadMetadataAfterMark = true;
          await tester.tap(find.text('正确').first);
        } else {
          key.currentState!.applyMetadata(0, captureTimeOffsetSeconds: 1);
        }
        await tester.pump();
        expect(
          _blockedValidation(),
          findsNothing,
          reason:
              'Loading metadata for one photo must not prepare all 23310 items again',
        );
        await tester.pump(const Duration(milliseconds: 150));
        expect(
          key.currentState!.detectionDataReads,
          lessThan(500),
          reason:
              'Unchanged photos must not have their grouping metadata read again',
        );
        expect(key.currentState!.metadataApplications, 1);
        expect(find.text('正确'), findsWidgets);
        await tester.tap(find.text('正确').first);
        await tester.pump();
        expect(
          key.currentState!.lastMarkedItem?.width,
          640,
          reason:
              'The next mark must receive the updated metadata from the bucket cache',
        );
        expect(tester.takeException(), isNull);
        await tester.pumpWidget(const SizedBox());
      },
    );
  }

  for (final unresolved in [false, true]) {
    testWidgets(
      'metadata refreshes changed rejection membership (unresolved=$unresolved)',
      (tester) async {
        tester.view.physicalSize = const Size(1280, 900);
        tester.view.devicePixelRatio = 1;
        addTearDown(tester.view.resetPhysicalSize);
        addTearDown(tester.view.resetDevicePixelRatio);
        var requests = 0;
        final api = NeriApiClient(
          httpClient: MockClient((request) async {
            expect(request.url.path, '/api/dinov2/feedback/rejection-clusters');
            requests++;
            final ids =
                (jsonDecode(request.body) as Map)['observation_ids'] as List;
            return http.Response(
              jsonEncode({
                for (final id in ids.where((id) => id != 'missing'))
                  id: {
                    'cluster_id': 'cluster-$id',
                    'candidate_number': id == 'a' ? 1 : (id == 'b' ? 2 : 3),
                    'model_fingerprint': 'fp',
                    'event_count': 10,
                    'min_cluster_events': 5,
                    'is_noise': false,
                    'density_verified': true,
                  },
              }),
              200,
              headers: {'content-type': 'application/json; charset=utf-8'},
            );
          }),
        );
        addTearDown(api.close);
        final key = GlobalKey<_IncrementalHarnessState>();
        await tester.pumpWidget(
          _IncrementalHarness(
            key: key,
            apiClient: api,
            tempDir: tempDir,
            rejected: true,
            initialUnassigned: unresolved,
          ),
        );
        Future<void> waitUntilReady() async {
          for (var attempt = 0; attempt < 500; attempt++) {
            if (_blockedValidation().evaluate().isEmpty) break;
            await tester.runAsync(
              () => Future<void>.delayed(const Duration(milliseconds: 10)),
            );
            await tester.pump(const Duration(milliseconds: 16));
          }
          expect(_blockedValidation(), findsNothing);
        }

        await waitUntilReady();
        expect(requests, 1);
        if (unresolved) {
          expect(find.text('拒识（未分簇）'), findsWidgets);
        }
        key.currentState!.applyMetadata(
          0,
          observationId: unresolved ? 'c' : 'b',
        );
        await tester.pump();
        await waitUntilReady();
        final label = unresolved ? '拒识簇 #3' : '拒识簇 #2';
        final tile = tester.widget<ListTile>(
          find
              .ancestor(of: find.text(label), matching: find.byType(ListTile))
              .first,
        );
        expect((tile.subtitle! as Text).data, unresolved ? '1 个文件' : '301 个文件');
        expect(requests, unresolved ? 2 : 1);
        expect(find.text('拒识（未分簇）'), findsNothing);
        expect(tester.takeException(), isNull);
        await tester.pumpWidget(const SizedBox());
      },
    );
  }

  for (final burstSize in [20, 1]) {
    testWidgets(
      'multi-cluster data keeps marking work local (burst=$burstSize)',
      (tester) async {
        tester.view.physicalSize = const Size(1280, 900);
        tester.view.devicePixelRatio = 1;
        addTearDown(tester.view.resetPhysicalSize);
        addTearDown(tester.view.resetDevicePixelRatio);
        final api = NeriApiClient(
          httpClient: MockClient((request) async {
            final ids =
                (jsonDecode(request.body) as Map)['observation_ids'] as List;
            return http.Response(
              jsonEncode({
                for (final id in ids)
                  id: {
                    'cluster_id': 'cluster-$id',
                    'candidate_number': id == 'a' ? 1 : 2,
                    'model_fingerprint': 'fp',
                    'event_count': 10,
                    'min_cluster_events': 5,
                    'is_noise': false,
                    'density_verified': true,
                  },
              }),
              200,
              headers: {'content-type': 'application/json; charset=utf-8'},
            );
          }),
        );
        addTearDown(api.close);
        final key = GlobalKey<_IncrementalHarnessState>();
        await tester.pumpWidget(
          _IncrementalHarness(
            key: key,
            apiClient: api,
            tempDir: tempDir,
            itemCount: 23310,
            multiClusterFile: true,
            burstSize: burstSize,
          ),
        );
        for (var attempt = 0; attempt < 500; attempt++) {
          if (_blockedValidation().evaluate().isEmpty) break;
          await tester.runAsync(
            () => Future<void>.delayed(const Duration(milliseconds: 10)),
          );
          await tester.pump(const Duration(milliseconds: 16));
        }
        expect(_blockedValidation(), findsNothing);
        expect(find.textContaining('拒识簇 #1'), findsWidgets);
        expect(find.textContaining('拒识簇 #2'), findsWidgets);
        key.currentState!.resetReadCount();
        await tester.tap(find.text('正确').first);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 150));
        expect(_blockedValidation(), findsNothing);
        expect(key.currentState!.lastMarkedItem, isNotNull);
        expect(
          key.currentState!.detectionDataReads,
          lessThan(500),
          reason:
              'An unrelated multi-cluster file or a completed group must not rescan 23310 photos',
        );
        expect(find.textContaining('拒识簇 #1'), findsWidgets);
        expect(find.textContaining('拒识簇 #2'), findsWidgets);
        expect(tester.takeException(), isNull);
        await tester.pumpWidget(const SizedBox());
      },
    );
  }

  testWidgets('marking work stays bounded with 600 validation items', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final key = GlobalKey<_IncrementalHarnessState>();
    await tester.pumpWidget(
      _IncrementalHarness(key: key, apiClient: apiClient, tempDir: tempDir),
    );
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    for (var attempt = 0; attempt < 500; attempt++) {
      if (find
          .byWidgetPredicate(
            (widget) =>
                widget is AbsorbPointer &&
                widget.key == const ValueKey('validation-directory-content') &&
                widget.absorbing,
          )
          .evaluate()
          .isEmpty)
        break;
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 10)),
      );
      await tester.pump(const Duration(milliseconds: 16));
    }
    expect(
      find.byWidgetPredicate(
        (widget) =>
            widget is AbsorbPointer &&
            widget.key == const ValueKey('validation-directory-content') &&
            widget.absorbing,
      ),
      findsNothing,
    );

    key.currentState!.resetReadCount();
    await tester.tap(find.text('正确').first);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 150));

    expect(
      key.currentState!.readsBeforeParentEcho,
      lessThan(50),
      reason:
          'Preparing one mark should use cached grouping state instead of '
          'rescanning all $_widgetItemCount items.',
    );
    expect(
      key.currentState!.detectionDataReads,
      lessThan(500),
      reason:
          'Applying the parent echo should stay bounded to the affected group '
          'instead of recomputing grouping data for all $_widgetItemCount items.',
    );

    await tester.pump(const Duration(seconds: 5));
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
    this.itemCount = _widgetItemCount,
    this.rejected = false,
    this.initialUnassigned = false,
    this.multiClusterFile = false,
    this.burstSize = 20,
    super.key,
  });

  final NeriApiClient apiClient;
  final Directory tempDir;
  final int itemCount;
  final bool rejected;
  final bool initialUnassigned;
  final bool multiClusterFile;
  final int burstSize;

  @override
  State<_IncrementalHarness> createState() => _IncrementalHarnessState();
}

class _IncrementalHarnessState extends State<_IncrementalHarness> {
  late List<DetectionItem> _items;
  int detectionDataReads = 0;
  int readsBeforeParentEcho = 0;
  bool loadMetadataAfterMark = false;
  int metadataApplications = 0;
  DetectionItem? lastMarkedItem;

  void applyMetadata(
    int index, {
    String? observationId,
    int captureTimeOffsetSeconds = 0,
  }) {
    final item = _items[index];
    final updated = DetectionItem(
      filename: item.filename,
      path: item.path,
      fileType: item.fileType,
      dateTaken: DateTime.parse(
        item.dateTaken ?? item.modifiedAt!,
      ).add(Duration(seconds: captureTimeOffsetSeconds)).toIso8601String(),
      modifiedAt: item.modifiedAt,
      width: 640,
      height: 480,
      species: item.species,
      confidence: item.confidence,
      detectionBoxes: observationId == null
          ? item.detectionBoxes
          : [
              DetectionBox(
                species: '拒识/Unknown',
                confidence: 0.95,
                bbox: const [0.1, 0.1, 0.9, 0.9],
                observationId: observationId,
              ),
            ],
      detectionData: item.detectionData,
      error: item.error,
      validated: item.validated,
    );
    setState(() {
      _items = List.of(_items)..[index] = updated;
      metadataApplications++;
    });
  }

  Future<void> _loadMetadata(DetectionItem item) async {
    if (!loadMetadataAfterMark || metadataApplications != 0) return;
    applyMetadata(
      _items.indexWhere((entry) => entry.path == item.path),
      captureTimeOffsetSeconds: 1,
    );
  }

  @override
  void initState() {
    super.initState();
    _items = <DetectionItem>[
      for (var index = 0; index < widget.itemCount; index++)
        DetectionItem(
          filename: 'image-$index.jpg',
          path: widget.rejected
              ? '${widget.tempDir.path}/rejected/image-$index.jpg'
              : '${widget.tempDir.path}/image-$index.jpg',
          fileType: 'jpg',
          modifiedAt: DateTime.utc(
            2026,
            9,
            14,
          ).add(Duration(seconds: index * 2)).toIso8601String(),
          species: widget.rejected ? const ['拒识/Unknown'] : const ['豹猫'],
          confidence: 0.95,
          detectionBoxes:
              widget.multiClusterFile && index == widget.itemCount - 1
              ? [
                  for (final id in ['a', 'b'])
                    DetectionBox(
                      species: '拒识/Unknown',
                      confidence: 0.95,
                      bbox: const [0.1, 0.1, 0.9, 0.9],
                      observationId: id,
                    ),
                ]
              : widget.rejected
              ? [
                  DetectionBox(
                    species: '拒识/Unknown',
                    confidence: 0.95,
                    bbox: const [0.1, 0.1, 0.9, 0.9],
                    observationId: widget.initialUnassigned && index == 0
                        ? 'missing'
                        : (index.isEven ? 'a' : 'b'),
                  ),
                ]
              : const [],
          detectionData: _CountingMap(<String, dynamic>{
            '物种名称': widget.rejected ? '拒识/Unknown' : '豹猫',
            '物种数量': '1',
          }, () => detectionDataReads += 1),
        ),
    ];
  }

  void resetReadCount() {
    detectionDataReads = 0;
    readsBeforeParentEcho = 0;
  }

  void _beginParentEchoMeasurement() {
    readsBeforeParentEcho = detectionDataReads;
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
    lastMarkedItem = item;
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
          rootInputPath: widget.tempDir.path,
          rootItems: _items,
          exportFormatMode: 'ask',
          items: _items,
          classificationModelPath: widget.rejected || widget.multiClusterFile
              ? 'model.neri.json'
              : null,
          loading: false,
          refreshVersion: 0,
          speciesTypes: const <String, String>{'豹猫': '兽类'},
          useCombinedConfidence: false,
          minFrameRatio: 0,
          autoGroup: !widget.rejected,
          collapseGroups: false,
          autoGroupDetectBurst: false,
          autoGroupBurstSize: widget.burstSize,
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
          onLoadMetadata: _loadMetadata,
          onOpenExternal: (_) {},
          onMarkItem: _markItem,
          onMarkItems: _markItems,
          onQuickMarkUsed: (_) async {},
          onQuickMarkReverted: (_) async {},
          onRedetectItems: (_, {required confidence}) async {},
          onFavoritePhotoPathsChanged: (_) async {},
          onFavoritePhotoExportModeChanged: (_) async {},
          onExportFormatModeChanged: (_) async {},
          onEmptyPhotoDeleteModeChanged: (_) async {},
          onAutoGroupInferredBurstSizeChanged: (_) {},
        ),
      ),
    );
  }
}

Finder _blockedValidation() => find.byWidgetPredicate(
  (widget) =>
      widget is AbsorbPointer &&
      widget.key == const ValueKey('validation-directory-content') &&
      widget.absorbing,
);
