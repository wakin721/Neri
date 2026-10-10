import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';

void main() {
  testWidgets('refresh and regroup settings retain media without a central bar', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1;
    final api = NeriApiClient(
      httpClient: MockClient((_) async => http.Response('{}', 200)),
    );
    await tester.pumpWidget(
      _GroupExpansionHarness(
        apiClient: api,
        initialItems: [
          for (var i = 0; i < 1200; i++)
            DetectionItem(
              filename: '$i.jpg',
              path:
                  '${Directory('regroup-settings-input').absolute.path}/$i.jpg',
              fileType: 'jpg',
              dateTaken: '2026-10-10T12:00:00',
            ),
        ],
      ),
    );
    final state = tester.state<_GroupExpansionHarnessState>(
      find.byType(_GroupExpansionHarness),
    );
    await _waitForValidationPreparation(tester);
    for (final trigger in ['refresh', 'burst', 'gap', 'confidence']) {
      final previousAction = tester.element(find.text('正确').first);
      if (trigger == 'confidence') {
        final slider = tester.widget<Slider>(find.byType(Slider).first);
        slider.onChanged!(slider.value == 0.75 ? 0.25 : 0.75);
      } else {
        state.regroup(trigger);
      }
      await tester.pump();
      expect(_preparingContent(), findsOneWidget, reason: trigger);
      expect(
        find.byType(LinearProgressIndicator),
        findsNothing,
        reason: trigger,
      );
      expect(
        identical(tester.element(find.text('正确').first), previousAction),
        isTrue,
        reason: trigger,
      );
      await _waitForValidationPreparation(tester);
      expect(find.text('正确'), findsWidgets);
    }
    expect(find.textContaining('400 组'), findsWidgets);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox());
    api.close();
    tester.view.resetPhysicalSize();
    tester.view.resetDevicePixelRatio();
  });
  testWidgets(
    'an initial empty placeholder still shows first media preparation',
    (tester) async {
      tester.view.physicalSize = const Size(1280, 900);
      tester.view.devicePixelRatio = 1;
      final api = NeriApiClient(
        httpClient: MockClient((_) async => http.Response('{}', 200)),
      );
      await tester.pumpWidget(
        _GroupExpansionHarness(apiClient: api, initialItems: const []),
      );
      final state = tester.state<_GroupExpansionHarnessState>(
        find.byType(_GroupExpansionHarness),
      );
      expect(find.byType(LinearProgressIndicator), findsNothing);
      state.setLoading(true);
      await tester.pump();
      _expectInitialValidationLoading(tester);
      state.switchItems([
        for (var i = 0; i < 600; i++)
          DetectionItem(
            filename: '$i.jpg',
            path: '${Directory('initial-empty-input').absolute.path}/$i.jpg',
            fileType: 'jpg',
            dateTaken: '2026-10-10T12:00:00',
          ),
      ]);
      await tester.pump();
      _expectInitialValidationLoading(tester);
      for (var attempt = 0; attempt < 500; attempt++) {
        if (_preparingContent().evaluate().isEmpty) break;
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 10)),
        );
        await tester.pump(const Duration(milliseconds: 16));
      }
      expect(find.byType(LinearProgressIndicator), findsNothing);
      expect(find.text('正确'), findsWidgets);
      expect(find.textContaining('150 组'), findsWidgets);
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox());
      api.close();
      tester.view.resetPhysicalSize();
      tester.view.resetDevicePixelRatio();
    },
  );
  testWidgets('first large validation shows progress until grouping is ready', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1;
    final api = NeriApiClient(
      httpClient: MockClient((_) async => http.Response('{}', 200)),
    );
    await tester.pumpWidget(
      _GroupExpansionHarness(
        apiClient: api,
        initialItems: [
          for (var i = 0; i < 12000; i++)
            DetectionItem(
              filename: '$i.jpg',
              path: '${Directory('preparation-input').absolute.path}/$i.jpg',
              fileType: 'jpg',
              dateTaken: '2026-10-10T12:00:00',
            ),
        ],
      ),
    );
    _expectInitialValidationLoading(tester);
    for (var attempt = 0; attempt < 500; attempt++) {
      if (_preparingContent().evaluate().isEmpty) break;
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 10)),
      );
      await tester.pump(const Duration(milliseconds: 16));
    }
    expect(find.byType(LinearProgressIndicator), findsNothing);
    expect(find.text('正确'), findsWidgets);
    expect(find.textContaining('3000 组'), findsWidgets);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox());
    api.close();
    tester.view.resetPhysicalSize();
    tester.view.resetDevicePixelRatio();
  });
  testWidgets(
    'large regrouping yields frames and retains every per-camera group',
    (tester) async {
      tester.view.physicalSize = const Size(1280, 900);
      tester.view.devicePixelRatio = 1;
      final api = NeriApiClient(
        httpClient: MockClient((_) async => http.Response('{}', 200)),
      );
      final root = Directory('preparation-input').absolute.path;
      final items = [
        for (var i = 0; i < 12000; i++)
          DetectionItem(
            filename: '$i.jpg',
            path: '$root/camera-${i % 4}/$i.jpg',
            fileType: 'jpg',
            dateTaken: '2026-10-10T12:00:00',
          ),
      ];
      await tester.pumpWidget(
        _GroupExpansionHarness(
          apiClient: api,
          initialItems: items.take(4).toList(),
        ),
      );
      final state = tester.state<_GroupExpansionHarnessState>(
        find.byType(_GroupExpansionHarness),
      );
      final previousAction = tester.element(find.text('正确').first);
      state.switchItems(items);
      final watch = Stopwatch()..start();
      await tester.pump();
      watch.stop();
      expect(
        find.text('正确'),
        findsWidgets,
        reason: 'Directory loading must retain the previous media and actions',
      );
      expect(
        identical(tester.element(find.text('正确').first), previousAction),
        isTrue,
      );
      expect(find.textContaining('4 组'), findsWidgets);
      expect(find.byType(LinearProgressIndicator), findsNothing);
      await tester.tap(find.text('正确').first, warnIfMissed: false);
      expect(
        state.markCalls,
        0,
        reason:
            'Retained contents must not mark the new directory during loading',
      );
      expect(
        _preparingContent(),
        findsOneWidget,
        reason:
            'A large regroup must defer computation instead of blocking its first frame',
      );
      expect(watch.elapsedMilliseconds, lessThan(300));
      var framesWhilePreparing = 0;
      for (var attempt = 0; attempt < 500; attempt++) {
        if (_preparingContent().evaluate().isEmpty) {
          break;
        }
        framesWhilePreparing++;
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 10)),
        );
        await tester.pump(const Duration(milliseconds: 16));
      }
      expect(framesWhilePreparing, greaterThan(1));
      expect(_preparingContent(), findsNothing);
      expect(find.textContaining('3000 组'), findsWidgets);
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox());
      api.close();
      tester.view.resetPhysicalSize();
      tester.view.resetDevicePixelRatio();
    },
  );
  testWidgets(
    'a newer directory supersedes preparation and disposal cancels safely',
    (tester) async {
      tester.view.physicalSize = const Size(1280, 900);
      tester.view.devicePixelRatio = 1;
      final api = NeriApiClient(
        httpClient: MockClient((_) async => http.Response('{}', 200)),
      );
      List<DetectionItem> photos(String camera, int count) => [
        for (var i = 0; i < count; i++)
          DetectionItem(
            filename: '$i.jpg',
            path:
                '${Directory('preparation-input').absolute.path}/$camera/$i.jpg',
            fileType: 'jpg',
            dateTaken: '2026-10-10T12:00:00',
          ),
      ];
      await tester.pumpWidget(
        _GroupExpansionHarness(
          apiClient: api,
          initialItems: photos('initial', 4),
        ),
      );
      final state = tester.state<_GroupExpansionHarnessState>(
        find.byType(_GroupExpansionHarness),
      );
      state.switchItems(photos('old', 12000));
      await tester.pump();
      expect(_preparingContent(), findsOneWidget);
      state.switchItems(photos('new', 800));
      await tester.pump();
      for (var attempt = 0; attempt < 500; attempt++) {
        if (_preparingContent().evaluate().isEmpty) {
          break;
        }
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 10)),
        );
        await tester.pump(const Duration(milliseconds: 16));
      }
      expect(find.textContaining('200 组'), findsWidgets);
      expect(find.textContaining('3000 组'), findsNothing);
      state.switchItems(photos('cancel-to-small', 12000));
      await tester.pump();
      state.switchItems(photos('small', 4));
      await tester.pump();
      await tester.pump(const Duration(seconds: 1));
      expect(_preparingContent(), findsNothing);
      expect(find.textContaining('1 组'), findsWidgets);
      state.switchItems(photos('disposing', 12000));
      await tester.pump();
      await tester.pumpWidget(const SizedBox());
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 300)),
      );
      await tester.pump(const Duration(seconds: 1));
      await tester.pump(const Duration(seconds: 1));
      expect(tester.takeException(), isNull);
      api.close();
      tester.view.resetPhysicalSize();
      tester.view.resetDevicePixelRatio();
    },
  );
  testWidgets('a failed large preparation recovers on a small directory', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1;
    final api = NeriApiClient(
      httpClient: MockClient((_) async => http.Response('{}', 200)),
    );

    List<DetectionItem> photos(int count, {bool fail = false}) => [
      for (var i = 0; i < count; i++)
        DetectionItem(
          filename: '$i.jpg',
          path: '${Directory('recovery-input').absolute.path}/$i.jpg',
          fileType: 'jpg',
          dateTaken: '2026-10-10T12:00:00',
          detectionData: fail && i == 0
              ? {'拍摄时间': _InvalidCaptureTime()}
              : const {},
        ),
    ];
    await tester.pumpWidget(
      _GroupExpansionHarness(apiClient: api, initialItems: photos(4)),
    );
    final state = tester.state<_GroupExpansionHarnessState>(
      find.byType(_GroupExpansionHarness),
    );
    state.switchItems(photos(600, fail: true));
    await tester.pump();
    for (var attempt = 0; attempt < 100; attempt++) {
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 10)),
      );
      await tester.pump(const Duration(milliseconds: 16));
      if (find.text('目录加载失败，点击重试').evaluate().isNotEmpty) break;
    }
    expect(find.text('目录加载失败，点击重试'), findsOneWidget);
    state.switchItems(photos(4));
    await tester.pump();
    expect(find.text('目录加载失败，点击重试'), findsNothing);
    expect(find.textContaining('1 组'), findsWidgets);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox());

    api.close();
    tester.view.resetPhysicalSize();
    tester.view.resetDevicePixelRatio();
  });
}

class _GroupExpansionHarness extends StatefulWidget {
  const _GroupExpansionHarness({
    required this.apiClient,
    required this.initialItems,
  });

  final NeriApiClient apiClient;
  final List<DetectionItem> initialItems;

  @override
  State<_GroupExpansionHarness> createState() => _GroupExpansionHarnessState();
}

class _GroupExpansionHarnessState extends State<_GroupExpansionHarness> {
  late List<DetectionItem> _items = widget.initialItems;
  int markCalls = 0;
  bool _loading = false;
  int _refreshVersion = 0;
  int _burstSize = 4;
  int _gapSeconds = 1800;
  void regroup(String trigger) => setState(() {
    switch (trigger) {
      case 'refresh':
        _refreshVersion++;
      case 'burst':
        _burstSize = 3;
      case 'gap':
        _gapSeconds = 60;
    }
  });
  void setLoading(bool value) => setState(() => _loading = value);
  void switchItems(List<DetectionItem> next) {
    setState(() {
      _items = next;
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: SpeciesValidationScreen(
          apiClient: widget.apiClient,
          inputPath: r'I:\原始照片\安息',
          items: _items,
          loading: _loading,
          refreshVersion: _refreshVersion,
          speciesTypes: const <String, String>{'豹猫': '兽类'},
          useCombinedConfidence: false,
          minFrameRatio: 0,
          autoGroup: true,
          collapseGroups: true,
          autoGroupDetectBurst: false,
          autoGroupBurstSize: _burstSize,
          autoGroupGapSeconds: _gapSeconds,
          autoSortQuickMarks: false,
          undoSteps: 20,
          quickMarkSpecies: const <String>['豹猫'],
          quickMarkRecentHistory: const <String>[],
          quickMarkUsageCounts: const <String, int>{},
          quantityButtons: const <String>['1'],
          exportColumns: const <String>[],
          favoritePhotoPaths: const <String>[],
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
              }) async {
                markCalls++;
                return candidate;
              },
          onMarkItems:
              (
                candidates,
                action, {
                speciesName,
                speciesCount,
                speciesType,
                remark,
              }) async {
                markCalls++;
                return candidates;
              },
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

class _InvalidCaptureTime {
  @override
  String toString() => throw StateError('Invalid capture time');
}

Finder _preparingContent() => find.byWidgetPredicate(
  (widget) =>
      widget is AbsorbPointer &&
      widget.key == const ValueKey('validation-directory-content') &&
      widget.absorbing,
);

void _expectInitialValidationLoading(WidgetTester tester) {
  expect(find.byType(LinearProgressIndicator), findsNothing);
  expect(find.byIcon(Icons.fact_check_outlined), findsOneWidget);
  expect(find.text('暂无待校验图像。'), findsOneWidget);
  final refresh = find.widgetWithText(FilledButton, '重新获取');
  expect(tester.widget<FilledButton>(refresh).onPressed, isNull);
  expect(
    find.descendant(
      of: refresh,
      matching: find.byType(CircularProgressIndicator),
    ),
    findsOneWidget,
  );
}

Future<void> _waitForValidationPreparation(WidgetTester tester) async {
  for (var attempt = 0; attempt < 500; attempt++) {
    if (_preparingContent().evaluate().isEmpty) break;
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 10)),
    );
    await tester.pump(const Duration(milliseconds: 16));
  }
  expect(_preparingContent(), findsNothing);
}
