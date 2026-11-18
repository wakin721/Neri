import 'dart:io';
import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/main_window.dart';
import 'package:neri_flutter/src/frontend_version.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/privacy/privacy_status.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';
import 'package:neri_flutter/src/utils/preview_snapshot_cache.dart';
import 'package:shared_preferences/shared_preferences.dart';

class StartupBackend {
  StartupBackend({PreviewSnapshotCache? previewCache})
    : _providedPreviewCache = previewCache;

  final PreviewSnapshotCache? _providedPreviewCache;
  late final cacheDirectory = Directory.systemTemp.createTempSync(
    'neri-preview-test-',
  );
  late final previewCache =
      _providedPreviewCache ?? PreviewSnapshotCache(directory: cacheDirectory);
  Completer<http.Response>? healthResponse;
  Completer<http.Response>? gpuSettingsResponse;
  final preview = Completer<http.Response>();
  Completer<http.Response>? refreshedPreview;
  final scopedPreviews = <String, Completer<http.Response>>{};
  final previewPaths = <String>[];
  final savedSettings = <String, dynamic>{};
  final backgroundPreviewPaths = <String>[];
  List<String>? directories;
  final details = <String>[];
  Map<String, Object?>? previewMetadata;
  int dinoStatusRequests = 0;
  int settingsRequests = 0;
  final gpuProbeRequests = <bool>[];
  int summaryRequests = 0;
  int previewRequests = 0;
  bool consent = true;
  bool stopped = false;
  bool offline = false;
  bool failDetails = false;
  bool advanceDuringDetail = false;
  int processed = 1;
  bool pytorchAvailable = false;
  List<Map<String, dynamic>> availableModels = const <Map<String, dynamic>>[];
  List<Map<String, dynamic>> availableClassificationModels =
      const <Map<String, dynamic>>[];
  Map<String, dynamic> dinoStatusPayload = <String, dynamic>{
    'installed': false,
    'healthy': false,
    'message': 'Neri1.1 未安装。',
  };

  Map<String, dynamic> job(String id) => {
    'id': id,
    'input_dir': id,
    'state': id == 'running' ? 'running' : 'completed',
    'processed': id == 'running' ? processed : 1,
    'total': 3008,
    'created_at': 'created',
    'updated_at': id == 'running' ? '$processed' : 'completed',
  };

  late final api = NeriApiClient(
    httpClient: MockClient((request) async {
      final path = request.url.path;
      if (path == '/api/health') {
        if (healthResponse != null && !stopped) return healthResponse!.future;
        return http.Response('{}', stopped ? 503 : 200);
      }
      if (path == '/api/shutdown') {
        stopped = true;
        return http.Response('{}', 200);
      }
      if (path == '/api/privacy') {
        return http.Response(
          jsonEncode({
            'agreement_version': privacyAgreementVersion,
            'agreement_accepted': consent,
            'participation_decided': consent,
          }),
          200,
        );
      }
      if (path == '/api/environment/dinov3-status') {
        if (offline) throw http.ClientException('Backend offline');
        dinoStatusRequests++;
        return http.Response(
          jsonEncode(dinoStatusPayload),
          200,
          headers: {'content-type': 'application/json; charset=utf-8'},
        );
      }
      if (path == '/api/settings') {
        if (offline) throw http.ClientException('Backend offline');
        settingsRequests++;
        if (request.method == 'GET') {
          final probeGpu = request.url.queryParameters['probe_gpu'] != 'false';
          gpuProbeRequests.add(probeGpu);
          if (probeGpu && gpuSettingsResponse != null) {
            return gpuSettingsResponse!.future;
          }
        }
        if (request.method == 'PUT') {
          savedSettings.addAll(
            (jsonDecode(request.body) as Map<String, dynamic>)['settings']
                as Map<String, dynamic>,
          );
        }
        return http.Response(
          jsonEncode({
            'settings': savedSettings,
            'pytorch_available': pytorchAvailable,
            'available_models': availableModels,
            'available_classification_models': availableClassificationModels,
          }),
          200,
          headers: {'content-type': 'application/json; charset=utf-8'},
        );
      }
      if (path == '/api/jobs') {
        if (offline) throw http.ClientException('Backend offline');
        summaryRequests++;
        expectSync(request.url.queryParameters['include_results'], 'false');
        return http.Response(jsonEncode([job('history'), job('running')]), 200);
      }
      if (path.startsWith('/api/jobs/')) {
        final id = path.split('/').last;
        details.add(id);
        if (failDetails) return http.Response('{}', 500);
        if (advanceDuringDetail) processed++;
        return http.Response(jsonEncode(job(id)), 200);
      }
      if (path == '/api/preview') {
        previewRequests++;
        final inputPath = request.url.queryParameters['input_path']!;
        previewPaths.add(inputPath);
        if (request.url.queryParameters['background'] == 'true') {
          backgroundPreviewPaths.add(inputPath);
        }
        final scoped = scopedPreviews[inputPath];
        if (scoped != null) return scoped.future;
        return (refreshedPreview ?? preview).future;
      }
      if (path == '/api/preview/directories' && directories != null) {
        return http.Response(jsonEncode(directories), 200);
      }
      if (path == '/api/preview/item' && previewMetadata != null) {
        return http.Response(
          jsonEncode(previewMetadata),
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }
      // Optional component checks and update-source lookup fail safely.
      return http.Response('{}', 503);
    }),
  );
}

Future<void> settleStartup(WidgetTester tester) async {
  // Maintenance status and local preferences use real filesystem futures.
  for (var i = 0; i < 12; i++) {
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 10)),
    );
    await tester.pump(const Duration(milliseconds: 1));
  }
}

Future<void> mount(
  WidgetTester tester,
  StartupBackend backend, {
  Map<String, Object>? preferences,
}) async {
  SharedPreferences.setMockInitialValues(
    preferences ?? {'last_input_path': 'startup-test-input'},
  );
  tester.view.physicalSize = const Size(1400, 1100);
  tester.view.devicePixelRatio = 1;
  await tester.pumpWidget(
    MaterialApp(
      home: MainWindow(
        apiClient: backend.api,
        themeNotifier: ValueNotifier(const ThemeSettings()),
        previewCache: backend.previewCache,
      ),
    ),
  );
  await settleStartup(tester);
}

Future<void> flushPreviewCache(
  WidgetTester tester,
  PreviewSnapshotCache cache,
) async {
  // Cache operations queue their continuations in the widget test's fake-async
  // zone, but compute() completes on real isolates. Keep pumping that zone
  // between real async turns instead of awaiting the queue inside runAsync().
  final pending = cache.flush();
  var completed = false;
  pending.then<void>(
    (_) => completed = true,
    onError: (Object error, StackTrace stackTrace) {
      completed = true;
    },
  );
  final stopwatch = Stopwatch()..start();
  while (!completed && stopwatch.elapsed < const Duration(seconds: 10)) {
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 10)),
    );
    await tester.pump();
  }
  expect(completed, isTrue, reason: 'Preview cache writes did not finish.');
  await pending;
}

Future<void> cleanup(WidgetTester tester, StartupBackend backend) async {
  if (backend.gpuSettingsResponse?.isCompleted == false) {
    backend.gpuSettingsResponse!.complete(http.Response('{}', 200));
  }
  for (final preview in backend.scopedPreviews.values) {
    if (!preview.isCompleted) preview.complete(http.Response('[]', 200));
  }
  if (backend.refreshedPreview?.isCompleted == false) {
    backend.refreshedPreview!.complete(http.Response('[]', 200));
  }
  if (!backend.preview.isCompleted) {
    backend.preview.complete(http.Response('[]', 200));
  }
  await tester.pump();
  await tester.pumpWidget(const SizedBox());
  await tester.pump(const Duration(seconds: 1));
  await tester.runAsync(
    () => Future<void>.delayed(const Duration(milliseconds: 50)),
  );
  await flushPreviewCache(tester, backend.previewCache);
  await tester.runAsync(() async {
    if (backend._providedPreviewCache == null) {
      await backend.cacheDirectory.delete(recursive: true);
    }
  });
  tester.view.resetPhysicalSize();
  tester.view.resetDevicePixelRatio();
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('window_manager'),
          (call) async => call.method == 'isMaximized' ? false : null,
        );
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('neri/windows_shell'),
          (_) async => null,
        );
  });

  testWidgets(
    'cleanup drains queued preview cache writes',
    (tester) async {
      final backend = StartupBackend();
      for (var i = 0; i < 8; i++) {
        unawaited(backend.previewCache.write('queued-$i', const []));
      }
      await cleanup(tester, backend);
      expect(backend.cacheDirectory.existsSync(), isFalse);
    },
    timeout: const Timeout(Duration(seconds: 30)),
  );

  testWidgets(
    'startup and idle polling keep histories out of shell critical path',
    (tester) async {
      final backend = StartupBackend()
        ..healthResponse = Completer<http.Response>();
      await mount(tester, backend);
      expect(backend.settingsRequests, 0);
      expect(backend.dinoStatusRequests, 0);
      expect(
        find.text('Neri ${frontendVersion.split(RegExp(r'[-_+(]')).first}'),
        findsOneWidget,
      );
      backend.healthResponse!.complete(http.Response('{}', 200));
      await settleStartup(tester);
      expect(
        find.text('Neri ${frontendVersion.split(RegExp(r'[-_+(]')).first}'),
        findsOneWidget,
      );
      expect(backend.settingsRequests, 2);
      expect(backend.dinoStatusRequests, 2);
      expect(backend.summaryRequests, 1);
      expect(backend.previewRequests, 1);
      expect(backend.details, isEmpty);
      expect(find.text('程序启动中'), findsNothing);
      // Histories render only in their dialog; startup has no global spinner.
      expect(find.byType(LinearProgressIndicator), findsNothing);
      await tester.pump(const Duration(seconds: 2));
      await settleStartup(tester);
      expect(backend.summaryRequests, greaterThanOrEqualTo(2));
      expect(backend.details, isEmpty);
      final history = find.byTooltip('任务历史');
      await tester.ensureVisible(history);
      await tester.pump();
      await tester.tap(history);
      await tester.pump(const Duration(milliseconds: 300));
      expect(find.byType(LinearProgressIndicator), findsNWidgets(2));
      expect(backend.details, isEmpty);
      await tester.tap(find.text('history'));
      await tester.pump();
      await settleStartup(tester);
      expect(backend.details, ['history']);
      expect(
        tester.widget<StartScreen>(find.byType(StartScreen)).failedJobDetailIds,
        isEmpty,
      );
      await tester.tap(find.byTooltip('关闭任务历史'));
      await tester.pump(const Duration(milliseconds: 300));
      backend.processed++;
      await tester.pump(const Duration(seconds: 2));
      await settleStartup(tester);
      expect(backend.details, ['history']);
      await cleanup(tester, backend);
    },
  );

  testWidgets('slow GPU detection does not hold the shell or history polling', (
    tester,
  ) async {
    final backend = StartupBackend()
      ..gpuSettingsResponse = Completer<http.Response>();
    await mount(tester, backend);
    expect(backend.gpuProbeRequests, [false, true]);
    expect(find.text('程序启动中'), findsNothing);
    final start = tester.widget<StartScreen>(find.byType(StartScreen));
    expect(start.settings, isNotNull);
    expect(start.settings!.gpuAvailable, isFalse);
    final summaries = backend.summaryRequests;
    await tester.pump(const Duration(seconds: 2));
    await settleStartup(tester);
    expect(backend.summaryRequests, greaterThan(summaries));
    expect(backend.details, isEmpty);
    backend.gpuSettingsResponse!.complete(
      http.Response('{"gpu_available":true}', 200),
    );
    await settleStartup(tester);
    expect(
      tester
          .widget<StartScreen>(find.byType(StartScreen))
          .settings!
          .gpuAvailable,
      isTrue,
    );
    await cleanup(tester, backend);
  });

  testWidgets(
    'running detail advancing during request succeeds and retry recovers',
    (tester) async {
      final backend = StartupBackend();
      await mount(tester, backend);
      var screen = tester.widget<StartScreen>(find.byType(StartScreen));
      backend.failDetails = true;
      screen.onJobExpansionChanged!(screen.jobs.last, true);
      await tester.pump();
      await settleStartup(tester);
      screen = tester.widget<StartScreen>(find.byType(StartScreen));
      expect(screen.failedJobDetailIds, contains('running'));
      expect(screen.loadingJobDetailIds, isEmpty);
      backend.failDetails = false;
      backend.advanceDuringDetail = true;
      screen.onRetryJobDetails!(screen.jobs.last);
      await tester.pump();
      await settleStartup(tester);
      screen = tester.widget<StartScreen>(find.byType(StartScreen));
      expect(screen.failedJobDetailIds, isEmpty);
      expect(screen.loadingJobDetailIds, isEmpty);
      expect(backend.details, ['running', 'running']);
      await cleanup(tester, backend);
    },
  );

  testWidgets('opening preview then validation loads current job details', (
    tester,
  ) async {
    final backend = StartupBackend();
    await mount(tester, backend);
    expect(backend.details, isEmpty);
    await tester.tap(find.text('预览').first);
    await tester.pump();
    await settleStartup(tester);
    expect(backend.details.toSet(), {'history', 'running'});
    backend.processed++;
    await tester.tap(find.text('校验').first);
    await tester.pump();
    await settleStartup(tester);
    expect(backend.details.where((id) => id == 'running').length, 2);
    await cleanup(tester, backend);
  });

  for (final page in ['预览', '校验']) {
    testWidgets('first opening $page shows loading until media arrives', (
      tester,
    ) async {
      final backend = StartupBackend();
      await mount(tester, backend);
      await tester.tap(find.text(page).first);
      await tester.pump();
      await settleStartup(tester);
      if (page == '预览') {
        expect(find.byType(LinearProgressIndicator), findsNothing);
        expect(find.byIcon(Icons.image_search_rounded), findsOneWidget);
        expect(find.text('暂无预览图像。'), findsOneWidget);
        final refresh = find.widgetWithText(FilledButton, '重新获取');
        expect(tester.widget<FilledButton>(refresh).onPressed, isNull);
        expect(
          find.descendant(
            of: refresh,
            matching: find.byType(CircularProgressIndicator),
          ),
          findsOneWidget,
        );
      } else {
        expect(find.byType(LinearProgressIndicator), findsNothing);
        expect(find.byIcon(Icons.fact_check_outlined), findsOneWidget);
        expect(find.text('正在读取待校验图像…'), findsOneWidget);
        expect(find.text('暂无待校验图像。'), findsNothing);
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
      backend.preview.complete(
        http.Response(
          jsonEncode([
            {
              'filename': 'first.jpg',
              'path':
                  '${Directory('startup-test-input').absolute.path}/first.jpg',
              'file_type': 'jpg',
            },
          ]),
          200,
        ),
      );
      await settleStartup(tester);
      expect(find.byType(LinearProgressIndicator), findsNothing);
      expect(find.byIcon(Icons.open_in_new_rounded), findsWidgets);
      expect(tester.takeException(), isNull);
      await cleanup(tester, backend);
    });
  }

  testWidgets('privacy consent continues to gate settings and jobs', (
    tester,
  ) async {
    final backend = StartupBackend()..consent = false;
    await mount(tester, backend);
    expect(backend.settingsRequests, 0);
    expect(backend.summaryRequests, 0);
    expect(backend.previewRequests, 0);
    await cleanup(tester, backend);
  });

  testWidgets(
    'silent poll after backend disconnect refreshes settings and component status even when jobs are unchanged',
    (tester) async {
      final backend = StartupBackend()
        ..dinoStatusPayload = <String, dynamic>{
          'installed': false,
          'healthy': false,
          'display_name': 'Neri1.2preview',
          'message': 'Neri1.2preview 未安装。',
        };
      await mount(tester, backend);
      backend.preview.complete(http.Response('[]', 200));
      await settleStartup(tester);

      await tester.tap(find.text('设置').first);
      await settleStartup(tester);
      await tester.tap(find.text('环境维护').first);
      await settleStartup(tester);
      expect(find.text('Neri1.2preview 未安装。'), findsOneWidget);

      final summariesBefore = backend.summaryRequests;
      final settingsBefore = backend.settingsRequests;
      final dinoBefore = backend.dinoStatusRequests;

      // Normal online silent poll must not reload settings or re-check DINO status.
      await tester.pump(const Duration(seconds: 2));
      await settleStartup(tester);
      expect(backend.summaryRequests, greaterThan(summariesBefore));
      expect(backend.settingsRequests, settingsBefore);
      expect(backend.dinoStatusRequests, dinoBefore);

      // Backend goes offline during silent poll -> UI marks backend not ready.
      backend.offline = true;
      await tester.pump(const Duration(seconds: 2));
      await settleStartup(tester);
      expect(find.text('等待后端就绪...'), findsOneWidget);

      final settingsOffline = backend.settingsRequests;
      final dinoOffline = backend.dinoStatusRequests;

      // Backend returns with identical jobs, updated torch/models, and installed DINOv3.
      backend.offline = false;
      backend.pytorchAvailable = true;
      backend.availableModels = <Map<String, dynamic>>[
        {
          'name': 'MDV6-yolov10-c',
          'path': 'res/model/detect/sync/MDV6-yolov10-c.pt',
          'source': 'sync',
          'kind': 'detect',
          'backend': 'yolo',
        },
      ];
      backend.availableClassificationModels = <Map<String, dynamic>>[
        {
          'name': 'Neri1.2preview',
          'path': 'res/model/DINOv3/Neri1.2preview.neri.json',
          'source': 'user',
          'kind': 'cls',
          'backend': 'dinov3',
          'encoder_backend': 'pytorch',
        },
      ];
      backend.dinoStatusPayload = <String, dynamic>{
        'installed': true,
        'healthy': true,
        'display_name': 'Neri1.2preview',
        'encoder_backend': 'pytorch',
        'runtime_choice': 'gpu',
        'resolved_runtime_choice': 'gpu',
        'message': 'Neri1.2preview 已安装。',
      };

      await tester.pump(const Duration(seconds: 2));
      await settleStartup(tester);

      expect(find.text('等待后端就绪...'), findsNothing);
      expect(find.text('Neri1.2preview 已安装。'), findsOneWidget);
      expect(backend.settingsRequests, settingsOffline + 1);
      expect(backend.dinoStatusRequests, dinoOffline + 1);

      final settingsScreen = tester.widget<SettingsScreen>(
        find.byType(SettingsScreen),
      );
      expect(settingsScreen.backendReady, isTrue);
      expect(settingsScreen.settings?.pytorchAvailable, isTrue);
      expect(
        settingsScreen.settings?.availableClassificationModels.any(
          (model) => model.path == 'res/model/DINOv3/Neri1.2preview.neri.json',
        ),
        isTrue,
      );

      // Subsequent normal poll while online remains quiet on settings & DINO status.
      await tester.pump(const Duration(seconds: 2));
      await settleStartup(tester);
      expect(backend.settingsRequests, settingsOffline + 1);
      expect(backend.dinoStatusRequests, dinoOffline + 1);

      await cleanup(tester, backend);
    },
  );
}
