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
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/privacy/privacy_status.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';
import 'package:shared_preferences/shared_preferences.dart';

class StartupBackend {
  final preview = Completer<http.Response>();
  Completer<http.Response>? refreshedPreview;
  List<String>? directories;
  final details = <String>[];
  Map<String, Object?>? previewMetadata;
  int settingsRequests = 0;
  int summaryRequests = 0;
  int previewRequests = 0;
  bool consent = true;
  bool stopped = false;
  bool failDetails = false;
  bool advanceDuringDetail = false;
  int processed = 1;

  Map<String, dynamic> job(String id) => {
    'id': id,
    'input_dir': id,
    'state': id == 'running' ? 'running' : 'completed',
    'processed': processed,
    'total': 3008,
    'created_at': 'created',
    'updated_at': '$processed',
  };

  late final api = NeriApiClient(
    httpClient: MockClient((request) async {
      final path = request.url.path;
      if (path == '/api/health') {
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
      if (path == '/api/settings') {
        settingsRequests++;
        return http.Response('{}', 200);
      }
      if (path == '/api/jobs') {
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
        return (refreshedPreview ?? preview).future;
      }
      if (path == '/api/preview/directories' && directories != null) {
        return http.Response(jsonEncode(directories), 200);
      }
      if (path == '/api/preview/item' && previewMetadata != null) {
        return http.Response(jsonEncode(previewMetadata), 200);
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

Future<void> mount(WidgetTester tester, StartupBackend backend) async {
  SharedPreferences.setMockInitialValues({
    'last_input_path': 'startup-test-input',
  });
  tester.view.physicalSize = const Size(1400, 1100);
  tester.view.devicePixelRatio = 1;
  await tester.pumpWidget(
    MaterialApp(
      home: MainWindow(
        apiClient: backend.api,
        themeNotifier: ValueNotifier(const ThemeSettings()),
      ),
    ),
  );
  await settleStartup(tester);
}

Future<void> cleanup(WidgetTester tester, StartupBackend backend) async {
  if (!backend.preview.isCompleted) {
    backend.preview.complete(http.Response('[]', 200));
  }
  await tester.pump();
  await tester.pumpWidget(const SizedBox());
  await tester.pump(const Duration(seconds: 1));
  await tester.runAsync(
    () => Future<void>.delayed(const Duration(milliseconds: 50)),
  );
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
    'startup and idle polling keep histories out of shell critical path',
    (tester) async {
      final backend = StartupBackend();
      await mount(tester, backend);
      expect(backend.settingsRequests, 1);
      expect(backend.summaryRequests, 1);
      expect(backend.previewRequests, 1);
      expect(backend.details, isEmpty);
      expect(find.text('程序启动中'), findsNothing);
      // The preview is still pending but the global shell progress has ended.
      expect(find.byType(LinearProgressIndicator), findsNWidgets(2));
      await tester.pump(const Duration(seconds: 2));
      await settleStartup(tester);
      expect(backend.summaryRequests, greaterThanOrEqualTo(2));
      expect(backend.details, isEmpty);
      final screen = tester.widget<StartScreen>(find.byType(StartScreen));
      screen.onJobExpansionChanged!(screen.jobs.first, true);
      await tester.pump();
      await settleStartup(tester);
      expect(backend.details, ['history']);
      expect(
        tester.widget<StartScreen>(find.byType(StartScreen)).failedJobDetailIds,
        isEmpty,
      );
      await cleanup(tester, backend);
    },
  );

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
        expect(find.byType(LinearProgressIndicator), findsOneWidget);
        expect(find.text('暂无可校验图像。'), findsNothing);
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
}
