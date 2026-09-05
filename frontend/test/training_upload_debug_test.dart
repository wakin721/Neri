import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';

Map<String, dynamic> diagnostics({bool enabled = true, int uploaded = 4}) => {
  'status': {
    'agreement_version': '2026-09-05',
    'agreement_accepted': true,
    'participation_decided': true,
    'training_enabled': enabled,
    'stats': {
      'pending': 2,
      'uploading': 1,
      'uploaded': uploaded,
      'failed': 0,
      'skipped': 3,
    },
  },
  'worker_running': true,
  'debounce_seconds': 17,
  'poll_seconds': 2,
  'request_timeout_seconds': 91,
  'retry_initial_seconds': 30,
  'retry_max_seconds': 3600,
  'default_chunk_bytes': 5242880,
  'max_empty_per_folder': 3,
  'max_image_edge': 2048,
  'jpeg_quality': 82,
  'max_image_bytes': 16777216,
  'image_suffixes': ['.jpg', '.png'],
  // Even unexpected fields must not reach the UI or clipboard.
  'broker_url': 'https://private.example/secret-destination',
};

void main() {
  Future<void> openDebug(WidgetTester tester, NeriApiClient client) async {
    tester.view.physicalSize = const Size(1100, 850);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final theme = ValueNotifier(const ThemeSettings());
    addTearDown(theme.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SettingsScreen(
            settings: NeriSettings.fromJson({
              'settings': {'debug_mode': true},
            }),
            autoGroupInferredBurstSize: null,
            apiClient: client,
            themeNotifier: theme,
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
    await tester.pumpAndSettle();
    await tester.tap(find.text('调试模式'));
    await tester.pumpAndSettle();
    expect(find.text('上传详细设置'), findsOneWidget);
    await tester.tap(find.text('上传详细设置'));
    await tester.pumpAndSettle();
  }

  testWidgets(
    'debug upload details poll, copy only visible facts and stop after closing',
    (tester) async {
      var reads = 0;
      final slowResponse = Completer<http.Response>();
      final client = NeriApiClient(
        httpClient: MockClient((request) async {
          if (request.url.path == '/api/debug/training-upload') {
            expect(request.method, 'GET');
            reads++;
            if (reads == 2) return slowResponse.future;
            return http.Response(jsonEncode(diagnostics()), 200);
          }
          return http.Response('{}', 200);
        }),
      );
      addTearDown(client.close);
      String? copied;
      tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
        SystemChannels.platform,
        (call) async {
          if (call.method == 'Clipboard.setData') {
            copied = (call.arguments as Map)['text'] as String;
          }
          return null;
        },
      );
      addTearDown(
        () => tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
          SystemChannels.platform,
          null,
        ),
      );
      await openDebug(tester, client);
      final dialog = find.byType(AlertDialog);
      expect(
        find.descendant(
          of: dialog,
          matching: find.byIcon(Icons.refresh_rounded),
        ),
        findsNothing,
      );
      expect(find.textContaining('17 秒'), findsOneWidget);
      expect(find.textContaining('91 秒'), findsOneWidget);
      await tester.pump(const Duration(seconds: 2));
      await tester.pump(const Duration(seconds: 4));
      expect(reads, 2);
      expect(
        find.descendant(
          of: dialog,
          matching: find.byType(CircularProgressIndicator),
        ),
        findsNothing,
      );
      slowResponse.complete(
        http.Response(
          jsonEncode(diagnostics(enabled: false, uploaded: 9)),
          200,
        ),
      );
      await tester.pumpAndSettle();
      expect(find.text('未参加'), findsOneWidget);
      expect(find.textContaining('已上传 9'), findsOneWidget);
      await tester.tap(find.text('复制上传设置'));
      await tester.pumpAndSettle();
      expect(copied, contains('已上传 9'));
      expect(copied, contains('2048'));
      expect(copied, isNot(contains('private.example')));
      expect(copied, isNot(contains('broker_url')));
      await tester.tap(find.text('关闭'));
      await tester.pumpAndSettle();
      final readsAfterClose = reads;
      await tester.pump(const Duration(seconds: 6));
      expect(reads, readsAfterClose);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets('debug upload diagnostics retry automatically after failure', (
    tester,
  ) async {
    var reads = 0;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/debug/training-upload') {
          reads++;
          return reads == 1
              ? http.Response('offline', 503)
              : http.Response(jsonEncode(diagnostics()), 200);
        }
        return http.Response('{}', 200);
      }),
    );
    addTearDown(client.close);
    await openDebug(tester, client);
    expect(find.textContaining('自动重试'), findsOneWidget);
    await tester.pump(const Duration(seconds: 2));
    await tester.pumpAndSettle();
    expect(find.text('已参加'), findsOneWidget);
    expect(find.textContaining('读取失败'), findsNothing);
    await tester.tap(find.text('关闭'));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
  });
}
