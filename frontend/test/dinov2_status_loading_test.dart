import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';

http.Response statusResponse({bool installed = false}) => http.Response(
  jsonEncode({
    'installed': installed,
    'healthy': installed,
    'message': installed ? 'DINOv2 ViT-B/14 已安装。' : 'DINOv2 未安装。',
  }),
  200,
  headers: {'content-type': 'application/json; charset=utf-8'},
);

class StatusHarness {
  StatusHarness(Future<http.Response> Function(int) response) {
    api = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/environment/dinov2-status') {
          return response(++requests);
        }
        return http.Response('{}', 200);
      }),
    );
  }

  late final NeriApiClient api;
  final theme = ValueNotifier(const ThemeSettings());
  NeriSettings? settings;
  bool ready = true;
  bool active = true;
  int generation = 1;
  int requests = 0;

  Widget screen() => MaterialApp(
    home: Scaffold(
      body: SettingsScreen(
        settings: settings,
        backendReady: ready,
        backendGeneration: generation,
        isActive: active,
        autoGroupInferredBurstSize: null,
        apiClient: api,
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
  );
}

Future<void> mount(WidgetTester tester, StatusHarness harness) async {
  await tester.binding.setSurfaceSize(const Size(1400, 1100));
  addTearDown(() => tester.binding.setSurfaceSize(null));
  addTearDown(harness.api.close);
  addTearDown(harness.theme.dispose);
  await tester.pumpWidget(harness.screen());
  await tester.pump();
  await tester.tap(find.text('环境维护').first);
  await tester.pump();
}

void main() {
  testWidgets(
    'waits for backend readiness and avoids requests on settings polling',
    (tester) async {
      final harness = StatusHarness((_) async => statusResponse())
        ..ready = false
        ..generation = 0
        ..active = false;
      await mount(tester, harness);
      expect(harness.requests, 0);
      expect(find.text('等待后端就绪...'), findsOneWidget);
      harness.ready = true;
      harness.generation = 1;
      await tester.pumpWidget(harness.screen());
      await tester.pumpAndSettle();
      expect(harness.requests, 1);
      harness.settings = NeriSettings.fromJson({});
      harness.active = true;
      await tester.pumpWidget(harness.screen());
      await tester.pumpAndSettle();
      expect(harness.requests, 1);
      expect(find.text('DINOv2 未安装。'), findsOneWidget);
      expect(find.byKey(const Key('dinov2-status-retry')), findsNothing);
      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets('failed status can retry without offering installation', (
    tester,
  ) async {
    final retry = Completer<http.Response>();
    final harness = StatusHarness((request) async {
      if (request == 1) throw http.ClientException('Connection refused');
      return retry.future;
    });
    await mount(tester, harness);
    expect(find.text('无法读取 DINOv2 组件状态。'), findsOneWidget);
    expect(find.byKey(const Key('dinov2-component-action')), findsNothing);
    await tester.tap(find.byKey(const Key('dinov2-status-retry')));
    await tester.pump();
    expect(harness.requests, 2);
    expect(find.text('正在读取 DINOv2 组件状态...'), findsOneWidget);
    expect(find.byKey(const Key('dinov2-status-retry')), findsNothing);
    retry.complete(statusResponse(installed: true));
    await tester.pumpAndSettle();
    expect(find.text('DINOv2 ViT-B/14 已安装。'), findsOneWidget);
    expect(find.widgetWithText(OutlinedButton, '删除'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets(
    'new backend generation reloads and ignores old request failure',
    (tester) async {
      final oldRequest = Completer<http.Response>();
      final harness = StatusHarness(
        (request) async =>
            request == 1 ? oldRequest.future : statusResponse(installed: true),
      );
      await mount(tester, harness);
      harness.generation = 2;
      await tester.pumpWidget(harness.screen());
      await tester.pumpAndSettle();
      expect(harness.requests, 2);
      oldRequest.completeError(http.ClientException('Old backend stopped'));
      await tester.pumpAndSettle();
      expect(find.text('DINOv2 ViT-B/14 已安装。'), findsOneWidget);
      expect(find.byKey(const Key('dinov2-status-retry')), findsNothing);
      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets('timed out status can retry and ignores late response', (
    tester,
  ) async {
    final slowRequest = Completer<http.Response>();
    final harness = StatusHarness(
      (request) async => request == 1 ? slowRequest.future : statusResponse(),
    );
    await mount(tester, harness);
    await tester.pump(const Duration(seconds: 10));
    await tester.pump();
    expect(find.text('无法读取 DINOv2 组件状态。'), findsOneWidget);
    await tester.tap(find.byKey(const Key('dinov2-status-retry')));
    await tester.pumpAndSettle();
    slowRequest.complete(statusResponse(installed: true));
    await tester.pumpAndSettle();
    expect(harness.requests, 2);
    expect(find.text('DINOv2 未安装。'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('backend shutdown invalidates a pending status response', (
    tester,
  ) async {
    final oldRequest = Completer<http.Response>();
    final harness = StatusHarness(
      (request) async => request == 1 ? oldRequest.future : statusResponse(),
    );
    await mount(tester, harness);
    harness.ready = false;
    await tester.pumpWidget(harness.screen());
    oldRequest.complete(statusResponse(installed: true));
    await tester.pump();
    expect(harness.requests, 1);
    expect(find.text('等待后端就绪...'), findsOneWidget);
    expect(find.text('DINOv2 ViT-B/14 已安装。'), findsNothing);
    harness.ready = true;
    harness.generation = 2;
    await tester.pumpWidget(harness.screen());
    await tester.pumpAndSettle();
    expect(harness.requests, 2);
    expect(find.text('DINOv2 未安装。'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('opening settings retries a failed status', (tester) async {
    final harness = StatusHarness((request) async {
      if (request == 1) throw http.ClientException('Temporary failure');
      return statusResponse();
    })..active = false;
    await mount(tester, harness);
    expect(harness.requests, 1);
    harness.active = true;
    await tester.pumpWidget(harness.screen());
    await tester.pumpAndSettle();
    expect(harness.requests, 2);
    expect(find.text('DINOv2 未安装。'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });
}
