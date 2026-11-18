import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/app_theme.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';
import 'package:neri_flutter/src/screens/settings_screen.dart';
import 'package:neri_flutter/src/widgets/settings_progress_indicator.dart';

http.Response statusResponse({bool installed = false, bool? healthy}) =>
    http.Response(
      jsonEncode({
        'installed': installed,
        'healthy': healthy ?? installed,
        'message': installed ? 'Neri1.1 已安装。' : 'Neri1.1 未安装。',
      }),
      200,
      headers: {'content-type': 'application/json; charset=utf-8'},
    );

class StatusHarness {
  StatusHarness(
    Future<http.Response> Function(int) response, {
    Future<http.Response>? classes,
  }) {
    api = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/models/classes' && classes != null)
          return classes;
        if (request.url.path == '/api/environment/dinov3-status') {
          return response(++requests);
        }
        if (request.url.path == '/api/environment/remove-dinov3') {
          removeRequests += 1;
          return http.Response(
            '{"operation":"remove_dinov3","state":"starting","message":"正在删除","progress":0}',
            202,
          );
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
  int removeRequests = 0;

  Widget screen() => MaterialApp(
    theme: buildNeriTheme(ColorScheme.fromSeed(seedColor: kDefaultSeedColor)),
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
  testWidgets('species and component settings paint a text-sized loading ring', (
    tester,
  ) async {
    final classes = Completer<http.Response>();
    final component = Completer<http.Response>();
    final harness =
        StatusHarness((_) => component.future, classes: classes.future)
          ..settings = NeriSettings.fromJson({
            'selected_model': 'detector.pt',
            'available_models': [
              {'name': 'Detector', 'path': 'detector.pt'},
            ],
          });
    await tester.binding.setSurfaceSize(const Size(1400, 1000));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    addTearDown(harness.api.close);
    addTearDown(harness.theme.dispose);
    await tester.pumpWidget(harness.screen());
    await tester.pump();
    await tester.ensureVisible(find.byType(SettingsProgressIndicator));
    await tester.pump();

    void expectRing() {
      final ring = find.descendant(
        of: find.byType(SettingsProgressIndicator),
        matching: find.byType(CircularProgressIndicator),
      );
      expect(tester.widget<CircularProgressIndicator>(ring).strokeWidth, 2);
      final paint = find.descendant(
        of: ring,
        matching: find.byType(CustomPaint),
      );
      expect(tester.getSize(paint), const Size(14, 14));
    }

    expectRing();
    expect(
      tester
          .widget<CircularProgressIndicator>(
            find.byType(CircularProgressIndicator),
          )
          .semanticsLabel,
      '正在读取识别物种',
    );
    classes.complete(http.Response('[]', 200));
    await tester.pump();
    expect(find.byType(SettingsProgressIndicator), findsNothing);
    await tester.tap(find.text('环境维护').first);
    await tester.pump();
    expectRing();
    component.complete(statusResponse(installed: true));
    await tester.pumpAndSettle();
    expect(find.byType(SettingsProgressIndicator), findsNothing);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('unhealthy installed component offers deletion beside repair', (
    tester,
  ) async {
    final harness = StatusHarness(
      (_) async => statusResponse(installed: true, healthy: false),
    );
    await mount(tester, harness);
    await tester.pumpAndSettle();
    final repair = find.widgetWithText(FilledButton, '安装/修复');
    final remove = find.byKey(const Key('dino-component-remove'));
    expect(repair, findsOneWidget);
    expect(remove, findsOneWidget);
    expect(
      tester.getCenter(remove).dx,
      greaterThan(tester.getRect(repair).right),
    );
    await tester.tap(remove);
    await tester.pumpAndSettle();
    expect(find.text('删除联合模型组件'), findsOneWidget);
    expect(find.textContaining('人工校验记录与补图数据会保留'), findsOneWidget);
    final confirmation = find.descendant(
      of: find.byType(AlertDialog),
      matching: find.text('删除'),
    );
    await tester.tap(confirmation);
    await tester.pumpAndSettle();
    expect(harness.removeRequests, 1);
    await tester.pumpWidget(const SizedBox());
  });

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
      expect(find.text('等待后端就绪'), findsNothing);
      expect(
        tester.getRect(find.text('等待后端就绪...')).right,
        lessThan(
          tester.getRect(find.byKey(const Key('dino-component-loading'))).left,
        ),
      );
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
      expect(find.text('Neri1.1 未安装。'), findsOneWidget);
      expect(find.byKey(const Key('dino-component-remove')), findsNothing);
      expect(find.byKey(const Key('dino-status-retry')), findsNothing);
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
    expect(find.text('无法读取 DINOv3 组件状态。'), findsOneWidget);
    expect(find.byKey(const Key('dino-component-action')), findsNothing);
    await tester.tap(find.byKey(const Key('dino-status-retry')));
    await tester.pump();
    expect(harness.requests, 2);
    expect(find.text('正在检查组件...'), findsOneWidget);
    expect(find.text('正在检查组件'), findsNothing);
    expect(find.byKey(const Key('dino-status-retry')), findsNothing);
    retry.complete(statusResponse(installed: true));
    await tester.pumpAndSettle();
    expect(find.text('Neri1.1 已安装。'), findsOneWidget);
    expect(find.widgetWithText(OutlinedButton, '删除'), findsOneWidget);
    expect(find.byKey(const Key('dino-component-remove')), findsNothing);
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
      expect(find.text('Neri1.1 已安装。'), findsOneWidget);
      expect(find.byKey(const Key('dino-status-retry')), findsNothing);
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
    expect(find.text('无法读取 DINOv3 组件状态。'), findsOneWidget);
    await tester.tap(find.byKey(const Key('dino-status-retry')));
    await tester.pumpAndSettle();
    slowRequest.complete(statusResponse(installed: true));
    await tester.pumpAndSettle();
    expect(harness.requests, 2);
    expect(find.text('Neri1.1 未安装。'), findsOneWidget);
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
    expect(find.text('Neri1.1 已安装。'), findsNothing);
    harness.ready = true;
    harness.generation = 2;
    await tester.pumpWidget(harness.screen());
    await tester.pumpAndSettle();
    expect(harness.requests, 2);
    expect(find.text('Neri1.1 未安装。'), findsOneWidget);
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
    expect(find.text('Neri1.1 未安装。'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });
}
