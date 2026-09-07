import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/screens/model_sync_settings_card.dart';
import 'package:neri_flutter/src/screens/model_sync_settings_host.dart';

void main() {
  testWidgets('model sync settings card renders progress and cloud counts', (
    tester,
  ) async {
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.method, 'GET');
        expect(request.url.path, '/api/model-sync/status');
        return http.Response(
          '{"state":"downloading","run_id":"run-1",'
          '"current_file":"detect/bird.pt","received_bytes":25,'
          '"total_bytes":100,"cloud_detect_count":4,"cloud_cls_count":2}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    addTearDown(client.close);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ModelSyncSettingsCard(
            apiClient: client,
            pollInterval: const Duration(hours: 1),
            onCatalogChanged: () async {},
          ),
        ),
      ),
    );
    await tester.pump();
    await tester.pump();

    expect(find.text('模型同步'), findsOneWidget);
    expect(find.textContaining('正在同步'), findsOneWidget);
    expect(find.textContaining('detect/bird.pt'), findsOneWidget);
    expect(find.textContaining('探测 4'), findsOneWidget);
    expect(find.textContaining('分类 2'), findsOneWidget);
    final progress = tester.widget<LinearProgressIndicator>(
      find.byType(LinearProgressIndicator),
    );
    expect(progress.value, 0.25);
  });

  testWidgets('failed model sync shows retry and invokes local run API', (
    tester,
  ) async {
    var postCount = 0;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.method == 'POST') {
          postCount++;
          return http.Response(
            '{"state":"checking","run_id":"run-2"}',
            202,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        return http.Response(
          '{"state":"failed","error":"网络不可用",'
          '"last_successful_sync":"2026-09-07T01:00:00+00:00"}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    addTearDown(client.close);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ModelSyncSettingsCard(
            apiClient: client,
            pollInterval: const Duration(hours: 1),
            onCatalogChanged: () async {},
          ),
        ),
      ),
    );
    await tester.pump();
    await tester.pump();

    expect(find.text('同步失败'), findsOneWidget);
    expect(find.textContaining('网络不可用'), findsOneWidget);
    expect(find.text('重试'), findsOneWidget);
    expect(find.textContaining('不会影响已安装的本地模型'), findsOneWidget);

    await tester.tap(find.text('重试'));
    await tester.pump();
    expect(postCount, 1);
  });

  testWidgets('disabled card waits for backend readiness before status read', (
    tester,
  ) async {
    var requestCount = 0;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        requestCount++;
        return http.Response(
          '{"state":"checking","run_id":"run-ready"}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    addTearDown(client.close);

    Widget buildCard(bool enabled) {
      return MaterialApp(
        home: Scaffold(
          body: ModelSyncSettingsCard(
            apiClient: client,
            enabled: enabled,
            pollInterval: const Duration(hours: 1),
            onCatalogChanged: () async {},
          ),
        ),
      );
    }

    await tester.pumpWidget(buildCard(false));
    await tester.pump();
    expect(requestCount, 0);
    expect(find.text('等待本地服务'), findsOneWidget);

    await tester.pumpWidget(buildCard(true));
    await tester.pump();
    await tester.pump();
    expect(requestCount, 1);
    expect(find.textContaining('正在同步'), findsOneWidget);
  });

  testWidgets('offstage settings host observes startup sync and refreshes catalog', (
    tester,
  ) async {
    var statusReads = 0;
    var catalogRefreshes = 0;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        statusReads++;
        if (statusReads == 1) {
          return http.Response(
            '{"state":"downloading","run_id":"startup-run",'
            '"current_file":"detect/cloud.pt","received_bytes":1,'
            '"total_bytes":2}',
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        return http.Response(
          '{"state":"completed","run_id":"startup-run",'
          '"received_bytes":2,"total_bytes":2}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    addTearDown(client.close);

    Widget buildHost(bool backendReady) {
      return MaterialApp(
        home: Scaffold(
          body: IndexedStack(
            index: 0,
            children: [
              const SizedBox.expand(),
              ModelSyncSettingsHost(
                apiClient: client,
                enabled: backendReady,
                pollInterval: const Duration(milliseconds: 10),
                onCatalogChanged: () async => catalogRefreshes++,
                child: const SizedBox.expand(),
              ),
            ],
          ),
        ),
      );
    }

    await tester.pumpWidget(buildHost(false));
    await tester.pump();
    expect(statusReads, 0);

    await tester.pumpWidget(buildHost(true));
    await tester.pump();
    expect(statusReads, 1);
    await tester.pump(const Duration(milliseconds: 15));
    await tester.pump(const Duration(milliseconds: 15));

    expect(statusReads, greaterThanOrEqualTo(2));
    expect(catalogRefreshes, 1);
  });
}
