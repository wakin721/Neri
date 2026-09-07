import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/screens/model_sync_settings_card.dart';

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

    expect(find.textContaining('同步失败'), findsOneWidget);
    expect(find.textContaining('网络不可用'), findsOneWidget);
    expect(find.text('重试'), findsOneWidget);
    expect(find.textContaining('不会影响已安装的本地模型'), findsOneWidget);

    await tester.tap(find.text('重试'));
    await tester.pump();
    expect(postCount, 1);
  });
}
