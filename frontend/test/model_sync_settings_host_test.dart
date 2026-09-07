import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/screens/model_sync_settings_host.dart';

void main() {
  for (final scenario in [
    (state: 'idle', ready: true, installing: false, tone: 'red'),
    (state: 'failed', ready: true, installing: false, tone: 'red'),
    (state: 'completed', ready: false, installing: false, tone: 'red'),
    (state: 'downloading', ready: true, installing: false, tone: 'yellow'),
    (state: 'completed', ready: false, installing: true, tone: 'yellow'),
    (state: 'idle', ready: false, installing: true, tone: 'yellow'),
    (state: 'completed', ready: true, installing: false, tone: 'green'),
  ]) {
    testWidgets('combined readiness card: $scenario', (tester) async {
      final client = NeriApiClient(
        httpClient: MockClient(
          (_) async => http.Response(
            '{"state":"${scenario.state}"}',
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          ),
        ),
      );
      addTearDown(client.close);
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: ModelSyncSettingsHost(
              apiClient: client,
              enabled: true,
              onCatalogChanged: () async {},
              child: ModelSyncSettingsRow(
                dependenciesReady: scenario.ready,
                installingDependencies: scenario.installing,
                missingDependencies: 'torch',
              ),
            ),
          ),
        ),
      );
      await tester.pump();
      await tester.pump();
      final finder = find.byKey(const Key('model-readiness-card'));
      final card = tester.widget<Container>(finder);
      final scheme = Theme.of(tester.element(finder)).colorScheme;
      final color = switch (scenario.tone) {
        'green' => Colors.green.shade700,
        'yellow' => Colors.amber.shade800,
        _ => scheme.error,
      };
      expect(
        (card.decoration! as BoxDecoration).color,
        color.withValues(alpha: 0.10),
      );
      expect(
        find.descendant(
          of: finder,
          matching: find.text(scenario.state == 'failed' ? '重试' : '立即同步'),
        ),
        scenario.state == 'downloading' ? findsNothing : findsOneWidget,
      );
      await tester.pumpWidget(const SizedBox.shrink());
    });
  }

  testWidgets(
    'inline status can recover when the initial status request fails',
    (tester) async {
      final client = NeriApiClient(
        httpClient: MockClient((request) async {
          if (request.method == 'GET') throw http.ClientException('offline');
          return http.Response(
            '{"state":"completed","run_id":"recovered"}',
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }),
      );
      addTearDown(client.close);
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: ModelSyncSettingsHost(
              apiClient: client,
              enabled: true,
              onCatalogChanged: () async {},
              child: const Column(
                children: [
                  ModelSyncSettingsRow(noticeOnly: true),
                  ModelSyncSettingsRow(),
                ],
              ),
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();
      expect(find.text('模型未同步'), findsOneWidget);
      await tester.tap(find.text('重试'));
      await tester.pumpAndSettle();
      expect(find.text('模型已同步'), findsOneWidget);
    },
  );

  testWidgets(
    'offstage settings host observes startup sync and refreshes catalog',
    (tester) async {
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
              headers: const {
                'content-type': 'application/json; charset=utf-8',
              },
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
    },
  );

  testWidgets('settings host shows a floating message card while models sync', (
    tester,
  ) async {
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        return http.Response(
          '{"state":"downloading","run_id":"float-run",'
          '"current_file":"detect/cloud-bird.pt","received_bytes":30,'
          '"total_bytes":100,"cloud_detect_count":3,"cloud_cls_count":1}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );
    addTearDown(client.close);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ModelSyncSettingsHost(
            apiClient: client,
            enabled: true,
            pollInterval: const Duration(hours: 1),
            onCatalogChanged: () async {},
            child: const SizedBox.expand(),
          ),
        ),
      ),
    );
    await tester.pump();
    await tester.pump();

    final messageCard = find.byKey(const Key('model-sync-message-card'));
    expect(messageCard, findsOneWidget);
    expect(
      find.descendant(of: messageCard, matching: find.text('模型同步')),
      findsOneWidget,
    );
    expect(
      find.descendant(
        of: messageCard,
        matching: find.textContaining('cloud-bird.pt'),
      ),
      findsOneWidget,
    );
    final progress = tester.widget<LinearProgressIndicator>(
      find.descendant(
        of: messageCard,
        matching: find.byType(LinearProgressIndicator),
      ),
    );
    expect(progress.value, 0.30);
  });

  testWidgets(
    'completed sync is still announced when the first read is terminal',
    (tester) async {
      final client = NeriApiClient(
        httpClient: MockClient((request) async {
          return http.Response(
            '{"state":"completed","run_id":"fast-run",'
            '"cloud_detect_count":5,"cloud_cls_count":2,'
            '"last_successful_sync":"2026-09-07T06:30:00+00:00"}',
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }),
      );
      addTearDown(client.close);

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: ModelSyncSettingsHost(
              apiClient: client,
              enabled: true,
              pollInterval: const Duration(hours: 1),
              onCatalogChanged: () async {},
              child: const SizedBox.expand(),
            ),
          ),
        ),
      );
      await tester.pump();
      await tester.pump();

      final messageCard = find.byKey(const Key('model-sync-message-card'));
      expect(messageCard, findsOneWidget);
      expect(
        find.descendant(of: messageCard, matching: find.text('模型同步完成')),
        findsOneWidget,
      );
      expect(
        find.descendant(of: messageCard, matching: find.textContaining('探测 5')),
        findsOneWidget,
      );
      expect(
        find.descendant(of: messageCard, matching: find.textContaining('分类 2')),
        findsOneWidget,
      );
    },
  );

  testWidgets(
    'floating failed sync message retries through the shared controller',
    (tester) async {
      var postCount = 0;
      final client = NeriApiClient(
        httpClient: MockClient((request) async {
          if (request.method == 'POST') {
            postCount++;
            return http.Response(
              '{"state":"checking","run_id":"retry-run"}',
              202,
              headers: const {
                'content-type': 'application/json; charset=utf-8',
              },
            );
          }
          return http.Response(
            '{"state":"failed","run_id":"failed-run",'
            '"error":"NeriCloud 暂时不可用"}',
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }),
      );
      addTearDown(client.close);

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: ModelSyncSettingsHost(
              apiClient: client,
              enabled: true,
              pollInterval: const Duration(hours: 1),
              onCatalogChanged: () async {},
              child: const SizedBox.expand(),
            ),
          ),
        ),
      );
      await tester.pump();
      await tester.pump();

      final messageCard = find.byKey(const Key('model-sync-message-card'));
      expect(messageCard, findsOneWidget);
      expect(
        find.descendant(of: messageCard, matching: find.text('同步失败')),
        findsOneWidget,
      );
      final retry = find.descendant(of: messageCard, matching: find.text('重试'));
      expect(retry, findsOneWidget);

      await tester.tap(retry);
      await tester.pump();
      expect(postCount, 1);
    },
  );
}
