import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/widgets/dinov3_registry_dialog.dart';

Map<String, dynamic> _entry({
  required int id,
  required String status,
  required String name,
}) {
  return {
    'id': id,
    'candidate_number': id.abs(),
    'status': status,
    'common_name': name,
    'scientific_name': '',
    'event_count': status == 'checkpoint' ? 0 : 1,
    'camera_count': status == 'checkpoint' ? 0 : 1,
    'prototype_count': status == 'checkpoint' ? 1 : 0,
    'cluster_purity': 1.0,
    'embedding_consistency': 1.0,
    'conditions': <String, bool>{},
    'can_register': false,
    'display_name': name,
    'feedback_event_count': 0,
    'feedback_prototype_count': 0,
    'learning_status': status,
    'clusters': <Map<String, dynamic>>[],
  };
}

void main() {
  testWidgets('clear unregistered events deletes candidates and keeps registered states', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1200, 800));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    var cleared = false;
    var deleteCalls = 0;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.method == 'GET' &&
            request.url.path == '/api/dinov3/registry/catalog') {
          return http.Response(
            jsonEncode([
              _entry(id: -1, status: 'checkpoint', name: '盘羊'),
              if (!cleared)
                _entry(id: 7, status: 'candidate', name: '未知物种 #7'),
              _entry(id: 8, status: 'provisional', name: '赤麂'),
            ]),
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        if (request.method == 'DELETE' &&
            request.url.path == '/api/dinov3/registry/candidates') {
          deleteCalls += 1;
          cleared = true;
          return http.Response(
            jsonEncode({'deleted': 1}),
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        if (request.method == 'GET' && request.url.path.endsWith('/events')) {
          return http.Response('[]', 200);
        }
        return http.Response('{}', 200);
      }),
    );
    addTearDown(client.close);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: DinoV3RegistryDialog(
            apiClient: client,
            modelPath: 'head.neri.json',
            onContinueValidation: (_) {},
          ),
        ),
      ),
    );
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 350));

    expect(find.text('清除未注册事件'), findsOneWidget);
    expect(find.text('未知物种 #7'), findsWidgets);
    expect(find.text('赤麂'), findsWidgets);

    await tester.tap(find.text('清除未注册事件'));
    await tester.pumpAndSettle();
    expect(find.text('清除所有未注册事件？'), findsOneWidget);
    expect(find.textContaining('1 个 Candidate'), findsOneWidget);

    await tester.tap(find.text('确认清除'));
    await tester.pumpAndSettle();

    expect(deleteCalls, 1);
    expect(find.text('未知物种 #7'), findsNothing);
    expect(find.text('赤麂'), findsWidgets);
    expect(find.text('盘羊'), findsWidgets);
  });
}
