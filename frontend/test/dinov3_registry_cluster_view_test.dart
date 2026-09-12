import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/dinov3_registry.dart';
import 'package:neri_flutter/src/widgets/dinov3_registry_dialog.dart';

Map<String, dynamic> _entryJson({
  required int id,
  required String status,
  required String name,
  List<Map<String, dynamic>> clusters = const [],
  int feedbackEventCount = 0,
  int feedbackPrototypeCount = 0,
  String? learningStatus,
}) {
  return {
    'id': id,
    'candidate_number': id.abs(),
    'status': status,
    'common_name': name,
    'scientific_name': '',
    'event_count': status == 'checkpoint' ? 0 : 4,
    'camera_count': status == 'checkpoint' ? 0 : 1,
    'prototype_count': status == 'checkpoint' ? 1 : 1,
    'cluster_purity': 1.0,
    'embedding_consistency': 1.0,
    'conditions': <String, bool>{},
    'can_register': false,
    'display_name': name,
    'feedback_event_count': feedbackEventCount,
    'feedback_prototype_count': feedbackPrototypeCount,
    if (learningStatus != null) 'learning_status': learningStatus,
    'clusters': clusters,
  };
}

Map<String, dynamic> _clusterJson({
  required String id,
  required String label,
  required String source,
  required int prototypeIndex,
  int eventCount = 0,
  int cameraCount = 0,
  bool active = true,
  double? meanSquaredDistance,
}) {
  return {
    'id': id,
    'label': label,
    'source': source,
    'prototype_index': prototypeIndex,
    'event_count': eventCount,
    'camera_count': cameraCount,
    'sample_count': eventCount,
    'active': active,
    if (meanSquaredDistance != null)
      'mean_squared_distance': meanSquaredDistance,
    'example_refs': <Map<String, dynamic>>[],
  };
}

DinoV3RegistryEntry _staleEntry() {
  return DinoV3RegistryEntry.fromJson(
    _entryJson(id: -1, status: 'checkpoint', name: '旧目录'),
  );
}

void main() {
  test('registry entry parses feedback learning state and clusters', () {
    final entry = DinoV3RegistryEntry.fromJson(
      _entryJson(
        id: -1,
        status: 'checkpoint',
        name: '盘羊',
        feedbackEventCount: 4,
        feedbackPrototypeCount: 1,
        learningStatus: 'provisional',
        clusters: [
          _clusterJson(
            id: 'checkpoint:-1:0',
            label: 'Base #1',
            source: 'checkpoint',
            prototypeIndex: 0,
          ),
          _clusterJson(
            id: 'feedback:盘羊:0',
            label: 'Feedback Cluster #1',
            source: 'feedback',
            prototypeIndex: 0,
            eventCount: 4,
            cameraCount: 2,
            meanSquaredDistance: 0.02,
          ),
        ],
      ),
    );

    expect(entry.feedbackEventCount, 4);
    expect(entry.feedbackPrototypeCount, 1);
    expect(entry.learningStatus, 'provisional');
    expect(entry.clusters, hasLength(2));
    expect(entry.clusters.last.source, 'feedback');
    expect(entry.clusters.last.eventCount, 4);
  });

  testWidgets('opening registry dialog refreshes stale initial catalog', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1200, 800));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    var catalogRequests = 0;
    final fresh = _entryJson(
      id: 7,
      status: 'candidate',
      name: '赤麂',
      clusters: [
        _clusterJson(
          id: 'registry:7:0',
          label: 'Cluster #1',
          source: 'registry',
          prototypeIndex: 0,
          eventCount: 1,
          cameraCount: 1,
        ),
      ],
    );
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.method == 'GET' &&
            request.url.path == '/api/dinov3/registry/catalog') {
          catalogRequests += 1;
          return http.Response(
            jsonEncode([fresh]),
            200,
            headers: const {'content-type': 'application/json; charset=utf-8'},
          );
        }
        if (request.method == 'GET' && request.url.path.endsWith('/events')) {
          return http.Response('[]', 200);
        }
        return http.Response('[]', 200);
      }),
    );
    addTearDown(client.close);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: DinoV3RegistryDialog(
            apiClient: client,
            modelPath: 'head.neri.json',
            initialEntries: [_staleEntry()],
            onContinueValidation: (_) {},
          ),
        ),
      ),
    );
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 350));

    expect(catalogRequests, 1);
    expect(find.text('赤麂'), findsWidgets);
    expect(find.text('旧目录'), findsNothing);
  });

  testWidgets('species rows start collapsed and expose clusters after expansion', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1200, 800));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final entry = DinoV3RegistryEntry.fromJson(
      _entryJson(
        id: -1,
        status: 'checkpoint',
        name: '盘羊',
        feedbackEventCount: 4,
        feedbackPrototypeCount: 1,
        learningStatus: 'provisional',
        clusters: [
          _clusterJson(
            id: 'checkpoint:-1:0',
            label: 'Base #1',
            source: 'checkpoint',
            prototypeIndex: 0,
          ),
          _clusterJson(
            id: 'feedback:盘羊:0',
            label: 'Feedback Cluster #1',
            source: 'feedback',
            prototypeIndex: 0,
            eventCount: 4,
            cameraCount: 2,
            meanSquaredDistance: 0.02,
          ),
        ],
      ),
    );
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/dinov3/registry/catalog') {
          return http.Response(
            jsonEncode([
              _entryJson(
                id: -1,
                status: 'checkpoint',
                name: '盘羊',
                feedbackEventCount: 4,
                feedbackPrototypeCount: 1,
                learningStatus: 'provisional',
                clusters: [
                  _clusterJson(
                    id: 'checkpoint:-1:0',
                    label: 'Base #1',
                    source: 'checkpoint',
                    prototypeIndex: 0,
                  ),
                  _clusterJson(
                    id: 'feedback:盘羊:0',
                    label: 'Feedback Cluster #1',
                    source: 'feedback',
                    prototypeIndex: 0,
                    eventCount: 4,
                    cameraCount: 2,
                    meanSquaredDistance: 0.02,
                  ),
                ],
              ),
            ]),
            200,
          );
        }
        return http.Response('[]', 200);
      }),
    );
    addTearDown(client.close);

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: DinoV3RegistryDialog(
            apiClient: client,
            modelPath: 'head.neri.json',
            initialEntries: [entry],
            onContinueValidation: (_) {},
          ),
        ),
      ),
    );
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 350));

    expect(find.text('盘羊'), findsWidgets);
    expect(find.text('Base #1'), findsNothing);
    expect(find.text('Feedback Cluster #1'), findsNothing);

    await tester.tap(find.text('盘羊').first);
    await tester.pumpAndSettle();

    expect(find.text('Base #1'), findsOneWidget);
    expect(find.text('Feedback Cluster #1'), findsOneWidget);
    expect(find.textContaining('4 事件'), findsWidgets);
    expect(find.textContaining('provisional'), findsWidgets);
  });
}
