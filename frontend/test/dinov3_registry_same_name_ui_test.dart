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
  required int candidateNumber,
  required String name,
  required int eventCount,
  String status = 'candidate',
}) {
  return <String, dynamic>{
    'id': id,
    'candidate_number': candidateNumber,
    'status': status,
    'common_name': name,
    'scientific_name': '',
    'event_count': eventCount,
    'camera_count': 1,
    'prototype_count': eventCount >= 4 ? 1 : 0,
    'cluster_purity': 1.0,
    'embedding_consistency': 1.0,
    'conditions': <String, bool>{},
    'can_register': false,
    'display_name': name.isEmpty ? '未知物种 #$candidateNumber' : name,
    'clusters': <Map<String, dynamic>>[],
  };
}

http.Response _jsonResponse(Object? value) {
  return http.Response.bytes(
    utf8.encode(jsonEncode(value)),
    200,
    headers: const {'content-type': 'application/json; charset=utf-8'},
  );
}

List<DinoV3RegistryEntry> _entries(List<Map<String, dynamic>> values) =>
    values.map(DinoV3RegistryEntry.fromJson).toList();

List<String> _speciesTileLabels(WidgetTester tester) {
  return tester
      .widgetList<ExpansionTile>(find.byType(ExpansionTile))
      .map((tile) => (tile.title as Text).data ?? '')
      .toList();
}

Future<void> _pumpDialog(
  WidgetTester tester,
  NeriApiClient client,
  List<Map<String, dynamic>> catalog,
) async {
  await tester.binding.setSurfaceSize(const Size(1200, 850));
  await tester.pumpWidget(
    MaterialApp(
      home: Scaffold(
        body: DinoV3RegistryDialog(
          apiClient: client,
          modelPath: 'head.neri.json',
          initialEntries: _entries(catalog),
          onContinueValidation: (_) {},
        ),
      ),
    ),
  );
  await tester.pump();
  await tester.pump(const Duration(milliseconds: 350));
}

void main() {
  testWidgets('named registry entries sort before unnamed then by event count', (
    tester,
  ) async {
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final catalog = <Map<String, dynamic>>[
      _entryJson(id: 3, candidateNumber: 3, name: '', eventCount: 6),
      _entryJson(id: 2, candidateNumber: 2, name: 'B物种', eventCount: 4),
      _entryJson(id: 4, candidateNumber: 4, name: '', eventCount: 3),
      _entryJson(id: 1, candidateNumber: 1, name: 'A物种', eventCount: 5),
    ];
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.url.path == '/api/dinov3/registry/catalog') {
          return _jsonResponse(catalog);
        }
        if (request.url.path.endsWith('/events')) {
          return _jsonResponse(const <Object>[]);
        }
        return _jsonResponse(const <Object>[]);
      }),
    );
    addTearDown(client.close);

    await _pumpDialog(tester, client, catalog);

    expect(
      _speciesTileLabels(tester),
      <String>['A物种', 'B物种', '未知物种 #3', '未知物种 #4'],
    );
  });

  testWidgets('declining same-name candidate merge keeps both with stable suffixes', (
    tester,
  ) async {
    addTearDown(() => tester.binding.setSurfaceSize(null));
    var catalog = <Map<String, dynamic>>[
      _entryJson(id: 1, candidateNumber: 1, name: '家骆驼', eventCount: 4),
      _entryJson(id: 2, candidateNumber: 2, name: '', eventCount: 1),
    ];
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.method == 'PATCH' &&
            request.url.path == '/api/dinov3/registry/2/identity') {
          final body = jsonDecode(request.body) as Map<String, dynamic>;
          expect(body['common_name'], '家骆驼');
          final updated = _entryJson(
            id: 2,
            candidateNumber: 2,
            name: '家骆驼',
            eventCount: 1,
          );
          catalog = <Map<String, dynamic>>[catalog.first, updated];
          return _jsonResponse(updated);
        }
        if (request.url.path == '/api/dinov3/registry/catalog') {
          return _jsonResponse(catalog);
        }
        if (request.url.path.endsWith('/events')) {
          return _jsonResponse(const <Object>[]);
        }
        return _jsonResponse(const <Object>[]);
      }),
    );
    addTearDown(client.close);

    await _pumpDialog(tester, client, catalog);
    await tester.tap(find.text('未知物种 #2').first);
    await tester.pumpAndSettle();
    expect(find.text('1 个独立事件 · 1 台相机'), findsOneWidget);
    await tester.enterText(find.byType(TextField).first, '家骆驼');
    await tester.tap(find.text('保存物种名称'));
    await tester.pumpAndSettle();

    expect(find.text('已存在同名物种“家骆驼”'), findsOneWidget);
    expect(find.text('不合并'), findsOneWidget);
    expect(find.text('合并'), findsOneWidget);

    await tester.tap(find.text('不合并'));
    await tester.pumpAndSettle();

    expect(
      _speciesTileLabels(tester),
      containsAllInOrder(<String>['家骆驼 #1', '家骆驼 #2']),
    );
  });

  testWidgets('accepting same-name candidate merge combines into existing entry', (
    tester,
  ) async {
    addTearDown(() => tester.binding.setSurfaceSize(null));
    var mergeCalled = false;
    var catalog = <Map<String, dynamic>>[
      _entryJson(id: 1, candidateNumber: 1, name: '家骆驼', eventCount: 4),
      _entryJson(id: 2, candidateNumber: 2, name: '', eventCount: 1),
    ];
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        if (request.method == 'PATCH' &&
            request.url.path == '/api/dinov3/registry/2/identity') {
          final updated = _entryJson(
            id: 2,
            candidateNumber: 2,
            name: '家骆驼',
            eventCount: 1,
          );
          catalog = <Map<String, dynamic>>[catalog.first, updated];
          return _jsonResponse(updated);
        }
        if (request.method == 'POST' &&
            request.url.path == '/api/dinov3/registry/2/merge-candidate') {
          final body = jsonDecode(request.body) as Map<String, dynamic>;
          expect(body['target_registration_id'], 1);
          mergeCalled = true;
          final merged = _entryJson(
            id: 1,
            candidateNumber: 1,
            name: '家骆驼',
            eventCount: 5,
          );
          catalog = <Map<String, dynamic>>[merged];
          return _jsonResponse(merged);
        }
        if (request.url.path == '/api/dinov3/registry/catalog') {
          return _jsonResponse(catalog);
        }
        if (request.url.path.endsWith('/events')) {
          return _jsonResponse(const <Object>[]);
        }
        return _jsonResponse(const <Object>[]);
      }),
    );
    addTearDown(client.close);

    await _pumpDialog(tester, client, catalog);
    await tester.tap(find.text('未知物种 #2').first);
    await tester.pumpAndSettle();
    expect(find.text('1 个独立事件 · 1 台相机'), findsOneWidget);
    await tester.enterText(find.byType(TextField).first, '家骆驼');
    await tester.tap(find.text('保存物种名称'));
    await tester.pumpAndSettle();

    await tester.tap(find.text('合并'));
    await tester.pumpAndSettle();

    expect(mergeCalled, isTrue);
    expect(_speciesTileLabels(tester), <String>['家骆驼']);
  });
}
