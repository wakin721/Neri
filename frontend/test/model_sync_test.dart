import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/model_sync_status.dart';
import 'package:neri_flutter/src/models/settings.dart';

void main() {
  test('model metadata exposes user and NeriCloud source labels', () {
    const user = ModelInfo(
      name: 'bird.pt',
      path: r'C:\Neri\res\Model\detect\user\bird.pt',
      source: 'user',
      kind: 'detect',
    );
    const synced = ModelInfo(
      name: 'bird.pt',
      path: r'C:\Neri\res\Model\detect\sync\bird.pt',
      source: 'sync',
      kind: 'detect',
    );

    expect(user.sourceLabel, '用户模型');
    expect(user.displayName, '用户模型 / bird.pt');
    expect(synced.sourceLabel, 'NeriCloud');
    expect(synced.displayName, 'NeriCloud / bird.pt');
    expect(user.path, isNot(synced.path));
  });

  test('settings JSON defaults to canonical model roots', () {
    final settings = NeriSettings.fromJson(const <String, dynamic>{});
    expect(settings.modelDirectory, 'res/Model/detect');
    expect(settings.classificationModelDirectory, 'res/Model/cls');
  });

  test('model sync status parses active download progress', () {
    final status = ModelSyncStatus.fromJson(<String, dynamic>{
      'state': 'downloading',
      'run_id': 'abc',
      'current_file': 'detect/bird.pt',
      'total_files': 3,
      'completed_files': 1,
      'received_bytes': 25,
      'total_bytes': 100,
      'last_successful_sync': '2026-09-07T01:00:00+00:00',
      'manifest_id': 'a' * 64,
      'error': null,
      'cloud_detect_count': 4,
      'cloud_cls_count': 2,
    });

    expect(status.isActive, isTrue);
    expect(status.progress, 0.25);
    expect(status.currentFile, 'detect/bird.pt');
    expect(status.cloudDetectCount, 4);
    expect(status.cloudClsCount, 2);
  });

  test('model sync API reads status and starts a manual run', () async {
    final requests = <http.Request>[];
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        requests.add(request);
        return http.Response(
          '{"state":"checking","run_id":"run-1","total_files":2,'
          '"completed_files":0,"received_bytes":0,'
          '"cloud_detect_count":1,"cloud_cls_count":1}',
          request.method == 'GET' || request.method == 'POST' ? 200 : 405,
          headers: const {'content-type': 'application/json'},
        );
      }),
    );

    final status = await client.fetchModelSyncStatus();
    final started = await client.runModelSync();

    expect(status.state, 'checking');
    expect(started.runId, 'run-1');
    expect(requests.map((request) => request.method), <String>['GET', 'POST']);
    expect(
      requests.map((request) => request.url.path),
      <String>['/api/model-sync/status', '/api/model-sync/run'],
    );
  });
}
