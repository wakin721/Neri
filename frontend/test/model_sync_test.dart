import 'package:flutter_test/flutter_test.dart';
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
}
