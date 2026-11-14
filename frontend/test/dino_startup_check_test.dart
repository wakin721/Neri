import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/dino_startup_check.dart';

void main() {
  const component = r'D:\Application\Neri\res\model\DINOv3\model.neri.json';
  final unhealthy = DinoComponentStatus.fromJson({
    'installed': true,
    'healthy': false,
    'message': '未找到有效的 Neri1.1 模型清单。',
  });
  test(
    'optional legacy component does not warn for LiteRT or independent preview models',
    () {
      for (final selected in [
        '',
        r'D:\Application\Neri\res\model\cls\user\dinov3-experiment-local.neri.json',
        r'D:\Application\Neri\res\model\detect\sync\lite.tflite',
      ]) {
        expect(
          dinoStartupStatusMessage(unhealthy, selectedModelPath: selected),
          isNull,
        );
      }
    },
  );
  test(
    'selected managed component still reports corruption and missing dependencies',
    () {
      expect(
        dinoStartupStatusMessage(unhealthy, selectedModelPath: component),
        contains('未找到有效'),
      );
      final dependency = DinoComponentStatus.fromJson({
        'installed': true,
        'healthy': false,
        'message': "No module named 'torch'",
      });
      expect(
        dinoStartupStatusMessage(dependency, selectedModelPath: component),
        contains('推理依赖不完整'),
      );
      final missing = DinoComponentStatus.fromJson({
        'installed': false,
        'healthy': false,
      });
      expect(
        dinoStartupStatusMessage(missing, selectedModelPath: component),
        contains('尚未安装'),
      );
      final healthy = DinoComponentStatus.fromJson({
        'installed': true,
        'healthy': true,
      });
      expect(
        dinoStartupStatusMessage(healthy, selectedModelPath: component),
        isNull,
      );
    },
  );
  test(
    'startup status stays generation-bound and honors selected model scope',
    () async {
      var requests = 0;
      final api = NeriApiClient(
        httpClient: MockClient((request) async {
          requests++;
          return http.Response(
            jsonEncode({
              'installed': true,
              'healthy': false,
              'message': '旧版多原型组件需迁移',
            }),
            200,
            headers: {'content-type': 'application/json'},
          );
        }),
      );
      addTearDown(api.close);
      final check = DinoStartupCheck();
      final messages = <String>[];
      await check.run(
        generation: 1,
        apiClient: api,
        onMessage: messages.add,
        selectedModelPath:
            r'D:\Application\Neri\res\model\cls\user\preview.neri.json',
      );
      expect(messages, isEmpty);
      await check.run(
        generation: 1,
        apiClient: api,
        onMessage: messages.add,
        selectedModelPath: component,
      );
      expect(requests, 1);
      await check.run(
        generation: 2,
        apiClient: api,
        onMessage: messages.add,
        selectedModelPath: component,
      );
      expect(requests, 2);
      expect(messages.single, contains('旧版多原型'));
    },
  );
}
