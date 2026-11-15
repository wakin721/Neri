import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';

void main() {
  test('missing component identity uses a neutral label', () {
    final status = DinoComponentStatus.fromJson({
      'installed': false,
      'healthy': false,
    });
    expect(status.displayName, '联合模型');
  });

  test('parses DINOv3 environment status', () async {
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.url.path, '/api/environment/dinov3-status');
        return http.Response.bytes(
          utf8.encode(
            jsonEncode({
              'installed': true,
              'healthy': true,
              'architecture': 'DINOv3 ViT-B/16',
              'component_version': 1,
              'display_name': 'Neri1.1',
              'class_count': 51,
              'feature_dim': 768,
              'manifest': 'dinov3-ridge-within.neri.json',
              'message': 'Neri1.1 已安装。',
            }),
          ),
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );

    final status = await client.fetchDinoComponentStatus();

    expect(status.installed, isTrue);
    expect(status.healthy, isTrue);
    expect(status.architecture, 'DINOv3 ViT-B/16');
    expect(status.componentVersion, 1);
    expect(status.displayName, 'Neri1.1');
    expect(status.classCount, 51);
    expect(status.featureDim, 768);
    expect(status.manifest, 'dinov3-ridge-within.neri.json');
    expect(status.message, 'Neri1.1 已安装。');
  });

  test('starts DINOv3 install with selected package source', () async {
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/api/environment/install-dinov3');
        expect(jsonDecode(request.body), {
          'env_choice': 'CPU Only',
          'package_source': 'nju',
        });
        return http.Response(
          jsonEncode({
            'operation': 'install_dinov3',
            'state': 'starting',
            'message': 'started',
            'progress': 0,
          }),
          202,
        );
      }),
    );

    final response = await client.installDinoV3(
      envChoice: 'CPU Only',
      packageSource: 'nju',
    );

    expect(response.operation, 'install_dinov3');
  });

  test('starts DINOv3 removal', () async {
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/api/environment/remove-dinov3');
        return http.Response(
          jsonEncode({
            'operation': 'remove_dinov3',
            'state': 'starting',
            'message': 'started',
            'progress': 0,
          }),
          202,
        );
      }),
    );

    final response = await client.removeDinoV3();

    expect(response.operation, 'remove_dinov3');
  });
}
