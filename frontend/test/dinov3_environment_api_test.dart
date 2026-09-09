import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';

void main() {
  test('parses DINOv3 environment status', () async {
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.url.path, '/api/environment/dinov3-status');
        return http.Response.bytes(
          utf8.encode(
            jsonEncode({
              'installed': true,
              'healthy': false,
              'architecture': 'DINOv3 ViT-B/16',
              'component_version': 1,
              'source_commit': '6876159a11b4df116f30f667f8c9888617df0751',
              'message': '需要修复',
            }),
          ),
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );

    final status = await client.fetchDinoV3ComponentStatus();

    expect(status.installed, isTrue);
    expect(status.healthy, isFalse);
    expect(status.architecture, 'DINOv3 ViT-B/16');
    expect(status.message, '需要修复');
  });

  test('starts DINOv3 install with selected environment source', () async {
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
            'accepted': true,
            'operation': 'install_dinov3',
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
            'accepted': true,
            'operation': 'remove_dinov3',
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
