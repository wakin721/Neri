import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';

void main() {
  test('parses DINOv2 environment status', () async {
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.url.path, '/api/environment/dinov2-status');
        return http.Response.bytes(
          utf8.encode(
            jsonEncode({
              'installed': true,
              'healthy': true,
              'architecture': 'DINOv2 ViT-B/14',
              'component_version': 1,
              'classifier_filename': 'classifier.pt',
              'classifier_fingerprint': 'a' * 64,
              'classifier_head_type': 'multi_prototype',
              'prototype_count': 75,
              'message': 'DINOv2 ViT-B/14 已安装。',
            }),
          ),
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        );
      }),
    );

    final status = await client.fetchDinoV2ComponentStatus();

    expect(status.installed, isTrue);
    expect(status.healthy, isTrue);
    expect(status.architecture, 'DINOv2 ViT-B/14');
    expect(status.componentVersion, 1);
    expect(status.classifierFilename, 'classifier.pt');
    expect(status.classifierFingerprint, 'a' * 64);
    expect(status.classifierHeadType, 'multi_prototype');
    expect(status.prototypeCount, 75);
    expect(status.message, 'DINOv2 ViT-B/14 已安装。');
  });

  test('starts DINOv2 install with selected package source', () async {
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/api/environment/install-dinov2');
        expect(jsonDecode(request.body), {
          'env_choice': 'CPU Only',
          'package_source': 'nju',
        });
        return http.Response(
          jsonEncode({
            'operation': 'install_dinov2',
            'state': 'starting',
            'message': 'started',
            'progress': 0,
          }),
          202,
        );
      }),
    );

    final response = await client.installDinoV2(
      envChoice: 'CPU Only',
      packageSource: 'nju',
    );

    expect(response.operation, 'install_dinov2');
  });

  test('starts DINOv2 removal', () async {
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/api/environment/remove-dinov2');
        return http.Response(
          jsonEncode({
            'operation': 'remove_dinov2',
            'state': 'starting',
            'message': 'started',
            'progress': 0,
          }),
          202,
        );
      }),
    );

    final response = await client.removeDinoV2();

    expect(response.operation, 'remove_dinov2');
  });
}
