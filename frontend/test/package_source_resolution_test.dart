import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';

void main() {
  test('reads the resolved package source and backend label', () {
    final resolution = PackageSourceResolution.fromJson({
      'source': 'aliyun',
      'label': '阿里源',
    });

    expect(resolution.source, 'aliyun');
    expect(resolution.label, '阿里源');
  });

  test('uses a readable fallback when the backend omits the label', () {
    final resolution = PackageSourceResolution.fromJson({'source': 'tsinghua'});

    expect(resolution.source, 'tsinghua');
    expect(resolution.label, '清华源');
  });

  test('manual update sources bypass automatic IP detection', () async {
    var requestCount = 0;
    final client = NeriApiClient(
      httpClient: MockClient((_) async {
        requestCount++;
        return http.Response('{"mainland_china":false}', 200);
      }),
    );

    try {
      expect(await client.shouldUseMainlandUpdateSource('domestic'), isTrue);
      expect(await client.shouldUseMainlandUpdateSource('github'), isFalse);
      expect(requestCount, 0);
    } finally {
      client.close();
    }
  });

  test('automatic update source uses the shared backend detector', () async {
    var requestCount = 0;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        requestCount++;
        expect(request.url.path, '/api/environment/update-source');
        return http.Response('{"mainland_china":true}', 200);
      }),
    );

    try {
      expect(await client.shouldUseMainlandUpdateSource(), isTrue);
      expect(requestCount, 1);
    } finally {
      client.close();
    }
  });
}
