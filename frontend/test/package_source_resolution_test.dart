import 'package:flutter_test/flutter_test.dart';
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
}
