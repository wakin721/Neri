import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/app_updater.dart';

void main() {
  group('compareNeriVersions', () {
    test('orders stable and preview releases', () {
      expect(compareNeriVersions('3.0.5', '3.0.5-alpha10'), greaterThan(0));
      expect(
        compareNeriVersions('3.0.5-alpha10', '3.0.5-alpha2'),
        greaterThan(0),
      );
    });

    test('ignores display build codes', () {
      expect(compareNeriVersions('3.0.5-alpha2(b5f6a6)', 'v3.0.5-alpha2'), 0);
      expect(compareNeriVersions('v3.0.4-release(dfc280)', '3.0.4'), 0);
    });

    test('compares numeric release components before prerelease labels', () {
      expect(compareNeriVersions('3.1.0-alpha1', '3.0.9'), greaterThan(0));
      expect(compareNeriVersions('3.0.6', '3.0.10'), lessThan(0));
    });
  });
}
