import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/close_behavior.dart';

void main() {
  test('normalizes supported close behaviors', () {
    expect(normalizeCloseBehavior(closeBehaviorAsk), closeBehaviorAsk);
    expect(
      normalizeCloseBehavior(closeBehaviorHideToTray),
      closeBehaviorHideToTray,
    );
    expect(normalizeCloseBehavior(closeBehaviorExit), closeBehaviorExit);
  });

  test('falls back to asking for missing or unknown values', () {
    expect(normalizeCloseBehavior(null), closeBehaviorAsk);
    expect(normalizeCloseBehavior('unexpected'), closeBehaviorAsk);
  });
}
