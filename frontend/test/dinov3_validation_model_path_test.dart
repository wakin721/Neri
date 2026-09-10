import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/main_window.dart';
import 'package:neri_flutter/src/models/settings.dart';

void main() {
  test('validation feedback model path is exposed only for DINOv3', () {
    const dino = ModelInfo(
      name: 'DINOv3',
      path: 'model.neri.json',
      kind: 'classification',
      backend: 'dinov3',
    );
    const other = ModelInfo(
      name: 'Other classifier',
      path: 'other.pt',
      kind: 'classification',
      backend: 'yolo',
    );

    expect(
      resolveDinoV3ValidationModelPath(dino, ' model.neri.json '),
      'model.neri.json',
    );
    expect(resolveDinoV3ValidationModelPath(other, 'other.pt'), isNull);
    expect(resolveDinoV3ValidationModelPath(dino, '   '), isNull);
    expect(resolveDinoV3ValidationModelPath(null, 'model.neri.json'), isNull);
  });
}
