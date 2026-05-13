import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/main.dart';

void main() {
  testWidgets('Neri app builds Material 3 shell', (tester) async {
    await tester.pumpWidget(const NeriApp());
    expect(find.text('Neri'), findsOneWidget);
  });
}
