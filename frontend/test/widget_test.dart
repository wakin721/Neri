import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/main.dart';
import 'package:neri_flutter/src/frontend_version.dart';
import 'package:neri_flutter/src/models/theme_settings.dart';

void main() {
  testWidgets('Neri app builds Material 3 shell', (tester) async {
    await tester.pumpWidget(
      NeriApp(
        themeNotifier: ValueNotifier(
          const ThemeSettings(
            themeMode: ThemeMode.light,
            useDynamicColor: false,
            seedColor: Colors.teal,
          ),
        ),
      ),
    );
    expect(
      find.descendant(
        of: find.byType(AppBar),
        matching: find.text('Neri ${frontendVersion.split('-').first}'),
      ),
      findsOneWidget,
    );
    await tester.pumpWidget(const SizedBox.shrink());
    await tester.pump(const Duration(milliseconds: 300));
  });
}
