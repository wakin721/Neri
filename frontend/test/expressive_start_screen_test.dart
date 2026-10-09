import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/app_theme.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';
import 'package:neri_flutter/src/widgets/expressive_action_button.dart';

void main() {
  for (final brightness in Brightness.values) {
    for (final width in [600.0, 1280.0]) {
      testWidgets(
        'Expressive start screen: $brightness at $width with large text',
        (tester) async {
          tester.view.physicalSize = Size(width, 1000);
          tester.view.devicePixelRatio = 1;
          addTearDown(tester.view.resetPhysicalSize);
          addTearDown(tester.view.resetDevicePixelRatio);
          final input = TextEditingController();
          addTearDown(input.dispose);
          var submitted = false;
          await tester.pumpWidget(
            MaterialApp(
              theme: buildNeriTheme(
                ColorScheme.fromSeed(
                  seedColor: const Color(0xFFFA9D85),
                  brightness: brightness,
                ),
              ),
              builder: (context, child) => MediaQuery(
                data: MediaQuery.of(
                  context,
                ).copyWith(textScaler: const TextScaler.linear(1.5)),
                child: child!,
              ),
              home: Scaffold(
                body: StartScreen(
                  settings: null,
                  inputController: input,
                  selectedModelPath: '',
                  onModelChanged: (_) {},
                  selectedClassificationModelPath: '',
                  onClassificationModelChanged: (_) {},
                  videoMode: 'all',
                  onVideoModeChanged: (_) {},
                  vidStride: 1,
                  onVidStrideChanged: (_) {},
                  useFp16: false,
                  onUseFp16Changed: (_) {},
                  confidence: 0.25,
                  onConfidenceChanged: (_) {},
                  iou: 0.3,
                  onIouChanged: (_) {},
                  submitting: false,
                  onCreateJob: () => submitted = true,
                  onCancelJob: (_) {},
                  onResumeJob: (_) {},
                  onDeleteJob: (_) {},
                  onClearJobs: () {},
                  pendingStartJobIds: const {},
                  pendingStopJobIds: const {},
                  jobs: const [],
                ),
              ),
            ),
          );
          await tester.pumpAndSettle();
          expect(tester.takeException(), isNull);
          await tester.ensureVisible(find.byType(ExpressiveActionButton));
          await tester.pumpAndSettle();
          await tester.tap(find.text('开始处理'));
          await tester.pumpAndSettle();
          expect(submitted, isTrue);
          expect(tester.takeException(), isNull);
        },
      );
    }
  }
}
