import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/app_theme.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';
import 'package:neri_flutter/src/widgets/app_menu_style.dart';
import 'package:neri_flutter/src/widgets/expressive_action_button.dart';

void main() {
  testWidgets('long form menu does not cover its trigger in a short window', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(600, 400);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: Center(
            child: SizedBox(
              width: 240,
              child: AppFormMenu<int>(
                value: 0,
                label: '探测模型',
                helperText: '选择模型',
                leadingIcon: const Icon(Icons.memory_rounded),
                options: [
                  for (var index = 0; index < 20; index++)
                    DropdownMenuEntry(value: index, label: '模型 $index'),
                ],
                onSelected: (_) {},
              ),
            ),
          ),
        ),
      ),
    );
    final trigger = find.widgetWithText(InkWell, '模型 0');
    await tester.tap(trigger);
    await tester.pumpAndSettle();

    final triggerRect = tester.getRect(find.byType(AppFormMenu<int>));
    final menuRect = tester.getRect(find.byType(SingleChildScrollView).last);
    expect(menuRect.overlaps(triggerRect), isFalse);
  });

  testWidgets('start screen video menu uses selected pill and checkmark', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1280, 1000);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final input = TextEditingController();
    addTearDown(input.dispose);
    String? selectedMode;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: StartScreen(
            settings: null,
            inputController: input,
            selectedModelPath: '',
            onModelChanged: (_) {},
            selectedClassificationModelPath: '',
            onClassificationModelChanged: (_) {},
            videoMode: 'fast',
            onVideoModeChanged: (value) => selectedMode = value,
            vidStride: 1,
            onVidStrideChanged: (_) {},
            useFp16: false,
            onUseFp16Changed: (_) {},
            confidence: 0.25,
            onConfidenceChanged: (_) {},
            iou: 0.3,
            onIouChanged: (_) {},
            submitting: false,
            onCreateJob: () {},
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
    final selector = find.widgetWithText(InkWell, '快速识别');
    await tester.ensureVisible(selector);
    await tester.tap(selector);
    await tester.pumpAndSettle();

    final selectedItem = find.widgetWithText(MenuItemButton, '快速识别');
    expect(selectedItem, findsOneWidget);
    final selectedButton = tester.widget<MenuItemButton>(selectedItem);
    expect(
      selectedButton.style?.backgroundColor?.resolve({}),
      Theme.of(tester.element(selectedItem)).colorScheme.secondaryContainer,
    );
    expect(
      find.descendant(
        of: selectedItem,
        matching: find.byIcon(Icons.check_rounded),
      ),
      findsOneWidget,
    );
    await tester.tap(find.widgetWithText(MenuItemButton, '跳过视频'));
    await tester.pumpAndSettle();
    expect(selectedMode, 'skip');
  });

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
