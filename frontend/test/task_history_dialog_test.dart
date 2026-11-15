import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';

StartScreen screen(
  TextEditingController input, {
  List<ProcessingJob> jobs = const [],
  ValueChanged<ProcessingJob>? onCancel,
  ValueChanged<ProcessingJob>? onResume,
  ValueChanged<ProcessingJob>? onDelete,
  VoidCallback? onClear,
  void Function(ProcessingJob, bool)? onExpand,
}) => StartScreen(
  settings: null,
  inputController: input,
  selectedModelPath: '',
  onModelChanged: (_) {},
  selectedClassificationModelPath: '',
  onClassificationModelChanged: (_) {},
  videoMode: 'skip',
  onVideoModeChanged: (_) {},
  vidStride: 1,
  onVidStrideChanged: (_) {},
  useFp16: false,
  onUseFp16Changed: (_) {},
  confidence: .25,
  onConfidenceChanged: (_) {},
  iou: .3,
  onIouChanged: (_) {},
  submitting: false,
  onCreateJob: () {},
  onCancelJob: onCancel ?? (_) {},
  onResumeJob: onResume ?? (_) {},
  onDeleteJob: onDelete ?? (_) {},
  onClearJobs: onClear ?? () {},
  onJobExpansionChanged: onExpand,
  pendingStartJobIds: const {},
  pendingStopJobIds: const {},
  jobs: jobs,
);

ProcessingJob job({
  String state = 'running',
  int processed = 10,
  int total = 100,
}) => ProcessingJob(
  id: 'job',
  state: state,
  inputDir: r'C:\photos',
  createdAt: 'created',
  updatedAt: '$processed',
  total: total,
  processed: processed,
  message: '正在处理 $processed/100',
  results: const [
    DetectionItem(
      filename: 'detail-photo.jpg',
      path: r'C:\photos\detail-photo.jpg',
      fileType: 'jpg',
    ),
  ],
);

Future<void> openHistory(WidgetTester tester) async {
  final button = find.byTooltip('任务历史');
  await tester.ensureVisible(button);
  await tester.pump();
  await tester.tap(button);
  await tester.pump(const Duration(milliseconds: 300));
}

void main() {
  late TextEditingController input;
  setUp(() {
    input = TextEditingController();
  });
  tearDown(() {
    input.dispose();
  });

  testWidgets(
    'round history button sits beside start and opens an empty dialog',
    (tester) async {
      await tester.pumpWidget(MaterialApp(home: Scaffold(body: screen(input))));
      expect(find.text('任务进度'), findsNothing);
      final history = find.byTooltip('任务历史');
      await tester.ensureVisible(history);
      await tester.pump();
      final button = tester.widget<IconButton>(
        find.widgetWithIcon(IconButton, Icons.history_rounded),
      );
      expect(button.style!.shape!.resolve({}), isA<CircleBorder>());
      expect(
        tester.getCenter(history).dx,
        greaterThan(tester.getCenter(find.text('开始处理')).dx),
      );
      await openHistory(tester);
      expect(find.byType(Dialog), findsOneWidget);
      expect(find.text('任务进度'), findsOneWidget);
      expect(find.text('创建任务后，处理进度和识别结果会显示在这里。'), findsOneWidget);
      await tester.tap(find.byTooltip('关闭任务历史'));
      await tester.pumpAndSettle();
      expect(find.byType(Dialog), findsNothing);
      await openHistory(tester);
      expect(find.byType(Dialog), findsOneWidget);
    },
  );

  testWidgets(
    'open dialog receives new progress from its parent and preserves expansion',
    (tester) async {
      final latest = ValueNotifier(job());
      addTearDown(latest.dispose);
      final expansion = <bool>[];
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: ValueListenableBuilder<ProcessingJob>(
              valueListenable: latest,
              builder: (_, value, _) => screen(
                input,
                jobs: [value],
                onExpand: (_, expanded) => expansion.add(expanded),
              ),
            ),
          ),
        ),
      );
      await openHistory(tester);
      expect(
        tester
            .widget<LinearProgressIndicator>(
              find.byType(LinearProgressIndicator),
            )
            .value,
        .1,
      );
      await tester.tap(find.text(r'C:\photos'));
      await tester.pumpAndSettle();
      expect(expansion, [true]);
      latest.value = job(processed: 60);
      await tester.pump();
      await tester.pump();
      expect(find.text('正在处理 60/100'), findsOneWidget);
      expect(
        tester
            .widget<LinearProgressIndicator>(
              find.byType(LinearProgressIndicator),
            )
            .value,
        .6,
      );
      expect(find.text('detail-photo.jpg'), findsOneWidget);
      await tester.tap(find.byTooltip('关闭任务历史'));
      await tester.pumpAndSettle();
      expect(expansion, [true, false]);
    },
  );

  testWidgets(
    'stop, resume, delete and clear callbacks remain available in history',
    (tester) async {
      final latest = ValueNotifier(job());
      addTearDown(latest.dispose);
      final actions = <String>[];
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: ValueListenableBuilder<ProcessingJob>(
              valueListenable: latest,
              builder: (_, value, _) => screen(
                input,
                jobs: [value],
                onCancel: (_) => actions.add('cancel'),
                onResume: (_) => actions.add('resume'),
                onDelete: (_) => actions.add('delete'),
                onClear: () => actions.add('clear'),
              ),
            ),
          ),
        ),
      );
      await openHistory(tester);
      await tester.tap(find.byTooltip('停止任务'));
      expect(actions, ['cancel']);
      latest.value = job(state: 'cancelled');
      await tester.pump();
      await tester.pump();
      await tester.tap(find.byTooltip('继续任务'));
      expect(actions, ['cancel', 'resume']);
      await tester.tap(find.byTooltip('删除任务'));
      await tester.pumpAndSettle();
      await tester.tap(find.widgetWithText(FilledButton, '删除'));
      await tester.pumpAndSettle();
      expect(actions.last, 'delete');
      await tester.tap(find.text('清除所有任务'));
      await tester.pumpAndSettle();
      await tester.tap(find.widgetWithText(FilledButton, '清除'));
      await tester.pumpAndSettle();
      expect(actions.last, 'clear');
    },
  );

  testWidgets('dialog fits a narrow window with large text', (tester) async {
    tester.view.physicalSize = const Size(600, 700);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.pumpWidget(
      MaterialApp(
        home: MediaQuery(
          data: const MediaQueryData(
            size: Size(600, 700),
            textScaler: TextScaler.linear(1.6),
          ),
          child: Scaffold(
            body: screen(input, jobs: [job(processed: 400, total: 17487)]),
          ),
        ),
      ),
    );
    await openHistory(tester);
    expect(find.byType(Dialog), findsOneWidget);
    expect(tester.takeException(), isNull);
    await tester.tap(find.byTooltip('关闭任务历史'));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
  });

  testWidgets('removing the source screen closes its history route safely', (
    tester,
  ) async {
    final visible = ValueNotifier(true);
    addTearDown(visible.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ValueListenableBuilder<bool>(
            valueListenable: visible,
            builder: (_, show, _) =>
                show ? screen(input) : const SizedBox.shrink(),
          ),
        ),
      ),
    );
    await openHistory(tester);
    visible.value = false;
    await tester.pump();
    await tester.pumpAndSettle();
    expect(find.byType(Dialog), findsNothing);
    expect(tester.takeException(), isNull);
  });
}
