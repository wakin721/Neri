import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';

void main() {
  Future<void> showJob(
    WidgetTester tester,
    ProcessingJob job, {
    Set<String> pendingStartJobIds = const {},
    Set<String> pendingStopJobIds = const {},
  }) async {
    final input = TextEditingController();
    addTearDown(input.dispose);
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
            videoMode: 'skip',
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
            onCreateJob: () {},
            onCancelJob: (_) {},
            onResumeJob: (_) {},
            onDeleteJob: (_) {},
            onClearJobs: () {},
            pendingStartJobIds: pendingStartJobIds,
            pendingStopJobIds: pendingStopJobIds,
            jobs: [job],
          ),
        ),
      ),
    );
    final history = find.byTooltip('任务历史');
    await tester.ensureVisible(history);
    await tester.pump();
    await tester.tap(history);
    await tester.pump(const Duration(milliseconds: 300));
  }

  ProcessingJob job(String stage, int done, int total) =>
      ProcessingJob.fromJson({
        'id': 'job',
        'state': 'running',
        'input_dir': r'C:\photos',
        'total': 100,
        'processed': 100,
        'sequence_stage': stage,
        'sequence_processed': done,
        'sequence_total': total,
        'speed': 15,
        'remaining_seconds': 0,
        'created_at': 'created',
        'updated_at': 'updated',
      });

  testWidgets(
    'resumed photo job shows actual progress despite a stale start flag',
    (tester) async {
      final running = ProcessingJob.fromJson({
        'id': 'job',
        'state': 'running',
        'input_dir': r'C:\photos',
        'total': 17487,
        'processed': 400,
        'message': '正在批处理照片 400/17487',
        'created_at': 'created',
        'updated_at': 'updated',
      });
      await showJob(tester, running, pendingStartJobIds: {'job'});
      final bar = tester.widget<LinearProgressIndicator>(
        find.byType(LinearProgressIndicator).last,
      );
      expect(bar.value, closeTo(400 / 17487, 1e-10));
    },
  );

  testWidgets('queued resume still shows an activity indicator until running', (
    tester,
  ) async {
    final queued = ProcessingJob.fromJson({
      'id': 'job',
      'state': 'queued',
      'input_dir': r'C:\photos',
      'total': 17487,
      'processed': 17487,
      'created_at': 'created',
      'updated_at': 'updated',
    });
    await showJob(tester, queued, pendingStartJobIds: {'job'});
    final bar = tester.widget<LinearProgressIndicator>(
      find.byType(LinearProgressIndicator).last,
    );
    expect(bar.value, isNull);
  });

  testWidgets('summary displays stage progress after all photos finish', (
    tester,
  ) async {
    await showJob(tester, job('构建因果历史', 50, 200), pendingStartJobIds: {'job'});
    final bar = tester.widget<LinearProgressIndicator>(
      find.byType(LinearProgressIndicator).last,
    );
    expect(bar.value, 0.25);
    expect(find.textContaining('构建因果历史'), findsOneWidget);
    expect(find.textContaining('50/200 (25%)'), findsOneWidget);
    expect(find.textContaining('剩余'), findsNothing);
    expect(find.textContaining('个/秒'), findsNothing);
    expect(find.text('100/100'), findsOneWidget);
  });

  testWidgets(
    'classification uses indeterminate progress without a fake count',
    (tester) async {
      await showJob(tester, job('全序列分类', 0, 0));
      final bar = tester.widget<LinearProgressIndicator>(
        find.byType(LinearProgressIndicator).last,
      );
      expect(bar.value, isNull);
      expect(find.textContaining('全序列分类'), findsOneWidget);
    },
  );
}
