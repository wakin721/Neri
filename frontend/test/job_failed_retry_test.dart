import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';

void main() {
  test('completed job exposes failed item retry state', () {
    final job = ProcessingJob.fromJson({
      'id': 'job',
      'state': 'completed',
      'input_dir': r'C:\photos',
      'total': 3,
      'processed': 3,
      'failed_count': 1,
      'message': '处理完成，1 个文件失败',
      'created_at': 'created',
      'updated_at': 'updated',
    });

    expect(job.failedCount, 1);
    expect(job.hasFailedItems, isTrue);
    expect(job.canRetryFailed, isTrue);
    expect(job.canResume, isFalse);
    expect(
      ProcessingJob(
        id: 'active',
        state: 'completed',
        inputDir: 'input',
        createdAt: 'created',
        updatedAt: 'updated',
        failedCount: 1,
        active: true,
      ).canRetryFailed,
      isFalse,
    );
    final failed = ProcessingJob.fromJson({
      'id': 'failed',
      'state': 'failed',
      'input_dir': r'C:\photos',
      'failed_count': 1,
      'created_at': 'created',
      'updated_at': 'updated',
    });
    expect(failed.canResume, isTrue);
    expect(failed.canRetryFailed, isFalse);
  });

  test('failed files are not presented as pending sequence summaries', () {
    const item = DetectionItem(
      filename: 'broken.jpg',
      path: 'broken.jpg',
      fileType: 'jpg',
      error: '无法读取图片',
      detectionData: {'sequence_status': 'failed', 'sequence_pending': true},
    );

    expect(item.isSequencePending, isFalse);
  });

  test('retryFailedJob posts to the source job endpoint', () async {
    var requests = 0;
    final client = NeriApiClient(
      httpClient: MockClient((request) async {
        requests++;
        expect(request.method, 'POST');
        expect(request.url.path, '/api/jobs/source-job/retry-failed');
        return http.Response.bytes(
          utf8.encode(
            jsonEncode({
              'id': 'retry-job',
              'state': 'queued',
              'input_dir': r'C:\photos\broken.jpg',
              'total': 1,
              'processed': 0,
              'failed_count': 0,
              'message': '任务已加入队列',
              'created_at': 'created',
              'updated_at': 'updated',
            }),
          ),
          202,
        );
      }),
    );
    addTearDown(client.close);

    final job = await client.retryFailedJob('source-job');

    expect(requests, 1);
    expect(job.id, 'retry-job');
    expect(job.state, 'queued');
  });

  testWidgets('completed job shows failed count, error, and retry action', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(1280, 1000));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final input = TextEditingController();
    addTearDown(input.dispose);
    ProcessingJob? resumedJob;
    final job = ProcessingJob(
      id: 'job',
      state: 'completed',
      inputDir: r'C:\photos',
      total: 14,
      processed: 14,
      failedCount: 1,
      message: '处理完成，1 个文件失败',
      createdAt: 'created',
      updatedAt: 'updated',
      results: [
        for (var index = 1; index <= 13; index++)
          DetectionItem(
            filename: 'success$index.jpg',
            path:
                r'C:\photos\success'
                '$index.jpg',
            fileType: 'jpg',
          ),
        const DetectionItem(
          filename: 'broken.jpg',
          path: r'C:\photos\broken.jpg',
          fileType: 'jpg',
          error: '文件为空，无法读取图像',
        ),
      ],
    );

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
            onResumeJob: (value) => resumedJob = value,
            onDeleteJob: (_) {},
            onClearJobs: () {},
            pendingStartJobIds: const {},
            pendingStopJobIds: const {},
            jobs: [job],
          ),
        ),
      ),
    );

    final history = find.byTooltip('任务历史');
    await tester.ensureVisible(history);
    await tester.pump();
    await tester.tap(history);
    await tester.pumpAndSettle();
    final jobTitle = find.text(r'C:\photos');
    expect(find.text('处理完成，1 个文件失败'), findsOneWidget);
    await tester.tap(jobTitle);
    await tester.pumpAndSettle();

    expect(find.text('1 个文件处理失败'), findsOneWidget);
    expect(find.text('broken.jpg'), findsOneWidget);
    expect(find.text('文件为空，无法读取图像'), findsOneWidget);
    expect(find.text('还有 2 条结果未在此处展开显示。'), findsOneWidget);
    expect(find.textContaining('可在导出的 CSV'), findsNothing);
    final retry = find.text('重试失败文件');
    expect(retry, findsOneWidget);
    await tester.tap(retry);
    expect(resumedJob?.id, 'job');
  });
}
