import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/utils/job_result_refresh.dart';
import 'package:neri_flutter/src/models/job.dart';

void main() {
  group('job result refresh policy', () {
    test('streams complete results while a results page is visible', () {
      expect(
        shouldFetchCompleteJobResults(
          includeJobResults: true,
          silent: true,
          resultsPageVisible: true,
        ),
        isTrue,
      );
    });

    test('uses summaries for a background job outside results pages', () {
      expect(
        shouldFetchCompleteJobResults(
          includeJobResults: true,
          silent: true,
          resultsPageVisible: false,
        ),
        isFalse,
      );
    });

    test('keeps idle historical jobs as summaries outside results pages', () {
      expect(
        shouldFetchCompleteJobResults(
          includeJobResults: true,
          silent: true,
          resultsPageVisible: false,
        ),
        isFalse,
      );
    });

    test('honors an explicit summaries-only request', () {
      expect(
        shouldFetchCompleteJobResults(
          includeJobResults: false,
          silent: false,
          resultsPageVisible: true,
        ),
        isFalse,
      );
    });

    ProcessingJob job(String state, int processed, String updatedAt) =>
        ProcessingJob(
          id: 'job',
          state: state,
          inputDir: 'input',
          createdAt: 'created',
          updatedAt: updatedAt,
          processed: processed,
        );

    test('opening a historical job requests only its missing detail', () {
      final summary = job('completed', 3008, 'old');
      expect(
        shouldFetchJobResults(
          summary: summary,
          previous: summary,
          complete: null,
          resultsRequested: false,
        ),
        isFalse,
      );
      expect(
        shouldFetchJobResults(
          summary: summary,
          previous: summary,
          complete: null,
          resultsRequested: true,
        ),
        isTrue,
      );
      expect(
        shouldFetchJobResults(
          summary: summary,
          previous: summary,
          complete: summary,
          resultsRequested: true,
        ),
        isFalse,
      );
    });

    test('hidden job gets final results on completion or stop', () {
      final running = job('running', 1, 'before');
      for (final state in ['completed', 'cancelled', 'failed']) {
        expect(
          shouldFetchJobResults(
            summary: job(state, 2, 'after'),
            previous: running,
            complete: null,
            resultsRequested: false,
          ),
          isTrue,
        );
      }
      expect(
        shouldFetchJobResults(
          summary: job('running', 2, 'after'),
          previous: running,
          complete: null,
          resultsRequested: false,
        ),
        isFalse,
      );
    });

    test('expanded resumed jobs stream results and skip unchanged data', () {
      final stopped = job('cancelled', 2, 'before');
      final resumed = job('running', 3, 'after');
      expect(
        shouldFetchJobResults(
          summary: resumed,
          previous: stopped,
          complete: stopped,
          resultsRequested: true,
        ),
        isTrue,
      );
      expect(
        shouldFetchJobResults(
          summary: resumed,
          previous: resumed,
          complete: resumed,
          resultsRequested: true,
        ),
        isFalse,
      );
    });

    test('keeps visible preview content during a same-path refresh', () {
      expect(
        shouldClearPreviewItemsBeforeRefresh(
          loadedPath: r'D:\photos',
          inputPath: r'D:\photos',
        ),
        isFalse,
      );
    });

    test('clears preview content when the input path changes', () {
      expect(
        shouldClearPreviewItemsBeforeRefresh(
          loadedPath: r'D:\old-photos',
          inputPath: r'D:\new-photos',
        ),
        isTrue,
      );
    });
  });
}
