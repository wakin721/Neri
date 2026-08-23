import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/utils/job_result_refresh.dart';

void main() {
  group('job result refresh policy', () {
    test('streams complete results while a results page is visible', () {
      expect(
        shouldFetchCompleteJobResults(
          includeJobResults: true,
          silent: true,
          jobProcessingBusy: true,
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
          jobProcessingBusy: true,
          resultsPageVisible: false,
        ),
        isFalse,
      );
    });

    test('fetches complete results after processing is no longer busy', () {
      expect(
        shouldFetchCompleteJobResults(
          includeJobResults: true,
          silent: true,
          jobProcessingBusy: false,
          resultsPageVisible: false,
        ),
        isTrue,
      );
    });

    test('honors an explicit summaries-only request', () {
      expect(
        shouldFetchCompleteJobResults(
          includeJobResults: false,
          silent: false,
          jobProcessingBusy: false,
          resultsPageVisible: true,
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
