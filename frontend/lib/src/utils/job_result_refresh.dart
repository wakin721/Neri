import '../models/job.dart';

bool shouldFetchCompleteJobResults({
  required bool includeJobResults,
  required bool silent,
  required bool resultsPageVisible,
}) {
  if (!includeJobResults) return false;
  if (!silent) return true;
  return resultsPageVisible;
}

bool shouldClearPreviewItemsBeforeRefresh({
  required String? loadedPath,
  required String inputPath,
}) {
  return loadedPath != inputPath;
}

bool jobResultsNeedRefresh(ProcessingJob summary, ProcessingJob? complete) {
  return complete == null ||
      summary.updatedAt != complete.updatedAt ||
      summary.processed != complete.processed ||
      summary.state != complete.state ||
      summary.active != complete.active;
}

// Hidden historical jobs stay summaries. A job that just finished still gets
// its final results so processing completion updates previews promptly.
bool shouldFetchJobResults({
  required ProcessingJob summary,
  required ProcessingJob? previous,
  required ProcessingJob? complete,
  required bool resultsRequested,
}) {
  if (!jobResultsNeedRefresh(summary, complete)) return false;
  return resultsRequested ||
      (!summary.isWorkerActive &&
          previous != null &&
          jobResultsNeedRefresh(summary, previous));
}
