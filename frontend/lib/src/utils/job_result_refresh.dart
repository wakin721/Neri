import '../models/job.dart';

bool shouldFetchCompleteJobResults({
  required bool includeJobResults,
  required bool silent,
  required bool jobProcessingBusy,
  required bool resultsPageVisible,
}) {
  if (!includeJobResults) return false;
  if (!silent || !jobProcessingBusy) return true;
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
