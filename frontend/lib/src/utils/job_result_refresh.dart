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
