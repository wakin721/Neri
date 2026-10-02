/// Coalesces timer ticks while preserving explicit refreshes after mutations.
class AsyncRefreshGate {
  Future<void>? _inFlight;

  Future<void> run(
    Future<void> Function() refresh, {
    bool coalesce = false,
  }) async {
    while (_inFlight != null) {
      final pending = _inFlight!;
      try {
        await pending;
      } catch (_) {
        // The original caller owns its error; queued explicit work must still
        // get a chance to recover after that request fails.
      } finally {
        if (identical(_inFlight, pending)) _inFlight = null;
      }
      if (coalesce) return;
    }
    final task = Future<void>.sync(refresh);
    _inFlight = task;
    try {
      await task;
    } finally {
      if (identical(_inFlight, task)) _inFlight = null;
    }
  }
}
