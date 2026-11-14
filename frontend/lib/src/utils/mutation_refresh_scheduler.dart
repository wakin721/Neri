import 'dart:async';

/// Debounces directory changes outside annotation writes. A save that starts
/// during the debounce postpones the scan; dirty events are retained once.
class MutationRefreshScheduler {
  MutationRefreshScheduler(this.refresh);
  final Future<void> Function() refresh;
  Timer? _timer;
  int _mutations = 0;
  bool _dirty = false;
  bool _running = false;
  bool _disposed = false;

  void request() {
    if (_disposed) return;
    _dirty = true;
    _schedule();
  }

  void beginMutation() {
    _mutations++;
    _timer?.cancel();
    _timer = null;
  }

  void endMutation() {
    if (_mutations > 0) _mutations--;
    _schedule();
  }

  void _schedule() {
    _timer?.cancel();
    _timer = null;
    if (_disposed || !_dirty || _mutations > 0 || _running) return;
    _timer = Timer(const Duration(milliseconds: 350), () async {
      _timer = null;
      if (_disposed || _mutations > 0) return;
      _dirty = false;
      _running = true;
      try {
        await refresh();
      } finally {
        _running = false;
        _schedule();
      }
    });
  }

  void cancel() {
    _dirty = false;
    _timer?.cancel();
    _timer = null;
  }

  void dispose() {
    _disposed = true;
    cancel();
  }
}
