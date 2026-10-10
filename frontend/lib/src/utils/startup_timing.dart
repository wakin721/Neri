import 'dart:io';

import '../crash_reporter.dart';

/// Local timings contain stage names and durations, never media or user paths.
class StartupTiming {
  StartupTiming() : _startedAt = DateTime.now().toIso8601String() {
    _clock.start();
  }

  final String _startedAt;
  final _clock = Stopwatch();
  int _lastElapsed = 0;
  Future<void> _pending = Future<void>.value();

  void mark(String stage) {
    final elapsed = _clock.elapsedMilliseconds;
    final line =
        '$_startedAt $stage stage_ms=${elapsed - _lastElapsed} '
        'total_ms=$elapsed\n';
    _lastElapsed = elapsed;
    // Keep diagnostics off the critical startup path and bound disk usage.
    _pending = _pending.then((_) async {
      try {
        final file = File(
          '${CrashReporter.logsDirectory.path}'
          '${Platform.pathSeparator}startup_timing.log',
        );
        if (await file.exists() && await file.length() > 128 * 1024) {
          await file.writeAsString('');
        }
        await file.writeAsString(line, mode: FileMode.append);
      } catch (_) {
        // Diagnostics must never prevent startup or shutdown.
      }
    });
  }
}
