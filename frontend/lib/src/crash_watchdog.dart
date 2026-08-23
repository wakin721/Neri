import 'dart:convert';
import 'dart:io';

import 'crash_reporter.dart';

class CrashWatchdog {
  CrashWatchdog._();

  static const crashReportModeArg = '--neri-crash-report';

  static File? _sessionFile;
  static Future<bool>? _startTask;
  static bool _started = false;

  static Future<void> start() async {
    await ensureStarted();
  }

  static Future<bool> ensureStarted() {
    if (_started) return Future<bool>.value(true);
    final activeTask = _startTask;
    if (activeTask != null) return activeTask;

    late final Future<bool> task;
    task = _start().whenComplete(() {
      if (!_started && identical(_startTask, task)) {
        _startTask = null;
      }
    });
    _startTask = task;
    return task;
  }

  static Future<bool> _start() async {
    final projectRoot = _resolveProjectRoot();
    if (projectRoot == null) {
      _appendDiagnostic(
        null,
        'Project root not found. cwd=${Directory.current.absolute.path}; '
        'executable=${Platform.resolvedExecutable}',
      );
      return false;
    }

    final pythonWatcherScript = File(
      [
        projectRoot.path,
        'system',
        'crash_watcher.py',
      ].join(Platform.pathSeparator),
    );
    final watcherScript = File(
      [
        projectRoot.path,
        'system',
        'crash_watcher.ps1',
      ].join(Platform.pathSeparator),
    );
    final pythonExe = _resolvePython(projectRoot);
    final powershellExe = _resolvePowerShell();
    if (!Platform.isWindows) {
      _appendDiagnostic(
        projectRoot,
        'Crash watchdog is only enabled on Windows.',
      );
      return false;
    }
    if (!pythonWatcherScript.existsSync() && !watcherScript.existsSync()) {
      _appendDiagnostic(
        projectRoot,
        'Watcher script not found: ${pythonWatcherScript.path} or '
        '${watcherScript.path}',
      );
      return false;
    }

    final logsDirectory = Directory(
      [projectRoot.path, 'logs'].join(Platform.pathSeparator),
    );
    try {
      logsDirectory.createSync(recursive: true);
      CrashReporter.prepareForNewCrashSession();
    } catch (error, stackTrace) {
      _appendDiagnostic(
        projectRoot,
        'Failed to create watchdog directories.',
        error,
        stackTrace,
      );
      return false;
    }

    final frontendLog = File(
      [
        logsDirectory.path,
        CrashReporter.crashLogFileName,
      ].join(Platform.pathSeparator),
    );
    final backendLog = frontendLog;
    final crashLogStartOffset = CrashReporter.crashLogSize();
    final sessionFile = File(
      [
        logsDirectory.path,
        'crash_watchdog_$pid.json',
      ].join(Platform.pathSeparator),
    );
    _sessionFile = sessionFile;
    _writeSession(
      sessionFile,
      'running',
      crashLogStartOffset: crashLogStartOffset,
    );
    final diagnosticLog = File(
      [logsDirectory.path, 'crash_watchdog.log'].join(Platform.pathSeparator),
    );
    final startupReport = File(
      [
        logsDirectory.path,
        'crash_startup_report.json',
      ].join(Platform.pathSeparator),
    );
    final restartExecutable = File(Platform.resolvedExecutable).absolute;

    if (pythonExe != null && pythonWatcherScript.existsSync()) {
      final started = await _startPythonWatcher(
        projectRoot: projectRoot,
        pythonExe: pythonExe,
        watcherScript: pythonWatcherScript,
        sessionFile: sessionFile,
        frontendLog: frontendLog,
        backendLog: backendLog,
        diagnosticLog: diagnosticLog,
        startupReport: startupReport,
        restartExecutable: restartExecutable,
      );
      if (started) return true;
    }

    if (powershellExe == null) {
      _appendDiagnostic(projectRoot, 'PowerShell executable not found.');
      return false;
    }
    if (!watcherScript.existsSync()) {
      _appendDiagnostic(
        projectRoot,
        'PowerShell watcher script not found: ${watcherScript.path}',
      );
      return false;
    }

    _appendDiagnostic(
      projectRoot,
      'Starting watchdog. mainPid=$pid; session=${sessionFile.path}; '
      'script=${watcherScript.path}; powershell=${powershellExe.path}',
    );

    try {
      final process = await Process.start(
        powershellExe.path,
        <String>[
          '-NoProfile',
          '-ExecutionPolicy',
          'Bypass',
          '-WindowStyle',
          'Hidden',
          '-File',
          watcherScript.path,
          '-MainPid',
          pid.toString(),
          '-Session',
          sessionFile.path,
          '-FrontendLog',
          frontendLog.path,
          '-BackendLog',
          backendLog.path,
          '-DiagnosticLog',
          diagnosticLog.path,
          '-Title',
          'Neri',
        ],
        workingDirectory: projectRoot.path,
        mode: ProcessStartMode.detached,
        runInShell: false,
      );
      _started = true;
      _appendDiagnostic(
        projectRoot,
        'Watchdog process started. pid=${process.pid}',
      );
      return true;
    } catch (error, stackTrace) {
      _appendDiagnostic(
        projectRoot,
        'Failed to start watchdog process.',
        error,
        stackTrace,
      );
      return false;
    }
  }

  static Future<bool> _startPythonWatcher({
    required Directory projectRoot,
    required File pythonExe,
    required File watcherScript,
    required File sessionFile,
    required File frontendLog,
    required File backendLog,
    required File diagnosticLog,
    required File startupReport,
    required File restartExecutable,
  }) async {
    _appendDiagnostic(
      projectRoot,
      'Starting Python watchdog. mainPid=$pid; session=${sessionFile.path}; '
      'script=${watcherScript.path}; python=${pythonExe.path}',
    );

    try {
      final process = await Process.start(
        pythonExe.path,
        <String>[
          watcherScript.path,
          '--main-pid',
          pid.toString(),
          '--session',
          sessionFile.path,
          '--frontend-log',
          frontendLog.path,
          '--backend-log',
          backendLog.path,
          '--diagnostic-log',
          diagnosticLog.path,
          '--startup-report',
          startupReport.path,
          '--restart-executable',
          restartExecutable.path,
          '--restart-arg=$crashReportModeArg',
          '--restart-cwd',
          projectRoot.path,
          '--title',
          'Neri',
        ],
        workingDirectory: projectRoot.path,
        mode: ProcessStartMode.detached,
        runInShell: false,
      );
      _started = true;
      _appendDiagnostic(
        projectRoot,
        'Python watchdog process started. pid=${process.pid}',
      );
      return true;
    } catch (error, stackTrace) {
      _appendDiagnostic(
        projectRoot,
        'Failed to start Python watchdog process.',
        error,
        stackTrace,
      );
      return false;
    }
  }

  static void markNormalExit() {
    final sessionFile = _sessionFile;
    if (sessionFile == null) return;
    _writeSession(sessionFile, 'normal');
  }

  static void _writeSession(
    File file,
    String status, {
    int? crashLogStartOffset,
  }) {
    try {
      var payload = <String, Object?>{};
      if (file.existsSync()) {
        try {
          final decoded = jsonDecode(file.readAsStringSync(encoding: utf8));
          if (decoded is Map<String, dynamic>) {
            payload = Map<String, Object?>.from(decoded);
          }
        } catch (_) {}
      }
      payload
        ..['status'] = status
        ..['pid'] = pid
        ..['updated_at'] = DateTime.now().toIso8601String();
      if (crashLogStartOffset != null) {
        payload['crash_log_start_offset'] = crashLogStartOffset;
      }
      file.writeAsStringSync(jsonEncode(payload), encoding: utf8, flush: true);
    } catch (_) {}
  }

  static void _appendDiagnostic(
    Directory? projectRoot,
    String message, [
    Object? error,
    StackTrace? stackTrace,
  ]) {
    try {
      final directory = projectRoot == null
          ? Directory(
              [
                Directory.current.absolute.path,
                'logs',
              ].join(Platform.pathSeparator),
            )
          : Directory([projectRoot.path, 'logs'].join(Platform.pathSeparator));
      directory.createSync(recursive: true);
      final file = File(
        [directory.path, 'crash_watchdog.log'].join(Platform.pathSeparator),
      );
      final buffer = StringBuffer()
        ..writeln('[${DateTime.now().toIso8601String()}] $message');
      if (error != null) {
        buffer.writeln('Error: $error');
      }
      if (stackTrace != null) {
        buffer
          ..writeln('Stack trace:')
          ..writeln(stackTrace);
      }
      file.writeAsStringSync(
        buffer.toString(),
        encoding: utf8,
        mode: FileMode.append,
        flush: true,
      );
    } catch (_) {}
  }

  static Directory? _resolveProjectRoot() {
    final candidates = <Directory>[
      Directory.current.absolute,
      File(Platform.resolvedExecutable).parent.absolute,
    ];
    final visited = <String>{};
    for (final candidate in candidates) {
      var current = candidate;
      for (var depth = 0; depth < 8; depth++) {
        final path = current.path;
        if (visited.add(path) && _looksLikeProjectRoot(current)) {
          return current;
        }
        final parent = current.parent;
        if (parent.path == current.path) break;
        current = parent;
      }
    }
    return null;
  }

  static File? _resolvePython(Directory projectRoot) {
    final candidates = <File>[
      File(
        [
          projectRoot.path,
          'toolkit',
          Platform.isWindows ? 'pythonw.exe' : 'python',
        ].join(Platform.pathSeparator),
      ),
      File(
        [
          projectRoot.path,
          'toolkit',
          Platform.isWindows ? 'python.exe' : 'python',
        ].join(Platform.pathSeparator),
      ),
    ];
    for (final candidate in candidates) {
      if (candidate.existsSync()) return candidate;
    }
    return null;
  }

  static File? _resolvePowerShell() {
    final systemRoot = Platform.environment['SystemRoot'] ?? r'C:\Windows';
    final candidates = <File>[
      File(
        [
          systemRoot,
          'System32',
          'WindowsPowerShell',
          'v1.0',
          'powershell.exe',
        ].join(Platform.pathSeparator),
      ),
      File('powershell.exe'),
    ];
    for (final candidate in candidates) {
      if (candidate.existsSync() || candidate.path == 'powershell.exe') {
        return candidate;
      }
    }
    return null;
  }

  static bool _looksLikeProjectRoot(Directory directory) {
    final separator = Platform.pathSeparator;
    return File(
          [directory.path, 'system', 'backend', 'main.py'].join(separator),
        ).existsSync() &&
        (File(
              [directory.path, 'system', 'crash_watcher.py'].join(separator),
            ).existsSync() ||
            File(
              [directory.path, 'system', 'crash_watcher.ps1'].join(separator),
            ).existsSync());
  }
}
