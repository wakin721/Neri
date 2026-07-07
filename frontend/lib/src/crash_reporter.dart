import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';

class CrashReport {
  const CrashReport({
    required this.title,
    required this.message,
    required this.details,
    required this.logPath,
    required this.createdAt,
  });

  final String title;
  final String message;
  final String details;
  final String logPath;
  final DateTime createdAt;
}

class CrashReporter {
  CrashReporter._();

  static const int simulatedFrontendCrashExitCode = 86;
  static const String crashLogFileName = 'crash.log';
  static const String _legacyCrashLogFileName = 'frontend_crash.log';

  static final GlobalKey<NavigatorState> navigatorKey =
      GlobalKey<NavigatorState>();
  static final ValueNotifier<CrashReport?> latestReport =
      ValueNotifier<CrashReport?>(null);

  static Directory? _logsDirectory;
  static bool _initialized = false;

  static void initialize() {
    if (_initialized) return;
    _logsDirectory = _resolveLogsDirectory();
    _initialized = true;
    _loadPendingStartupReport();
  }

  static void prepareForNewCrashSession() {
    initialize();
    _deleteLogFile(_legacyCrashLogFileName);
    _deleteLogFile('backend_runtime.log');
    _deleteLogFile('crash_watchdog.log');
    _deleteBackendProcessLogs();
  }

  static int crashLogSize() {
    initialize();
    try {
      final file = _logFile(crashLogFileName);
      return file.existsSync() ? file.lengthSync() : 0;
    } catch (_) {
      return 0;
    }
  }

  static void recordFlutterError(FlutterErrorDetails details) {
    record(
      details.exception,
      details.stack,
      origin: 'Flutter framework',
      details: details.toString(),
    );
  }

  static void record(
    Object error,
    StackTrace? stackTrace, {
    String origin = 'Dart runtime',
    String? details,
  }) {
    initialize();
    final now = DateTime.now();
    final message = _compactMessage(error.toString());
    final fullDetails = _buildDetails(
      origin: origin,
      message: message,
      details: details,
      stackTrace: stackTrace,
    );
    final logPath = _appendCrashLog('Frontend crash', fullDetails);
    latestReport.value = CrashReport(
      title: '程序发生异常',
      message: message,
      details: _trimDialogDetails(fullDetails),
      logPath: logPath,
      createdAt: now,
    );
  }

  static Never simulateFrontendCrash() {
    initialize();
    final fullDetails = _buildDetails(
      origin: '调试模式模拟崩溃',
      message: '调试模式模拟前端崩溃。',
      details: '这是由调试模式手动触发的前端进程退出，用于验证真实前端崩溃后的 watchdog 提示和日志记录。',
      stackTrace: StackTrace.current,
    );
    _appendCrashLog('Frontend crash', fullDetails);
    exit(simulatedFrontendCrashExitCode);
  }

  static File createBackendLogFile() {
    initialize();
    _deleteBackendProcessLogs();
    final file = _logFile('backend_${_timestampForFilename()}.log');
    file.writeAsStringSync(
      '[${DateTime.now().toIso8601String()}] Backend process log started\n',
      encoding: utf8,
      mode: FileMode.write,
      flush: true,
    );
    return file;
  }

  static void deleteBackendLogFile(File file) {
    _deleteFile(file);
  }

  static void recordBackendCrash({
    required int exitCode,
    required String logPath,
    required String outputTail,
  }) {
    initialize();
    final reason = _backendReason(outputTail, exitCode);
    final fullDetails = StringBuffer()
      ..writeln('Origin: Python backend process')
      ..writeln('Exit code: $exitCode')
      ..writeln('Backend log: $logPath')
      ..writeln()
      ..writeln('Recent backend output:')
      ..writeln(
        outputTail.trim().isEmpty ? '(no output captured)' : outputTail.trim(),
      );
    final crashLogPath = _appendCrashLog(
      'Backend crash',
      fullDetails.toString(),
    );
    latestReport.value = CrashReport(
      title: 'Python 后端崩溃',
      message: reason,
      details: _trimDialogDetails(fullDetails.toString()),
      logPath: crashLogPath,
      createdAt: DateTime.now(),
    );
  }

  static String _appendCrashLog(String title, String details) {
    final file = _logFile(crashLogFileName);
    final entry = StringBuffer()
      ..writeln('============================================================')
      ..writeln('[$title] ${DateTime.now().toIso8601String()}')
      ..writeln(details.trimRight())
      ..writeln();
    try {
      file.writeAsStringSync(
        entry.toString(),
        encoding: utf8,
        mode: FileMode.append,
        flush: true,
      );
    } catch (_) {}
    return file.path;
  }

  static void _loadPendingStartupReport() {
    final projectRoot = _resolveProjectRoot();
    if (projectRoot == null) return;
    final candidates = <File>[
      File(
        [
          projectRoot.path,
          'logs',
          'crash_startup_report.json',
        ].join(Platform.pathSeparator),
      ),
      File(
        [
          projectRoot.path,
          'temp',
          'crash_startup_report.json',
        ].join(Platform.pathSeparator),
      ),
    ];
    final file = candidates.firstWhere(
      (candidate) => candidate.existsSync(),
      orElse: () => candidates.first,
    );
    if (!file.existsSync()) return;

    try {
      final decoded = jsonDecode(file.readAsStringSync(encoding: utf8));
      if (decoded is! Map<String, dynamic>) return;
      final createdAtText = decoded['createdAt'];
      latestReport.value = CrashReport(
        title: _jsonString(decoded['title'], '程序崩溃提示'),
        message: _jsonString(decoded['message'], 'Neri 上次运行异常退出。'),
        details: _trimDialogDetails(_jsonString(decoded['details'], '')),
        logPath: _jsonString(
          decoded['logPath'],
          _logFile(crashLogFileName).path,
        ),
        createdAt: createdAtText is String
            ? DateTime.tryParse(createdAtText) ?? DateTime.now()
            : DateTime.now(),
      );
    } catch (error, stackTrace) {
      _appendCrashLog(
        'Startup crash report',
        _buildDetails(
          origin: 'Startup crash report loader',
          message: '读取崩溃重启提示失败：$error',
          details: null,
          stackTrace: stackTrace,
        ),
      );
    } finally {
      try {
        file.deleteSync();
      } catch (_) {}
    }
  }

  static String _jsonString(Object? value, String fallback) {
    return value is String && value.trim().isNotEmpty ? value : fallback;
  }

  static String _buildDetails({
    required String origin,
    required String message,
    required String? details,
    required StackTrace? stackTrace,
  }) {
    final buffer = StringBuffer()
      ..writeln('Origin: $origin')
      ..writeln('Message: $message');
    final extraDetails = details?.trim();
    if (extraDetails != null && extraDetails.isNotEmpty) {
      buffer
        ..writeln()
        ..writeln(extraDetails);
    }
    if (stackTrace != null) {
      buffer
        ..writeln()
        ..writeln('Stack trace:')
        ..writeln(stackTrace);
    }
    return buffer.toString();
  }

  static Directory _resolveLogsDirectory() {
    final projectRoot = _resolveProjectRoot();
    final baseDirectory = projectRoot ?? Directory.current.absolute;
    final logsDirectory = Directory(
      [baseDirectory.path, 'logs'].join(Platform.pathSeparator),
    );
    try {
      logsDirectory.createSync(recursive: true);
      return logsDirectory;
    } catch (_) {
      return Directory.current.absolute;
    }
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

  static bool _looksLikeProjectRoot(Directory directory) {
    final separator = Platform.pathSeparator;
    return File(
          '${directory.path}${separator}system${separator}backend${separator}main.py',
        ).existsSync() ||
        File(
          '${directory.path}${separator}res${separator}logo.png',
        ).existsSync();
  }

  static File _logFile(String name) {
    initialize();
    final directory = _logsDirectory ?? Directory.current.absolute;
    return File([directory.path, name].join(Platform.pathSeparator));
  }

  static void _deleteLogFile(String name) {
    _deleteFile(_logFile(name));
  }

  static void _deleteBackendProcessLogs() {
    final directory = _logsDirectory;
    if (directory == null || !directory.existsSync()) return;
    try {
      for (final entity in directory.listSync()) {
        if (entity is! File) continue;
        final name = entity.path.split(Platform.pathSeparator).last;
        if (name.startsWith('backend_') && name.endsWith('.log')) {
          _deleteFile(entity);
        }
      }
    } catch (_) {}
  }

  static void _deleteFile(File file) {
    try {
      if (file.existsSync()) {
        file.deleteSync();
      }
    } catch (_) {}
  }

  static String _timestampForFilename() {
    return DateTime.now()
        .toIso8601String()
        .replaceAll(':', '-')
        .replaceAll('.', '-');
  }

  static String _compactMessage(String message) {
    final trimmed = message.trim();
    if (trimmed.isEmpty) return '未知异常';
    return trimmed.length <= 600 ? trimmed : '${trimmed.substring(0, 600)}...';
  }

  static String _backendReason(String outputTail, int exitCode) {
    final lines = outputTail
        .split('\n')
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty)
        .toList();
    if (lines.isEmpty) {
      return 'Python 后端异常退出，退出码 $exitCode。';
    }
    final errorLine = lines.lastWhere(
      _looksLikeErrorLine,
      orElse: () => lines.last,
    );
    return _compactMessage(errorLine);
  }

  static bool _looksLikeErrorLine(String line) {
    final lower = line.toLowerCase();
    return lower.contains('error') ||
        lower.contains('exception') ||
        lower.contains('traceback') ||
        lower.contains('failed') ||
        line.contains('错误') ||
        line.contains('异常') ||
        line.contains('失败');
  }

  static String _trimDialogDetails(String details) {
    const maxLength = 5000;
    final trimmed = details.trim();
    if (trimmed.length <= maxLength) return trimmed;
    return '... 仅显示最后 $maxLength 个字符\n${trimmed.substring(trimmed.length - maxLength)}';
  }
}

class CrashDialogListener extends StatefulWidget {
  const CrashDialogListener({required this.child, super.key});

  final Widget child;

  @override
  State<CrashDialogListener> createState() => _CrashDialogListenerState();
}

class _CrashDialogListenerState extends State<CrashDialogListener> {
  CrashReport? _shownReport;

  @override
  void initState() {
    super.initState();
    CrashReporter.latestReport.addListener(_handleCrashReportChanged);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _showReportIfNeeded(CrashReporter.latestReport.value);
    });
  }

  @override
  void dispose() {
    CrashReporter.latestReport.removeListener(_handleCrashReportChanged);
    super.dispose();
  }

  void _handleCrashReportChanged() {
    _showReportIfNeeded(CrashReporter.latestReport.value);
  }

  void _showReportIfNeeded(CrashReport? report) {
    if (!mounted || report == null || identical(report, _shownReport)) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final navigatorContext = CrashReporter.navigatorKey.currentContext;
      if (navigatorContext == null) {
        Future<void>.delayed(
          const Duration(milliseconds: 100),
          () => _showReportIfNeeded(report),
        );
        return;
      }
      if (identical(report, _shownReport)) return;
      _shownReport = report;
      showDialog<void>(
        context: navigatorContext,
        builder: (context) => _CrashReportDialog(report: report),
      );
    });
  }

  @override
  Widget build(BuildContext context) => widget.child;
}

class _CrashReportDialog extends StatelessWidget {
  const _CrashReportDialog({required this.report});

  final CrashReport report;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final colorScheme = Theme.of(context).colorScheme;
    return AlertDialog(
      title: Text(report.title),
      content: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 620),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(report.message),
              const SizedBox(height: 16),
              Text(
                '崩溃日志',
                style: textTheme.titleSmall?.copyWith(
                  color: colorScheme.onSurface,
                ),
              ),
              const SizedBox(height: 6),
              SelectableText(
                report.logPath,
                style: textTheme.bodySmall?.copyWith(
                  color: colorScheme.onSurfaceVariant,
                ),
              ),
              if (report.details.isNotEmpty) ...[
                const SizedBox(height: 16),
                Text(
                  '详细原因',
                  style: textTheme.titleSmall?.copyWith(
                    color: colorScheme.onSurface,
                  ),
                ),
                const SizedBox(height: 6),
                DecoratedBox(
                  decoration: BoxDecoration(
                    color: colorScheme.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: SelectableText(
                      report.details,
                      style: textTheme.bodySmall?.copyWith(
                        fontFamily: 'Consolas',
                        color: colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('关闭'),
        ),
      ],
    );
  }
}
