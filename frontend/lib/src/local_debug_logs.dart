import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as path_util;

import 'api_client.dart';
import 'crash_reporter.dart';

class LocalDebugLogReader {
  LocalDebugLogReader({Directory? logsDirectory})
    : _overrideLogsDirectory = logsDirectory;

  static const _allowedExtensions = <String>{'.log', '.txt', '.json'};
  final Directory? _overrideLogsDirectory;

  Future<List<DebugLogInfo>> listLogs() async {
    final logsDirectory = _logsDirectory;
    if (!await logsDirectory.exists()) return const <DebugLogInfo>[];

    final allowedRoot = await _allowedRoot(logsDirectory);
    final logs = <DebugLogInfo>[];
    await for (final entity in logsDirectory.list(
      recursive: true,
      followLinks: false,
    )) {
      if (entity is! File || !_hasAllowedExtension(entity.path)) continue;
      try {
        final canonicalPath = await entity.resolveSymbolicLinks();
        if (!_isWithin(canonicalPath, allowedRoot)) continue;
        final stat = await entity.stat();
        logs.add(
          DebugLogInfo(
            name: _displayName(canonicalPath, allowedRoot),
            path: canonicalPath,
            sizeBytes: stat.size,
            modifiedAt: stat.modified.toUtc().toIso8601String(),
          ),
        );
      } on FileSystemException {
        // A log can be rotated or deleted while the directory is enumerated.
      }
    }

    logs.sort(
      (left, right) =>
          (right.modifiedAt ?? '').compareTo(left.modifiedAt ?? ''),
    );
    return logs;
  }

  Future<DebugLogContent> readLog(
    DebugLogInfo log, {
    int maxBytes = 32000,
  }) async {
    if (maxBytes <= 0) {
      throw ArgumentError.value(maxBytes, 'maxBytes', '必须大于 0');
    }

    final logsDirectory = _logsDirectory;
    final allowedRoot = await _allowedRoot(logsDirectory);
    final file = File(log.path);
    if (!await file.exists()) {
      throw const FileSystemException('日志文件不存在');
    }
    if (!_hasAllowedExtension(file.path)) {
      throw const FileSystemException('不支持查看该类型的日志文件');
    }

    final canonicalPath = await file.resolveSymbolicLinks();
    if (!_isWithin(canonicalPath, allowedRoot)) {
      throw const FileSystemException('日志路径不在允许的调试目录中');
    }

    final canonicalFile = File(canonicalPath);
    final stat = await canonicalFile.stat();
    final truncated = stat.size > maxBytes;
    final readLength = truncated ? maxBytes : stat.size;
    final handle = await canonicalFile.open();
    late final List<int> bytes;
    try {
      if (truncated) await handle.setPosition(stat.size - maxBytes);
      bytes = await handle.read(readLength);
    } finally {
      await handle.close();
    }

    var content = utf8.decode(bytes, allowMalformed: true);
    if (truncated) {
      content = '... 仅显示最后 $maxBytes 字节\n$content';
    }
    return DebugLogContent(
      name: _displayName(canonicalPath, allowedRoot),
      path: canonicalPath,
      sizeBytes: stat.size,
      modifiedAt: stat.modified.toUtc().toIso8601String(),
      content: content,
      truncated: truncated,
    );
  }

  bool _hasAllowedExtension(String filePath) {
    return _allowedExtensions.contains(
      path_util.extension(filePath).toLowerCase(),
    );
  }

  Directory get _logsDirectory =>
      _overrideLogsDirectory ?? CrashReporter.logsDirectory;

  Future<String> _allowedRoot(Directory logsDirectory) {
    if (path_util.basename(logsDirectory.absolute.path).toLowerCase() !=
        'logs') {
      throw const FileSystemException('无法定位软件日志目录');
    }
    return logsDirectory.resolveSymbolicLinks();
  }

  bool _isWithin(String filePath, String rootPath) {
    return path_util.equals(filePath, rootPath) ||
        path_util.isWithin(rootPath, filePath);
  }

  String _displayName(String filePath, String rootPath) {
    return path_util.join('logs', path_util.relative(filePath, from: rootPath));
  }
}
