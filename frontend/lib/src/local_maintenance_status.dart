import 'dart:convert';
import 'dart:io';

import 'api_client.dart';
import 'crash_reporter.dart';

class LocalMaintenanceStatusStore {
  static const activeStates = <String>{
    'starting',
    'waiting_for_backend',
    'running',
    'restarting',
  };

  File get statusFile => File(
    [
      CrashReporter.logsDirectory.path,
      'backend_maintenance_status.json',
    ].join(Platform.pathSeparator),
  );

  bool isActive(MaintenanceStatus status) =>
      activeStates.contains(status.state);

  Future<MaintenanceStatus?> tryRead({String? path}) async {
    try {
      return await read(path: path);
    } on FileSystemException {
      return null;
    } on FormatException {
      return null;
    }
  }

  Future<MaintenanceStatus> read({String? path}) async {
    final file = path == null || path.isEmpty ? statusFile : File(path);
    final decoded = jsonDecode(await file.readAsString(encoding: utf8));
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('维护状态文件格式无效');
    }
    return MaintenanceStatus.fromJson(decoded);
  }

  Future<bool> isWorkerRunning(MaintenanceStatus status) async {
    try {
      if (Platform.isWindows) {
        final pidFilter = status.maintenancePid == null
            ? ''
            : ' -and \$_.ProcessId -eq ${status.maintenancePid}';
        final script =
            r'$process = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | '
            r'''Where-Object { $_.CommandLine -match '[\\/]maintenance\.py' '''
            '$pidFilter }; '
            r'if ($null -ne $process) { exit 0 }; exit 1';
        final result = await Process.run('powershell', [
          '-NoProfile',
          '-ExecutionPolicy',
          'Bypass',
          '-Command',
          script,
        ]).timeout(const Duration(seconds: 5));
        return result.exitCode == 0;
      }

      final result = await Process.run('ps', [
        '-eo',
        'pid=,args=',
      ]).timeout(const Duration(seconds: 5));
      if (result.exitCode != 0) return true;
      final expectedPid = status.maintenancePid;
      return result.stdout.toString().split('\n').any((line) {
        if (!line.contains('maintenance.py') || !line.contains('--operation')) {
          return false;
        }
        if (expectedPid == null) return true;
        return line.trimLeft().startsWith('$expectedPid ');
      });
    } catch (_) {
      // If process inspection is unavailable, waiting is safer than starting a
      // second backend while package files may still be changing.
      return true;
    }
  }

  Future<void> markInterrupted(MaintenanceStatus status) async {
    final file = status.statusPath == null || status.statusPath!.isEmpty
        ? statusFile
        : File(status.statusPath!);
    Map<String, dynamic> data = <String, dynamic>{};
    try {
      final decoded = jsonDecode(await file.readAsString(encoding: utf8));
      if (decoded is Map<String, dynamic>) data = decoded;
    } catch (_) {}
    data.addAll({
      'state': 'failed',
      'message': '上次环境维护任务已中断，可以重新安装。',
      'updated_at': DateTime.now().toUtc().toIso8601String(),
      'return_code': 1,
      'error': 'maintenance worker is no longer running',
    });
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(data),
      encoding: utf8,
      flush: true,
    );
  }
}
