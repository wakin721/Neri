import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/python_runtime.dart';

void main() {
  test('preserves OS and app settings while removing host Python paths', () {
    final parent = <String, String>{
      'Path': 'system-tools',
      'APPDATA': 'user-data',
      'NERI_PARENT_PID': '123',
      'HTTPS_PROXY': 'http://proxy.test',
      'PythonPath': 'foreign-packages',
      'PYTHONHOME': 'foreign-python',
      'pythonNOUSERsite': '0',
    };
    expect(pythonProcessEnvironment(parent: parent), <String, String>{
      'Path': 'system-tools',
      'APPDATA': 'user-data',
      'NERI_PARENT_PID': '123',
      'HTTPS_PROXY': 'http://proxy.test',
      'PYTHONNOUSERSITE': '1',
    });
    expect(parent['PythonPath'], 'foreign-packages');
  });

  test(
    'launcher arguments prevent imports from a foreign Python path',
    () async {
      final temporary = Directory.systemTemp.createTempSync('neri-python-');
      addTearDown(() => temporary.deleteSync(recursive: true));
      File(
        '${temporary.path}/neri_foreign_python_probe.py',
      ).writeAsStringSync('POLLUTED = True\n');
      final environment = Map<String, String>.of(Platform.environment)
        ..removeWhere((key, _) => key.toUpperCase() == 'PYTHONNOUSERSITE')
        ..['PYTHONPATH'] = temporary.path
        ..['PYTHONHOME'] = '${temporary.path}/nonexistent-python-home';
      final toolkit = File('../toolkit/python.exe').absolute;
      final executable = toolkit.existsSync()
          ? toolkit.path
          : (Platform.isWindows ? 'python.exe' : 'python3');
      final result = await Process.run(
        executable,
        <String>[
          ...pythonIsolationArguments,
          '-B',
          '-c',
          'import importlib.util,json,site,sys; '
              'print(json.dumps([sys.flags.ignore_environment, '
              'sys.flags.no_user_site, site.ENABLE_USER_SITE, '
              'importlib.util.find_spec("neri_foreign_python_probe") is None]))',
        ],
        environment: environment,
        includeParentEnvironment: false,
      );
      expect(result.exitCode, 0, reason: '${result.stderr}');
      expect(jsonDecode('${result.stdout}'), <Object>[1, 1, false, true]);
    },
  );
}
