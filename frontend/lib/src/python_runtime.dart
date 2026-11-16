import 'dart:io';

// Ignore host Python configuration while retaining application module paths.
const pythonIsolationArguments = <String>['-E', '-s'];

Map<String, String> pythonProcessEnvironment({Map<String, String>? parent}) {
  final environment = <String, String>{
    for (final entry in (parent ?? Platform.environment).entries)
      if (!const <String>{
        'PYTHONPATH',
        'PYTHONHOME',
        'PYTHONNOUSERSITE',
      }.contains(entry.key.toUpperCase()))
        entry.key: entry.value,
    'PYTHONNOUSERSITE': '1',
  };
  return environment;
}
