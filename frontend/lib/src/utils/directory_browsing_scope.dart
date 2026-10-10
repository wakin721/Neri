import 'dart:io';

import '../models/job.dart';
import 'local_detection_items.dart';

/// Derives the directory tree from the already-loaded media snapshot.
/// No filesystem enumeration is performed while the UI rebuilds.
class DirectoryBrowsingScope {
  List<DetectionItem>? _source;
  String _root = '';
  String? _selected;
  List<String> directories = const [];
  List<DetectionItem> _items = const [];

  String get inputPath => _selected ?? _root;
  String? get selected => _selected;

  void update(List<DetectionItem> source, String root) {
    if (identical(source, _source) && root == _root) return;
    if (root != _root) _selected = null;
    _root = root;
    _source = source;
    final paths = <String>{};
    if (root.isNotEmpty) {
      final rootKey = _key(Directory(root).absolute.path);
      for (final item in source) {
        var directory = File(item.path).absolute.parent;
        while (_key(directory.path).startsWith('$rootKey/')) {
          paths.add(directory.path);
          directory = directory.parent;
        }
      }
    }
    directories = paths.toList()..sort((a, b) => _key(a).compareTo(_key(b)));
    if (_selected != null && !directories.contains(_selected)) _selected = null;
    _filter();
  }

  void select(String? path) {
    _selected = path;
    _filter();
  }

  List<DetectionItem> get items => _items;

  String label(String path) {
    final root = Directory(
      _root,
    ).absolute.path.replaceAll('\\', '/').replaceAll(RegExp(r'/+$'), '');
    return path.replaceAll('\\', '/').substring(root.length + 1);
  }

  void _filter() {
    _items = inputPath.isEmpty
        ? (_source ?? const [])
        : validationItemsInInputFolder(_source ?? const [], inputPath);
  }

  String _key(String path) {
    final normalized = path
        .replaceAll('\\', '/')
        .replaceAll(RegExp(r'/+$'), '');
    return Platform.isWindows ? normalized.toLowerCase() : normalized;
  }
}
