import 'dart:io';

import '../models/job.dart';
import 'local_detection_items.dart';

/// Keeps a small per-root directory cache independently of media loading.
class DirectoryBrowsingScope {
  final _directoryCache = <String, List<String>>{};
  List<DetectionItem>? _source;
  String _root = '';
  String? _selected;
  List<String> directories = const [];
  List<DetectionItem> _items = const [];

  String get inputPath => _selected ?? _root;
  String? get selected => _selected;

  void cacheDirectories(List<String> paths, String root) {
    _directoryCache.remove(root);
    _directoryCache[root] = List.unmodifiable(paths);
    if (_directoryCache.length > 8) {
      _directoryCache.remove(_directoryCache.keys.first);
    }
    if (root != _root) return;
    directories = _directoryCache[root]!;
    if (_selected != null && !directories.contains(_selected)) _selected = null;
    _filter();
  }

  void update(List<DetectionItem> source, String root) {
    if (identical(source, _source) && root == _root) return;
    if (root != _root) _selected = null;
    _root = root;
    _source = source;
    final cached = _directoryCache[root];
    if (cached != null) {
      directories = cached;
      if (_selected != null && !directories.contains(_selected)) {
        _selected = null;
      }
      _filter();
      return;
    }
    final paths = <String, String>{};
    final ancestors = <String>{};
    if (root.isNotEmpty) {
      final rootKey = _key(Directory(root).absolute.path);
      for (final item in source) {
        if (item.path.trim().isEmpty) continue;
        var directory = File(item.path).absolute.parent;
        while (_key(directory.path).startsWith('$rootKey/')) {
          paths.putIfAbsent(_key(directory.path), () => directory.path);
          directory = directory.parent;
          ancestors.add(_key(directory.path));
        }
      }
    }
    directories = [
      for (final entry in paths.entries)
        if (!ancestors.contains(entry.key)) entry.value,
    ]..sort((a, b) => _key(a).compareTo(_key(b)));
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
