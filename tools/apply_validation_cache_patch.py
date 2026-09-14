from pathlib import Path


PATH = Path("frontend/lib/src/screens/species_validation_screen.dart")
text = PATH.read_text(encoding="utf-8")


def replace_once(label: str, old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    text = text.replace(old, new, 1)


replace_once(
    "import",
    "import '../utils/quick_mark_sort.dart';\n",
    "import '../utils/quick_mark_sort.dart';\n"
    "import '../utils/validation_cache_delta.dart';\n",
)

replace_once(
    "cache fields",
    "  List<DetectionItem>? _bucketCacheItemsIdentity;\n"
    "  List<_SpeciesBucket> _bucketCache = const <_SpeciesBucket>[];\n",
    "  List<DetectionItem>? _bucketCacheItemsIdentity;\n"
    "  Map<String, int> _bucketCacheItemIndexByPath = const <String, int>{};\n"
    "  Map<String, _SpeciesBucket> _bucketCacheBucketByPath =\n"
    "      const <String, _SpeciesBucket>{};\n"
    "  Map<String, int> _bucketCacheBucketItemIndexByPath =\n"
    "      const <String, int>{};\n"
    "  final Set<String> _pendingValidationEchoPaths = <String>{};\n"
    "  List<_SpeciesBucket> _bucketCache = const <_SpeciesBucket>[];\n",
)

replace_once(
    "input reset",
    "    if (oldWidget.inputPath != widget.inputPath) {\n"
    "      _markHistory.clear();\n"
    "    }\n",
    "    if (oldWidget.inputPath != widget.inputPath) {\n"
    "      _markHistory.clear();\n"
    "      _pendingValidationEchoPaths.clear();\n"
    "    }\n",
)

replace_once(
    "grouping settings reset",
    "      _expandedGroupSignatures.clear();\n"
    "      _selectedGroupSignature = null;\n"
    "    }\n"
    "    if (widget.items.isEmpty) {\n",
    "      _expandedGroupSignatures.clear();\n"
    "      _selectedGroupSignature = null;\n"
    "      _pendingValidationEchoPaths.clear();\n"
    "    }\n"
    "    if (oldWidget.refreshVersion != widget.refreshVersion) {\n"
    "      _pendingValidationEchoPaths.clear();\n"
    "    }\n"
    "    if (widget.items.isEmpty) {\n",
)

replace_once(
    "empty cache reset",
    "      _bucketCacheItemsIdentity = null;\n"
    "      _bucketCache = const <_SpeciesBucket>[];\n"
    "      _bucketCacheItemByPath = const <String, DetectionItem>{};\n",
    "      _bucketCacheItemsIdentity = null;\n"
    "      _bucketCacheItemIndexByPath = const <String, int>{};\n"
    "      _bucketCacheBucketByPath = const <String, _SpeciesBucket>{};\n"
    "      _bucketCacheBucketItemIndexByPath = const <String, int>{};\n"
    "      _pendingValidationEchoPaths.clear();\n"
    "      _bucketCache = const <_SpeciesBucket>[];\n"
    "      _bucketCacheItemByPath = const <String, DetectionItem>{};\n",
)

replace_once(
    "current buckets fast path",
    "    if (hasSameGroupingSettings &&\n"
    "        identical(_bucketCacheItemsIdentity, widget.items)) {\n"
    "      if (!_hasCompletedDeferredGroup()) {\n"
    "        return _bucketCache;\n"
    "      }\n"
    "    }\n\n"
    "    final pathSignature = _itemsPathSignature(widget.items);\n",
    "    if (hasSameGroupingSettings &&\n"
    "        identical(_bucketCacheItemsIdentity, widget.items)) {\n"
    "      if (!_hasCompletedDeferredGroup()) {\n"
    "        return _bucketCache;\n"
    "      }\n"
    "    }\n\n"
    "    if (hasSameGroupingSettings &&\n"
    "        _tryAdoptValidationEchoWithoutGlobalSignatures()) {\n"
    "      if (!_hasCompletedDeferredGroup()) {\n"
    "        return _bucketCache;\n"
    "      }\n"
    "    }\n\n"
    "    final pathSignature = _itemsPathSignature(widget.items);\n",
)

replace_once(
    "full rebuild clears pending echo",
    "    _bucketCache = _buildBuckets(groups);\n"
    "    _rebuildBucketLookups();\n"
    "    _deferredRegroupGroupSignatures.clear();\n",
    "    _bucketCache = _buildBuckets(groups);\n"
    "    _rebuildBucketLookups();\n"
    "    _pendingValidationEchoPaths.clear();\n"
    "    _deferredRegroupGroupSignatures.clear();\n",
)

HELPER = r'''  bool _sameGroupingBoundaryMetadata(
    DetectionItem previous,
    DetectionItem next,
  ) {
    return previous.path == next.path &&
        previous.filename == next.filename &&
        previous.fileType == next.fileType &&
        previous.dateTaken == next.dateTaken &&
        previous.modifiedAt == next.modifiedAt &&
        previous.error == next.error &&
        previous.detectionData['拍摄时间']?.toString() ==
            next.detectionData['拍摄时间']?.toString();
  }

  bool _tryAdoptValidationEchoWithoutGlobalSignatures() {
    if (_pendingValidationEchoPaths.isEmpty) return false;
    final previousItems = _bucketCacheItemsIdentity;
    if (previousItems == null || previousItems.length != widget.items.length) {
      return false;
    }
    if (!ValidationCacheDelta.canAdoptPendingEcho<DetectionItem>(
      nextItems: widget.items,
      cachedLength: _bucketCacheItemIndexByPath.length,
      cachedIndexByPath: _bucketCacheItemIndexByPath,
      pendingPaths: _pendingValidationEchoPaths,
      pathOf: (item) => item.path,
      groupingSettingsUnchanged: true,
      refreshVersionUnchanged: true,
    )) {
      return false;
    }

    final pendingIndexes = <int>{};
    for (final path in _pendingValidationEchoPaths) {
      final index = _bucketCacheItemIndexByPath[path];
      if (index == null) return false;
      pendingIndexes.add(index);
    }
    for (var index = 0; index < widget.items.length; index++) {
      if (pendingIndexes.contains(index)) continue;
      if (!identical(previousItems[index], widget.items[index])) return false;
    }

    final replacements = <String, DetectionItem>{};
    final groupItemIndexByPath = <String, int>{};
    final affectedGroups = <String, _ValidationMediaGroup>{};

    for (final path in _pendingValidationEchoPaths) {
      final globalIndex = _bucketCacheItemIndexByPath[path];
      final previous = _bucketCacheItemByPath[path];
      final groupEntry = _bucketCacheGroupByPath[path];
      final bucket = _bucketCacheBucketByPath[path];
      final bucketIndex = _bucketCacheBucketItemIndexByPath[path];
      if (globalIndex == null ||
          previous == null ||
          groupEntry == null ||
          bucket == null ||
          bucketIndex == null ||
          globalIndex < 0 ||
          globalIndex >= widget.items.length ||
          bucketIndex < 0 ||
          bucketIndex >= bucket.items.length) {
        return false;
      }

      final next = widget.items[globalIndex];
      if (!_sameGroupingBoundaryMetadata(previous, next) ||
          bucket.items[bucketIndex].path != path) {
        return false;
      }
      final groupItemIndex = groupEntry.group.items.indexWhere(
        (item) => item.path == path,
      );
      if (groupItemIndex < 0) return false;

      replacements[path] = next;
      groupItemIndexByPath[path] = groupItemIndex;
      affectedGroups[groupEntry.signature] = groupEntry.group;
    }

    for (final entry in affectedGroups.entries) {
      final group = entry.value;
      final nextGroup = _ValidationMediaGroup(<DetectionItem>[
        for (final item in group.items) replacements[item.path] ?? item,
      ]);
      if (_primarySpeciesForGroup(group) != _primarySpeciesForGroup(nextGroup)) {
        return false;
      }
      final wasComplete = _groupComplete(group);
      final isComplete = _groupComplete(nextGroup);
      if (wasComplete != isComplete) return false;
      if (_deferredRegroupGroupSignatures.contains(entry.key) && isComplete) {
        return false;
      }
    }

    for (final entry in replacements.entries) {
      final path = entry.key;
      final next = entry.value;
      final groupEntry = _bucketCacheGroupByPath[path]!;
      final bucket = _bucketCacheBucketByPath[path]!;
      final bucketIndex = _bucketCacheBucketItemIndexByPath[path]!;
      final groupItemIndex = groupItemIndexByPath[path]!;
      groupEntry.group.items[groupItemIndex] = next;
      bucket.items[bucketIndex] = next;
      _bucketCacheItemByPath[path] = next;
    }
    for (final signature in affectedGroups.keys) {
      _groupSpeciesLabelCache.remove(signature);
    }
    _pendingValidationEchoPaths.removeAll(replacements.keys);
    _bucketCacheItemsIdentity = widget.items;
    _bucketCacheGroupingSignature = null;
    return true;
  }

'''

replace_once(
    "incremental helper insertion",
    "  void _updateCachedBucketItems() {\n",
    HELPER + "  void _updateCachedBucketItems() {\n",
)

replace_once(
    "full cache refresh consumes pending echo",
    "    _rebuildBucketLookups();\n"
    "  }\n\n"
    "  void _rebuildBucketLookups() {\n",
    "    _rebuildBucketLookups();\n"
    "    _pendingValidationEchoPaths.clear();\n"
    "  }\n\n"
    "  void _rebuildBucketLookups() {\n",
)

OLD_LOOKUP = r'''  void _rebuildBucketLookups() {
    _bucketCacheItemByPath = <String, DetectionItem>{
      for (final item in widget.items) item.path: item,
    };
    _mediaTimestampCache.removeWhere(
      (path, _) => !_bucketCacheItemByPath.containsKey(path),
    );
    final groupByPath = <String, _ValidationGroupIndexEntry>{};
    final groupBySignature = <String, _ValidationMediaGroup>{};
    for (final bucket in _bucketCache) {
      for (var index = 0; index < bucket.groups.length; index++) {
        final group = bucket.groups[index];
        final signature = _groupSignature(group);
        groupBySignature[signature] = group;
        final entry = _ValidationGroupIndexEntry(
          index: _globalGroupIndexForSignature(signature, fallback: index),
          group: group,
          signature: signature,
        );
        for (final item in group.items) {
          groupByPath[item.path] = entry;
        }
      }
    }
    _bucketCacheGroupByPath = groupByPath;
    _bucketCacheGroupBySignature = groupBySignature;
  }
'''

NEW_LOOKUP = r'''  void _rebuildBucketLookups() {
    _bucketCacheItemByPath = <String, DetectionItem>{
      for (final item in widget.items) item.path: item,
    };
    _bucketCacheItemIndexByPath = <String, int>{
      for (var index = 0; index < widget.items.length; index++)
        widget.items[index].path: index,
    };
    _mediaTimestampCache.removeWhere(
      (path, _) => !_bucketCacheItemByPath.containsKey(path),
    );
    final bucketByPath = <String, _SpeciesBucket>{};
    final bucketItemIndexByPath = <String, int>{};
    final groupByPath = <String, _ValidationGroupIndexEntry>{};
    final groupBySignature = <String, _ValidationMediaGroup>{};
    for (final bucket in _bucketCache) {
      for (var itemIndex = 0; itemIndex < bucket.items.length; itemIndex++) {
        final item = bucket.items[itemIndex];
        bucketByPath[item.path] = bucket;
        bucketItemIndexByPath[item.path] = itemIndex;
      }
      for (var index = 0; index < bucket.groups.length; index++) {
        final group = bucket.groups[index];
        final signature = _groupSignature(group);
        groupBySignature[signature] = group;
        final entry = _ValidationGroupIndexEntry(
          index: _globalGroupIndexForSignature(signature, fallback: index),
          group: group,
          signature: signature,
        );
        for (final item in group.items) {
          groupByPath[item.path] = entry;
        }
      }
    }
    _bucketCacheBucketByPath = bucketByPath;
    _bucketCacheBucketItemIndexByPath = bucketItemIndexByPath;
    _bucketCacheGroupByPath = groupByPath;
    _bucketCacheGroupBySignature = groupBySignature;
  }
'''
replace_once("bucket lookup indexes", OLD_LOOKUP, NEW_LOOKUP)

replace_once(
    "batch echo registration",
    "      final lastUpdated = updatedItems.isEmpty ? null : updatedItems.last;\n"
    "      if (!mounted) return;\n"
    "      final usedQuickSpecies =\n",
    "      final lastUpdated = updatedItems.isEmpty ? null : updatedItems.last;\n"
    "      if (!mounted) return;\n"
    "      _pendingValidationEchoPaths.addAll(\n"
    "        updatedItems.map((item) => item.path),\n"
    "      );\n"
    "      final usedQuickSpecies =\n",
)

replace_once(
    "undo echo registration",
    "      final lastUpdated = updatedItems.isEmpty ? null : updatedItems.last;\n"
    "      if (!mounted) return;\n"
    "      final targetPaths = targets.map((item) => item.path).toSet();\n",
    "      final lastUpdated = updatedItems.isEmpty ? null : updatedItems.last;\n"
    "      if (!mounted) return;\n"
    "      _pendingValidationEchoPaths.addAll(\n"
    "        (updatedItems.isEmpty ? targets : updatedItems).map(\n"
    "          (item) => item.path,\n"
    "        ),\n"
    "      );\n"
    "      final targetPaths = targets.map((item) => item.path).toSet();\n",
)

replace_once(
    "single echo registration",
    "      );\n"
    "      if (!mounted) return;\n"
    "      final usedQuickSpecies =\n"
    "          action == 'update' &&\n",
    "      );\n"
    "      if (!mounted) return;\n"
    "      _pendingValidationEchoPaths.add(updated.path);\n"
    "      final usedQuickSpecies =\n"
    "          action == 'update' &&\n",
)

PATH.write_text(text, encoding="utf-8")
