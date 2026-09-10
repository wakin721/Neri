from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str, marker: str | None = None) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if marker and marker in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:120]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


model_path = Path("frontend/lib/src/models/dinov3_registry.dart")
model_path.write_text(
    '''class DinoV3ClusterExampleRef {
  const DinoV3ClusterExampleRef({
    required this.kind,
    this.registrationId,
    this.eventId,
    this.observationId,
  });

  factory DinoV3ClusterExampleRef.fromJson(Map<String, dynamic> json) {
    return DinoV3ClusterExampleRef(
      kind: json['kind']?.toString() ?? '',
      registrationId: (json['registration_id'] as num?)?.toInt(),
      eventId: (json['event_id'] as num?)?.toInt(),
      observationId: json['observation_id']?.toString(),
    );
  }

  final String kind;
  final int? registrationId;
  final int? eventId;
  final String? observationId;
}

class DinoV3RegistryCluster {
  const DinoV3RegistryCluster({
    required this.id,
    required this.label,
    required this.source,
    required this.prototypeIndex,
    required this.eventCount,
    required this.cameraCount,
    required this.sampleCount,
    required this.active,
    required this.exampleRefs,
    this.meanSquaredDistance,
    this.learningStatus,
  });

  factory DinoV3RegistryCluster.fromJson(Map<String, dynamic> json) {
    final rawRefs = json['example_refs'] as List<dynamic>? ?? const <dynamic>[];
    return DinoV3RegistryCluster(
      id: json['id']?.toString() ?? '',
      label: json['label']?.toString() ?? 'Cluster',
      source: json['source']?.toString() ?? '',
      prototypeIndex: (json['prototype_index'] as num?)?.toInt() ?? 0,
      eventCount: (json['event_count'] as num?)?.toInt() ?? 0,
      cameraCount: (json['camera_count'] as num?)?.toInt() ?? 0,
      sampleCount: (json['sample_count'] as num?)?.toInt() ?? 0,
      meanSquaredDistance: (json['mean_squared_distance'] as num?)?.toDouble(),
      active: json['active'] != false,
      learningStatus: json['learning_status']?.toString(),
      exampleRefs: rawRefs
          .whereType<Map<String, dynamic>>()
          .map(DinoV3ClusterExampleRef.fromJson)
          .toList(growable: false),
    );
  }

  final String id;
  final String label;
  final String source;
  final int prototypeIndex;
  final int eventCount;
  final int cameraCount;
  final int sampleCount;
  final double? meanSquaredDistance;
  final bool active;
  final String? learningStatus;
  final List<DinoV3ClusterExampleRef> exampleRefs;

  bool get isCheckpoint => source == 'checkpoint';
  bool get isFeedback => source == 'feedback' || source == 'feedback_evidence';
  bool get isRegistry => source == 'registry';
}

class DinoV3RegistryEntry {
  const DinoV3RegistryEntry({
    required this.id,
    required this.candidateNumber,
    required this.status,
    required this.displayName,
    required this.commonName,
    required this.scientificName,
    required this.eventCount,
    required this.cameraCount,
    required this.prototypeCount,
    required this.clusterPurity,
    required this.embeddingConsistency,
    required this.conditions,
    required this.canRegister,
    this.feedbackEventCount = 0,
    this.feedbackPrototypeCount = 0,
    this.learningStatus,
    this.clusters = const <DinoV3RegistryCluster>[],
  });

  factory DinoV3RegistryEntry.fromJson(Map<String, dynamic> json) {
    final rawConditions =
        json['conditions'] as Map<String, dynamic>? ?? const {};
    final rawClusters = json['clusters'] as List<dynamic>? ?? const <dynamic>[];
    return DinoV3RegistryEntry(
      id: (json['id'] as num?)?.toInt() ?? 0,
      candidateNumber: (json['candidate_number'] as num?)?.toInt() ?? 0,
      status: json['status']?.toString() ?? 'candidate',
      displayName: json['display_name']?.toString() ?? '',
      commonName: json['common_name']?.toString() ?? '',
      scientificName: json['scientific_name']?.toString() ?? '',
      eventCount: (json['event_count'] as num?)?.toInt() ?? 0,
      cameraCount: (json['camera_count'] as num?)?.toInt() ?? 0,
      prototypeCount: (json['prototype_count'] as num?)?.toInt() ?? 0,
      clusterPurity: (json['cluster_purity'] as num?)?.toDouble() ?? 0,
      embeddingConsistency:
          (json['embedding_consistency'] as num?)?.toDouble() ?? 0,
      conditions: rawConditions.map(
        (key, value) => MapEntry(key, value == true),
      ),
      canRegister: json['can_register'] == true,
      feedbackEventCount: (json['feedback_event_count'] as num?)?.toInt() ?? 0,
      feedbackPrototypeCount:
          (json['feedback_prototype_count'] as num?)?.toInt() ?? 0,
      learningStatus: json['learning_status']?.toString(),
      clusters: rawClusters
          .whereType<Map<String, dynamic>>()
          .map(DinoV3RegistryCluster.fromJson)
          .toList(growable: false),
    );
  }

  final int id;
  final int candidateNumber;
  final String status;
  final String displayName;
  final String commonName;
  final String scientificName;
  final int eventCount;
  final int cameraCount;
  final int prototypeCount;
  final double clusterPurity;
  final double embeddingConsistency;
  final Map<String, bool> conditions;
  final bool canRegister;
  final int feedbackEventCount;
  final int feedbackPrototypeCount;
  final String? learningStatus;
  final List<DinoV3RegistryCluster> clusters;

  bool get isCandidate => status == 'candidate';
  bool get isCheckpoint => status.toLowerCase() == 'checkpoint';
  bool get hasFeedbackLearning => feedbackEventCount > 0;

  bool get canDelete => const <String>{
    'candidate',
    'provisional',
    'confirmed',
    'mature',
  }.contains(status.toLowerCase());
}

class DinoV3RegistryEvent {
  const DinoV3RegistryEvent({
    required this.eventKey,
    required this.sourcePath,
    required this.cameraId,
    required this.sampleCount,
    required this.timestampMissing,
    this.id = 0,
    this.bbox = const <double>[],
    this.frameIndex,
    this.timestampSeconds,
    this.hasExample = false,
    this.startedAt,
    this.endedAt,
  });

  factory DinoV3RegistryEvent.fromJson(Map<String, dynamic> json) {
    return DinoV3RegistryEvent(
      id: (json['id'] as num?)?.toInt() ?? 0,
      eventKey: json['event_key']?.toString() ?? '',
      sourcePath: json['source_path']?.toString() ?? '',
      cameraId: json['camera_id']?.toString() ?? '',
      sampleCount: (json['sample_count'] as num?)?.toInt() ?? 1,
      timestampMissing: json['timestamp_missing'] == true,
      bbox: (json['bbox'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<num>()
          .map((value) => value.toDouble())
          .toList(),
      frameIndex: (json['frame_index'] as num?)?.toInt(),
      timestampSeconds: (json['timestamp_seconds'] as num?)?.toDouble(),
      hasExample: json['has_example'] == true,
      startedAt: json['started_at']?.toString(),
      endedAt: json['ended_at']?.toString(),
    );
  }

  final int id;
  final String eventKey;
  final String sourcePath;
  final String cameraId;
  final int sampleCount;
  final bool timestampMissing;
  final List<double> bbox;
  final int? frameIndex;
  final double? timestampSeconds;
  final bool hasExample;
  final String? startedAt;
  final String? endedAt;
}
''',
    encoding="utf-8",
)

path = "frontend/lib/src/widgets/dinov3_registry_dialog.dart"
replace_once(
    path,
    '''  DinoV3RegistryEntry? _selected;
  bool _loading = false;
''',
    '''  DinoV3RegistryEntry? _selected;
  DinoV3RegistryCluster? _selectedCluster;
  List<Uint8List> _clusterExampleBytes = const <Uint8List>[];
  bool _loading = false;
''',
    marker="DinoV3RegistryCluster? _selectedCluster;",
)
replace_once(
    path,
    '''    _entries = widget.initialEntries ?? const [];
    if (_entries.isNotEmpty) _select(_entries.first, notify: false);
    if (widget.initialEntries == null) unawaited(_load());
''',
    '''    _entries = widget.initialEntries ?? const [];
    if (_entries.isNotEmpty) _select(_entries.first, notify: false);
    unawaited(_load(preferredId: _selected?.id));
''',
    marker="unawaited(_load(preferredId: _selected?.id));",
)
replace_once(
    path,
    '''      _selected = entry;
      _commonNameController.text = entry.commonName;
''',
    '''      _selected = entry;
      _selectedCluster = null;
      _clusterExampleBytes = const <Uint8List>[];
      _commonNameController.text = entry.commonName;
''',
    marker="_selectedCluster = null;",
)
replace_once(
    path,
    '''  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final entries = await widget.apiClient.fetchDinoV3RegistryCatalog(
        widget.modelPath,
      );
      if (!mounted) return;
      setState(() {
        _entries = entries;
        _loading = false;
      });
      if (entries.isNotEmpty) _select(entries.first);
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = '读取注册状态失败：$error';
      });
    }
  }
''',
    '''  Future<void> _load({int? preferredId}) async {
    if (mounted) {
      setState(() {
        _loading = true;
        _error = null;
      });
    }
    try {
      final fetched = await widget.apiClient.fetchDinoV3RegistryCatalog(
        widget.modelPath,
      );
      if (!mounted) return;
      // A real DINOv3 catalog always contains checkpoint classes. Keeping the
      // supplied snapshot only when a legacy/mock backend returns [] preserves
      // backward compatibility while every normal dialog open still refreshes.
      final entries = fetched.isEmpty && _entries.isNotEmpty ? _entries : fetched;
      final wantedId = preferredId ?? _selected?.id;
      setState(() {
        _entries = entries;
        _loading = false;
      });
      if (entries.isEmpty) {
        setState(() {
          _selected = null;
          _selectedCluster = null;
          _clusterExampleBytes = const <Uint8List>[];
        });
        return;
      }
      DinoV3RegistryEntry selected = entries.first;
      if (wantedId != null) {
        for (final entry in entries) {
          if (entry.id == wantedId) {
            selected = entry;
            break;
          }
        }
      }
      _select(selected);
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = '读取注册状态失败：$error';
      });
    }
  }
''',
    marker="A real DINOv3 catalog always contains checkpoint classes.",
)
replace_once(
    path,
    '''  void _replaceEntry(DinoV3RegistryEntry entry) {
    setState(() {
      _entries = [
        for (final current in _entries)
          if (current.id == entry.id) entry else current,
      ];
      _selected = entry;
      _commonNameController.text = entry.commonName;
      _scientificNameController.text = entry.scientificName;
      _error = null;
    });
  }

''',
    '',
    marker="Future<void> _loadClusterExamples",
)
replace_once(
    path,
    '''      if (mounted) _replaceEntry(updated);
''',
    '''      if (mounted) await _load(preferredId: updated.id);
''',
)
replace_once(
    path,
    '''      if (mounted) _replaceEntry(updated);
''',
    '''      if (mounted) await _load(preferredId: updated.id);
''',
)
cluster_methods = '''  Future<void> _loadClusterExamples(
    DinoV3RegistryEntry entry,
    DinoV3RegistryCluster cluster,
  ) async {
    final requestId = ++_examplesRequestId;
    if (mounted && _selected?.id == entry.id && _selectedCluster?.id == cluster.id) {
      setState(() {
        _examplesLoading = true;
        _clusterExampleBytes = const <Uint8List>[];
      });
    }
    final loaded = <Uint8List>[];
    for (final ref in cluster.exampleRefs.take(3)) {
      try {
        Uint8List? bytes;
        if (ref.kind == 'registry' &&
            ref.registrationId != null &&
            ref.eventId != null) {
          bytes = await widget.apiClient.fetchDinoV3RegistryExample(
            widget.modelPath,
            ref.registrationId!,
            ref.eventId!,
          );
        } else if (ref.kind == 'observation' &&
            (ref.observationId?.isNotEmpty ?? false)) {
          bytes = await widget.apiClient.fetchDinoV3ObservationExample(
            widget.modelPath,
            ref.observationId!,
          );
        }
        if (bytes != null && bytes.isNotEmpty) loaded.add(bytes);
      } catch (_) {
        // A deleted local source file must not hide other cluster examples.
      }
    }
    if (!mounted ||
        requestId != _examplesRequestId ||
        _selected?.id != entry.id ||
        _selectedCluster?.id != cluster.id) {
      return;
    }
    setState(() {
      _clusterExampleBytes = loaded;
      _examplesLoading = false;
    });
  }

  void _selectCluster(
    DinoV3RegistryEntry entry,
    DinoV3RegistryCluster cluster,
  ) {
    setState(() {
      _selected = entry;
      _selectedCluster = cluster;
      _clusterExampleBytes = const <Uint8List>[];
      _events = const <DinoV3RegistryEvent>[];
      _exampleBytes = const <int, Uint8List>{};
      _commonNameController.text = entry.commonName;
      _scientificNameController.text = entry.scientificName;
      _error = null;
    });
    unawaited(_loadClusterExamples(entry, cluster));
  }

'''
replace_once(
    path,
    "  Future<void> _loadExamples(DinoV3RegistryEntry entry) async {\n",
    cluster_methods + "  Future<void> _loadExamples(DinoV3RegistryEntry entry) async {\n",
    marker="Future<void> _loadClusterExamples",
)
replace_once(
    path,
    '''  Widget _buildExampleGallery() {
    if (_selected?.isCheckpoint == true) {
''',
    '''  Widget _buildExampleGallery() {
    final selectedCluster = _selectedCluster;
    if (selectedCluster != null) {
      if (_examplesLoading) {
        return const SizedBox(
          height: 104,
          child: Center(child: CircularProgressIndicator(strokeWidth: 2)),
        );
      }
      if (_clusterExampleBytes.isEmpty) {
        final label = selectedCluster.isCheckpoint
            ? 'Checkpoint prototype 不包含原始训练图片'
            : '该 Cluster 暂无可用裁切例图';
        return SizedBox(
          height: 72,
          child: Align(
            alignment: Alignment.centerLeft,
            child: Text(
              label,
              style: TextStyle(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        );
      }
      return SizedBox(
        height: 112,
        child: ListView.separated(
          scrollDirection: Axis.horizontal,
          itemCount: _clusterExampleBytes.length,
          separatorBuilder: (_, __) => const SizedBox(width: 10),
          itemBuilder: (context, index) {
            return ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: SizedBox(
                width: 112,
                height: 112,
                child: Image.memory(
                  _clusterExampleBytes[index],
                  key: ValueKey('dinov3-cluster-example-$index'),
                  fit: BoxFit.cover,
                  errorBuilder: (_, __, ___) =>
                      const Center(child: Icon(Icons.broken_image_outlined)),
                ),
              ),
            );
          },
        ),
      );
    }
    if (_selected?.isCheckpoint == true) {
''',
    marker="dinov3-cluster-example-$index",
)
replace_once(
    path,
    '''                if (entry.isCheckpoint) ...[
                  const SizedBox(height: 8),
                  Text(
                    '来自不可变 DINOv3 Checkpoint · ${entry.prototypeCount} 个 prototype。'
                    '分类头保存特征中心而非原始训练影像。',
                  ),
                ],
''',
    '''                if (entry.isCheckpoint) ...[
                  const SizedBox(height: 8),
                  Text(
                    '来自不可变 DINOv3 Checkpoint · ${entry.prototypeCount} 个 base prototype。'
                    '分类头保存特征中心而非原始训练影像。',
                  ),
                  if (entry.hasFeedbackLearning)
                    Text(
                      '人工纠正学习：${entry.feedbackEventCount} 个事件 · '
                      '${entry.feedbackPrototypeCount} 个 learned prototype · '
                      '${entry.learningStatus ?? 'collecting'}',
                    ),
                ],
                if (_selectedCluster != null) ...[
                  const SizedBox(height: 12),
                  Text(
                    _selectedCluster!.label,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  Text(
                    '来源 ${_selectedCluster!.source} · '
                    '${_selectedCluster!.eventCount} 事件 · '
                    '${_selectedCluster!.cameraCount} 相机 · '
                    '${_selectedCluster!.sampleCount} 样本',
                  ),
                  if (_selectedCluster!.learningStatus != null)
                    Text('学习状态：${_selectedCluster!.learningStatus}'),
                  if (_selectedCluster!.meanSquaredDistance != null)
                    Text(
                      'Cluster 内平均距离² '
                      '${_selectedCluster!.meanSquaredDistance!.toStringAsFixed(4)}',
                    ),
                ],
''',
    marker="人工纠正学习：${entry.feedbackEventCount}",
)
replace_once(
    path,
    '''                                    return ListTile(
                                      selected: selected?.id == entry.id,
                                      title: Text(entry.displayName),
                                      subtitle: Text(
                                        entry.isCheckpoint
                                            ? '分类头基础物种 · ${entry.prototypeCount} prototypes'
                                            : '${entry.status} · ${entry.eventCount} 事件 · ${entry.cameraCount} 相机',
                                      ),
                                      onTap: () => _select(entry),
                                    );
''',
    '''                                    final subtitle = entry.isCheckpoint
                                        ? entry.hasFeedbackLearning
                                              ? '分类头基础物种 · ${entry.prototypeCount} base · '
                                                    '${entry.feedbackPrototypeCount} learned · '
                                                    '${entry.learningStatus ?? 'collecting'}'
                                              : '分类头基础物种 · ${entry.prototypeCount} base prototypes'
                                        : '${entry.status} · ${entry.eventCount} 事件 · ${entry.cameraCount} 相机';
                                    return ExpansionTile(
                                      key: PageStorageKey<String>(
                                        'dinov3-registry-species-${entry.id}',
                                      ),
                                      initiallyExpanded:
                                          entry.clusters.isNotEmpty &&
                                          (!entry.isCheckpoint || entry.hasFeedbackLearning),
                                      title: Text(entry.displayName),
                                      subtitle: Text(subtitle),
                                      onExpansionChanged: (expanded) {
                                        if (expanded) _select(entry);
                                      },
                                      children: [
                                        for (final cluster in entry.clusters)
                                          ListTile(
                                            dense: true,
                                            contentPadding:
                                                const EdgeInsets.only(left: 28, right: 8),
                                            selected:
                                                selected?.id == entry.id &&
                                                _selectedCluster?.id == cluster.id,
                                            leading: Icon(
                                              cluster.isCheckpoint
                                                  ? Icons.lock_outline_rounded
                                                  : Icons.scatter_plot_rounded,
                                              size: 18,
                                            ),
                                            title: Text(cluster.label),
                                            subtitle: Text(
                                              cluster.isCheckpoint
                                                  ? 'checkpoint prototype'
                                                  : '${cluster.source} · ${cluster.eventCount} 事件 · '
                                                        '${cluster.cameraCount} 相机'
                                                        '${cluster.learningStatus == null ? '' : ' · ${cluster.learningStatus}'}',
                                            ),
                                            onTap: () => _selectCluster(entry, cluster),
                                          ),
                                      ],
                                    );
''',
    marker="dinov3-registry-species-${entry.id}",
)

# Move class names from the top edge to an explicit horizontal-axis legend.
scatter_path = "frontend/lib/src/widgets/dinov3_feature_scatter.dart"
replace_once(
    scatter_path,
    '''          Row(
            children: [
              Expanded(
                child: Text(
                  first,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
              Expanded(
                child: Text(
                  second,
                  textAlign: TextAlign.end,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
''',
    '''          Text(
            '局部特征投影',
            style: TextStyle(
              fontSize: 12,
              color: colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 6),
''',
    marker="'局部特征投影'",
)
replace_once(
    scatter_path,
    '''          const SizedBox(height: 6),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.diamond_rounded, size: 14, color: colorScheme.error),
''',
    '''          const SizedBox(height: 6),
          Row(
            children: [
              Expanded(
                child: Text(
                  first,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
              Text(
                '← 物种判别轴 →',
                style: TextStyle(
                  fontSize: 11,
                  color: colorScheme.onSurfaceVariant,
                ),
              ),
              Expanded(
                child: Text(
                  second,
                  textAlign: TextAlign.end,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.diamond_rounded, size: 14, color: colorScheme.error),
''',
    marker="'← 物种判别轴 →'",
)

print("Applied DINOv3 registry cluster frontend implementation")
