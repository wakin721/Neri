import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../api_client.dart';
import '../models/dinov3_registry.dart';

String dinov3RegistrySummary(List<DinoV3RegistryEntry> entries) {
  int count(String status) =>
      entries.where((entry) => entry.status == status).length;
  return 'Checkpoint ${count('checkpoint')} · Candidate ${count('candidate')} · '
      'Provisional ${count('provisional')} · '
      'Confirmed ${count('confirmed')} · '
      'Mature ${count('mature')}';
}

class DinoV3RegistryButton extends StatefulWidget {
  const DinoV3RegistryButton({
    required this.apiClient,
    required this.modelPath,
    required this.onContinueValidation,
    this.onShowMessage,
    super.key,
  });

  final NeriApiClient apiClient;
  final String modelPath;
  final ValueChanged<Set<String>> onContinueValidation;
  final ValueChanged<String>? onShowMessage;

  @override
  State<DinoV3RegistryButton> createState() => _DinoV3RegistryButtonState();
}

class _DinoV3RegistryButtonState extends State<DinoV3RegistryButton> {
  List<DinoV3RegistryEntry> _entries = const [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    unawaited(_refresh());
  }

  @override
  void didUpdateWidget(covariant DinoV3RegistryButton oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.modelPath != widget.modelPath) {
      _entries = const [];
      _loading = true;
      unawaited(_refresh());
    }
  }

  Future<void> _refresh() async {
    try {
      final entries = await widget.apiClient.fetchDinoV3RegistryCatalog(
        widget.modelPath,
      );
      if (!mounted) return;
      setState(() {
        _entries = entries;
        _loading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _loading = false);
      widget.onShowMessage?.call('读取 DINOv3 物种注册状态失败：$error');
    }
  }

  Future<void> _open() async {
    await showDialog<void>(
      context: context,
      builder: (context) => DinoV3RegistryDialog(
        apiClient: widget.apiClient,
        modelPath: widget.modelPath,
        initialEntries: _entries,
        onContinueValidation: widget.onContinueValidation,
      ),
    );
    if (mounted) unawaited(_refresh());
  }

  @override
  Widget build(BuildContext context) {
    return ListTile(
      contentPadding: EdgeInsets.zero,
      leading: const Icon(Icons.hub_rounded),
      title: const Text('物种注册状态'),
      subtitle: Text(_loading ? '读取中…' : dinov3RegistrySummary(_entries)),
      trailing: const Icon(Icons.chevron_right_rounded),
      onTap: _open,
    );
  }
}

class DinoV3RegistryDialog extends StatefulWidget {
  const DinoV3RegistryDialog({
    required this.apiClient,
    required this.modelPath,
    required this.onContinueValidation,
    this.initialEntries,
    super.key,
  });

  final NeriApiClient apiClient;
  final String modelPath;
  final ValueChanged<Set<String>> onContinueValidation;
  final List<DinoV3RegistryEntry>? initialEntries;

  @override
  State<DinoV3RegistryDialog> createState() => _DinoV3RegistryDialogState();
}

class _DinoV3RegistryDialogState extends State<DinoV3RegistryDialog> {
  final _commonNameController = TextEditingController();
  final _scientificNameController = TextEditingController();
  List<DinoV3RegistryEntry> _entries = const [];
  DinoV3RegistryEntry? _selected;
  bool _loading = false;
  bool _saving = false;
  bool _examplesLoading = false;
  int _examplesRequestId = 0;
  List<DinoV3RegistryEvent> _events = const <DinoV3RegistryEvent>[];
  Map<int, Uint8List> _exampleBytes = const <int, Uint8List>{};
  String? _error;

  @override
  void initState() {
    super.initState();
    _entries = widget.initialEntries ?? const [];
    if (_entries.isNotEmpty) _select(_entries.first, notify: false);
    if (widget.initialEntries == null) unawaited(_load());
  }

  @override
  void dispose() {
    _commonNameController.dispose();
    _scientificNameController.dispose();
    super.dispose();
  }

  void _select(DinoV3RegistryEntry entry, {bool notify = true}) {
    void update() {
      _selected = entry;
      _commonNameController.text = entry.commonName;
      _scientificNameController.text = entry.scientificName;
      _events = const <DinoV3RegistryEvent>[];
      _exampleBytes = const <int, Uint8List>{};
      _error = null;
    }

    if (notify) {
      setState(update);
    } else {
      update();
    }
    if (entry.isCheckpoint) {
      ++_examplesRequestId;
      if (notify && mounted) setState(() => _examplesLoading = false);
    } else {
      unawaited(_loadExamples(entry));
    }
  }

  Future<void> _load() async {
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

  void _replaceEntry(DinoV3RegistryEntry entry) {
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

  Future<void> _saveIdentity() async {
    final selected = _selected;
    if (selected?.isCheckpoint == true) return;
    final commonName = _commonNameController.text.trim();
    if (selected == null || commonName.isEmpty) {
      setState(() => _error = '请填写人工确认物种名称。');
      return;
    }
    setState(() => _saving = true);
    try {
      final updated = await widget.apiClient.updateDinoV3RegistryIdentity(
        classificationModelPath: widget.modelPath,
        registrationId: selected.id,
        commonName: commonName,
        scientificName: _scientificNameController.text.trim(),
      );
      if (mounted) _replaceEntry(updated);
    } catch (error) {
      if (mounted) setState(() => _error = '保存物种名称失败：$error');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Future<void> _register() async {
    final selected = _selected;
    if (selected == null || !selected.canRegister) return;
    setState(() => _saving = true);
    try {
      final updated = await widget.apiClient.registerDinoV3Species(
        classificationModelPath: widget.modelPath,
        registrationId: selected.id,
      );
      if (mounted) _replaceEntry(updated);
    } catch (error) {
      if (mounted) setState(() => _error = '注册新物种失败：$error');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Future<void> _loadExamples(DinoV3RegistryEntry entry) async {
    if (entry.isCheckpoint) {
      ++_examplesRequestId;
      if (mounted && _selected?.id == entry.id) {
        setState(() {
          _events = const <DinoV3RegistryEvent>[];
          _exampleBytes = const <int, Uint8List>{};
          _examplesLoading = false;
        });
      }
      return;
    }
    final requestId = ++_examplesRequestId;
    if (mounted && _selected?.id == entry.id) {
      setState(() => _examplesLoading = true);
    }
    try {
      final events = await widget.apiClient.fetchDinoV3RegistryEvents(
        widget.modelPath,
        entry.id,
      );
      final bytes = <int, Uint8List>{};
      for (final event
          in events
              .where((event) => event.id > 0 && event.hasExample)
              .take(3)) {
        try {
          final data = await widget.apiClient.fetchDinoV3RegistryExample(
            widget.modelPath,
            entry.id,
            event.id,
          );
          if (data.isNotEmpty) bytes[event.id] = data;
        } catch (_) {
          // One stale source file must not hide the other representative crops.
        }
      }
      if (!mounted ||
          requestId != _examplesRequestId ||
          _selected?.id != entry.id) {
        return;
      }
      setState(() {
        _events = events;
        _exampleBytes = bytes;
        _examplesLoading = false;
      });
    } catch (_) {
      if (!mounted ||
          requestId != _examplesRequestId ||
          _selected?.id != entry.id) {
        return;
      }
      setState(() {
        _events = const <DinoV3RegistryEvent>[];
        _exampleBytes = const <int, Uint8List>{};
        _examplesLoading = false;
      });
    }
  }

  Future<void> _deleteSelected() async {
    final selected = _selected;
    if (selected == null || !selected.canDelete || _saving) return;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('删除“${selected.displayName}”？'),
        content: const Text(
          '将删除该物种的本地注册记录、事件关联和本地 prototypes。\n'
          '此操作不可撤销。\n\n'
          '历史 human-feedback / audit 数据将保留。',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('确认删除'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;

    setState(() {
      _saving = true;
      _error = null;
    });
    try {
      await widget.apiClient.deleteDinoV3RegistryEntry(
        widget.modelPath,
        selected.id,
      );
      if (!mounted) return;
      final remaining = _entries
          .where((entry) => entry.id != selected.id)
          .toList(growable: false);
      ++_examplesRequestId;
      setState(() {
        _entries = remaining;
        _selected = null;
        _events = const <DinoV3RegistryEvent>[];
        _exampleBytes = const <int, Uint8List>{};
        _examplesLoading = false;
        _commonNameController.clear();
        _scientificNameController.clear();
      });
      if (remaining.isNotEmpty) _select(remaining.first);
    } catch (error) {
      if (mounted) setState(() => _error = '删除物种失败：$error');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Widget _buildExampleGallery() {
    if (_selected?.isCheckpoint == true) {
      return SizedBox(
        height: 72,
        child: Align(
          alignment: Alignment.centerLeft,
          child: Text(
            'Checkpoint 不包含原始训练图片；暂无 Registry 裁切例图',
            style: TextStyle(
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
        ),
      );
    }
    if (_examplesLoading) {
      return const SizedBox(
        height: 104,
        child: Center(child: CircularProgressIndicator(strokeWidth: 2)),
      );
    }
    final examples = _events
        .where(
          (event) => event.hasExample && _exampleBytes.containsKey(event.id),
        )
        .take(3)
        .toList(growable: false);
    if (examples.isEmpty) {
      return SizedBox(
        height: 72,
        child: Align(
          alignment: Alignment.centerLeft,
          child: Text(
            '暂无裁切例图',
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
        itemCount: examples.length,
        separatorBuilder: (_, __) => const SizedBox(width: 10),
        itemBuilder: (context, index) {
          final event = examples[index];
          final bytes = _exampleBytes[event.id]!;
          return ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: SizedBox(
              width: 112,
              height: 112,
              child: Image.memory(
                bytes,
                key: ValueKey('dinov3-registry-example-${event.id}'),
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

  Future<void> _continueValidation() async {
    final selected = _selected;
    if (selected == null || selected.isCheckpoint) return;
    setState(() => _saving = true);
    try {
      final events = await widget.apiClient.fetchDinoV3RegistryEvents(
        widget.modelPath,
        selected.id,
      );
      final paths = events
          .map((event) => event.sourcePath.trim())
          .where((path) => path.isNotEmpty)
          .toSet();
      if (paths.isEmpty) {
        if (mounted) setState(() => _error = '该候选物种没有可用于继续验证的事件文件。');
        return;
      }
      if (!mounted) return;
      Navigator.of(context).maybePop();
      widget.onContinueValidation(paths);
    } catch (error) {
      if (mounted) setState(() => _error = '读取候选事件失败：$error');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Widget _conditionRow(String label, bool passed) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          Icon(
            passed ? Icons.check_circle_rounded : Icons.radio_button_unchecked,
            size: 18,
            color: passed
                ? Theme.of(context).colorScheme.primary
                : Theme.of(context).colorScheme.outline,
          ),
          const SizedBox(width: 8),
          Expanded(child: Text(label)),
        ],
      ),
    );
  }

  Widget _detail(DinoV3RegistryEntry entry) {
    final editable = entry.isCandidate && !entry.isCheckpoint;
    final conditions = entry.conditions;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Expanded(
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  entry.displayName,
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 4),
                Text(entry.isCheckpoint ? '状态：分类头基础物种' : '状态：${entry.status}'),
                if (entry.isCheckpoint) ...[
                  const SizedBox(height: 8),
                  Text(
                    '来自不可变 DINOv3 Checkpoint · ${entry.prototypeCount} 个 prototype。'
                    '分类头保存特征中心而非原始训练影像。',
                  ),
                ],
                const SizedBox(height: 16),
                TextField(
                  controller: _commonNameController,
                  enabled: editable && !_saving,
                  decoration: const InputDecoration(
                    labelText: '人工确认物种',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: _scientificNameController,
                  enabled: editable && !_saving,
                  decoration: const InputDecoration(
                    labelText: '学名',
                    border: OutlineInputBorder(),
                  ),
                ),
                if (editable) ...[
                  const SizedBox(height: 8),
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton(
                      onPressed: _saving ? null : _saveIdentity,
                      child: const Text('保存物种名称'),
                    ),
                  ),
                ],
                const SizedBox(height: 8),
                Text('${entry.eventCount} 个独立事件 · ${entry.cameraCount} 台相机'),
                Text(
                  'cluster purity ${entry.clusterPurity.toStringAsFixed(3)} · '
                  'embedding consistency ${entry.embeddingConsistency.toStringAsFixed(3)}',
                ),
                const SizedBox(height: 14),
                Text('裁切例图', style: Theme.of(context).textTheme.titleSmall),
                const SizedBox(height: 8),
                _buildExampleGallery(),
                if (!entry.isCheckpoint) ...[
                  const SizedBox(height: 14),
                  Text('注册条件', style: Theme.of(context).textTheme.titleSmall),
                  _conditionRow('≥5 个独立事件', conditions['events'] == true),
                  _conditionRow('≥2 台相机', conditions['cameras'] == true),
                  _conditionRow(
                    'cluster purity ≥ threshold',
                    conditions['cluster_purity'] == true,
                  ),
                  _conditionRow(
                    'embedding consistency ≥ threshold',
                    conditions['embedding_consistency'] == true,
                  ),
                  _conditionRow('已确认物种名称', conditions['identity'] == true),
                ],
                if (_error != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    _error!,
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.error,
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        Wrap(
          alignment: WrapAlignment.end,
          spacing: 8,
          runSpacing: 8,
          children: [
            if (entry.canDelete) ...[
              OutlinedButton.icon(
                onPressed: _saving ? null : _deleteSelected,
                icon: const Icon(Icons.delete_outline_rounded),
                label: const Text('删除物种'),
              ),
              const SizedBox(width: 8),
            ],
            if (!entry.isCheckpoint) ...[
              OutlinedButton.icon(
                onPressed: _saving ? null : _continueValidation,
                icon: const Icon(Icons.fact_check_outlined),
                label: const Text('继续验证'),
              ),
              const SizedBox(width: 8),
            ],
            FilledButton.icon(
              onPressed: !_saving && entry.canRegister ? _register : null,
              icon: const Icon(Icons.add_circle_outline_rounded),
              label: const Text('注册为新物种'),
            ),
          ],
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final selected = _selected;
    return AlertDialog(
      title: const Text('DINOv3 物种注册状态'),
      content: SizedBox(
        width: 860,
        height: 560,
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  SizedBox(
                    width: 270,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Text(dinov3RegistrySummary(_entries)),
                        const SizedBox(height: 8),
                        Expanded(
                          child: _entries.isEmpty
                              ? const Center(child: Text('暂无候选或已注册物种'))
                              : ListView.builder(
                                  itemCount: _entries.length,
                                  itemBuilder: (context, index) {
                                    final entry = _entries[index];
                                    return ListTile(
                                      selected: selected?.id == entry.id,
                                      title: Text(entry.displayName),
                                      subtitle: Text(
                                        entry.isCheckpoint
                                            ? '分类头基础物种 · ${entry.prototypeCount} prototypes'
                                            : '${entry.status} · ${entry.eventCount} 事件 · ${entry.cameraCount} 相机',
                                      ),
                                      onTap: () => _select(entry),
                                    );
                                  },
                                ),
                        ),
                      ],
                    ),
                  ),
                  const VerticalDivider(width: 24),
                  Expanded(
                    child: selected == null
                        ? Center(child: Text(_error ?? '选择一个候选物种查看详情'))
                        : _detail(selected),
                  ),
                ],
              ),
      ),
      actions: [
        TextButton(
          onPressed: _saving ? null : () => Navigator.of(context).maybePop(),
          child: const Text('关闭'),
        ),
      ],
    );
  }
}
