import 'package:flutter/material.dart';

import '../models/settings.dart';
import '../models/theme_settings.dart';
import '../widgets/section_card.dart';

const _defaultExportColumns = <String>[
  '文件名',
  '格式',
  '拍摄日期',
  '拍摄时间',
  '工作天数',
  '物种名称',
  '学名',
  '目名',
  '目拉丁名',
  '科名',
  '科拉丁名',
  '属名',
  '属拉丁名',
  '物种类型',
  '物种数量',
  '最低置信度',
  '独立探测首只',
  '备注',
];

const _defaultQuickMarks = <String>[
  '灰雁',
  '大白鹭',
  '绿头鸭',
  '赤嘴潜鸭',
  '赤麻鸭',
  '青头潜鸭',
  '马',
  '黑翅长脚鹬',
  '未知鸟',
];

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({
    required this.settings,
    required this.themeNotifier,
    required this.onUpdateTheme,
    required this.onSaveSettings,
    required this.onShowMessage,
    super.key,
  });

  final NeriSettings? settings;
  final ValueNotifier<ThemeSettings> themeNotifier;
  final ValueChanged<ThemeSettings> onUpdateTheme;
  final Future<void> Function(Map<String, dynamic> settings) onSaveSettings;
  final ValueChanged<String> onShowMessage;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _packageController = TextEditingController();
  final _newQuickMarkController = TextEditingController();
  Map<String, dynamic> _draft = <String, dynamic>{};
  int _sectionIndex = 0;
  bool _saving = false;
  String? _speciesToAdd;

  @override
  void initState() {
    super.initState();
    _resetDraft();
  }

  @override
  void didUpdateWidget(covariant SettingsScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.settings != widget.settings) {
      _resetDraft();
    }
  }

  @override
  void dispose() {
    _packageController.dispose();
    _newQuickMarkController.dispose();
    super.dispose();
  }

  void _resetDraft() {
    final settings = widget.settings;
    final saved = Map<String, dynamic>.from(
      settings?.settings ?? const <String, dynamic>{},
    );
    _draft = <String, dynamic>{
      ...saved,
      'selected_model': _stringSetting(
        saved,
        'selected_model',
        settings?.selectedModel ?? '',
      ),
      'selected_classification_model': _stringSetting(
        saved,
        'selected_classification_model',
        settings?.selectedClassificationModel ?? '',
      ),
      'confidence': _doubleSetting(saved, 'confidence', 0.25),
      'iou': _doubleSetting(saved, 'iou', 0.30),
      'batch_size': settings?.gpuAvailable == true
          ? _intSetting(saved, 'batch_size', 16)
          : 1,
      'use_fp16':
          settings?.gpuAvailable == true &&
          _boolSetting(saved, 'use_fp16', settings?.gpuAvailable == true),
      'video_mode': _stringSetting(saved, 'video_mode', 'all'),
      'vid_stride': _intSetting(saved, 'vid_stride', 1),
      'min_frame_ratio': _doubleSetting(saved, 'min_frame_ratio', 0.0),
      'auto_group': _boolSetting(saved, 'auto_group', true),
      'auto_sort': _boolSetting(saved, 'auto_sort', false),
      'quick_mark_list': _stringListSetting(
        saved,
        'quick_mark_list',
        _defaultQuickMarks,
      ),
      'export_columns': _stringListSetting(
        saved,
        'export_columns',
        _defaultExportColumns,
      ),
      'selected_species_names': _stringListSetting(
        saved,
        'selected_species_names',
        const <String>[],
      ),
      'update_channel': _stringSetting(saved, 'update_channel', 'Preview'),
      'update_mirror': _stringSetting(saved, 'update_mirror', 'KKGitHub'),
    };
    _packageController.text = _stringSetting(saved, 'package', '');
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        SizedBox(
          width: 260,
          child: NavigationDrawer(
            selectedIndex: _sectionIndex,
            onDestinationSelected: (index) {
              setState(() => _sectionIndex = index);
            },
            children: const [
              Padding(
                padding: EdgeInsets.fromLTRB(28, 20, 16, 12),
                child: Text('设置', style: TextStyle(fontSize: 18)),
              ),
              NavigationDrawerDestination(
                icon: Icon(Icons.memory_outlined),
                selectedIcon: Icon(Icons.memory_rounded),
                label: Text('模型参数'),
              ),
              NavigationDrawerDestination(
                icon: Icon(Icons.movie_filter_outlined),
                selectedIcon: Icon(Icons.movie_filter_rounded),
                label: Text('视频检测'),
              ),
              NavigationDrawerDestination(
                icon: Icon(Icons.construction_outlined),
                selectedIcon: Icon(Icons.construction_rounded),
                label: Text('环境维护'),
              ),
              NavigationDrawerDestination(
                icon: Icon(Icons.fact_check_outlined),
                selectedIcon: Icon(Icons.fact_check_rounded),
                label: Text('基础设置'),
              ),
              NavigationDrawerDestination(
                icon: Icon(Icons.bookmark_add_outlined),
                selectedIcon: Icon(Icons.bookmark_add_rounded),
                label: Text('快速标记'),
              ),
              NavigationDrawerDestination(
                icon: Icon(Icons.table_chart_outlined),
                selectedIcon: Icon(Icons.table_chart_rounded),
                label: Text('导出设置'),
              ),
              NavigationDrawerDestination(
                icon: Icon(Icons.palette_outlined),
                selectedIcon: Icon(Icons.palette_rounded),
                label: Text('外观主题'),
              ),
              NavigationDrawerDestination(
                icon: Icon(Icons.system_update_alt_outlined),
                selectedIcon: Icon(Icons.system_update_alt_rounded),
                label: Text('软件更新'),
              ),
              NavigationDrawerDestination(
                icon: Icon(Icons.info_outline_rounded),
                selectedIcon: Icon(Icons.info_rounded),
                label: Text('项目信息'),
              ),
            ],
          ),
        ),
        const VerticalDivider(width: 1),
        Expanded(
          child: ListView(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: const EdgeInsets.all(16),
            children: [
              _buildSaveStrip(),
              _buildSelectedSection(),
              const SizedBox(height: 8),
              Align(
                alignment: Alignment.centerRight,
                child: FilledButton.icon(
                  onPressed: _saving ? null : _save,
                  icon: _saving
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.save_rounded),
                  label: Text(_saving ? '保存中' : '保存设置'),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildSelectedSection() {
    return switch (_sectionIndex) {
      0 => _buildModelParameters(),
      1 => _buildVideoSettings(),
      2 => _buildEnvironmentMaintenance(),
      3 => _buildBasicSettings(),
      4 => _buildQuickMarkSettings(),
      5 => _buildExportSettings(),
      6 => _buildAppearanceSettings(),
      7 => _buildUpdateSettingsSection(),
      _ => _buildProjectSettings(),
    };
  }

  Widget _buildSaveStrip() {
    return Card.outlined(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            const Icon(Icons.tune_rounded),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                widget.settings == null
                    ? '尚未读取到后端配置，部分选项会使用默认值。'
                    : '高级设置会保存到 Python 后端的配置文件，并在后续处理任务中复用。',
              ),
            ),
            const SizedBox(width: 12),
            FilledButton.icon(
              onPressed: _saving ? null : _save,
              icon: const Icon(Icons.save_rounded),
              label: const Text('保存'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModelParameters() {
    final settings = widget.settings;
    final selectedModel = _validModelValue(
      _string('selected_model'),
      settings?.availableModels ?? const <ModelInfo>[],
    );
    final selectedClassificationModel = _validModelValue(
      _string('selected_classification_model'),
      settings?.availableClassificationModels ?? const <ModelInfo>[],
      allowEmpty: true,
    );

    return SectionCard(
      title: '模型参数设置',
      subtitle: '模型管理、识别物种、阈值和推理加速',
      icon: Icons.memory_rounded,
      child: Column(
        children: [
          _SettingsPanel(
            title: '模型管理',
            subtitle: '选择探测模型和二次分类模型',
            icon: Icons.build_rounded,
            child: Column(
              children: [
                DropdownButtonFormField<String>(
                  key: ValueKey('model-$selectedModel'),
                  initialValue: selectedModel,
                  isExpanded: true,
                  decoration: const InputDecoration(
                    labelText: '选择探测模型',
                    border: OutlineInputBorder(),
                  ),
                  items: (settings?.availableModels ?? const <ModelInfo>[])
                      .map(
                        (model) => DropdownMenuItem(
                          value: model.path,
                          child: Text(model.name),
                        ),
                      )
                      .toList(),
                  onChanged: (value) => _set('selected_model', value ?? ''),
                ),
                const SizedBox(height: 12),
                DropdownButtonFormField<String>(
                  key: ValueKey('cls-model-$selectedClassificationModel'),
                  initialValue: selectedClassificationModel,
                  isExpanded: true,
                  decoration: const InputDecoration(
                    labelText: '选择分类模型',
                    border: OutlineInputBorder(),
                  ),
                  items: [
                    const DropdownMenuItem(value: '', child: Text('不使用')),
                    ...(settings?.availableClassificationModels ??
                            const <ModelInfo>[])
                        .map(
                          (model) => DropdownMenuItem(
                            value: model.path,
                            child: Text(model.name),
                          ),
                        ),
                  ],
                  onChanged: (value) =>
                      _set('selected_classification_model', value ?? ''),
                ),
                const SizedBox(height: 8),
                _MutedText(
                  '分类模型目录：${settings?.classificationModelDirectory ?? 'res/model_cls'}',
                ),
              ],
            ),
          ),
          _SettingsPanel(
            title: '识别物种设置',
            subtitle: '不选择时默认识别全部物种',
            icon: Icons.pets_rounded,
            child: _buildSpeciesSelector(),
          ),
          _SettingsPanel(
            title: '检测阈值设置',
            subtitle: '控制目标检测的置信度和 IOU 阈值',
            icon: Icons.speed_rounded,
            child: Column(
              children: [
                _LabeledSlider(
                  label: '置信度',
                  value: _double('confidence'),
                  min: 0.05,
                  max: 0.95,
                  divisions: 90,
                  onChanged: (value) => _set('confidence', value),
                ),
                _LabeledSlider(
                  label: 'IOU',
                  value: _double('iou'),
                  min: 0.10,
                  max: 0.90,
                  divisions: 80,
                  onChanged: (value) => _set('iou', value),
                ),
              ],
            ),
          ),
          _SettingsPanel(
            title: '模型加速选项',
            subtitle: 'CPU 模式会锁定 batch=1 并禁用 FP16',
            icon: Icons.bolt_rounded,
            child: Column(
              children: [
                _LabeledSlider(
                  label: '批处理大小',
                  value: _int('batch_size').toDouble(),
                  min: 1,
                  max: 32,
                  divisions: 31,
                  valueLabel: _int('batch_size').toString(),
                  enabled: settings?.gpuAvailable == true,
                  onChanged: (value) => _set('batch_size', value.round()),
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  value: _bool('use_fp16') && settings?.gpuAvailable == true,
                  onChanged: settings?.gpuAvailable == true
                      ? (value) => _set('use_fp16', value)
                      : null,
                  title: const Text('使用 FP16 加速'),
                  subtitle: Text(
                    settings?.gpuAvailable == true
                        ? '需要支持半精度推理的 GPU'
                        : '当前为 CPU 模式，已禁用 FP16 并锁定 batch=1',
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSpeciesSelector() {
    final speciesNames =
        (widget.settings?.speciesTypes.keys.toList() ?? const <String>[])
          ..sort();
    final selected = _stringList('selected_species_names');

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: DropdownButtonFormField<String>(
                key: ValueKey('species-add-${selected.length}-$_speciesToAdd'),
                initialValue: speciesNames.contains(_speciesToAdd)
                    ? _speciesToAdd
                    : null,
                isExpanded: true,
                decoration: const InputDecoration(
                  labelText: '添加识别物种',
                  border: OutlineInputBorder(),
                ),
                items: speciesNames
                    .map(
                      (name) =>
                          DropdownMenuItem(value: name, child: Text(name)),
                    )
                    .toList(),
                onChanged: (value) => setState(() => _speciesToAdd = value),
              ),
            ),
            const SizedBox(width: 10),
            FilledButton.icon(
              onPressed: _speciesToAdd == null
                  ? null
                  : () {
                      final next = {...selected, _speciesToAdd!}.toList();
                      _set('selected_species_names', next);
                      setState(() => _speciesToAdd = null);
                    },
              icon: const Icon(Icons.add_rounded),
              label: const Text('添加'),
            ),
            const SizedBox(width: 8),
            TextButton(
              onPressed: selected.isEmpty
                  ? null
                  : () => _set('selected_species_names', <String>[]),
              child: const Text('全部识别'),
            ),
          ],
        ),
        const SizedBox(height: 12),
        if (selected.isEmpty)
          const _MutedText('当前未限制物种，处理时默认识别全部模型类别。')
        else
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: selected.map((name) {
              return InputChip(
                label: Text(name),
                onDeleted: () {
                  final next = selected.where((item) => item != name).toList();
                  _set('selected_species_names', next);
                },
              );
            }).toList(),
          ),
      ],
    );
  }

  Widget _buildVideoSettings() {
    final videoMode = _string('video_mode', 'all');
    final strideLabel = videoMode == 'all' ? '帧间隔' : '快速识别帧数';

    return SectionCard(
      title: '视频检测设置',
      subtitle: '视频处理模式、跳帧和检测过滤',
      icon: Icons.movie_filter_rounded,
      child: Column(
        children: [
          _SettingsPanel(
            title: '视频处理模式',
            subtitle: '全部识别会按帧间隔处理，快速识别只抽取关键帧',
            icon: Icons.play_circle_outline_rounded,
            child: SegmentedButton<String>(
              segments: const [
                ButtonSegment(
                  value: 'all',
                  icon: Icon(Icons.video_collection_rounded),
                  label: Text('全部识别'),
                ),
                ButtonSegment(
                  value: 'fast',
                  icon: Icon(Icons.flash_on_rounded),
                  label: Text('快速识别'),
                ),
              ],
              selected: {videoMode},
              onSelectionChanged: (selection) =>
                  _set('video_mode', selection.first),
            ),
          ),
          _SettingsPanel(
            title: '跳帧设置',
            subtitle: videoMode == 'all' ? '全部识别模式下表示帧间隔' : '快速识别模式下表示抽取检测的帧数',
            icon: Icons.skip_next_rounded,
            child: _LabeledSlider(
              label: strideLabel,
              value: _int('vid_stride').toDouble(),
              min: 1,
              max: 30,
              divisions: 29,
              valueLabel: _int('vid_stride').toString(),
              onChanged: (value) => _set('vid_stride', value.round()),
            ),
          ),
          _SettingsPanel(
            title: '检测过滤',
            subtitle: '设置检测到的最低帧数比例',
            icon: Icons.filter_alt_rounded,
            child: _LabeledSlider(
              label: '最低帧数比例',
              value: _double('min_frame_ratio'),
              min: 0,
              max: 0.30,
              divisions: 30,
              valueLabel: '${(_double('min_frame_ratio') * 100).round()}%',
              onChanged: (value) => _set('min_frame_ratio', value),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEnvironmentMaintenance() {
    return SectionCard(
      title: '环境维护',
      subtitle: 'PyTorch 与单个 Python 包维护入口',
      icon: Icons.construction_rounded,
      child: Column(
        children: [
          _SettingsPanel(
            title: '安装 PyTorch',
            subtitle: '选择安装环境后执行安装或修复',
            icon: Icons.download_rounded,
            child: Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    key: ValueKey('pytorch-${_string('pytorch_version')}'),
                    initialValue: _string('pytorch_version', '自动检测'),
                    decoration: const InputDecoration(
                      labelText: '安装环境',
                      border: OutlineInputBorder(),
                    ),
                    items: const [
                      DropdownMenuItem(value: '自动检测', child: Text('自动检测')),
                      DropdownMenuItem(
                        value: 'CUDA 13.0',
                        child: Text('CUDA 13.0'),
                      ),
                      DropdownMenuItem(
                        value: 'CUDA 12.8',
                        child: Text('CUDA 12.8'),
                      ),
                      DropdownMenuItem(
                        value: 'CUDA 12.6',
                        child: Text('CUDA 12.6'),
                      ),
                      DropdownMenuItem(
                        value: 'CUDA 12.4',
                        child: Text('CUDA 12.4'),
                      ),
                      DropdownMenuItem(
                        value: 'CUDA 12.1',
                        child: Text('CUDA 12.1'),
                      ),
                      DropdownMenuItem(
                        value: 'CUDA 11.8',
                        child: Text('CUDA 11.8'),
                      ),
                      DropdownMenuItem(
                        value: 'Intel XPU',
                        child: Text('Intel XPU'),
                      ),
                      DropdownMenuItem(
                        value: 'CPU Only',
                        child: Text('CPU Only'),
                      ),
                    ],
                    onChanged: (value) =>
                        _set('pytorch_version', value ?? '自动检测'),
                  ),
                ),
                const SizedBox(width: 10),
                FilledButton(
                  onPressed: () => widget.onShowMessage(
                    'Flutter 端暂未直接执行 PyTorch 安装，请在 Python GUI 或命令行中操作。',
                  ),
                  child: const Text('安装'),
                ),
              ],
            ),
          ),
          _SettingsPanel(
            title: '重新安装单个 Python 包',
            subtitle: '包名可留空备用',
            icon: Icons.extension_rounded,
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _packageController,
                    decoration: const InputDecoration(
                      labelText: '包名',
                      hintText: '例如 numpy',
                      border: OutlineInputBorder(),
                    ),
                    onChanged: (value) => _draft['package'] = value.trim(),
                  ),
                ),
                const SizedBox(width: 10),
                FilledButton(
                  onPressed: () {
                    final package = _packageController.text.trim();
                    widget.onShowMessage(
                      package.isEmpty
                          ? '已保留 Python 包维护入口，输入包名后可用于后续安装。'
                          : 'Flutter 端暂未直接执行 pip 安装：$package',
                    );
                  },
                  child: const Text('安装'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBasicSettings() {
    return SectionCard(
      title: '基础设置',
      subtitle: '控制物种校验界面的辅助行为',
      icon: Icons.fact_check_rounded,
      child: Column(
        children: [
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            value: _bool('auto_group'),
            onChanged: (value) => _set('auto_group', value),
            title: const Text('物种校验界面自动分组'),
            subtitle: const Text(
              '开启后，将根据视频文件自动对连续拍摄的照片和视频进行分组，并在文件列表中按所在组校验最多的物种统一归类显示。',
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuickMarkSettings() {
    return SectionCard(
      title: '快速标记设置',
      subtitle: '自动排序或手动增减快速标记物种',
      icon: Icons.bookmark_add_rounded,
      child: _buildQuickMarkEditor(),
    );
  }

  Widget _buildExportSettings() {
    return SectionCard(
      title: '导出设置',
      subtitle: '自定义导出表格中的列',
      icon: Icons.table_chart_rounded,
      child: _buildExportColumns(),
    );
  }

  Widget _buildAppearanceSettings() {
    return SectionCard(
      title: '外观与主题',
      subtitle: '主题模式、动态颜色和调色板',
      icon: Icons.palette_rounded,
      child: _AppearanceControls(
        themeNotifier: widget.themeNotifier,
        onUpdateTheme: widget.onUpdateTheme,
      ),
    );
  }

  Widget _buildUpdateSettingsSection() {
    return SectionCard(
      title: '软件更新',
      subtitle: '选择更新通道和镜像源',
      icon: Icons.system_update_alt_rounded,
      child: _buildUpdateSettings(),
    );
  }

  Widget _buildProjectSettings() {
    return SectionCard(
      title: '项目信息',
      subtitle: '后端版本与媒体格式',
      icon: Icons.info_rounded,
      child: _ProjectInfo(settings: widget.settings),
    );
  }

  Widget _buildQuickMarkEditor() {
    final marks = _stringList('quick_mark_list');
    return Column(
      children: [
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          value: _bool('auto_sort'),
          onChanged: (value) => _set('auto_sort', value),
          title: const Text('自动排序'),
        ),
        const Divider(),
        ...List.generate(marks.length, (index) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              children: [
                SizedBox(width: 36, child: Text('${index + 1}')),
                Expanded(
                  child: TextFormField(
                    key: ValueKey('quick-${marks[index]}-$index'),
                    initialValue: marks[index],
                    decoration: const InputDecoration(
                      isDense: true,
                      border: OutlineInputBorder(),
                    ),
                    onChanged: (value) {
                      final next = _stringList('quick_mark_list');
                      next[index] = value.trim();
                      _draft['quick_mark_list'] = next;
                    },
                  ),
                ),
                IconButton(
                  tooltip: '删除',
                  onPressed: () {
                    final next = _stringList('quick_mark_list')
                      ..removeAt(index);
                    _set('quick_mark_list', next);
                  },
                  icon: const Icon(Icons.delete_outline_rounded),
                ),
              ],
            ),
          );
        }),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _newQuickMarkController,
                decoration: const InputDecoration(
                  labelText: '新增快速标记',
                  border: OutlineInputBorder(),
                ),
                onSubmitted: (_) => _addQuickMark(),
              ),
            ),
            const SizedBox(width: 10),
            FilledButton.icon(
              onPressed: _addQuickMark,
              icon: const Icon(Icons.add_rounded),
              label: const Text('新增'),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildExportColumns() {
    final selected = _stringList('export_columns').toSet();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 8,
          children: [
            TextButton.icon(
              onPressed: () => _set('export_columns', _defaultExportColumns),
              icon: const Icon(Icons.select_all_rounded),
              label: const Text('全选'),
            ),
            TextButton.icon(
              onPressed: () => _set('export_columns', <String>[]),
              icon: const Icon(Icons.deselect_rounded),
              label: const Text('全不选'),
            ),
          ],
        ),
        Wrap(
          spacing: 8,
          runSpacing: 4,
          children: _defaultExportColumns.map((column) {
            return SizedBox(
              width: 180,
              child: CheckboxListTile(
                dense: true,
                contentPadding: EdgeInsets.zero,
                value: selected.contains(column),
                onChanged: (value) {
                  final next = selected.toSet();
                  if (value == true) {
                    next.add(column);
                  } else {
                    next.remove(column);
                  }
                  _set('export_columns', next.toList());
                },
                title: Text(column),
              ),
            );
          }).toList(),
        ),
      ],
    );
  }

  Widget _buildUpdateSettings() {
    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: DropdownButtonFormField<String>(
                key: ValueKey('channel-${_string('update_channel')}'),
                initialValue: _string('update_channel', 'Preview'),
                decoration: const InputDecoration(
                  labelText: '更新通道',
                  border: OutlineInputBorder(),
                ),
                items: const [
                  DropdownMenuItem(value: 'Release', child: Text('稳定版')),
                  DropdownMenuItem(value: 'Preview', child: Text('预览版')),
                ],
                onChanged: (value) =>
                    _set('update_channel', value ?? 'Preview'),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: DropdownButtonFormField<String>(
                key: ValueKey('mirror-${_string('update_mirror')}'),
                initialValue: _string('update_mirror', 'KKGitHub'),
                decoration: const InputDecoration(
                  labelText: '镜像源',
                  border: OutlineInputBorder(),
                ),
                items: const [
                  DropdownMenuItem(value: 'Official', child: Text('官方源')),
                  DropdownMenuItem(value: 'KKGitHub', child: Text('国内源')),
                ],
                onChanged: (value) =>
                    _set('update_mirror', value ?? 'KKGitHub'),
              ),
            ),
            const SizedBox(width: 10),
            FilledButton(
              onPressed: () => widget.onShowMessage('软件更新检查入口已保留，后续可接入更新服务。'),
              child: const Text('检查更新'),
            ),
          ],
        ),
      ],
    );
  }

  void _addQuickMark() {
    final value = _newQuickMarkController.text.trim();
    if (value.isEmpty) return;
    final next = _stringList('quick_mark_list')..add(value);
    _newQuickMarkController.clear();
    _set('quick_mark_list', next);
  }

  Future<void> _save() async {
    setState(() => _saving = true);
    try {
      _draft['package'] = _packageController.text.trim();
      if (widget.settings?.gpuAvailable != true) {
        _draft['batch_size'] = 1;
        _draft['use_fp16'] = false;
      }
      await widget.onSaveSettings(Map<String, dynamic>.from(_draft));
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  void _set(String key, dynamic value) {
    setState(() => _draft[key] = value);
  }

  bool _bool(String key, [bool fallback = false]) {
    return _boolSetting(_draft, key, fallback);
  }

  double _double(String key, [double fallback = 0]) {
    return _doubleSetting(_draft, key, fallback);
  }

  int _int(String key, [int fallback = 0]) {
    return _intSetting(_draft, key, fallback);
  }

  String _string(String key, [String fallback = '']) {
    return _stringSetting(_draft, key, fallback);
  }

  List<String> _stringList(String key) {
    return _stringListSetting(_draft, key, const <String>[]);
  }

  String? _validModelValue(
    String value,
    List<ModelInfo> models, {
    bool allowEmpty = false,
  }) {
    if (allowEmpty && value.isEmpty) return '';
    if (models.any((model) => model.path == value)) return value;
    if (allowEmpty) return '';
    return models.isEmpty ? null : models.first.path;
  }
}

class _SettingsPanel extends StatelessWidget {
  const _SettingsPanel({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.child,
  });

  final String title;
  final String subtitle;
  final IconData icon;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: Theme.of(context).textTheme.titleMedium),
                    Text(
                      subtitle,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Theme.of(context).colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          child,
        ],
      ),
    );
  }
}

class _LabeledSlider extends StatelessWidget {
  const _LabeledSlider({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.onChanged,
    this.valueLabel,
    this.enabled = true,
  });

  final String label;
  final double value;
  final double min;
  final double max;
  final int divisions;
  final ValueChanged<double> onChanged;
  final String? valueLabel;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    final display = valueLabel ?? value.toStringAsFixed(2);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(width: 110, child: Text(label)),
          Expanded(
            child: Slider(
              value: value.clamp(min, max).toDouble(),
              min: min,
              max: max,
              divisions: divisions,
              label: display,
              onChanged: enabled ? onChanged : null,
            ),
          ),
          SizedBox(width: 56, child: Text(display, textAlign: TextAlign.end)),
        ],
      ),
    );
  }
}

class _AppearanceControls extends StatelessWidget {
  const _AppearanceControls({
    required this.themeNotifier,
    required this.onUpdateTheme,
  });

  final ValueNotifier<ThemeSettings> themeNotifier;
  final ValueChanged<ThemeSettings> onUpdateTheme;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<ThemeSettings>(
      valueListenable: themeNotifier,
      builder: (context, settings, _) {
        final colorScheme = Theme.of(context).colorScheme;
        final paletteDisabled = settings.useDynamicColor;
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SegmentedButton<ThemeMode>(
              segments: const [
                ButtonSegment(
                  value: ThemeMode.light,
                  icon: Icon(Icons.light_mode_rounded),
                  label: Text('浅色'),
                ),
                ButtonSegment(
                  value: ThemeMode.dark,
                  icon: Icon(Icons.dark_mode_rounded),
                  label: Text('深色'),
                ),
                ButtonSegment(
                  value: ThemeMode.system,
                  icon: Icon(Icons.brightness_auto_rounded),
                  label: Text('自动'),
                ),
              ],
              selected: {settings.themeMode},
              onSelectionChanged: (selection) =>
                  onUpdateTheme(settings.copyWith(themeMode: selection.first)),
            ),
            const SizedBox(height: 12),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              value: settings.useDynamicColor,
              onChanged: (value) =>
                  onUpdateTheme(settings.copyWith(useDynamicColor: value)),
              title: const Text('使用系统动态颜色'),
            ),
            const SizedBox(height: 8),
            Text(
              paletteDisabled ? '调色板已由系统接管' : '调色板',
              style: Theme.of(context).textTheme.labelLarge?.copyWith(
                color: paletteDisabled
                    ? colorScheme.onSurface.withValues(alpha: 0.38)
                    : null,
              ),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: kSeedColorOptions.map((option) {
                final selected =
                    !paletteDisabled &&
                    settings.seedColor.toARGB32() == option.color.toARGB32();
                return Tooltip(
                  message: option.label,
                  child: _ColorSwatch(
                    color: option.color,
                    selected: selected,
                    disabled: paletteDisabled,
                    onTap: paletteDisabled
                        ? null
                        : () => onUpdateTheme(
                            settings.copyWith(seedColor: option.color),
                          ),
                  ),
                );
              }).toList(),
            ),
          ],
        );
      },
    );
  }
}

class _ProjectInfo extends StatelessWidget {
  const _ProjectInfo({required this.settings});

  final NeriSettings? settings;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.zero,
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        borderRadius: BorderRadius.circular(8),
      ),
      child: ListTile(
        leading: const Icon(Icons.info_outline_rounded),
        title: const Text('项目配置'),
        subtitle: Text(
          settings == null
              ? '未读取到后端配置'
              : '版本 ${settings!.appVersion} · 图像 ${settings!.supportedImageExtensions.join(' ')} · 视频 ${settings!.supportedVideoExtensions.join(' ')}',
        ),
      ),
    );
  }
}

class _MutedText extends StatelessWidget {
  const _MutedText(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Text(
        text,
        style: Theme.of(context).textTheme.bodySmall?.copyWith(
          color: Theme.of(context).colorScheme.onSurfaceVariant,
        ),
      ),
    );
  }
}

class _ColorSwatch extends StatelessWidget {
  const _ColorSwatch({
    required this.color,
    required this.selected,
    required this.disabled,
    required this.onTap,
  });

  final Color color;
  final bool selected;
  final bool disabled;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = ColorScheme.fromSeed(
      seedColor: color,
      brightness: Theme.of(context).brightness,
    );

    return GestureDetector(
      onTap: onTap,
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 200),
        opacity: disabled ? 0.35 : 1.0,
        child: Container(
          width: 72,
          height: 72,
          decoration: BoxDecoration(
            color: Theme.of(
              context,
            ).colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
            borderRadius: BorderRadius.circular(16),
          ),
          alignment: Alignment.center,
          child: SizedBox(
            width: 46,
            height: 46,
            child: ClipOval(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Column(
                    children: [
                      Expanded(
                        child: Container(color: scheme.primaryContainer),
                      ),
                      Expanded(
                        child: Row(
                          children: [
                            Expanded(
                              child: Container(color: scheme.tertiaryContainer),
                            ),
                            Expanded(child: Container(color: scheme.primary)),
                          ],
                        ),
                      ),
                    ],
                  ),
                  if (selected)
                    Center(
                      child: Icon(
                        Icons.check_rounded,
                        color:
                            ThemeData.estimateBrightnessForColor(color) ==
                                Brightness.dark
                            ? Colors.white
                            : Colors.black,
                        size: 24,
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

bool _boolSetting(Map<String, dynamic> values, String key, bool fallback) {
  final value = values[key];
  if (value is bool) return value;
  if (value is String) return value.toLowerCase() == 'true';
  return fallback;
}

double _doubleSetting(
  Map<String, dynamic> values,
  String key,
  double fallback,
) {
  final value = values[key];
  if (value is num) return value.toDouble();
  if (value is String) return double.tryParse(value) ?? fallback;
  return fallback;
}

int _intSetting(Map<String, dynamic> values, String key, int fallback) {
  final value = values[key];
  if (value is int) return value;
  if (value is num) return value.round();
  if (value is String) return int.tryParse(value) ?? fallback;
  return fallback;
}

String _stringSetting(
  Map<String, dynamic> values,
  String key,
  String fallback,
) {
  final value = values[key];
  if (value is String) return value;
  return fallback;
}

List<String> _stringListSetting(
  Map<String, dynamic> values,
  String key,
  List<String> fallback,
) {
  final value = values[key];
  if (value is List) {
    return value.map((item) => item.toString()).toList();
  }
  return List<String>.from(fallback);
}
