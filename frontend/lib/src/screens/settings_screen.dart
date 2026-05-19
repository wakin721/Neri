import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../api_client.dart';
import '../models/settings.dart';
import '../models/theme_settings.dart';
import '../widgets/app_menu_style.dart';
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
    required this.apiClient,
    required this.themeNotifier,
    required this.onUpdateTheme,
    required this.onSaveSettings,
    required this.onShowMessage,
    super.key,
  });

  final NeriSettings? settings;
  final NeriApiClient apiClient;
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
  bool _dirty = false;
  bool _resettingDraft = false;
  bool _loadingModelClasses = false;
  bool _installingPytorch = false;
  bool _reinstallingPackage = false;
  Timer? _maintenanceTimer;
  String? _maintenanceOperation;
  String? _maintenanceMessage;
  String? _modelClassesPath;
  List<ModelClassInfo> _modelClassOptions = const <ModelClassInfo>[];

  @override
  void initState() {
    super.initState();
    _packageController.addListener(_handlePackageChanged);
    _resetDraft();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) _loadModelClassesForSelection();
    });
  }

  @override
  void didUpdateWidget(covariant SettingsScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.settings != widget.settings && !_dirty && !_saving) {
      _resetDraft();
      _loadModelClassesForSelection();
    }
  }

  @override
  void dispose() {
    _maintenanceTimer?.cancel();
    _packageController.removeListener(_handlePackageChanged);
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
      'package_source': _stringSetting(saved, 'package_source', 'official'),
      'confidence': _doubleSetting(saved, 'confidence', 0.25),
      'iou': _doubleSetting(saved, 'iou', 0.30),
      'batch_size': settings?.gpuAvailable == true
          ? _intSetting(saved, 'batch_size', 16)
          : 1,
      'use_fp16':
          settings?.gpuAvailable == true &&
          _boolSetting(saved, 'use_fp16', settings?.gpuAvailable == true),
      'use_augment': _boolSetting(saved, 'use_augment', true),
      'use_agnostic_nms': _boolSetting(saved, 'use_agnostic_nms', true),
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
    _resettingDraft = true;
    _packageController.text = _stringSetting(saved, 'package', '');
    _resettingDraft = false;
    _dirty = false;
  }

  void _handlePackageChanged() {
    if (_resettingDraft) return;
    _draft['package'] = _packageController.text.trim();
    _markDirty();
  }

  void _markDirty() {
    if (_dirty || !mounted) return;
    setState(() => _dirty = true);
  }

  Future<void> _loadModelClassesForSelection([String? modelPath]) async {
    final path =
        modelPath ??
        _validModelValue(
          _string('selected_model'),
          widget.settings?.availableModels ?? const <ModelInfo>[],
        );
    if (path == null || path.isEmpty) {
      if (!mounted) return;
      setState(() {
        _modelClassesPath = null;
        _modelClassOptions = const <ModelClassInfo>[];
        _loadingModelClasses = false;
      });
      return;
    }
    if (_modelClassesPath == path && _modelClassOptions.isNotEmpty) return;

    setState(() {
      _modelClassesPath = path;
      _loadingModelClasses = true;
    });

    try {
      final classes = await widget.apiClient.fetchModelClasses(path);
      if (!mounted || _string('selected_model') != path) return;
      setState(() {
        _modelClassOptions = classes;
        _loadingModelClasses = false;
      });
    } catch (error) {
      if (!mounted || _string('selected_model') != path) return;
      setState(() {
        _modelClassOptions = const <ModelClassInfo>[];
        _loadingModelClasses = false;
      });
      widget.onShowMessage('读取模型物种列表失败：$error');
    }
  }

  Future<void> _installPytorch() async {
    final envChoice = _string('pytorch_version', '自动检测');
    final packageSource = _string('package_source', 'official');
    final confirmed = await _confirmMaintenance(
      title: '安装 PyTorch',
      message:
          '将先退出当前 Python 后端，然后使用 toolkit\\python.exe 重新安装适用于 $envChoice 的 PyTorch。\n\n安装完成后 Python 后端会自动重启，期间界面可能短暂显示后端离线。',
      confirmLabel: '开始安装',
    );
    if (confirmed != true) return;

    setState(() {
      _installingPytorch = true;
      _maintenanceOperation = 'install_pytorch';
      _maintenanceMessage = '正在启动 PyTorch 安装...';
    });
    try {
      final response = await widget.apiClient.installPytorch(
        envChoice,
        packageSource: packageSource,
      );
      if (!mounted) return;
      _startMaintenanceWatch(
        operation: response.operation.isEmpty
            ? 'install_pytorch'
            : response.operation,
        message: response.message,
      );
      widget.onShowMessage(response.message);
    } catch (error) {
      if (mounted) {
        setState(() {
          _installingPytorch = false;
          _maintenanceOperation = null;
          _maintenanceMessage = null;
        });
      }
      widget.onShowMessage('启动 PyTorch 安装失败：$error');
    }
  }

  Future<void> _reinstallPythonPackage() async {
    final packageSpec = _packageController.text.trim();
    final packageSource = _string('package_source', 'official');
    if (packageSpec.isEmpty) {
      widget.onShowMessage('请输入要重新安装的 Python 包名。');
      return;
    }

    final confirmed = await _confirmMaintenance(
      title: '重新安装 Python 包',
      message:
          '将使用 toolkit\\python.exe 对 $packageSpec 执行强制重新安装。\n\n安装完成后 Python 后端会自动重启，期间界面可能短暂显示后端离线。',
      confirmLabel: '开始安装',
    );
    if (confirmed != true) return;

    setState(() {
      _reinstallingPackage = true;
      _maintenanceOperation = 'reinstall_package';
      _maintenanceMessage = '正在启动 Python 包重新安装...';
    });
    try {
      final response = await widget.apiClient.reinstallPythonPackage(
        packageSpec,
        packageSource,
      );
      if (!mounted) return;
      _startMaintenanceWatch(
        operation: response.operation.isEmpty
            ? 'reinstall_package'
            : response.operation,
        message: response.message,
      );
      widget.onShowMessage(response.message);
    } catch (error) {
      if (mounted) {
        setState(() {
          _reinstallingPackage = false;
          _maintenanceOperation = null;
          _maintenanceMessage = null;
        });
      }
      widget.onShowMessage('启动 Python 包重新安装失败：$error');
    }
  }

  void _startMaintenanceWatch({
    required String operation,
    required String message,
  }) {
    _maintenanceTimer?.cancel();
    setState(() {
      _maintenanceOperation = operation;
      _maintenanceMessage = message.isEmpty ? '正在安装...' : message;
      _installingPytorch =
          operation == 'install_pytorch' ||
          operation == 'install_yolo_dependencies';
      _reinstallingPackage = operation == 'reinstall_package';
    });
    _maintenanceTimer = Timer.periodic(
      const Duration(seconds: 2),
      (_) => unawaited(_pollMaintenanceStatus()),
    );
    unawaited(_pollMaintenanceStatus());
  }

  Future<void> _pollMaintenanceStatus() async {
    if (!_maintenanceInProgress) return;
    try {
      final status = await widget.apiClient.fetchMaintenanceStatus();
      if (!mounted || !_maintenanceInProgress) return;
      final operation = status.operation ?? _maintenanceOperation;
      final message = status.message.isEmpty
          ? (_maintenanceMessage ?? '正在安装...')
          : status.message;
      if (_isMaintenanceActive(status.state)) {
        setState(() {
          _maintenanceOperation = operation;
          _maintenanceMessage = message;
          _installingPytorch =
              operation == 'install_pytorch' ||
              operation == 'install_yolo_dependencies';
          _reinstallingPackage = operation == 'reinstall_package';
        });
        return;
      }

      _finishMaintenanceWatch(message);
    } catch (_) {
      if (!mounted || !_maintenanceInProgress) return;
      setState(() {
        _maintenanceMessage = '正在安装，Python 后端暂时离线，完成后会自动重启...';
      });
    }
  }

  bool _isMaintenanceActive(String state) {
    return const {
      'starting',
      'waiting_for_backend',
      'running',
      'restarting',
    }.contains(state);
  }

  bool get _maintenanceInProgress => _installingPytorch || _reinstallingPackage;

  void _finishMaintenanceWatch(String message) {
    _maintenanceTimer?.cancel();
    _maintenanceTimer = null;
    setState(() {
      _installingPytorch = false;
      _reinstallingPackage = false;
      _maintenanceOperation = null;
      _maintenanceMessage = null;
    });
    if (message.isNotEmpty) widget.onShowMessage(message);
  }

  Future<bool?> _confirmMaintenance({
    required String title,
    required String message,
    required String confirmLabel,
  }) {
    return showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: Text(confirmLabel),
          ),
        ],
      ),
    );
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
                icon: Icon(Icons.tune_outlined),
                selectedIcon: Icon(Icons.tune_rounded),
                label: Text('检测设置'),
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
                icon: Icon(Icons.palette_outlined),
                selectedIcon: Icon(Icons.palette_rounded),
                label: Text('外观主题'),
              ),
              NavigationDrawerDestination(
                icon: Icon(Icons.system_update_alt_rounded),
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
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                child: _buildSaveStrip(),
              ),
              Expanded(
                child: ListView(
                  physics: const AlwaysScrollableScrollPhysics(),
                  padding: const EdgeInsets.all(16),
                  children: [_buildSelectedSection()],
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
      0 => _buildDetectionSettings(),
      1 => _buildEnvironmentMaintenance(),
      2 => _buildBasicSettings(),
      3 => _buildAppearanceSettings(),
      4 => _buildUpdateSettingsSection(),
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

  Widget _buildDetectionSettings() {
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
    final videoMode = _string('video_mode', 'all');
    final strideLabel = videoMode == 'all' ? '帧间隔' : '快速识别帧数';

    return SectionCard(
      title: '检测设置',
      subtitle: '模型管理、识别物种、阈值、推理加速和视频检测',
      icon: Icons.tune_rounded,
      child: Column(
        children: [
          _SettingsPanel(
            title: '探测模型',
            subtitle: '选择用于照片和视频目标检测的模型',
            icon: Icons.build_rounded,
            child: _SettingsMenuButton<String>(
              value: selectedModel,
              placeholder: '未发现探测模型',
              options: (settings?.availableModels ?? const <ModelInfo>[])
                  .map(
                    (model) => _SettingsOption<String>(
                      value: model.path,
                      label: model.name,
                    ),
                  )
                  .toList(),
              onChanged: (value) {
                _set('selected_model', value);
                _set('selected_species_names', <String>[]);
                _loadModelClassesForSelection(value);
              },
            ),
          ),
          _SettingsPanel(
            title: '分类模型',
            subtitle: '选择二次分类模型，或设为不使用',
            icon: Icons.account_tree_rounded,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _SettingsMenuButton<String>(
                  value: selectedClassificationModel,
                  options: [
                    const _SettingsOption<String>(value: '', label: '不使用'),
                    ...(settings?.availableClassificationModels ??
                            const <ModelInfo>[])
                        .map(
                          (model) => _SettingsOption<String>(
                            value: model.path,
                            label: model.name,
                          ),
                        ),
                  ],
                  onChanged: (value) =>
                      _set('selected_classification_model', value),
                ),
                const SizedBox(height: 8),
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
          _SettingsPanel(
            title: '高级检测选项',
            subtitle: 'TTA 可能提高准确率但会降低速度，NMS 控制跨类别框合并',
            icon: Icons.auto_awesome_rounded,
            child: Column(
              children: [
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  value: _bool('use_augment'),
                  onChanged: (value) => _set('use_augment', value),
                  title: const Text('使用 TTA 数据增强'),
                  subtitle: const Text('推理时使用测试时增强，通常更慢但可能更稳。'),
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  value: _bool('use_agnostic_nms'),
                  onChanged: (value) => _set('use_agnostic_nms', value),
                  title: const Text('使用跨类别 NMS'),
                  subtitle: const Text('合并不同类别间重叠较高的检测框。'),
                ),
              ],
            ),
          ),
          ..._buildVideoSettingPanels(videoMode, strideLabel),
        ],
      ),
    );
  }

  Widget _buildSpeciesSelector() {
    final speciesNames = _modelSpeciesNames();
    final selected = _stringList('selected_species_names');
    final summary = _loadingModelClasses
        ? '读取中...'
        : _speciesSelectionSummary(selected, speciesNames);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        if (_loadingModelClasses)
          const Padding(
            padding: EdgeInsets.only(bottom: 12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
                SizedBox(width: 8),
              ],
            ),
          ),
        if (!_loadingModelClasses)
          _SummaryDialogButton(
            summary: summary,
            onPressed: () => _showSpeciesSelectionDialog(speciesNames),
          ),
      ],
    );
  }

  List<String> _modelSpeciesNames() {
    final seen = <String>{};
    return [
      for (final species in _modelClassOptions)
        if (species.label.isNotEmpty && seen.add(species.label)) species.label,
    ];
  }

  String _selectionSummary(
    List<String> selected, {
    required String emptyLabel,
  }) {
    final cleaned = selected.where((item) => item.trim().isNotEmpty).toList();
    if (cleaned.isEmpty) return emptyLabel;
    final preview = cleaned.take(3).join(', ');
    return cleaned.length > 3 ? '$preview, ...' : preview;
  }

  String _speciesSelectionSummary(List<String> selected, List<String> options) {
    final optionSet = options.toSet();
    final cleaned = selected
        .where((item) => item.trim().isNotEmpty)
        .where((item) => optionSet.isEmpty || optionSet.contains(item))
        .toSet();
    if (cleaned.isEmpty ||
        (options.isNotEmpty && cleaned.length >= options.length)) {
      return '全部识别';
    }
    return _selectionSummary(cleaned.toList(), emptyLabel: '全部识别');
  }

  Future<void> _showSpeciesSelectionDialog(List<String> speciesNames) async {
    final initialSelection = _stringList('selected_species_names').toSet();
    final result = await showDialog<List<String>>(
      context: context,
      builder: (context) {
        final selected = initialSelection.toSet();
        return StatefulBuilder(
          builder: (context, setDialogState) {
            final hasOptions = speciesNames.isNotEmpty;
            return AlertDialog(
              title: const Text('选择识别物种'),
              content: SizedBox(
                width: 460,
                height: hasOptions ? 480 : 120,
                child: hasOptions
                    ? Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const _MutedText('不勾选时默认识别全部模型类别。'),
                          const SizedBox(height: 8),
                          Wrap(
                            spacing: 8,
                            children: [
                              TextButton.icon(
                                onPressed: () => setDialogState(() {
                                  selected
                                    ..clear()
                                    ..addAll(speciesNames);
                                }),
                                icon: const Icon(Icons.select_all_rounded),
                                label: const Text('全选'),
                              ),
                              TextButton.icon(
                                onPressed: selected.isEmpty
                                    ? null
                                    : () => setDialogState(() {
                                        selected.clear();
                                      }),
                                icon: const Icon(Icons.deselect_rounded),
                                label: const Text('清空'),
                              ),
                            ],
                          ),
                          const Divider(height: 20),
                          Expanded(
                            child: ListView.builder(
                              itemCount: speciesNames.length,
                              itemBuilder: (context, index) {
                                final name = speciesNames[index];
                                return CheckboxListTile(
                                  dense: true,
                                  value: selected.contains(name),
                                  title: Text(name),
                                  onChanged: (value) {
                                    setDialogState(() {
                                      if (value == true) {
                                        selected.add(name);
                                      } else {
                                        selected.remove(name);
                                      }
                                    });
                                  },
                                );
                              },
                            ),
                          ),
                        ],
                      )
                    : const Center(child: Text('当前模型暂未读取到可选物种，处理时会识别全部模型类别。')),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('取消'),
                ),
                FilledButton(
                  onPressed: hasOptions
                      ? () {
                          final ordered = speciesNames
                              .where(selected.contains)
                              .toList();
                          Navigator.of(context).pop(
                            ordered.length == speciesNames.length
                                ? <String>[]
                                : ordered,
                          );
                        }
                      : () => Navigator.of(context).pop(<String>[]),
                  child: const Text('确定'),
                ),
              ],
            );
          },
        );
      },
    );

    if (result != null) {
      _set('selected_species_names', result);
    }
  }

  Widget _buildVideoSettings() {
    final videoMode = _string('video_mode', 'all');
    final strideLabel = videoMode == 'all' ? '帧间隔' : '快速识别帧数';

    return SectionCard(
      title: '视频检测设置',
      subtitle: '视频处理模式、跳帧和检测过滤',
      icon: Icons.movie_filter_rounded,
      child: Column(children: _buildVideoSettingPanels(videoMode, strideLabel)),
    );
  }

  List<Widget> _buildVideoSettingPanels(String videoMode, String strideLabel) {
    return [
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
          onSelectionChanged: (selection) {
            if (selection.isEmpty) return;
            _set('video_mode', selection.first);
          },
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
        showDivider: false,
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
    ];
  }

  Widget _buildMaintenanceProgress() {
    if (!_maintenanceInProgress) return const SizedBox.shrink();
    final scheme = Theme.of(context).colorScheme;
    final message = _maintenanceMessage ?? '正在安装...';
    return Padding(
      padding: const EdgeInsets.fromLTRB(0, 4, 0, 12),
      child: Row(
        children: [
          SizedBox(
            width: 22,
            height: 22,
            child: CircularProgressIndicator(
              strokeWidth: 2.6,
              color: scheme.primary,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              message,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: scheme.primary,
                fontWeight: FontWeight.w600,
              ),
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
          if (_maintenanceInProgress) _buildMaintenanceProgress(),
          _SettingsPanel(
            title: 'Python 包安装源',
            subtitle: '用于 ultralytics 和单个 Python 包安装',
            icon: Icons.travel_explore_rounded,
            child: _SettingsMenuButton<String>(
              value: _string('package_source', 'official'),
              options: const [
                _SettingsOption<String>(value: 'official', label: '官方源'),
                _SettingsOption<String>(value: 'aliyun', label: '阿里源'),
                _SettingsOption<String>(value: 'tsinghua', label: '清华源'),
              ],
              onChanged: (value) => _set('package_source', value),
            ),
          ),
          _SettingsPanel(
            title: '安装 PyTorch',
            subtitle: '选择安装环境后执行安装或修复',
            icon: Icons.download_rounded,
            child: Row(
              children: [
                Expanded(
                  child: _SettingsMenuButton<String>(
                    value: _string('pytorch_version', '自动检测'),
                    options: const [
                      _SettingsOption<String>(value: '自动检测', label: '自动检测'),
                      _SettingsOption<String>(
                        value: 'CUDA 13.0',
                        label: 'CUDA 13.0',
                      ),
                      _SettingsOption<String>(
                        value: 'CUDA 12.8',
                        label: 'CUDA 12.8',
                      ),
                      _SettingsOption<String>(
                        value: 'CUDA 12.6',
                        label: 'CUDA 12.6',
                      ),
                      _SettingsOption<String>(
                        value: 'CUDA 12.4',
                        label: 'CUDA 12.4',
                      ),
                      _SettingsOption<String>(
                        value: 'CUDA 12.1',
                        label: 'CUDA 12.1',
                      ),
                      _SettingsOption<String>(
                        value: 'CUDA 11.8',
                        label: 'CUDA 11.8',
                      ),
                      _SettingsOption<String>(
                        value: 'Intel XPU',
                        label: 'Intel XPU',
                      ),
                      _SettingsOption<String>(
                        value: 'CPU Only',
                        label: 'CPU Only',
                      ),
                    ],
                    onChanged: (value) => _set('pytorch_version', value),
                  ),
                ),
                const SizedBox(width: 10),
                FilledButton(
                  onPressed: _installingPytorch || _reinstallingPackage
                      ? null
                      : _installPytorch,
                  child: _installingPytorch
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('安装'),
                ),
              ],
            ),
          ),
          _SettingsPanel(
            title: '重新安装单个 Python 包',
            subtitle: '使用 toolkit\\python.exe 强制重新安装指定包',
            icon: Icons.extension_rounded,
            showDivider: false,
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _packageController,
                    decoration: const InputDecoration(
                      labelText: '包名',
                      hintText: '例如 numpy 或 numpy==1.26.4',
                      border: OutlineInputBorder(),
                    ),
                    onChanged: (value) => _draft['package'] = value.trim(),
                  ),
                ),
                const SizedBox(width: 10),
                FilledButton(
                  onPressed: _installingPytorch || _reinstallingPackage
                      ? null
                      : _reinstallPythonPackage,
                  child: _reinstallingPackage
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('重新安装'),
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
      subtitle: '校验辅助、快速标记和导出字段',
      icon: Icons.fact_check_rounded,
      child: Column(
        children: [
          _SettingsPanel(
            title: '物种校验界面自动分组',
            subtitle: '开启后，将根据视频文件自动对连续拍摄的照片和视频进行分组，并在文件列表中按所在组校验最多的物种统一归类显示。',
            icon: Icons.auto_awesome_motion_rounded,
            child: Align(
              alignment: Alignment.centerRight,
              child: Switch(
                value: _bool('auto_group'),
                onChanged: (value) => _set('auto_group', value),
              ),
            ),
          ),
          _buildQuickMarkEditor(),
          Builder(
            builder: (context) => Padding(
              padding: const EdgeInsets.only(top: 16, bottom: 12),
              child: Divider(
                height: 1,
                color: Theme.of(context).colorScheme.outlineVariant,
              ),
            ),
          ),
          _buildExportColumns(),
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
    final configuredMarks = _stringList('quick_mark_list');
    final autoSort = _bool('auto_sort');
    final marks = autoSort
        ? _autoSortedQuickMarks(configuredMarks)
        : configuredMarks;
    return Column(
      children: [
        _SettingsPanel(
          title: '自动排序',
          subtitle: '根据最近常用物种调整快速标记列的显示顺序',
          icon: Icons.sort_rounded,
          showDivider: false,
          child: Align(
            alignment: Alignment.centerRight,
            child: Switch(
              value: autoSort,
              onChanged: (value) => _set('auto_sort', value),
            ),
          ),
        ),
        ...List.generate(marks.length, (index) {
          final configuredIndex = configuredMarks.indexOf(marks[index]);
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
                      if (configuredIndex < 0) return;
                      final next = _stringList('quick_mark_list');
                      next[configuredIndex] = value.trim();
                      _draft['quick_mark_list'] = next;
                      _markDirty();
                    },
                  ),
                ),
                IconButton(
                  tooltip: '删除',
                  onPressed: () {
                    if (configuredIndex < 0) return;
                    final next = _stringList('quick_mark_list')
                      ..removeAt(configuredIndex);
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
    final selected = _stringList('export_columns');
    final summary = selected.isEmpty
        ? '使用默认列'
        : selected.length == _defaultExportColumns.length
        ? '全部列'
        : _selectionSummary(selected, emptyLabel: '使用默认列');

    return _SettingsPanel(
      title: '导出表格列',
      subtitle: '选择校验数据导出时包含的字段',
      icon: Icons.view_column_rounded,
      showDivider: false,
      child: _SummaryDialogButton(
        summary: summary,
        onPressed: _showExportColumnsDialog,
      ),
    );
  }

  Future<void> _showExportColumnsDialog() async {
    final initialSelection = _stringList('export_columns').toSet();
    final result = await showDialog<List<String>>(
      context: context,
      builder: (context) {
        final selected = initialSelection.toSet();
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              title: const Text('选择导出表格列'),
              content: SizedBox(
                width: 460,
                height: 480,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Wrap(
                      spacing: 8,
                      children: [
                        TextButton.icon(
                          onPressed: () => setDialogState(() {
                            selected
                              ..clear()
                              ..addAll(_defaultExportColumns);
                          }),
                          icon: const Icon(Icons.select_all_rounded),
                          label: const Text('全选'),
                        ),
                        TextButton.icon(
                          onPressed: selected.isEmpty
                              ? null
                              : () => setDialogState(() {
                                  selected.clear();
                                }),
                          icon: const Icon(Icons.deselect_rounded),
                          label: const Text('清空'),
                        ),
                      ],
                    ),
                    const Divider(height: 20),
                    Expanded(
                      child: ListView.builder(
                        itemCount: _defaultExportColumns.length,
                        itemBuilder: (context, index) {
                          final column = _defaultExportColumns[index];
                          return CheckboxListTile(
                            dense: true,
                            value: selected.contains(column),
                            title: Text(column),
                            onChanged: (value) {
                              setDialogState(() {
                                if (value == true) {
                                  selected.add(column);
                                } else {
                                  selected.remove(column);
                                }
                              });
                            },
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('取消'),
                ),
                FilledButton(
                  onPressed: () {
                    final ordered = _defaultExportColumns
                        .where(selected.contains)
                        .toList();
                    Navigator.of(context).pop(ordered);
                  },
                  child: const Text('确定'),
                ),
              ],
            );
          },
        );
      },
    );

    if (result != null) {
      _set('export_columns', result);
    }
  }

  Widget _buildUpdateSettings() {
    return Column(
      children: [
        _SettingsPanel(
          title: '更新通道',
          subtitle: '选择稳定版或预览版更新来源',
          icon: Icons.commit_rounded,
          child: _SettingsMenuButton<String>(
            value: _string('update_channel', 'Preview'),
            options: const [
              _SettingsOption<String>(value: 'Release', label: '稳定版'),
              _SettingsOption<String>(value: 'Preview', label: '预览版'),
            ],
            onChanged: (value) => _set('update_channel', value),
          ),
        ),
        _SettingsPanel(
          title: '镜像源',
          subtitle: '选择检查更新时使用的下载源',
          icon: Icons.public_rounded,
          child: _SettingsMenuButton<String>(
            value: _string('update_mirror', 'KKGitHub'),
            options: const [
              _SettingsOption<String>(value: 'Official', label: '官方源'),
              _SettingsOption<String>(value: 'KKGitHub', label: '国内源'),
            ],
            onChanged: (value) => _set('update_mirror', value),
          ),
        ),
        _SettingsPanel(
          title: '检查更新',
          subtitle: '保留软件更新检查入口',
          icon: Icons.system_update_alt_rounded,
          showDivider: false,
          child: Align(
            alignment: Alignment.centerRight,
            child: FilledButton(
              onPressed: () => widget.onShowMessage('软件更新检查入口已保留，后续可接入更新服务。'),
              child: const Text('检查更新'),
            ),
          ),
        ),
      ],
    );
  }

  List<String> _autoSortedQuickMarks(List<String> configuredMarks) {
    final usageCounts = _intMap('quick_mark_usage_counts');
    final history = _stringList('quick_mark_recent_history');
    final latestIndex = <String, int>{};
    final recentCounts = <String, int>{};
    for (var index = 0; index < history.length; index++) {
      final species = history[index].trim();
      if (species.isEmpty || species == 'Unknown' || species == '空') {
        continue;
      }
      latestIndex[species] = index;
      recentCounts[species] = (recentCounts[species] ?? 0) + 1;
    }

    final originalIndex = <String, int>{};
    final sorted = <String>[];
    for (var index = 0; index < configuredMarks.length; index++) {
      final species = configuredMarks[index].trim();
      if (species.isEmpty) continue;
      originalIndex.putIfAbsent(species, () => index);
      sorted.add(species);
    }

    sorted.sort((a, b) {
      final recentCompare = (recentCounts[b] ?? 0).compareTo(
        recentCounts[a] ?? 0,
      );
      if (recentCompare != 0) return recentCompare;
      final usageCompare = (usageCounts[b] ?? 0).compareTo(usageCounts[a] ?? 0);
      if (usageCompare != 0) return usageCompare;
      final latestCompare = (latestIndex[b] ?? -1).compareTo(
        latestIndex[a] ?? -1,
      );
      if (latestCompare != 0) return latestCompare;
      return (originalIndex[a] ?? 0).compareTo(originalIndex[b] ?? 0);
    });

    return sorted;
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
    var saved = false;
    try {
      _draft['package'] = _packageController.text.trim();
      if (widget.settings?.gpuAvailable != true) {
        _draft['batch_size'] = 1;
        _draft['use_fp16'] = false;
      }
      await widget.onSaveSettings(Map<String, dynamic>.from(_draft));
      saved = true;
    } finally {
      if (mounted) {
        setState(() {
          _saving = false;
          if (saved) _dirty = false;
        });
      }
    }
  }

  void _set(String key, dynamic value) {
    setState(() {
      _draft[key] = value;
      _dirty = true;
    });
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

  Map<String, int> _intMap(String key) {
    return _intMapSetting(_draft, key);
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

class _SettingsOption<T> {
  const _SettingsOption({required this.value, required this.label});

  final T value;
  final String label;
}

class _SettingsMenuButton<T> extends StatelessWidget {
  const _SettingsMenuButton({
    required this.value,
    required this.options,
    required this.onChanged,
    this.placeholder = '请选择',
    this.minMenuWidth = 180,
    this.maxMenuWidth = 360,
  });

  final T? value;
  final List<_SettingsOption<T>> options;
  final ValueChanged<T> onChanged;
  final String placeholder;
  final double minMenuWidth;
  final double maxMenuWidth;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final selectedOption = _selectedOption;
    final label = selectedOption?.label ?? placeholder;
    final enabled = options.isNotEmpty;

    return Align(
      alignment: Alignment.centerRight,
      child: MenuAnchor(
        style: appMenuStyle(context, minWidth: minMenuWidth),
        menuChildren: [
          for (final option in options)
            MenuItemButton(
              style: appMenuItemStyle(
                context,
                selected: option.value == value,
                minWidth: minMenuWidth,
              ),
              leadingIcon: option.value == value
                  ? const Icon(Icons.check_rounded)
                  : const SizedBox(width: 24),
              onPressed: () => onChanged(option.value),
              child: ConstrainedBox(
                constraints: BoxConstraints(
                  minWidth: minMenuWidth,
                  maxWidth: maxMenuWidth,
                ),
                child: Text(option.label, overflow: TextOverflow.ellipsis),
              ),
            ),
        ],
        builder: (context, controller, child) {
          return TextButton(
            onPressed: enabled
                ? () {
                    if (controller.isOpen) {
                      controller.close();
                    } else {
                      controller.open();
                    }
                  }
                : null,
            style: TextButton.styleFrom(
              foregroundColor: scheme.primary,
              disabledForegroundColor: scheme.onSurfaceVariant,
              minimumSize: Size.zero,
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 16),
              alignment: Alignment.centerRight,
              textStyle: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w700),
            ),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 320),
              child: Text(
                label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                textAlign: TextAlign.end,
              ),
            ),
          );
        },
      ),
    );
  }

  _SettingsOption<T>? get _selectedOption {
    for (final option in options) {
      if (option.value == value) return option;
    }
    return null;
  }
}

class _SummaryDialogButton extends StatelessWidget {
  const _SummaryDialogButton({required this.summary, required this.onPressed});

  final String summary;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    return TextButton(
      onPressed: onPressed,
      style: TextButton.styleFrom(
        foregroundColor: scheme.primary,
        disabledForegroundColor: scheme.onSurfaceVariant,
        minimumSize: Size.zero,
        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 16),
        alignment: Alignment.centerRight,
        textStyle: Theme.of(
          context,
        ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w700),
      ),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 320),
        child: Text(
          summary,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          textAlign: TextAlign.end,
        ),
      ),
    );
  }
}

class _SettingsPanel extends StatelessWidget {
  const _SettingsPanel({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.child,
    this.showDivider = true,
  });

  final String title;
  final String subtitle;
  final IconData icon;
  final Widget child;
  final bool showDivider;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12),
      child: Column(
        children: [
          LayoutBuilder(
            builder: (context, constraints) {
              final label = Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(icon, size: 20, color: scheme.primary),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          title,
                          style: textTheme.titleSmall?.copyWith(
                            color: scheme.primary,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          subtitle,
                          style: textTheme.bodySmall?.copyWith(
                            color: scheme.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              );

              if (constraints.maxWidth < 720) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [label, const SizedBox(height: 12), child],
                );
              }

              return Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(flex: 4, child: label),
                  const SizedBox(width: 28),
                  Expanded(
                    flex: 5,
                    child: Align(
                      alignment: Alignment.centerRight,
                      child: child,
                    ),
                  ),
                ],
              );
            },
          ),
          if (showDivider)
            Padding(
              padding: const EdgeInsets.only(top: 16),
              child: Divider(height: 1, color: scheme.outlineVariant),
            ),
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
        final paletteDisabled = settings.useDynamicColor;
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _SettingsPanel(
              title: '主题模式',
              subtitle: '选择浅色、深色或跟随系统',
              icon: Icons.brightness_6_rounded,
              child: SegmentedButton<ThemeMode>(
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
                onSelectionChanged: (selection) {
                  if (selection.isEmpty) return;
                  onUpdateTheme(settings.copyWith(themeMode: selection.first));
                },
              ),
            ),
            _SettingsPanel(
              title: '使用系统动态颜色',
              subtitle: '开启后调色板由系统主题接管',
              icon: Icons.color_lens_rounded,
              child: Align(
                alignment: Alignment.centerRight,
                child: Switch(
                  value: settings.useDynamicColor,
                  onChanged: (value) =>
                      onUpdateTheme(settings.copyWith(useDynamicColor: value)),
                ),
              ),
            ),
            _SettingsPanel(
              title: paletteDisabled ? '调色板已由系统接管' : '调色板',
              subtitle: '选择应用主色调',
              icon: Icons.palette_rounded,
              showDivider: false,
              child: Wrap(
                alignment: WrapAlignment.end,
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
    return Card.outlined(
      margin: EdgeInsets.zero,
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
    final theme = Theme.of(context);
    final scheme = ColorScheme.fromSeed(
      seedColor: color,
      brightness: theme.brightness,
    );
    final tileColor = selected
        ? Color.alphaBlend(
            scheme.primary.withValues(alpha: 0.10),
            theme.colorScheme.surface,
          )
        : theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5);
    final checkColor =
        ThemeData.estimateBrightnessForColor(scheme.primary) == Brightness.dark
        ? Colors.white
        : theme.colorScheme.onSurface;

    final palette = SizedBox(
      width: 46,
      height: 46,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CustomPaint(
            size: const Size.square(46),
            painter: _PaletteCirclePainter(scheme),
          ),
          if (selected)
            Container(
              width: 29,
              height: 29,
              decoration: BoxDecoration(
                color: Color.alphaBlend(
                  scheme.primary.withValues(alpha: 0.08),
                  theme.colorScheme.surface,
                ),
                shape: BoxShape.circle,
              ),
              child: Icon(Icons.check_rounded, color: checkColor, size: 20),
            ),
        ],
      ),
    );

    return GestureDetector(
      onTap: onTap,
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 200),
        opacity: disabled ? 0.35 : 1.0,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          curve: Curves.easeOutCubic,
          width: 72,
          height: 72,
          decoration: BoxDecoration(
            color: tileColor,
            borderRadius: BorderRadius.circular(18),
          ),
          alignment: Alignment.center,
          child: palette,
        ),
      ),
    );
  }
}

class _PaletteCirclePainter extends CustomPainter {
  const _PaletteCirclePainter(this.scheme);

  final ColorScheme scheme;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final paint = Paint()..isAntiAlias = true;

    canvas.drawArc(
      rect,
      math.pi,
      math.pi,
      true,
      paint..color = scheme.primaryContainer,
    );
    canvas.drawArc(
      rect,
      math.pi / 2,
      math.pi / 2,
      true,
      paint..color = scheme.tertiaryContainer,
    );
    canvas.drawArc(rect, 0, math.pi / 2, true, paint..color = scheme.primary);
  }

  @override
  bool shouldRepaint(covariant _PaletteCirclePainter oldDelegate) {
    return oldDelegate.scheme != scheme;
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

Map<String, int> _intMapSetting(Map<String, dynamic> values, String key) {
  final value = values[key];
  if (value is! Map) return const <String, int>{};
  return value.map((key, value) {
    final parsed = switch (value) {
      int number => number,
      num number => number.round(),
      String text => int.tryParse(text) ?? 0,
      _ => 0,
    };
    return MapEntry(key.toString(), parsed);
  });
}
