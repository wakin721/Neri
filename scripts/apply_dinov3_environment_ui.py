from __future__ import annotations

from pathlib import Path


PATH = Path("frontend/lib/src/screens/settings_screen.dart")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = PATH.read_text(encoding="utf-8")

    text = replace_once(
        text,
        """  bool _installingPytorch = false;\n  bool _reinstallingPackage = false;\n  bool _debugModeSaving = false;\n""",
        """  bool _installingPytorch = false;\n  bool _reinstallingPackage = false;\n  bool _maintainingDinoV3 = false;\n  bool _debugModeSaving = false;\n""",
        "DINOv3 maintenance flag",
    )
    text = replace_once(
        text,
        """  String? _maintenanceStatusPath;\n  String? _modelClassesPath;\n""",
        """  String? _maintenanceStatusPath;\n  DinoV3ComponentStatus? _dinoV3Status;\n  bool _loadingDinoV3Status = true;\n  String? _modelClassesPath;\n""",
        "DINOv3 status fields",
    )
    text = replace_once(
        text,
        """      _loadModelClassesForSelection();\n      unawaited(_resumeMaintenanceWatchIfActive());\n""",
        """      _loadModelClassesForSelection();\n      unawaited(_loadDinoV3Status());\n      unawaited(_resumeMaintenanceWatchIfActive());\n""",
        "initial DINOv3 status load",
    )

    dino_methods = r'''  Future<void> _loadDinoV3Status() async {
    try {
      final status = await widget.apiClient.fetchDinoV3ComponentStatus();
      if (!mounted) return;
      setState(() {
        _dinoV3Status = status;
        _loadingDinoV3Status = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _dinoV3Status = null;
        _loadingDinoV3Status = false;
      });
    }
  }

  Future<void> _installDinoV3() async {
    if (_maintenanceBusy) return;
    _startMaintenancePreparation(
      operation: 'install_dinov3',
      message: '正在检查 DINOv3 安装环境...',
    );

    final envChoice = _normalizedPytorchVersion(
      _string('pytorch_version', '自动检测'),
    );
    PackageSourceResolution? packageSource;
    try {
      if (await _resumeMaintenanceWatchIfActive(announce: true)) return;
      _updateMaintenancePreparation('正在检测 Python 包安装源...');
      packageSource = await _resolvePackageSource();
    } finally {
      _finishMaintenancePreparation();
    }

    if (packageSource == null || !mounted) return;
    final repairing = _dinoV3Status?.installed == true;
    final confirmed = await _confirmPythonInstallation(
      title: repairing ? '修复 DINOv3' : '安装 DINOv3',
      message:
          '将从 NeriCloud 严格同步 DINOv3 ViT-B/16 组件到 res/model/DINOv3，'
          '并校验 Multi-prototype 模型、官方 source 与 SHA-256。'
          '如果共享 PyTorch 环境缺失，会按当前运行环境自动补齐。',
    );
    if (confirmed != true || !mounted) return;

    setState(() {
      _maintainingDinoV3 = true;
      _maintenanceOperation = 'install_dinov3';
      _maintenanceMessage = repairing
          ? '正在启动 DINOv3 修复...'
          : '正在启动 DINOv3 安装...';
    });
    try {
      final response = await widget.apiClient.installDinoV3(
        envChoice: envChoice,
        packageSource: packageSource.source,
      );
      if (!mounted) return;
      _startMaintenanceWatch(
        operation: response.operation.isEmpty
            ? 'install_dinov3'
            : response.operation,
        message: response.message,
        progress: response.progress,
        statusPath: response.statusPath,
      );
      widget.onShowMessage(response.message);
    } catch (error) {
      await _handleMaintenanceStartFailure(error, '启动 DINOv3 安装失败');
    }
  }

  Future<void> _removeDinoV3() async {
    if (_maintenanceBusy) return;
    final confirmed = await _confirmMaintenance(
      title: '删除 DINOv3',
      message:
          '将删除 res/model/DINOv3 中的 DINOv3 组件文件。\n\n'
          '共享 PyTorch 环境不会删除；本地已学习物种、人工校验事件和增量 prototype 数据也会保留。',
      confirmLabel: '删除',
    );
    if (confirmed != true || !mounted) return;

    setState(() {
      _maintainingDinoV3 = true;
      _maintenanceOperation = 'remove_dinov3';
      _maintenanceMessage = '正在启动 DINOv3 删除...';
    });
    try {
      final response = await widget.apiClient.removeDinoV3();
      if (!mounted) return;
      _startMaintenanceWatch(
        operation: response.operation.isEmpty
            ? 'remove_dinov3'
            : response.operation,
        message: response.message,
        progress: response.progress,
        statusPath: response.statusPath,
      );
      widget.onShowMessage(response.message);
    } catch (error) {
      await _handleMaintenanceStartFailure(error, '启动 DINOv3 删除失败');
    }
  }

'''
    text = replace_once(
        text,
        "  Future<bool?> _resolveIntelDriverInstall(String envChoice) async {\n",
        dino_methods + "  Future<bool?> _resolveIntelDriverInstall(String envChoice) async {\n",
        "DINOv3 maintenance methods",
    )

    text = replace_once(
        text,
        """      _installingPytorch = false;\n      _reinstallingPackage = false;\n      _maintenanceOperation = null;\n""",
        """      _installingPytorch = false;\n      _reinstallingPackage = false;\n      _maintainingDinoV3 = false;\n      _maintenanceOperation = null;\n""",
        "maintenance failure reset",
    )
    text = replace_once(
        text,
        """      'install_pytorch',\n      'install_yolo_dependencies',\n      'reinstall_package',\n""",
        """      'install_pytorch',\n      'install_yolo_dependencies',\n      'reinstall_package',\n      'install_dinov3',\n      'remove_dinov3',\n""",
        "resumable DINOv3 operations",
    )
    text = replace_once(
        text,
        """      _reinstallingPackage = operation == 'reinstall_package';\n    });\n    _maintenanceTimer = Timer.periodic(\n""",
        """      _reinstallingPackage = operation == 'reinstall_package';\n      _maintainingDinoV3 =\n          operation == 'install_dinov3' || operation == 'remove_dinov3';\n    });\n    _maintenanceTimer = Timer.periodic(\n""",
        "initial DINOv3 maintenance watch state",
    )
    text = replace_once(
        text,
        """          _reinstallingPackage = operation == 'reinstall_package';\n        });\n        return;\n""",
        """          _reinstallingPackage = operation == 'reinstall_package';\n          _maintainingDinoV3 =\n              operation == 'install_dinov3' || operation == 'remove_dinov3';\n        });\n        return;\n""",
        "polled DINOv3 maintenance state",
    )
    text = replace_once(
        text,
        """  bool get _maintenanceInProgress => _installingPytorch || _reinstallingPackage;\n""",
        """  bool get _maintenanceInProgress =>\n      _installingPytorch || _reinstallingPackage || _maintainingDinoV3;\n""",
        "DINOv3 maintenance progress",
    )
    text = replace_once(
        text,
        """  void _finishMaintenanceWatch(String message) {\n    _maintenanceTimer?.cancel();\n    _maintenanceTimer = null;\n    setState(() {\n      _installingPytorch = false;\n      _reinstallingPackage = false;\n      _maintenanceOperation = null;\n""",
        """  void _finishMaintenanceWatch(String message) {\n    final finishedOperation = _maintenanceOperation;\n    _maintenanceTimer?.cancel();\n    _maintenanceTimer = null;\n    setState(() {\n      _installingPytorch = false;\n      _reinstallingPackage = false;\n      _maintainingDinoV3 = false;\n      _maintenanceOperation = null;\n""",
        "maintenance finish reset",
    )
    text = replace_once(
        text,
        """    if (message.isNotEmpty) widget.onShowMessage(message);\n  }\n\n  Future<bool?> _confirmMaintenance({\n""",
        """    if (message.isNotEmpty) widget.onShowMessage(message);\n    if (finishedOperation == 'install_dinov3' ||\n        finishedOperation == 'remove_dinov3') {\n      unawaited(\n        Future<void>.delayed(\n          const Duration(seconds: 1),\n          _loadDinoV3Status,\n        ),\n      );\n    }\n  }\n\n  Future<bool?> _confirmMaintenance({\n""",
        "refresh DINOv3 status after maintenance",
    )

    text = replace_once(
        text,
        """      subtitle: 'PyTorch 与单个 Python 包维护入口',\n""",
        """      subtitle: 'PyTorch、DINOv3 与单个 Python 包维护入口',\n""",
        "environment maintenance subtitle",
    )

    dino_panel = r'''          _SettingsPanel(
            title: 'DINOv3 ViT-B/16',
            subtitle: _loadingDinoV3Status
                ? '正在读取 DINOv3 组件状态...'
                : _dinoV3Status == null
                ? '无法读取 DINOv3 组件状态。'
                : _dinoV3Status!.healthy && _dinoV3Status!.selectionK != null
                ? '${_dinoV3Status!.message} · Multi-prototype K=${_dinoV3Status!.selectionK}'
                : _dinoV3Status!.message,
            icon: Icons.hub_rounded,
            child: Builder(
              builder: (context) {
                if (_loadingDinoV3Status) {
                  return const SizedBox.square(
                    dimension: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  );
                }
                final status = _dinoV3Status;
                if (status?.healthy == true) {
                  return OutlinedButton(
                    onPressed: _maintenanceBusy ? null : _removeDinoV3,
                    child: const Text('删除'),
                  );
                }
                return FilledButton(
                  onPressed: _maintenanceBusy ? null : _installDinoV3,
                  child: Text(status?.installed == true ? '安装/修复' : '安装'),
                );
              },
            ),
          ),
'''
    text = replace_once(
        text,
        """          _SettingsPanel(\n            title: '重新安装单个 Python 包',\n""",
        dino_panel
        + """          _SettingsPanel(\n            title: '重新安装单个 Python 包',\n""",
        "DINOv3 environment panel",
    )

    PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
