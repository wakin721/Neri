from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_start_screen() -> None:
    path = Path("frontend/lib/src/screens/start_screen.dart")
    replace_once(
        path,
        "const _defaultModelDirectory = 'res/model';\n"
        "const _defaultClassificationModelDirectory = 'res/model_cls';",
        "const _defaultModelDirectory = 'res/Model/detect';\n"
        "const _defaultClassificationModelDirectory = 'res/Model/cls';",
    )
    replace_once(
        path,
        "  return '扫描 $directory';\n}\n\nclass StartScreen",
        "  return '扫描 $directory';\n}\n\n"
        "List<DropdownMenuEntry<String>> _modelDropdownEntries(\n"
        "  List<ModelInfo> models,\n"
        ") {\n"
        "  final userModels = models.where((model) => model.source != 'sync').toList();\n"
        "  final cloudModels = models.where((model) => model.source == 'sync').toList();\n"
        "  return <DropdownMenuEntry<String>>[\n"
        "    const DropdownMenuEntry<String>(value: '', label: '不使用'),\n"
        "    if (userModels.isNotEmpty) ...[\n"
        "      const DropdownMenuEntry<String>(\n"
        "        value: '__neri_header_user__',\n"
        "        label: '用户模型',\n"
        "        enabled: false,\n"
        "      ),\n"
        "      for (final model in userModels)\n"
        "        DropdownMenuEntry<String>(value: model.path, label: model.rawName),\n"
        "    ],\n"
        "    if (cloudModels.isNotEmpty) ...[\n"
        "      const DropdownMenuEntry<String>(\n"
        "        value: '__neri_header_sync__',\n"
        "        label: 'NeriCloud',\n"
        "        enabled: false,\n"
        "      ),\n"
        "      for (final model in cloudModels)\n"
        "        DropdownMenuEntry<String>(value: model.path, label: model.rawName),\n"
        "    ],\n"
        "  ];\n"
        "}\n\nclass StartScreen",
    )
    old_entries = (
        "      dropdownMenuEntries: [\n"
        "        const DropdownMenuEntry<String>(value: '', label: '不使用'),\n"
        "        for (final model in models)\n"
        "          DropdownMenuEntry<String>(value: model.path, label: model.name),\n"
        "      ],"
    )
    text = path.read_text(encoding="utf-8")
    if text.count(old_entries) != 2:
        raise RuntimeError(f"{path}: expected two flat model dropdown blocks")
    path.write_text(
        text.replace(old_entries, "      dropdownMenuEntries: _modelDropdownEntries(models),"),
        encoding="utf-8",
    )


def patch_settings_screen() -> None:
    path = Path("frontend/lib/src/screens/settings_screen.dart")
    replace_once(
        path,
        "import '../widgets/workspace_split_metrics.dart';\n",
        "import '../widgets/workspace_split_metrics.dart';\n"
        "import 'model_sync_settings_host.dart';\n",
    )
    replace_once(
        path,
        "  @override\n  Widget build(BuildContext context) {\n"
        "    final debugModeEnabled = _bool(_debugModeKey);",
        "  Future<void> _refreshModelCatalog() async {\n"
        "    await widget.onSaveSettings(Map<String, dynamic>.from(_draft));\n"
        "  }\n\n"
        "  @override\n  Widget build(BuildContext context) {\n"
        "    final debugModeEnabled = _bool(_debugModeKey);",
    )
    replace_once(
        path,
        "    return LayoutBuilder(\n      builder: (context, constraints) {",
        "    return ModelSyncSettingsHost(\n"
        "      apiClient: widget.apiClient,\n"
        "      enabled: widget.settings != null,\n"
        "      onCatalogChanged: _refreshModelCatalog,\n"
        "      child: LayoutBuilder(\n"
        "        builder: (context, constraints) {",
    )
    replace_once(
        path,
        "      },\n    );\n  }\n\n  Widget _buildSelectedSection()",
        "        },\n      ),\n    );\n  }\n\n  Widget _buildSelectedSection()",
    )
    detection_options = (
        "              options: [\n"
        "                const _SettingsOption<String>(value: '', label: '不使用'),\n"
        "                ...(settings?.availableModels ?? const <ModelInfo>[]).map(\n"
        "                  (model) => _SettingsOption<String>(\n"
        "                    value: model.path,\n"
        "                    label: model.name,\n"
        "                  ),\n"
        "                ),\n"
        "              ],"
    )
    replace_once(
        path,
        detection_options,
        "              options: _modelSettingsOptions(\n"
        "                settings?.availableModels ?? const <ModelInfo>[],\n"
        "              ),",
    )
    classification_options = (
        "                  options: [\n"
        "                    const _SettingsOption<String>(value: '', label: '不使用'),\n"
        "                    ...(settings?.availableClassificationModels ??\n"
        "                            const <ModelInfo>[])\n"
        "                        .map(\n"
        "                          (model) => _SettingsOption<String>(\n"
        "                            value: model.path,\n"
        "                            label: model.name,\n"
        "                          ),\n"
        "                        ),\n"
        "                  ],"
    )
    replace_once(
        path,
        classification_options,
        "                  options: _modelSettingsOptions(\n"
        "                    settings?.availableClassificationModels ??\n"
        "                        const <ModelInfo>[],\n"
        "                  ),",
    )
    replace_once(
        path,
        "class _SettingsOption<T> {\n"
        "  const _SettingsOption({required this.value, required this.label});\n\n"
        "  final T value;\n"
        "  final String label;\n"
        "}\n",
        "List<_SettingsOption<String>> _modelSettingsOptions(List<ModelInfo> models) {\n"
        "  final userModels = models.where((model) => model.source != 'sync').toList();\n"
        "  final cloudModels = models.where((model) => model.source == 'sync').toList();\n"
        "  return <_SettingsOption<String>>[\n"
        "    const _SettingsOption<String>(value: '', label: '不使用'),\n"
        "    if (userModels.isNotEmpty) ...[\n"
        "      const _SettingsOption<String>(\n"
        "        value: '__neri_header_user__',\n"
        "        label: '用户模型',\n"
        "        enabled: false,\n"
        "      ),\n"
        "      for (final model in userModels)\n"
        "        _SettingsOption<String>(value: model.path, label: model.rawName),\n"
        "    ],\n"
        "    if (cloudModels.isNotEmpty) ...[\n"
        "      const _SettingsOption<String>(\n"
        "        value: '__neri_header_sync__',\n"
        "        label: 'NeriCloud',\n"
        "        enabled: false,\n"
        "      ),\n"
        "      for (final model in cloudModels)\n"
        "        _SettingsOption<String>(value: model.path, label: model.rawName),\n"
        "    ],\n"
        "  ];\n"
        "}\n\n"
        "class _SettingsOption<T> {\n"
        "  const _SettingsOption({\n"
        "    required this.value,\n"
        "    required this.label,\n"
        "    this.enabled = true,\n"
        "  });\n\n"
        "  final T value;\n"
        "  final String label;\n"
        "  final bool enabled;\n"
        "}\n",
    )
    replace_once(
        path,
        "              onPressed: () => onChanged(option.value),",
        "              onPressed: option.enabled\n"
        "                  ? () => onChanged(option.value)\n"
        "                  : null,",
    )


def patch_main_window() -> None:
    path = Path("frontend/lib/src/main_window.dart")
    replace_once(
        path,
        "          modelDirectory: 'res/model',\n"
        "          classificationModelDirectory: 'res/model_cls',",
        "          modelDirectory: 'res/Model/detect',\n"
        "          classificationModelDirectory: 'res/Model/cls',",
    )


def main() -> None:
    patch_start_screen()
    patch_settings_screen()
    patch_main_window()


if __name__ == "__main__":
    main()
