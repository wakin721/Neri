from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor in {path}, found {count}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "system/dinov3/registry.py",
    '''    def delete(self, entry_id: int) -> None:\n        self._row(entry_id)\n        self._conn.execute("DELETE FROM registrations WHERE id=?", (entry_id,))\n        self._conn.commit()\n\n    def _event_rows(self, entry_id):\n''',
    '''    def delete(self, entry_id: int) -> None:\n        self._row(entry_id)\n        self._conn.execute("DELETE FROM registrations WHERE id=?", (entry_id,))\n        self._conn.commit()\n\n    def delete_candidates(self) -> int:\n        """Delete every unregistered Candidate and its cascaded local evidence."""\n        count = int(\n            self._conn.execute(\n                "SELECT COUNT(*) FROM registrations WHERE status='candidate'"\n            ).fetchone()[0]\n        )\n        if count:\n            self._conn.execute(\n                "DELETE FROM registrations WHERE status='candidate'"\n            )\n            self._conn.commit()\n        return count\n\n    def _event_rows(self, entry_id):\n''',
)

replace_once(
    "system/dinov3/api.py",
    '''    @router.get("/registry/{registration_id}", response_model=DinoV3RegistryEntryResponse)\n    def get_registry_entry(\n''',
    '''    @router.delete("/registry/candidates")\n    def delete_registry_candidates(\n        classification_model_path: str = Query(..., min_length=1),\n    ):\n        deleted = _run_with_registry(\n            classification_model_path,\n            lambda registry: registry.delete_candidates(),\n        )\n        return {"deleted": int(deleted)}\n\n    @router.get("/registry/{registration_id}", response_model=DinoV3RegistryEntryResponse)\n    def get_registry_entry(\n''',
)

replace_once(
    "frontend/lib/src/api_client_core.dart",
    '''  Future<DinoV3RegistryEntry> updateDinoV3RegistryIdentity({\n''',
    '''  Future<int> clearDinoV3UnregisteredCandidates(\n    String classificationModelPath,\n  ) async {\n    final uri = _uri('/api/dinov3/registry/candidates').replace(\n      queryParameters: {'classification_model_path': classificationModelPath},\n    );\n    final response = await _httpClient.delete(uri);\n    _ensureSuccess(response);\n    final decoded = jsonDecode(response.body);\n    if (decoded is Map<String, dynamic>) {\n      return (decoded['deleted'] as num?)?.toInt() ?? 0;\n    }\n    return 0;\n  }\n\n  Future<DinoV3RegistryEntry> updateDinoV3RegistryIdentity({\n''',
)

replace_once(
    "frontend/lib/src/widgets/dinov3_registry_dialog.dart",
    '''  Widget _buildExampleGallery() {\n''',
    '''  Future<void> _clearCandidates() async {\n    final candidateCount = _entries\n        .where((entry) => entry.status == 'candidate')\n        .length;\n    if (candidateCount == 0 || _saving) return;\n\n    final confirmed = await showDialog<bool>(\n      context: context,\n      builder: (context) => AlertDialog(\n        title: const Text('清除所有未注册事件？'),\n        content: Text(\n          '将删除 $candidateCount 个 Candidate 物种及其未注册事件和本地 prototypes。\\n'\n          'Provisional / Confirmed / Mature、Checkpoint 以及历史 human-feedback / audit 数据都会保留。\\n\\n'\n          '此操作不可撤销。',\n        ),\n        actions: [\n          TextButton(\n            onPressed: () => Navigator.of(context).pop(false),\n            child: const Text('取消'),\n          ),\n          FilledButton(\n            onPressed: () => Navigator.of(context).pop(true),\n            child: const Text('确认清除'),\n          ),\n        ],\n      ),\n    );\n    if (confirmed != true || !mounted) return;\n\n    setState(() {\n      _saving = true;\n      _error = null;\n    });\n    try {\n      await widget.apiClient.clearDinoV3UnregisteredCandidates(\n        widget.modelPath,\n      );\n      if (mounted) await _load();\n    } catch (error) {\n      if (mounted) setState(() => _error = '清除未注册事件失败：$error');\n    } finally {\n      if (mounted) setState(() => _saving = false);\n    }\n  }\n\n  Widget _buildExampleGallery() {\n''',
)

replace_once(
    "frontend/lib/src/widgets/dinov3_registry_dialog.dart",
    '''  Widget build(BuildContext context) {\n    final selected = _selected;\n    return AlertDialog(\n''',
    '''  Widget build(BuildContext context) {\n    final selected = _selected;\n    final candidateCount = _entries\n        .where((entry) => entry.status == 'candidate')\n        .length;\n    return AlertDialog(\n''',
)

replace_once(
    "frontend/lib/src/widgets/dinov3_registry_dialog.dart",
    '''      actions: [\n        TextButton(\n          onPressed: _saving ? null : () => Navigator.of(context).maybePop(),\n          child: const Text('关闭'),\n        ),\n      ],\n''',
    '''      actions: [\n        OutlinedButton.icon(\n          key: const ValueKey('dinov3-clear-unregistered-events'),\n          onPressed: !_saving && candidateCount > 0 ? _clearCandidates : null,\n          icon: const Icon(Icons.delete_sweep_outlined),\n          label: const Text('清除未注册事件'),\n        ),\n        TextButton(\n          onPressed: _saving ? null : () => Navigator.of(context).maybePop(),\n          child: const Text('关闭'),\n        ),\n      ],\n''',
)
