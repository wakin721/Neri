from pathlib import Path

path = Path('frontend/lib/src/screens/species_validation_screen.dart')
text = path.read_text(encoding='utf-8')

replacements = [
    (
        """  bool _marking = false;
  bool _exporting = false;
""",
        """  bool _marking = false;
  bool _exporting = false;
  String? _selectedObservationId;
""",
    ),
    (
        """  ) {
    final visibleBoxes = _filteredBoxes(selectedItem);
    // 将 220.0 修改为 200.0，与预览界面保持完全一致
""",
        """  ) {
    final visibleBoxes = _filteredBoxes(selectedItem);
    final selectedDinoBox = _selectedDinoBox(visibleBoxes);
    // 将 220.0 修改为 200.0，与预览界面保持完全一致
""",
    ),
    (
        """              Expanded(child: _buildImagePanel(selectedItem, visibleBoxes)),
              const SizedBox(height: 10),
              _buildSummaryPanel(selectedItem, visibleBoxes),
""",
        """              Expanded(child: _buildImagePanel(selectedItem, visibleBoxes)),
              if (selectedDinoBox != null) ...[
                const SizedBox(height: 10),
                _buildDinoFeedbackPanel(selectedDinoBox),
              ],
              const SizedBox(height: 10),
              _buildSummaryPanel(selectedItem, visibleBoxes),
""",
    ),
    (
        """  ) {
    final visibleBoxes = _filteredBoxes(selectedItem);
    return ListView(
""",
        """  ) {
    final visibleBoxes = _filteredBoxes(selectedItem);
    final selectedDinoBox = _selectedDinoBox(visibleBoxes);
    return ListView(
""",
    ),
    (
        """        SizedBox(
          height: 330,
          child: _buildImagePanel(selectedItem, visibleBoxes),
        ),
        const SizedBox(height: 10),
        SizedBox(height: 260, child: _buildLeftLists(buckets, visibleRows)),
""",
        """        SizedBox(
          height: 330,
          child: _buildImagePanel(selectedItem, visibleBoxes),
        ),
        if (selectedDinoBox != null) ...[
          const SizedBox(height: 10),
          _buildDinoFeedbackPanel(selectedDinoBox),
        ],
        const SizedBox(height: 10),
        SizedBox(height: 260, child: _buildLeftLists(buckets, visibleRows)),
""",
    ),
    (
        """        onOpenExternal: () => widget.onOpenExternal(item.path),
        isFavorite: _isFavoritePhoto(item),
""",
        """        onOpenExternal: () => widget.onOpenExternal(item.path),
        selectedObservationId: _selectedObservationId,
        onDetectionBoxSelected: (box) {
          setState(() => _selectedObservationId = box?.observationId);
        },
        isFavorite: _isFavoritePhoto(item),
""",
    ),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(f'anchor not found: {old!r}')
    text = text.replace(old, new, 1)

anchor = """  Widget _buildRightActions(DetectionItem item) {
"""
insert = """  DetectionBox? _selectedDinoBox(List<DetectionBox> visibleBoxes) {
    final observationId = _selectedObservationId?.trim();
    if (observationId == null || observationId.isEmpty) return null;
    for (final box in visibleBoxes) {
      if (box.observationId?.trim() == observationId) return box;
    }
    return null;
  }

  Widget _buildDinoFeedbackPanel(DetectionBox box) {
    final predicted = box.predictedSpecies?.trim();
    final title = predicted == null || predicted.isEmpty
        ? '检测框校验'
        : '检测框校验 · $predicted';
    return _ValidationPanel(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        child: Row(
          children: [
            Expanded(
              child: Text(
                title,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
            ),
            const SizedBox(width: 12),
            OutlinedButton(onPressed: null, child: const Text('正确')),
            const SizedBox(width: 8),
            OutlinedButton(onPressed: null, child: const Text('修改物种')),
            const SizedBox(width: 8),
            OutlinedButton(onPressed: null, child: const Text('空 / 误检')),
            const SizedBox(width: 8),
            OutlinedButton(onPressed: null, child: const Text('不参与学习')),
          ],
        ),
      ),
    );
  }

"""
if anchor not in text:
    raise SystemExit('right-actions anchor not found')
text = text.replace(anchor, insert + anchor, 1)

path.write_text(text, encoding='utf-8')
