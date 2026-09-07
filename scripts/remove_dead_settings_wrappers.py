from pathlib import Path

path = Path('frontend/lib/src/screens/settings_screen.dart')
text = path.read_text(encoding='utf-8')
blocks = [
'''  Widget _buildVideoSettings() {
    final videoMode = normalizeVideoProcessingMode(
      _string('video_mode', defaultVideoProcessingMode),
    );
    final strideLabel = videoMode == videoProcessingModeAll ? '帧间隔' : '快速识别帧数';

    return SectionCard(
      title: '视频检测设置',
      subtitle: '视频处理模式、跳帧和检测过滤',
      icon: Icons.movie_filter_rounded,
      child: Column(
        children: _buildVideoSettingPanels(
          videoMode,
          strideLabel,
          enabled: _detectionDependenciesReady,
        ),
      ),
    );
  }

''',
'''  Widget _buildQuickMarkSettings() {
    return SectionCard(
      title: '快速标记设置',
      subtitle: '物种按钮、数量按钮和自动排序',
      icon: Icons.bookmark_add_rounded,
      child: _buildQuickMarkEditor(),
    );
  }

''',
'''  Widget _buildExportSettings() {
    return SectionCard(
      title: '导出设置',
      subtitle: '自定义导出表格、收藏媒体同步和空照片删除策略',
      icon: Icons.table_chart_rounded,
      child: Column(
        children: [
          _buildExportColumns(showDivider: true),
          _buildFavoritePhotoExportMode(showDivider: true),
          _buildEmptyPhotoDeleteMode(),
        ],
      ),
    );
  }

''',
]
for block in blocks:
    count = text.count(block)
    if count != 1:
        raise RuntimeError(f'expected exactly one dead wrapper block, found {count}')
    text = text.replace(block, '', 1)
path.write_text(text, encoding='utf-8')
