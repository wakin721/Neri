import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'section_card.dart';
import 'settings_progress_indicator.dart';

typedef SpeciesLearningSnapshot = ({
  Map<String, dynamic>? status,
  String? error,
  bool busy,
});

String learningStateLabel(Map<String, dynamic> status) =>
    switch (status['state']) {
      'queued' => '等待学习',
      'running' => '正在学习',
      'ready' => '学习已完成',
      'blocked' => '学习暂不可用',
      'failed' => '学习失败',
      _ => '自动学习已就绪',
    };

class DinoSpeciesLearningDialog extends StatelessWidget {
  const DinoSpeciesLearningDialog({
    required this.data,
    required this.onRefresh,
    required this.onRetry,
    required this.onImportCalibration,
    super.key,
  });

  final ValueListenable<SpeciesLearningSnapshot> data;
  final VoidCallback onRefresh;
  final VoidCallback onRetry;
  final VoidCallback onImportCalibration;

  @override
  Widget build(BuildContext context) {
    return Dialog(
      insetPadding: const EdgeInsets.all(16),
      clipBehavior: Clip.antiAlias,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 600, maxHeight: 560),
        child: ValueListenableBuilder<SpeciesLearningSnapshot>(
          valueListenable: data,
          builder: (context, snapshot, _) {
            final status = snapshot.status;
            return Column(
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 10, 8, 8),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          '模型学习状态',
                          style: Theme.of(context).textTheme.titleLarge,
                        ),
                      ),
                      IconButton(
                        tooltip: '刷新学习状态',
                        onPressed: snapshot.busy ? null : onRefresh,
                        icon: const Icon(Icons.refresh_rounded),
                      ),
                      IconButton(
                        tooltip: '关闭明细',
                        onPressed: () => Navigator.pop(context),
                        icon: const Icon(Icons.close_rounded),
                      ),
                    ],
                  ),
                ),
                const Divider(height: 1),
                Expanded(
                  child: status != null
                      ? _LearningTabs(
                          status: status,
                          stale: snapshot.error != null,
                          overview: _LearningOverview(
                            status: status,
                            error: snapshot.error,
                            onRetry: snapshot.busy ? null : onRetry,
                            onImportCalibration: snapshot.busy
                                ? null
                                : onImportCalibration,
                          ),
                        )
                      : Center(
                          child: snapshot.error == null
                              ? const SettingsProgressIndicator()
                              : SingleChildScrollView(
                                  padding: const EdgeInsets.all(20),
                                  child: Text('读取学习状态失败：${snapshot.error}'),
                                ),
                        ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _LearningOverview extends StatelessWidget {
  const _LearningOverview({
    required this.status,
    required this.error,
    required this.onRetry,
    required this.onImportCalibration,
  });
  final Map<String, dynamic> status;
  final String? error;
  final VoidCallback? onRetry;
  final VoidCallback? onImportCalibration;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final colors = Theme.of(context).colorScheme;
    final state = status['state'];
    final active = state == 'queued' || state == 'running';
    final failed = state == 'failed' || state == 'blocked';
    final calibration = status['calibration_available'];
    final canImport = calibration == false;
    final canRetry = state == 'failed' || (state == 'blocked' && calibration == true);
    final last = DateTime.tryParse(
      status['last_update']?.toString() ?? '',
    )?.toLocal();
    final message = status['message']?.toString() ?? '';
    final reason = status['last_error']?.toString() ?? '';
    final rejection = status['rejection'] as Map? ?? const {};
    final experiment = rejection['gate_mode'] == 'sequence';
    String threshold(String key) =>
        (rejection[key] as num?)?.toStringAsFixed(3) ?? '未记录';

    Widget metric(String field, String label) => Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '${status[field] ?? '—'}',
          style: text.headlineSmall?.copyWith(color: colors.primary),
        ),
        Text(
          label,
          style: text.bodySmall?.copyWith(color: colors.onSurfaceVariant),
        ),
      ],
    );
    return ListView(
      key: const PageStorageKey('learning-overview'),
      padding: const EdgeInsets.all(12),
      children: [
        SectionCard(
          title: learningStateLabel(status),
          icon: failed
              ? Icons.error_outline_rounded
              : active
              ? Icons.sync_rounded
              : Icons.psychology_rounded,
          subtitle: '人工确认或修改物种后在后台自动学习，完成后用于下一次识别。',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Wrap(
                alignment: WrapAlignment.spaceBetween,
                spacing: 32,
                runSpacing: 12,
                children: [
                  metric('class_count', '模型物种'),
                  metric('learned_count', '已学习独立样本'),
                  metric('pending_count', '待处理检测框'),
                ],
              ),
              if (active)
                const Padding(
                  padding: EdgeInsets.only(top: 12),
                  child: LinearProgressIndicator(),
                ),
              if (last != null)
                Padding(
                  padding: const EdgeInsets.only(top: 14),
                  child: Text('最近完成：${last.toString().split('.').first}'),
                ),
              if (message.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Text(message),
                ),
              if (failed && reason.isNotEmpty && !message.contains(reason))
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Text(reason, style: TextStyle(color: colors.error)),
                ),
              if (error != null)
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Text(error!, style: TextStyle(color: colors.error)),
                ),
              if (canImport || canRetry)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Wrap(
                    spacing: 8,
                    children: [
                      if (canImport)
                        TextButton(
                          onPressed: onImportCalibration,
                          child: const Text('导入校准数据'),
                        ),
                      if (canRetry)
                        TextButton(
                          onPressed: onRetry,
                          child: const Text('重试自动学习'),
                        ),
                    ],
                  ),
                ),
            ],
          ),
        ),
        const SizedBox(height: 4),
        SectionCard(
          title: 'DINO 拒识门限（只读）',
          icon: Icons.tune_rounded,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                experiment
                    ? '实验事件门限 ${threshold('sequence_threshold')}'
                    : '单框门限 ${threshold('box_threshold')}  ·  序列门限 ${threshold('sequence_threshold')}',
                style: text.titleSmall,
              ),
              const SizedBox(height: 6),
              Text(
                experiment
                    ? '事件分数达到门限才接纳。门限由模型自动校准。'
                    : '单框和序列均达到门限才接纳。门限由模型自动校准。',
                style: text.bodySmall,
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _LearningTabs extends StatelessWidget {
  const _LearningTabs({
    required this.status,
    required this.stale,
    required this.overview,
  });
  final Map<String, dynamic> status;
  final bool stale;
  final Widget overview;

  @override
  Widget build(BuildContext context) {
    final learned = Map<String, int>.from(
      status['learned_species_counts'] as Map? ?? {},
    );
    final pending = Map<String, int>.from(
      status['species_counts'] as Map? ?? {},
    );
    final original = (status['original_species'] as List?)
        ?.whereType<String>()
        .toSet();
    final current =
        (status['current_species'] as List?)?.whereType<String>().toSet() ??
        <String>{};
    final withdrawals =
        (status['pending_count'] as int? ?? 0) -
        pending.values.fold(0, (total, count) => total + count);
    final messages = [
      if (original == null) '未找到基础模型物种清单，以下按物种列出学习记录。',
      if (stale) '刷新失败，显示上次读取的记录。',
      if (status['state'] == 'failed' || status['state'] == 'blocked')
        status['last_error']?.toString() ??
            status['message']?.toString() ??
            '自动学习暂不可用。',
      if (withdrawals > 0) '另有 $withdrawals 个检测框等待撤回。',
    ];

    Widget list(Set<String> names, String emptyText) {
      final sorted = names.toList()
        ..sort((a, b) {
          final byPending = (pending[b] ?? 0).compareTo(pending[a] ?? 0);
          if (byPending != 0) return byPending;
          final byLearned = (learned[b] ?? 0).compareTo(learned[a] ?? 0);
          return byLearned != 0 ? byLearned : a.compareTo(b);
        });
      final header = [...messages, if (sorted.isEmpty) emptyText];
      return ListView.builder(
        primary: false,
        padding: const EdgeInsets.fromLTRB(8, 8, 8, 16),
        itemCount: header.length + sorted.length,
        itemBuilder: (context, index) {
          if (index < header.length) {
            return Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              child: Text(header[index]),
            );
          }
          final name = sorted[index - header.length];
          final samples = learned[name] ?? 0;
          final boxes = pending[name] ?? 0;
          final isOriginal = original?.contains(name);
          final summary = [
            if (isOriginal == false) current.contains(name) ? '已加入模型' : '待加入模型',
            if (samples > 0)
              '${isOriginal == true ? '已补入' : '已学习'} $samples 个独立样本'
            else if (isOriginal == true)
              '尚未补入样本',
            if (boxes > 0) '待处理 $boxes 个检测框',
          ].join(' · ');
          return ListTile(title: Text(name), subtitle: Text(summary));
        },
      );
    }

    final recorded = {...learned.keys, ...pending.keys};
    final added = {...current, ...recorded}.difference(original ?? const {});
    return DefaultTabController(
      length: original == null ? 2 : 3,
      child: Column(
        children: [
          TabBar(
            isScrollable: true,
            tabAlignment: TabAlignment.start,
            tabs: [
              const Tab(text: '学习概览'),
              if (original == null)
                const Tab(text: '物种记录')
              else ...[
                Tab(text: '原有物种（${original.length}）'),
                Tab(text: '新物种（${added.length}）'),
              ],
            ],
          ),
          Expanded(
            child: TabBarView(
              children: [
                overview,
                if (original == null)
                  list(recorded, '暂无学习记录。')
                else ...[
                  list(original, '基础模型没有记录原有物种。'),
                  list(added, '暂无新物种学习记录。'),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
