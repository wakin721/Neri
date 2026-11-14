import 'package:flutter/material.dart';

import '../models/job.dart';

class SequencePendingNotice extends StatelessWidget {
  const SequencePendingNotice({required this.item, super.key});

  static const message = '临时结果 · 序列待汇总。中断任务可在开始页继续，完成后再确认识别结果。';

  final DetectionItem item;

  @override
  Widget build(BuildContext context) {
    if (!item.isSequencePending) return const SizedBox.shrink();
    return Padding(
      padding: const EdgeInsets.only(left: 6),
      child: Tooltip(
        message: message,
        child: Icon(
          Icons.info_outline_rounded,
          size: 16,
          color: Theme.of(context).colorScheme.onSurfaceVariant,
        ),
      ),
    );
  }
}
