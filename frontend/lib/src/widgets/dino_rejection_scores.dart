import 'package:flutter/material.dart';
import 'package:flutter/gestures.dart';

import '../models/job.dart';
import '../utils/dino_rejection.dart';

class DinoRejectionScores extends StatelessWidget {
  const DinoRejectionScores({
    required this.boxes,
    this.detectionData,
    super.key,
  });

  final List<DetectionBox> boxes;
  final Map<String, dynamic>? detectionData;

  Map<dynamic, dynamic>? _rawBoxFor(DetectionBox box, int index) =>
      findRawDinoBox(box, index, detectionData?['检测框']);

  String _gateComparison(
    String label,
    double? score,
    double? threshold,
    bool? passed,
  ) {
    if (score == null) {
      return '$label 未记录';
    }
    final scoreText = score.toStringAsFixed(3);
    if (threshold == null) {
      if (passed == null) return '$label $scoreText / 门限 未记录';
      return '$label $scoreText / 门限 未记录（${passed ? '通过' : '未通过'}）';
    }
    final compared = score >= threshold;
    final op = compared ? '≥' : '<';
    final thresholdText = threshold.toStringAsFixed(3);
    if (passed != null && passed != compared) {
      final savedText = passed ? '通过' : '未通过';
      return '$label $scoreText $op 门限 $thresholdText（记录$savedText，与比较不一致）';
    }
    final outcome = (passed ?? compared) ? '通过' : '未通过';
    return '$label $scoreText $op 门限 $thresholdText（$outcome）';
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      key: const ValueKey('dino-rejection-single-line'),
      height: 36,
      child: ScrollConfiguration(
        behavior: ScrollConfiguration.of(
          context,
        ).copyWith(dragDevices: PointerDeviceKind.values.toSet()),
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          primary: false,
          child: Row(
            children: [
              const Tooltip(
                message:
                    'DINO 拒识判定分数（只读）\n以下为该图本次识别时记录的原始分数与当时门限，自动学习后不会覆盖历史判定。\n分数越低越易拒识；单框和序列均达到各自门限才接纳。\n数字右侧为门限，悬停可查看完整比较；内容较多时可横向拖动。',
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text('拒识', style: TextStyle(fontWeight: FontWeight.w600)),
                    SizedBox(width: 4),
                    Icon(Icons.info_outline_rounded, size: 16),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              if (boxes.isEmpty) const Text('暂无检测框', maxLines: 1),
              for (var i = 0; i < boxes.length; i++) ...[
                if (i > 0) const SizedBox(width: 8),
                _buildBoxRow(
                  context,
                  i,
                  dinoRejectionInfo(boxes[i], rawBox: _rawBoxFor(boxes[i], i)),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _gate(
    BuildContext context,
    String label,
    double? score,
    double? threshold,
    bool? passed,
  ) {
    final value = score?.toStringAsFixed(3) ?? '—';
    final boundary = threshold?.toStringAsFixed(3) ?? '—';
    final comparison = score != null && threshold != null
        ? (score >= threshold ? '≥' : '<')
        : '/';
    return Tooltip(
      message: _gateComparison(label, score, threshold, passed),
      child: Text(
        '$label $value $comparison $boundary',
        maxLines: 1,
        softWrap: false,
        style: Theme.of(context).textTheme.bodySmall,
      ),
    );
  }

  Widget _buildBoxRow(
    BuildContext context,
    int index,
    DinoBoxRejectionInfo info,
  ) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final statusColor = info.finalRejected
        ? colorScheme.error
        : info.isImageLabeled
        ? colorScheme.secondary
        : colorScheme.primary;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest.withValues(alpha: 0.35),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '#${index + 1} ${info.finalLabel}',
            maxLines: 1,
            softWrap: false,
            style: theme.textTheme.bodySmall?.copyWith(
              fontWeight: FontWeight.w600,
              color: statusColor,
            ),
          ),
          const SizedBox(width: 10),
          if (info.isLegacySingleGate && info.legacyScore != null)
            _gate(
              context,
              '旧版门控',
              info.legacyScore,
              info.legacyThreshold,
              info.legacyPassed,
            )
          else ...[
            _gate(
              context,
              '单框',
              info.boxScore,
              info.boxThreshold,
              info.boxPassed,
            ),
            const SizedBox(width: 10),
            _gate(
              context,
              '序列',
              info.sequenceScore,
              info.sequenceThreshold,
              info.sequencePassed,
            ),
          ],
          if (info.hasInconsistentRecord) ...[
            const SizedBox(width: 8),
            Tooltip(
              message: '记录不一致，建议重新识别',
              child: Icon(
                Icons.warning_amber_rounded,
                size: 16,
                color: colorScheme.error,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
