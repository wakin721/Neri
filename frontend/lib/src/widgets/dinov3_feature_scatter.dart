import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../api_client.dart';
import '../models/dinov3_explanation.dart';

class DinoV3FeatureScatter extends StatelessWidget {
  const DinoV3FeatureScatter({required this.explanation, super.key});

  final DinoV3FeatureExplanation explanation;

  @override
  Widget build(BuildContext context) {
    final species = explanation.projection.species;
    final first = species.isEmpty ? '最近类别 1' : species.first;
    final second = species.length < 2 ? '最近类别 2' : species[1];
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      key: const ValueKey('dinov3-feature-scatter'),
      padding: const EdgeInsets.fromLTRB(10, 8, 10, 8),
      decoration: BoxDecoration(
        border: Border.all(color: colorScheme.outlineVariant),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            '局部特征投影',
            style: TextStyle(fontSize: 12, color: colorScheme.onSurfaceVariant),
          ),
          const SizedBox(height: 6),
          Expanded(
            child: CustomPaint(
              painter: _FeatureScatterPainter(
                points: explanation.projection.points,
                firstSpecies: first,
                secondSpecies: second,
                colors: colorScheme,
              ),
            ),
          ),
          const SizedBox(height: 6),
          Row(
            children: [
              Expanded(
                child: Text(
                  first,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
              Text(
                '← 物种判别轴 →',
                style: TextStyle(
                  fontSize: 11,
                  color: colorScheme.onSurfaceVariant,
                ),
              ),
              Expanded(
                child: Text(
                  second,
                  textAlign: TextAlign.end,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.diamond_rounded, size: 14, color: colorScheme.error),
              const SizedBox(width: 4),
              const Text('当前检测框'),
            ],
          ),
        ],
      ),
    );
  }
}

class DinoV3FeatureExplanationPanel extends StatefulWidget {
  const DinoV3FeatureExplanationPanel({
    required this.apiClient,
    required this.classificationModelPath,
    required this.observationId,
    super.key,
  });

  final NeriApiClient apiClient;
  final String classificationModelPath;
  final String observationId;

  @override
  State<DinoV3FeatureExplanationPanel> createState() =>
      _DinoV3FeatureExplanationPanelState();
}

class _DinoV3FeatureExplanationPanelState
    extends State<DinoV3FeatureExplanationPanel> {
  DinoV3FeatureExplanation? _explanation;
  Uint8List? _currentExample;
  Uint8List? _nearestExample;
  String? _error;
  bool _loading = true;
  int _requestId = 0;

  @override
  void initState() {
    super.initState();
    unawaited(_load());
  }

  @override
  void didUpdateWidget(covariant DinoV3FeatureExplanationPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.observationId != widget.observationId ||
        oldWidget.classificationModelPath != widget.classificationModelPath) {
      unawaited(_load());
    }
  }

  Future<void> _load() async {
    final requestId = ++_requestId;
    if (mounted) {
      setState(() {
        _loading = true;
        _error = null;
        _explanation = null;
        _currentExample = null;
        _nearestExample = null;
      });
    }
    try {
      final explanation = await widget.apiClient.fetchDinoV3FeatureExplanation(
        widget.classificationModelPath,
        widget.observationId,
      );
      Uint8List? currentBytes;
      Uint8List? nearestBytes;
      if (explanation.currentExampleAvailable) {
        try {
          currentBytes = await widget.apiClient.fetchDinoV3ObservationExample(
            widget.classificationModelPath,
            widget.observationId,
          );
        } catch (_) {}
      }
      final nearest = explanation.nearestExample;
      if (nearest != null) {
        try {
          if (nearest.kind == 'registry' &&
              nearest.registrationId != null &&
              nearest.eventId != null) {
            nearestBytes = await widget.apiClient.fetchDinoV3RegistryExample(
              widget.classificationModelPath,
              nearest.registrationId!,
              nearest.eventId!,
            );
          } else if (nearest.kind == 'observation' &&
              (nearest.observationId?.isNotEmpty ?? false)) {
            nearestBytes = await widget.apiClient.fetchDinoV3ObservationExample(
              widget.classificationModelPath,
              nearest.observationId!,
            );
          }
        } catch (_) {}
      }
      if (!mounted || requestId != _requestId) return;
      setState(() {
        _explanation = explanation;
        _currentExample = currentBytes;
        _nearestExample = nearestBytes;
        _loading = false;
      });
    } catch (error) {
      if (!mounted || requestId != _requestId) return;
      setState(() {
        _loading = false;
        _error = error.toString();
      });
    }
  }

  Widget _imageCard(String title, Uint8List? bytes, String emptyLabel) {
    final colors = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest.withValues(alpha: 0.45),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            title,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 6),
          SizedBox(
            height: 104,
            child: bytes == null || bytes.isEmpty
                ? Center(
                    child: Text(
                      emptyLabel,
                      textAlign: TextAlign.center,
                      style: TextStyle(color: colors.onSurfaceVariant),
                    ),
                  )
                : ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: Image.memory(bytes, fit: BoxFit.cover),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _content(DinoV3FeatureExplanation explanation) {
    final nearest = explanation.nearestSpecies;
    final closestName = nearest.isEmpty ? '暂无' : nearest.first.name;
    final nearestExampleLabel = explanation.nearestExample == null
        ? '暂无本地代表例图'
        : '代表例图不可用';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        LayoutBuilder(
          builder: (context, constraints) {
            final cards = [
              _imageCard('当前检测框裁切', _currentExample, '当前裁切不可用'),
              _imageCard(
                '最近类别代表图 · $closestName',
                _nearestExample,
                nearestExampleLabel,
              ),
            ];
            if (constraints.maxWidth >= 520) {
              return Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: cards[0]),
                  const SizedBox(width: 10),
                  Expanded(child: cards[1]),
                ],
              );
            }
            return Column(
              children: [cards[0], const SizedBox(height: 10), cards[1]],
            );
          },
        ),
        const SizedBox(height: 10),
        SizedBox(
          height: 210,
          child: DinoV3FeatureScatter(explanation: explanation),
        ),
        const SizedBox(height: 10),
        Wrap(
          spacing: 16,
          runSpacing: 4,
          children: [
            Text('判定：${explanation.species}'),
            Text('最近类：$closestName'),
            Text('cosine ${explanation.knownScore.toStringAsFixed(4)}'),
            Text('阈值 ${explanation.threshold.toStringAsFixed(4)}'),
            if (explanation.squaredDistance != null)
              Text('距离² ${explanation.squaredDistance!.toStringAsFixed(4)}'),
          ],
        ),
        if (nearest.isNotEmpty) ...[
          const SizedBox(height: 6),
          for (var index = 0; index < nearest.length; index++)
            Text(
              '${index + 1}. ${nearest[index].name} · '
              '距离² ${nearest[index].squaredDistance.toStringAsFixed(4)} · '
              'cosine ${nearest[index].cosineScore.toStringAsFixed(4)} · '
              '${nearest[index].source}',
            ),
        ],
        const SizedBox(height: 8),
        Text(
          '二维图为当前样本附近 768 维特征空间的解释性投影；实际分类仍使用完整高维特征与真实距离。',
          style: TextStyle(
            fontSize: 12,
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return ExpansionTile(
      key: ValueKey('dinov3-feature-explanation-${widget.observationId}'),
      initiallyExpanded: true,
      tilePadding: EdgeInsets.zero,
      childrenPadding: const EdgeInsets.only(bottom: 4),
      title: const Text('特征空间位置与最近类别'),
      subtitle: const Text('局部二维投影 + 真实高维距离'),
      children: [
        if (_loading)
          const Padding(
            padding: EdgeInsets.all(16),
            child: Center(child: CircularProgressIndicator(strokeWidth: 2)),
          )
        else if (_error != null)
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Text(
              '特征空间解释暂不可用：$_error',
              style: TextStyle(color: Theme.of(context).colorScheme.error),
            ),
          )
        else if (_explanation != null)
          _content(_explanation!),
      ],
    );
  }
}

class _FeatureScatterPainter extends CustomPainter {
  _FeatureScatterPainter({
    required this.points,
    required this.firstSpecies,
    required this.secondSpecies,
    required this.colors,
  });

  final List<DinoV3ProjectionPoint> points;
  final String firstSpecies;
  final String secondSpecies;
  final ColorScheme colors;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.width <= 0 || size.height <= 0 || points.isEmpty) return;
    const inset = 14.0;
    final plot = Rect.fromLTWH(
      inset,
      inset,
      math.max(1, size.width - inset * 2),
      math.max(1, size.height - inset * 2),
    );
    var minX = points.map((point) => point.x).reduce(math.min);
    var maxX = points.map((point) => point.x).reduce(math.max);
    var minY = points.map((point) => point.y).reduce(math.min);
    var maxY = points.map((point) => point.y).reduce(math.max);
    final xSpan = math.max((maxX - minX).abs(), 0.2);
    final ySpan = math.max((maxY - minY).abs(), 0.2);
    minX -= xSpan * 0.16;
    maxX += xSpan * 0.16;
    minY -= ySpan * 0.22;
    maxY += ySpan * 0.22;

    final domainWidth = math.max(maxX - minX, 1e-9);
    final domainHeight = math.max(maxY - minY, 1e-9);
    final scale = math.min(
      plot.width / domainWidth,
      plot.height / domainHeight,
    );
    final centerX = (minX + maxX) / 2;
    final centerY = (minY + maxY) / 2;

    Offset mapPoint(DinoV3ProjectionPoint point) => Offset(
      plot.center.dx + (point.x - centerX) * scale,
      plot.center.dy - (point.y - centerY) * scale,
    );

    double mapX(double x) => plot.center.dx + (x - centerX) * scale;
    double mapY(double y) => plot.center.dy - (y - centerY) * scale;

    final axisPaint = Paint()
      ..color = colors.outlineVariant
      ..strokeWidth = 1;
    if (minY <= 0 && maxY >= 0) {
      canvas.drawLine(
        Offset(plot.left, mapY(0)),
        Offset(plot.right, mapY(0)),
        axisPaint,
      );
    }
    if (minX <= 0 && maxX >= 0) {
      canvas.drawLine(
        Offset(mapX(0), plot.top),
        Offset(mapX(0), plot.bottom),
        axisPaint,
      );
    }

    for (final point in points) {
      final offset = mapPoint(point);
      if (point.isCurrent) {
        final paint = Paint()..color = colors.error;
        final path = Path()
          ..moveTo(offset.dx, offset.dy - 7)
          ..lineTo(offset.dx + 7, offset.dy)
          ..lineTo(offset.dx, offset.dy + 7)
          ..lineTo(offset.dx - 7, offset.dy)
          ..close();
        canvas.drawPath(path, paint);
        canvas.drawPath(
          path,
          Paint()
            ..color = colors.surface
            ..style = PaintingStyle.stroke
            ..strokeWidth = 2,
        );
        continue;
      }
      final color = point.species == firstSpecies
          ? colors.primary
          : point.species == secondSpecies
          ? colors.tertiary
          : colors.secondary;
      canvas.drawCircle(offset, 5, Paint()..color = color);
      canvas.drawCircle(
        offset,
        5,
        Paint()
          ..color = colors.surface
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _FeatureScatterPainter oldDelegate) {
    return oldDelegate.points != points ||
        oldDelegate.firstSpecies != firstSpecies ||
        oldDelegate.secondSpecies != secondSpecies ||
        oldDelegate.colors != colors;
  }
}
