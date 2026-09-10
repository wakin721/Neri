from __future__ import annotations

from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"missing anchor: {label}")
    return text.replace(old, new, 1)


EXPLANATION_MODEL = r'''class DinoV3NearestSpecies {
  const DinoV3NearestSpecies({
    required this.name,
    required this.nearestPrototypeIndex,
    required this.squaredDistance,
    required this.cosineScore,
    required this.source,
    this.registryId,
    this.registrationStatus,
  });

  factory DinoV3NearestSpecies.fromJson(Map<String, dynamic> json) {
    return DinoV3NearestSpecies(
      name: json['name']?.toString() ?? '',
      nearestPrototypeIndex:
          (json['nearest_prototype_index'] as num?)?.toInt() ?? -1,
      squaredDistance: (json['squared_distance'] as num?)?.toDouble() ?? 0,
      cosineScore: (json['cosine_score'] as num?)?.toDouble() ?? 0,
      source: json['source']?.toString() ?? '',
      registryId: (json['registry_id'] as num?)?.toInt(),
      registrationStatus: json['registration_status']?.toString(),
    );
  }

  final String name;
  final int nearestPrototypeIndex;
  final double squaredDistance;
  final double cosineScore;
  final String source;
  final int? registryId;
  final String? registrationStatus;
}

class DinoV3ProjectionPoint {
  const DinoV3ProjectionPoint({
    required this.kind,
    required this.species,
    required this.source,
    required this.prototypeIndex,
    required this.x,
    required this.y,
    this.registryId,
    this.registrationStatus,
  });

  factory DinoV3ProjectionPoint.fromJson(Map<String, dynamic> json) {
    return DinoV3ProjectionPoint(
      kind: json['kind']?.toString() ?? 'prototype',
      species: json['species']?.toString() ?? '',
      source: json['source']?.toString() ?? '',
      prototypeIndex: (json['prototype_index'] as num?)?.toInt(),
      x: (json['x'] as num?)?.toDouble() ?? 0,
      y: (json['y'] as num?)?.toDouble() ?? 0,
      registryId: (json['registry_id'] as num?)?.toInt(),
      registrationStatus: json['registration_status']?.toString(),
    );
  }

  final String kind;
  final String species;
  final String source;
  final int? prototypeIndex;
  final double x;
  final double y;
  final int? registryId;
  final String? registrationStatus;

  bool get isCurrent => kind == 'current';
}

class DinoV3FeatureProjection {
  const DinoV3FeatureProjection({
    required this.method,
    required this.species,
    required this.points,
  });

  factory DinoV3FeatureProjection.fromJson(Map<String, dynamic> json) {
    return DinoV3FeatureProjection(
      method: json['method']?.toString() ?? '',
      species: (json['species'] as List<dynamic>? ?? const <dynamic>[])
          .map((value) => value.toString())
          .toList(growable: false),
      points: (json['points'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<Map<String, dynamic>>()
          .map(DinoV3ProjectionPoint.fromJson)
          .toList(growable: false),
    );
  }

  final String method;
  final List<String> species;
  final List<DinoV3ProjectionPoint> points;
}

class DinoV3NearestExample {
  const DinoV3NearestExample({
    required this.kind,
    required this.species,
    this.registrationId,
    this.eventId,
    this.observationId,
  });

  factory DinoV3NearestExample.fromJson(Map<String, dynamic> json) {
    return DinoV3NearestExample(
      kind: json['kind']?.toString() ?? '',
      species: json['species']?.toString() ?? '',
      registrationId: (json['registration_id'] as num?)?.toInt(),
      eventId: (json['event_id'] as num?)?.toInt(),
      observationId: json['observation_id']?.toString(),
    );
  }

  final String kind;
  final String species;
  final int? registrationId;
  final int? eventId;
  final String? observationId;
}

class DinoV3FeatureExplanation {
  const DinoV3FeatureExplanation({
    required this.species,
    required this.accepted,
    required this.bestKnownSpecies,
    required this.knownScore,
    required this.threshold,
    required this.nearestSpecies,
    required this.projection,
    required this.currentExampleAvailable,
    this.nearestPrototypeIndex,
    this.squaredDistance,
    this.nearestExample,
  });

  factory DinoV3FeatureExplanation.fromJson(Map<String, dynamic> json) {
    final projection = json['projection'];
    final nearestExample = json['nearest_example'];
    return DinoV3FeatureExplanation(
      species: json['species']?.toString() ?? 'Unknown',
      accepted: json['accepted'] == true,
      bestKnownSpecies: json['best_known_species']?.toString() ?? '',
      knownScore: (json['known_score'] as num?)?.toDouble() ?? 0,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0,
      nearestPrototypeIndex:
          (json['nearest_prototype_index'] as num?)?.toInt(),
      squaredDistance: (json['squared_distance'] as num?)?.toDouble(),
      nearestSpecies:
          (json['nearest_species'] as List<dynamic>? ?? const <dynamic>[])
              .whereType<Map<String, dynamic>>()
              .map(DinoV3NearestSpecies.fromJson)
              .toList(growable: false),
      projection: DinoV3FeatureProjection.fromJson(
        projection is Map<String, dynamic>
            ? projection
            : const <String, dynamic>{},
      ),
      currentExampleAvailable: json['current_example_available'] == true,
      nearestExample: nearestExample is Map<String, dynamic>
          ? DinoV3NearestExample.fromJson(nearestExample)
          : null,
    );
  }

  final String species;
  final bool accepted;
  final String bestKnownSpecies;
  final double knownScore;
  final double threshold;
  final int? nearestPrototypeIndex;
  final double? squaredDistance;
  final List<DinoV3NearestSpecies> nearestSpecies;
  final DinoV3FeatureProjection projection;
  final bool currentExampleAvailable;
  final DinoV3NearestExample? nearestExample;
}
'''


SCATTER_WIDGET = r'''import 'dart:async';
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
          Row(
            children: [
              Expanded(
                child: Text(
                  first,
                  style: const TextStyle(fontWeight: FontWeight.w600),
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
              _imageCard('最近类别代表图 · $closestName', _nearestExample, nearestExampleLabel),
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

    Offset mapPoint(DinoV3ProjectionPoint point) {
      final nx = (point.x - minX) / (maxX - minX);
      final ny = (point.y - minY) / (maxY - minY);
      return Offset(
        plot.left + nx * plot.width,
        plot.bottom - ny * plot.height,
      );
    }

    double mapX(double x) =>
        plot.left + ((x - minX) / (maxX - minX)) * plot.width;
    double mapY(double y) =>
        plot.bottom - ((y - minY) / (maxY - minY)) * plot.height;

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
'''


def patch_api() -> None:
    path = "frontend/lib/src/api_client_core.dart"
    text = read(path)
    if "models/dinov3_explanation.dart" not in text:
        text = replace_once(
            text,
            "import 'models/dinov3_feedback.dart';\n",
            "import 'models/dinov3_explanation.dart';\nimport 'models/dinov3_feedback.dart';\n",
            "api explanation import",
        )
    if "fetchDinoV3RegistryCatalog" not in text:
        marker = "  Future<DinoV3RegistryEntry> fetchDinoV3RegistryEntry(\n"
        addition = r'''  Future<List<DinoV3RegistryEntry>> fetchDinoV3RegistryCatalog(
    String classificationModelPath,
  ) async {
    final uri = _uri('/api/dinov3/registry/catalog').replace(
      queryParameters: {'classification_model_path': classificationModelPath},
    );
    final response = await _httpClient.get(uri);
    _ensureSuccess(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(DinoV3RegistryEntry.fromJson)
        .toList();
  }

'''
        if marker not in text:
            raise RuntimeError("api catalog anchor missing")
        text = text.replace(marker, addition + marker, 1)
    if "fetchDinoV3FeatureExplanation" not in text:
        marker = "  Future<DinoV3FeedbackRevertResult> revertDinoV3Feedback({\n"
        addition = r'''  Future<DinoV3FeatureExplanation> fetchDinoV3FeatureExplanation(
    String classificationModelPath,
    String observationId,
  ) async {
    final uri = _uri(
      '/api/dinov3/feedback/observations/${Uri.encodeComponent(observationId)}/explain',
    ).replace(
      queryParameters: {'classification_model_path': classificationModelPath},
    );
    final response = await _httpClient.get(uri);
    _ensureSuccess(response);
    return DinoV3FeatureExplanation.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<Uint8List> fetchDinoV3ObservationExample(
    String classificationModelPath,
    String observationId,
  ) async {
    final uri = _uri(
      '/api/dinov3/feedback/observations/${Uri.encodeComponent(observationId)}/example',
    ).replace(
      queryParameters: {'classification_model_path': classificationModelPath},
    );
    final response = await _httpClient.get(uri);
    _ensureSuccess(response);
    return response.bodyBytes;
  }

'''
        if marker not in text:
            raise RuntimeError("api explain anchor missing")
        text = text.replace(marker, addition + marker, 1)
    write(path, text)


def patch_registry_model() -> None:
    path = "frontend/lib/src/models/dinov3_registry.dart"
    text = read(path)
    if "bool get isCheckpoint" not in text:
        marker = "  bool get isCandidate => status == 'candidate';\n\n"
        addition = "  bool get isCandidate => status == 'candidate';\n  bool get isCheckpoint => status.toLowerCase() == 'checkpoint';\n\n"
        text = replace_once(text, marker, addition, "registry isCheckpoint")
    write(path, text)


def patch_registry_dialog() -> None:
    path = "frontend/lib/src/widgets/dinov3_registry_dialog.dart"
    text = read(path)
    text = text.replace(
        "  return 'Candidate ${count('candidate')} · '",
        "  return 'Checkpoint ${count('checkpoint')} · Candidate ${count('candidate')} · '",
        1,
    )
    text = text.replace("fetchDinoV3Registry(\n", "fetchDinoV3RegistryCatalog(\n", 2)
    old = "    unawaited(_loadExamples(entry));\n  }\n\n  Future<void> _load() async {"
    new = "    if (entry.isCheckpoint) {\n      ++_examplesRequestId;\n      if (notify && mounted) setState(() => _examplesLoading = false);\n    } else {\n      unawaited(_loadExamples(entry));\n    }\n  }\n\n  Future<void> _load() async {"
    if old in text:
        text = text.replace(old, new, 1)
    old = "  Future<void> _saveIdentity() async {\n    final selected = _selected;\n"
    new = "  Future<void> _saveIdentity() async {\n    final selected = _selected;\n    if (selected?.isCheckpoint == true) return;\n"
    if old in text:
        text = text.replace(old, new, 1)
    old = "  Future<void> _loadExamples(DinoV3RegistryEntry entry) async {\n    final requestId = ++_examplesRequestId;\n"
    new = "  Future<void> _loadExamples(DinoV3RegistryEntry entry) async {\n    if (entry.isCheckpoint) {\n      ++_examplesRequestId;\n      if (mounted && _selected?.id == entry.id) {\n        setState(() {\n          _events = const <DinoV3RegistryEvent>[];\n          _exampleBytes = const <int, Uint8List>{};\n          _examplesLoading = false;\n        });\n      }\n      return;\n    }\n    final requestId = ++_examplesRequestId;\n"
    if old in text:
        text = text.replace(old, new, 1)
    old = "  Widget _buildExampleGallery() {\n    if (_examplesLoading) {"
    new = "  Widget _buildExampleGallery() {\n    if (_selected?.isCheckpoint == true) {\n      return SizedBox(\n        height: 72,\n        child: Align(\n          alignment: Alignment.centerLeft,\n          child: Text(\n            'Checkpoint 不包含原始训练图片；暂无 Registry 裁切例图',\n            style: TextStyle(\n              color: Theme.of(context).colorScheme.onSurfaceVariant,\n            ),\n          ),\n        ),\n      );\n    }\n    if (_examplesLoading) {"
    if old in text:
        text = text.replace(old, new, 1)
    old = "  Future<void> _continueValidation() async {\n    final selected = _selected;\n    if (selected == null) return;\n"
    new = "  Future<void> _continueValidation() async {\n    final selected = _selected;\n    if (selected == null || selected.isCheckpoint) return;\n"
    if old in text:
        text = text.replace(old, new, 1)
    text = text.replace(
        "    final editable = entry.isCandidate;\n",
        "    final editable = entry.isCandidate && !entry.isCheckpoint;\n",
        1,
    )
    text = text.replace(
        "                Text('状态：${entry.status}'),\n                const SizedBox(height: 16),\n",
        "                Text(\n                  entry.isCheckpoint ? '状态：分类头基础物种' : '状态：${entry.status}',\n                ),\n                if (entry.isCheckpoint) ...[\n                  const SizedBox(height: 8),\n                  Text(\n                    '来自不可变 DINOv3 Checkpoint · ${entry.prototypeCount} 个 prototype。'\n                    '分类头保存特征中心而非原始训练影像。',\n                  ),\n                ],\n                const SizedBox(height: 16),\n",
        1,
    )
    text = text.replace(
        "                const SizedBox(height: 14),\n                Text('注册条件', style: Theme.of(context).textTheme.titleSmall),\n                _conditionRow('≥5 个独立事件', conditions['events'] == true),\n",
        "                if (!entry.isCheckpoint) ...[\n                  const SizedBox(height: 14),\n                  Text('注册条件', style: Theme.of(context).textTheme.titleSmall),\n                  _conditionRow('≥5 个独立事件', conditions['events'] == true),\n",
        1,
    )
    text = text.replace(
        "                _conditionRow('已确认物种名称', conditions['identity'] == true),\n                if (_error != null) ...[",
        "                  _conditionRow('已确认物种名称', conditions['identity'] == true),\n                ],\n                if (_error != null) ...[",
        1,
    )
    text = text.replace(
        "            OutlinedButton.icon(\n              onPressed: _saving ? null : _continueValidation,\n              icon: const Icon(Icons.fact_check_outlined),\n              label: const Text('继续验证'),\n            ),\n            const SizedBox(width: 8),\n",
        "            if (!entry.isCheckpoint) ...[\n              OutlinedButton.icon(\n                onPressed: _saving ? null : _continueValidation,\n                icon: const Icon(Icons.fact_check_outlined),\n                label: const Text('继续验证'),\n              ),\n              const SizedBox(width: 8),\n            ],\n",
        1,
    )
    text = text.replace(
        "                                        '${entry.status} · ${entry.eventCount} 事件 · ${entry.cameraCount} 相机',\n",
        "                                        entry.isCheckpoint\n                                            ? '分类头基础物种 · ${entry.prototypeCount} prototypes'\n                                            : '${entry.status} · ${entry.eventCount} 事件 · ${entry.cameraCount} 相机',\n",
        1,
    )
    write(path, text)


def patch_validation() -> None:
    path = "frontend/lib/src/screens/species_validation_screen.dart"
    text = read(path)
    if "widgets/dinov3_feature_scatter.dart" not in text:
        text = replace_once(
            text,
            "import '../widgets/detection_media_viewer.dart';\n",
            "import '../widgets/detection_media_viewer.dart';\nimport '../widgets/dinov3_feature_scatter.dart';\n",
            "validation scatter import",
        )
    old_start = r'''    return _ValidationPanel(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        child: Row(
          children: [
'''
    new_start = r'''    return _ValidationPanel(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
'''
    if old_start in text:
        text = text.replace(old_start, new_start, 1)
    old_end = r'''            OutlinedButton(
              onPressed: canSubmit
                  ? () => unawaited(_submitDinoBoxFeedback(box, 'unverified'))
                  : null,
              child: const Text('不参与学习'),
            ),
          ],
        ),
      ),
    );
  }
'''
    new_end = r'''            OutlinedButton(
              onPressed: canSubmit
                  ? () => unawaited(_submitDinoBoxFeedback(box, 'unverified'))
                  : null,
              child: const Text('不参与学习'),
            ),
              ],
            ),
            if (classificationModelPath.isNotEmpty && observationId.isNotEmpty) ...[
              const SizedBox(height: 6),
              DinoV3FeatureExplanationPanel(
                apiClient: widget.apiClient,
                classificationModelPath: classificationModelPath,
                observationId: observationId,
              ),
            ],
          ],
        ),
      ),
    );
  }
'''
    if old_end not in text:
        raise RuntimeError("validation feedback panel end anchor missing")
    text = text.replace(old_end, new_end, 1)
    write(path, text)


if __name__ == "__main__":
    write("frontend/lib/src/models/dinov3_explanation.dart", EXPLANATION_MODEL)
    write("frontend/lib/src/widgets/dinov3_feature_scatter.dart", SCATTER_WIDGET)
    patch_api()
    patch_registry_model()
    patch_registry_dialog()
    patch_validation()
