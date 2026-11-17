import 'package:flutter/material.dart';

import '../models/job.dart';
import '../models/rejection_cluster.dart';
import 'dino_rejection.dart';

/// The source record remains Unknown. Cluster names are only visual labels.
RejectionCluster? detectionBoxDisplayCluster(DetectionBox box) {
  if (box.feedbackStatus == 'confirmed' ||
      box.feedbackStatus == 'image_label') {
    return null;
  }
  final name = dinoFinalBox(box).species.trim().toLowerCase();
  if (name != 'unknown' && !name.contains('拒识')) return null;
  if (box.candidates.any(
    (candidate) => candidate['sequence_pending'] == true,
  )) {
    return null;
  }
  if (box.rejectionCluster.isEmpty) return null;
  final cluster = RejectionCluster.fromJson(box.rejectionCluster);
  return cluster.isClustered ? cluster : null;
}

DetectionBox detectionBoxWithDisplayMetadata(
  DetectionItem item,
  DetectionBox box,
  int index,
) {
  if (item.isSequencePending) {
    return box.rejectionCluster.isEmpty
        ? box
        : box.withRejectionCluster(const {});
  }
  if (box.rejectionCluster.isNotEmpty) return box;
  final raw = findRawDinoBox(box, index, item.detectionData['检测框']);
  final metadata = raw?['rejection_cluster'];
  return metadata is Map
      ? box.withRejectionCluster(Map<String, dynamic>.from(metadata))
      : box;
}

String detectionBoxDisplayName(DetectionBox box) {
  final cluster = detectionBoxDisplayCluster(box);
  return cluster == null ? box.species : '簇${cluster.number}';
}

List<DetectionBox> detectionBoxesWithDisplayMetadata(
  DetectionItem item,
  List<DetectionBox> boxes,
) {
  List<DetectionBox>? updated;
  for (var i = 0; i < boxes.length; i++) {
    final next = detectionBoxWithDisplayMetadata(item, boxes[i], i);
    if (!identical(next, boxes[i])) {
      updated ??= List<DetectionBox>.of(boxes);
      updated[i] = next;
    }
  }
  return updated ?? boxes;
}

Color detectionBoxColor(DetectionBox box) {
  final cluster = detectionBoxDisplayCluster(box);
  // Golden-angle spacing gives adjacent cluster numbers contrasting hues.
  final hue = cluster == null
      ? (box.species.hashCode % 360).toDouble()
      : ((cluster.number - 1) * 137.508) % 360;
  return HSVColor.fromAHSV(1, hue, .85, .95).toColor();
}

String detectionBoxLabel(DetectionBox box, int index) {
  final track = box.trackId?.trim() ?? '';
  final number = RegExp(r'^\d{1,6}$').hasMatch(track) ? track : '${index + 1}';
  final confidence = box.confidence?.toStringAsFixed(2) ?? 'N/A';
  final kind = dinoConfidenceLabel(box);
  final prefix = kind == '分类证据'
      ? '分类'
      : kind == '探测证据'
      ? '探测'
      : kind == '事件接纳'
      ? '事件'
      : '';
  return '#$number ${detectionBoxDisplayName(box)} $prefix$confidence';
}
