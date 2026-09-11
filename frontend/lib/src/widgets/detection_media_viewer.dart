import 'package:flutter/material.dart';

import '../dino_validation_selection.dart';
import '../models/job.dart';
import 'detection_media_viewer_impl.dart' as impl;

export 'detection_media_viewer_impl.dart' hide DetectionMediaViewer;

class DetectionMediaViewer extends StatelessWidget {
  const DetectionMediaViewer({
    required this.item,
    required this.visibleBoxes,
    required this.showDetections,
    required this.onOpenExternal,
    this.isFavorite = false,
    this.onToggleFavorite,
    this.selectedObservationId,
    this.onDetectionBoxSelected,
    super.key,
  });

  final DetectionItem item;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
  final VoidCallback onOpenExternal;
  final bool isFavorite;
  final VoidCallback? onToggleFavorite;
  final String? selectedObservationId;
  final ValueChanged<DetectionBox?>? onDetectionBoxSelected;

  @override
  Widget build(BuildContext context) {
    final selectionCallback = onDetectionBoxSelected;
    return impl.DetectionMediaViewer(
      item: item,
      visibleBoxes: visibleBoxes,
      showDetections: showDetections,
      onOpenExternal: onOpenExternal,
      isFavorite: isFavorite,
      onToggleFavorite: onToggleFavorite,
      selectedObservationId: selectedObservationId,
      onDetectionBoxSelected: selectionCallback == null
          ? null
          : (box) {
              recordDinoValidationBoxSelection(item, box);
              selectionCallback(box);
            },
    );
  }
}
