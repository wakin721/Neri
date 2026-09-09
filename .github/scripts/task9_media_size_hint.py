from pathlib import Path

path = Path('frontend/lib/src/widgets/detection_media_viewer.dart')
text = path.read_text(encoding='utf-8')

replacements = [
    (
        """    } else {
      return _ImageMediaViewer(
        path: item.path,
        visibleBoxes: visibleBoxes,
        showDetections: showDetections,
        selectedObservationId: selectedObservationId,
        onDetectionBoxSelected: onDetectionBoxSelected,
      );
    }
""",
        """    } else {
      final itemWidth = item.width;
      final itemHeight = item.height;
      final mediaSizeHint =
          itemWidth != null &&
              itemWidth > 0 &&
              itemHeight != null &&
              itemHeight > 0
          ? Size(itemWidth.toDouble(), itemHeight.toDouble())
          : null;
      return _ImageMediaViewer(
        path: item.path,
        visibleBoxes: visibleBoxes,
        showDetections: showDetections,
        mediaSizeHint: mediaSizeHint,
        selectedObservationId: selectedObservationId,
        onDetectionBoxSelected: onDetectionBoxSelected,
      );
    }
""",
    ),
    (
        """    required this.showDetections,
    this.selectedObservationId,
    this.onDetectionBoxSelected,
  });

  final String path;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
  final String? selectedObservationId;
""",
        """    required this.showDetections,
    this.mediaSizeHint,
    this.selectedObservationId,
    this.onDetectionBoxSelected,
  });

  final String path;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
  final Size? mediaSizeHint;
  final String? selectedObservationId;
""",
    ),
    (
        """    if (oldWidget.path != widget.path) {
      _resolveImageSize();
    }
""",
        """    if (oldWidget.path != widget.path ||
        oldWidget.mediaSizeHint != widget.mediaSizeHint) {
      _resolveImageSize();
    }
""",
    ),
    (
        """    _removeImageStreamListener();
    _imageSize = null;
    _imageError = null;
""",
        """    _removeImageStreamListener();
    _imageSize = widget.mediaSizeHint;
    _imageError = null;
""",
    ),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(f'anchor not found: {old!r}')
    text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
