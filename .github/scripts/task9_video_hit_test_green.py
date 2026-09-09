from pathlib import Path

path = Path('frontend/lib/src/widgets/detection_media_viewer.dart')
text = path.read_text(encoding='utf-8')

insert_anchor = """class _ValidationVideoPlayer extends StatefulWidget {\n"""
overlay = r'''class DinoVideoDetectionOverlay extends StatelessWidget {
  const DinoVideoDetectionOverlay({
    required this.boxes,
    required this.mediaSize,
    required this.position,
    required this.duration,
    required this.detectionData,
    this.selectedObservationId,
    this.onDetectionBoxSelected,
    this.onBackgroundTap,
    super.key,
  });

  final List<DetectionBox> boxes;
  final Size mediaSize;
  final Duration position;
  final Duration duration;
  final Map<String, dynamic> detectionData;
  final String? selectedObservationId;
  final ValueChanged<DetectionBox?>? onDetectionBoxSelected;
  final VoidCallback? onBackgroundTap;

  @override
  Widget build(BuildContext context) {
    final currentBoxes = currentVideoDetectionBoxes(
      boxes: boxes,
      position: position,
      duration: duration,
      detectionData: detectionData,
    );
    return LayoutBuilder(
      builder: (context, constraints) {
        final viewport = Size(constraints.maxWidth, constraints.maxHeight);
        return GestureDetector(
          behavior: HitTestBehavior.translucent,
          onTapUp: onDetectionBoxSelected == null && onBackgroundTap == null
              ? null
              : (details) {
                  final selected = _hitTestDinoBox(
                    details.localPosition,
                    viewport,
                    mediaSize,
                    currentBoxes,
                  );
                  onDetectionBoxSelected?.call(selected);
                  if (selected == null) onBackgroundTap?.call();
                },
          child: CustomPaint(
            painter: _DetectionOverlayPainter(
              boxes: currentBoxes,
              mediaSize: mediaSize,
            ),
            child: const SizedBox.expand(),
          ),
        );
      },
    );
  }
}

'''
if insert_anchor not in text:
    raise SystemExit('video player class anchor not found')
text = text.replace(insert_anchor, overlay + insert_anchor, 1)

old = """        detectionData: item.detectionData,\n        onOpenExternal: onOpenExternal,\n      );\n"""
new = """        detectionData: item.detectionData,\n        onOpenExternal: onOpenExternal,\n        selectedObservationId: selectedObservationId,\n        onDetectionBoxSelected: onDetectionBoxSelected,\n      );\n"""
if old not in text:
    raise SystemExit('media-content video wiring anchor not found')
text = text.replace(old, new, 1)

old = """    required this.detectionData,\n    required this.onOpenExternal,\n  });\n\n  final String path;\n  final List<DetectionBox> visibleBoxes;\n  final bool showDetections;\n  final Map<String, dynamic> detectionData;\n  final VoidCallback onOpenExternal;\n"""
new = """    required this.detectionData,\n    required this.onOpenExternal,\n    this.selectedObservationId,\n    this.onDetectionBoxSelected,\n  });\n\n  final String path;\n  final List<DetectionBox> visibleBoxes;\n  final bool showDetections;\n  final Map<String, dynamic> detectionData;\n  final VoidCallback onOpenExternal;\n  final String? selectedObservationId;\n  final ValueChanged<DetectionBox?>? onDetectionBoxSelected;\n"""
if old not in text:
    raise SystemExit('video player constructor anchor not found')
text = text.replace(old, new, 1)

old = """                return RepaintBoundary(\n                  child: IgnorePointer(\n                    child: CustomPaint(\n                      painter: _DetectionOverlayPainter(\n                        boxes: _currentBoxes(position),\n                        mediaSize: _videoSize!,\n                      ),\n                    ),\n                  ),\n                );\n"""
new = """                return RepaintBoundary(\n                  child: DinoVideoDetectionOverlay(\n                    boxes: widget.visibleBoxes,\n                    mediaSize: _videoSize!,\n                    position: position,\n                    duration: _duration,\n                    detectionData: widget.detectionData,\n                    selectedObservationId: widget.selectedObservationId,\n                    onDetectionBoxSelected: widget.onDetectionBoxSelected,\n                    onBackgroundTap: _togglePlayPause,\n                  ),\n                );\n"""
if old not in text:
    raise SystemExit('video overlay paint anchor not found')
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
