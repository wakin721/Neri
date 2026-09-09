from pathlib import Path

path = Path('frontend/lib/src/widgets/detection_media_viewer.dart')
text = path.read_text(encoding='utf-8')
old = """  List<DetectionBox> _currentBoxes(Duration position) {\n    return currentVideoDetectionBoxes(\n      boxes: widget.visibleBoxes,\n      position: position,\n      duration: _duration,\n      detectionData: widget.detectionData,\n    );\n  }\n\n"""
if old not in text:
    raise SystemExit('unused video current-box helper not found')
path.write_text(text.replace(old, '', 1), encoding='utf-8')
