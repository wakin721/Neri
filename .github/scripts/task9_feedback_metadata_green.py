from pathlib import Path

path = Path('frontend/lib/src/screens/species_validation_screen.dart')
text = path.read_text(encoding='utf-8')
old = """        return DetectionBox(\n          species: species,\n          confidence: confidence,\n          bbox: box.bbox,\n          frameIndex: box.frameIndex,\n          timestamp: box.timestamp,\n          trackId: box.trackId,\n          candidates: box.candidates,\n        );\n"""
new = """        return DetectionBox(\n          species: species,\n          confidence: confidence,\n          bbox: box.bbox,\n          frameIndex: box.frameIndex,\n          timestamp: box.timestamp,\n          trackId: box.trackId,\n          candidates: box.candidates,\n          observationId: box.observationId,\n          registryId: box.registryId,\n          predictedSpecies: box.predictedSpecies,\n          feedbackStatus: box.feedbackStatus,\n        );\n"""
if old not in text:
    raise SystemExit('confidence-filter box reconstruction anchor not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
