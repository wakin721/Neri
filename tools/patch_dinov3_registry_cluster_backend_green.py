from pathlib import Path

path = Path("system/dinov3/feedback.py")
text = path.read_text(encoding="utf-8")
old = '''                _EvidenceObservation(
                    row_id=int(row["id"]),
                    source_path=str(payload.get("source_path") or ""),
'''
new = '''                _EvidenceObservation(
                    row_id=int(row["id"]),
                    observation_id=str(row["observation_id"]),
                    source_path=str(payload.get("source_path") or ""),
'''
if old not in text:
    if new not in text:
        raise RuntimeError("feedback evidence observation constructor anchor not found")
else:
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")

print("Patched DINOv3 registry cluster backend GREEN issues")
