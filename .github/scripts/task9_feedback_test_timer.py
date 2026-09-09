from pathlib import Path

path = Path('frontend/test/dinov3_box_feedback_ui_test.dart')
text = path.read_text(encoding='utf-8')
old = """    final operationId = feedbackBody['feedback_operation_id']?.toString() ?? '';\n    expect(operationId, isNotEmpty);\n  });\n}\n"""
new = """    final operationId = feedbackBody['feedback_operation_id']?.toString() ?? '';\n    expect(operationId, isNotEmpty);\n    await tester.pump(const Duration(seconds: 4));\n    await tester.pumpAndSettle();\n  });\n}\n"""
if old not in text:
    raise SystemExit('feedback timer test anchor not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
