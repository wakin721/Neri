"""Verify that a portable application's built-in LiteRT detector really runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def verify(application_root):
    root = Path(application_root).resolve(strict=True)
    sys.path.insert(0, str(root))
    import system
    if not Path(system.__file__).resolve().is_relative_to(root):
        raise ValueError('Verification imported a different application')
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    from system.model_sync.layout import get_model_layout
    from system.model_sync.catalog import discover_models
    from system.backend.model_services import _to_model_info
    expected = root/'res/model/detect/bundled/MDV6-yolov10-c.tflite'
    discovered = discover_models(get_model_layout(root/'res'), 'detect')
    model = next((row for row in discovered if Path(row.path) == expected), None)
    if model is None or model.source != 'bundled' or model.backend != 'litert':
        raise ValueError('Bundled detector is absent from the application catalog')
    _to_model_info(model)
    interpreter = Interpreter(model_path=str(expected), num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_details['index'],
        np.zeros(input_details['shape'], dtype=input_details['dtype']))
    interpreter.invoke()
    outputs = [interpreter.get_tensor(row['index']) for row in interpreter.get_output_details()]
    if not outputs or not all(np.isfinite(output).all() for output in outputs):
        raise ValueError('Bundled detector returned invalid outputs')
    report = dict(passed=True, model=model.name, source=model.source, backend=model.backend,
                  smoke_inference=True, outputs=[list(output.shape) for output in outputs])
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--application-root', type=Path, required=True)
    verify(parser.parse_args().application_root)