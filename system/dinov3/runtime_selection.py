"""Runtime selection and hardware probing for DINOv3 CPU (LiteRT) vs GPU (PyTorch)."""
from __future__ import annotations

from typing import Literal

RUNTIME_CHOICES = ("auto", "cpu", "gpu")
RESOLVED_RUNTIME_CHOICES = ("cpu", "gpu")

RuntimeChoice = Literal["auto", "cpu", "gpu"]
ResolvedRuntimeChoice = Literal["cpu", "gpu"]


def validate_runtime_choice(runtime_choice: object) -> RuntimeChoice:
    """Validate the three-value runtime selector strictly."""
    if not isinstance(runtime_choice, str) or runtime_choice not in RUNTIME_CHOICES:
        raise ValueError(
            f"Unsupported runtime_choice: {runtime_choice!r}; expected one of 'auto', 'cpu', 'gpu'."
        )
    return runtime_choice  # type: ignore[return-value]


def validate_resolved_runtime(runtime: object) -> ResolvedRuntimeChoice:
    """Validate a concrete target runtime ('cpu' or 'gpu')."""
    if not isinstance(runtime, str) or runtime not in RESOLVED_RUNTIME_CHOICES:
        raise ValueError(
            f"Unsupported runtime: {runtime!r}; expected 'cpu' or 'gpu'."
        )
    return runtime  # type: ignore[return-value]


def detect_supported_gpu() -> bool:
    """Probe supported CUDA/XPU GPU hardware without requiring torch."""
    from system.backend import maintenance as _maintenance

    cpu_index = _maintenance.PYTORCH_CPU_INDEX_URL.rstrip("/")
    try:
        _label, index_url = _maintenance._auto_detect_pytorch_index()
    except RuntimeError:
        if _maintenance.XPU_ENABLED:
            intel_choice = _maintenance._detect_intel_xpu_pytorch_index()
            if intel_choice is not None:
                return intel_choice[1].rstrip("/") != cpu_index
        return False
    return index_url.rstrip("/") != cpu_index


def resolve_runtime_choice(runtime_choice: str = "auto") -> ResolvedRuntimeChoice:
    """Resolve 'auto', 'cpu', or 'gpu' into a concrete 'cpu' or 'gpu' runtime."""
    choice = validate_runtime_choice(runtime_choice)
    if choice == "cpu":
        return "cpu"
    if choice == "gpu":
        return "gpu"
    return "gpu" if detect_supported_gpu() else "cpu"


def expected_encoder_backend(resolved_runtime: str) -> Literal["litert", "pytorch"]:
    """Return the manifest encoder_backend expected for a resolved runtime."""
    runtime = validate_resolved_runtime(resolved_runtime)
    return "litert" if runtime == "cpu" else "pytorch"


def runtime_choice_from_encoder_backend(
    encoder_backend: str | None,
) -> ResolvedRuntimeChoice | None:
    """Infer the concrete runtime ('cpu' or 'gpu') from manifest encoder_backend."""
    if encoder_backend == "litert":
        return "cpu"
    if encoder_backend == "pytorch":
        return "gpu"
    return None
