"""User-facing model names, independent of the shared inference backend."""
from collections.abc import Mapping


def model_display_name(manifest: Mapping | None) -> str:
    name = manifest.get("display_name") if isinstance(manifest, Mapping) else None
    return name.strip() if isinstance(name, str) and name.strip() else "联合模型"
