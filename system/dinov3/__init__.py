"""Native DINOv3 classification and open-world species registration."""
from pathlib import PurePosixPath

from .checkpoint import (
    DINO_BACKBONE,
    DINO_FEATURE_DIM,
    CheckpointValidationError,
    DinoV3Checkpoint,
    load_checkpoint,
    validate_checkpoint,
)
from .component import (
    DINO_BACKBONE_FILENAME,
    DINO_BACKBONE_SHA256,
    DINO_CLASSIFIER_FILENAME,
    DINO_CLASSIFIER_SHA256,
    DINO_MODEL_MANIFEST_FILENAME,
    DINO_SOURCE_COMMIT,
    DinoV3ComponentPaths,
    dinov3_component_paths,
    dinov3_component_status,
    install_dinov3_component as _install_dinov3_component,
    remove_dinov3_component,
)
from .public_distribution import (
    DinoV3DistributionClient,
    DinoV3DistributionError as DinoV3CloudError,
)


class DinoV3CloudClient(DinoV3DistributionClient):
    """Compatibility name for the anonymous DINOv3 distribution client."""

    @staticmethod
    def mount_path(relative_path: str = "") -> str:
        raw = str(relative_path or "").replace("\\", "/").strip("/")
        if not raw:
            return "/Neri_Data/Model/DINOv3"
        pure = PurePosixPath(raw)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError("DINOv3 NeriCloud 路径无效。")
        return "/Neri_Data/Model/DINOv3/" + pure.as_posix()


def install_dinov3_component(
    *,
    root=None,
    cloud_client=None,
    seed_dir=None,
    on_progress=None,
):
    """Install DINOv3 through the anonymous Neri model-distribution service."""
    return _install_dinov3_component(
        root=root,
        cloud_client=cloud_client or DinoV3CloudClient(),
        seed_dir=seed_dir,
        on_progress=on_progress,
    )


__all__ = [
    "DINO_BACKBONE",
    "DINO_FEATURE_DIM",
    "DINO_BACKBONE_FILENAME",
    "DINO_BACKBONE_SHA256",
    "DINO_CLASSIFIER_FILENAME",
    "DINO_CLASSIFIER_SHA256",
    "DINO_MODEL_MANIFEST_FILENAME",
    "DINO_SOURCE_COMMIT",
    "CheckpointValidationError",
    "DinoV3Checkpoint",
    "DinoV3CloudClient",
    "DinoV3CloudError",
    "DinoV3ComponentPaths",
    "load_checkpoint",
    "validate_checkpoint",
    "dinov3_component_paths",
    "dinov3_component_status",
    "install_dinov3_component",
    "remove_dinov3_component",
]
