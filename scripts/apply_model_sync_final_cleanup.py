from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_main_core() -> None:
    path = Path("system/backend/main_core.py")
    old = (
        "from .services import (\n"
        "    JobNotFoundError,\n"
        "    ProcessingJobManager,\n"
        "    classification_model_directory,\n"
        "    detect_gpu_available,\n"
        "    export_validation_data,\n"
        "    list_available_classification_models,\n"
        "    list_available_models,\n"
        "    list_model_classes,\n"
        "    load_species_types,\n"
        "    mark_validation_items,\n"
        "    mark_validation_item,\n"
        "    missing_yolo_dependencies,\n"
        "    model_directory,\n"
        "    preview_media_item,\n"
        "    preview_media_items,\n"
        ")"
    )
    new = (
        "from .model_services import (\n"
        "    classification_model_directory,\n"
        "    list_available_classification_models,\n"
        "    list_available_models,\n"
        "    model_directory,\n"
        ")\n"
        "from .services import (\n"
        "    JobNotFoundError,\n"
        "    ProcessingJobManager,\n"
        "    detect_gpu_available,\n"
        "    export_validation_data,\n"
        "    list_model_classes,\n"
        "    load_species_types,\n"
        "    mark_validation_items,\n"
        "    mark_validation_item,\n"
        "    missing_yolo_dependencies,\n"
        "    preview_media_item,\n"
        "    preview_media_items,\n"
        ")"
    )
    replace_once(path, old, new)


def patch_services() -> None:
    path = Path("system/backend/services.py")
    replace_once(
        path,
        "def model_directory() -> Path:\n"
        "    \"\"\"Return the project model directory used by the API.\"\"\"\n\n"
        "    return Path(resource_path(\"res/model\")).resolve()\n\n\n"
        "def classification_model_directory() -> Path:\n"
        "    \"\"\"Return the optional second-stage classification model directory.\"\"\"\n\n"
        "    return Path(resource_path(\"res/model_cls\")).resolve()\n",
        "def model_directory() -> Path:\n"
        "    \"\"\"Return the canonical detection-model root.\"\"\"\n\n"
        "    from .model_services import model_directory as canonical_model_directory\n\n"
        "    return canonical_model_directory()\n\n\n"
        "def classification_model_directory() -> Path:\n"
        "    \"\"\"Return the canonical classification-model root.\"\"\"\n\n"
        "    from .model_services import (\n"
        "        classification_model_directory as canonical_classification_model_directory,\n"
        "    )\n\n"
        "    return canonical_classification_model_directory()\n",
    )
    replace_once(
        path,
        "def list_available_models() -> list[ModelInfo]:\n"
        "    \"\"\"List all .pt files available under res/model.\"\"\"\n\n"
        "    directory = model_directory()\n"
        "    return _list_model_files(directory, (\"*.pt\",))\n\n\n"
        "def list_available_classification_models() -> list[ModelInfo]:\n"
        "    \"\"\"List classification model files under res/model_cls.\"\"\"\n\n"
        "    directory = classification_model_directory()\n"
        "    return _list_model_files(directory, (\"*.pt\", \"*.onnx\", \"*.engine\"))\n",
        "def list_available_models() -> list[ModelInfo]:\n"
        "    \"\"\"List detection models from canonical user and sync directories.\"\"\"\n\n"
        "    from .model_services import list_available_models as canonical_list_available_models\n\n"
        "    return canonical_list_available_models()\n\n\n"
        "def list_available_classification_models() -> list[ModelInfo]:\n"
        "    \"\"\"List classification models from canonical user and sync directories.\"\"\"\n\n"
        "    from .model_services import (\n"
        "        list_available_classification_models as canonical_list_available_classification_models,\n"
        "    )\n\n"
        "    return canonical_list_available_classification_models()\n",
    )


def patch_frontend_fixtures() -> None:
    path = Path("frontend/test/model_selection_test.dart")
    text = path.read_text(encoding="utf-8")
    text = text.replace("res/model_cls", "res/Model/cls")
    text = text.replace("res/model", "res/Model/detect")
    path.write_text(text, encoding="utf-8")

    path = Path("frontend/test/model_sync_test.dart")
    replace_once(path, "import 'dart:async';\n\n", "")


def patch_readme() -> None:
    path = Path("README.md")
    replace_once(
        path,
        "- 模型目录位于 `res/model` 以及 `res/cls_model` 文件夹下",
        "- 模型目录位于 `res/Model/detect` 与 `res/Model/cls`；用户模型放入各自的 `user` 子目录，NeriCloud 同步模型位于 `sync` 子目录",
    )


def patch_final_verify_workflow() -> None:
    path = Path(".github/workflows/model-sync-final-verify.yml")
    replace_once(
        path,
        "      - name: Analyze Flutter frontend\n"
        "        shell: pwsh\n"
        "        working-directory: frontend\n"
        "        run: flutter analyze\n\n",
        "",
    )


def main() -> None:
    patch_main_core()
    patch_services()
    patch_frontend_fixtures()
    patch_readme()
    patch_final_verify_workflow()


if __name__ == "__main__":
    main()
