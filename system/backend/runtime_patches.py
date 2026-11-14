from __future__ import annotations

from typing import Any

def install_runtime_patches(services_module: Any, *, patch_preview: bool = True) -> None:
    """Install compatibility-preserving backend patches before main_core imports."""

    if not patch_preview:
        return

    detection_loader = getattr(services_module, '_load_detection_index', None)
    if callable(detection_loader) and not getattr(
        detection_loader,
        '_neri_filtered_detection_sql',
        False,
    ):
        original_detection_loader = detection_loader

        def patched_detection_loader(
            roots,
            recursive=False,
            filenames=None,
        ):
            if filenames is None:
                return original_detection_loader(
                    roots,
                    recursive=recursive,
                    filenames=None,
                )
            from .preview_fast import load_detection_index_for_filenames

            db_paths = services_module._candidate_detection_dbs_for_roots(
                roots,
                recursive=recursive,
            )
            return load_detection_index_for_filenames(db_paths, set(filenames))

        setattr(patched_detection_loader, '_neri_filtered_detection_sql', True)
        services_module._load_detection_index = patched_detection_loader

    validation_loader = getattr(services_module, '_load_validation_index', None)
    if callable(validation_loader) and not getattr(
        validation_loader,
        '_neri_filtered_validation_sql',
        False,
    ):
        original_validation_loader = validation_loader

        def patched_validation_loader(
            roots,
            recursive=False,
            filenames=None,
        ):
            if filenames is None:
                return original_validation_loader(
                    roots,
                    recursive=recursive,
                )
            from .preview_fast import load_validation_index_for_filenames

            db_paths = services_module._candidate_detection_dbs_for_roots(
                roots,
                recursive=recursive,
            )
            return load_validation_index_for_filenames(db_paths, set(filenames))

        setattr(patched_validation_loader, '_neri_filtered_validation_sql', True)
        services_module._load_validation_index = patched_validation_loader

    preview = getattr(services_module, 'preview_media_items', None)
    if callable(preview) and not getattr(
        preview,
        '_neri_filtered_preview_sql',
        False,
    ):
        from .preview_fast import make_preview_media_items

        services_module.preview_media_items = make_preview_media_items(services_module)

    preview_item = getattr(services_module, 'preview_media_item', None)
    if callable(preview_item) and not getattr(
        preview_item,
        '_neri_filtered_single_preview_sql',
        False,
    ):
        from .preview_fast import make_preview_media_item

        services_module.preview_media_item = make_preview_media_item(services_module)

    reload_item = getattr(services_module, '_reload_validation_item', None)
    if callable(reload_item) and not getattr(
        reload_item,
        '_neri_filtered_validation_reload_sql',
        False,
    ):
        from .preview_fast import make_reload_validation_item

        services_module._reload_validation_item = make_reload_validation_item(
            services_module
        )

    mark_items = getattr(services_module, 'mark_validation_items', None)
    if callable(mark_items) and not getattr(
        mark_items,
        '_neri_batched_dino_feedback',
        False,
    ):
        from .validation_fast import make_mark_validation_items

        services_module.mark_validation_items = make_mark_validation_items(
            services_module
        )
