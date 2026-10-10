"""A root-folder job must not conflate cameras that reuse media filenames."""
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import pytest

from system import detection_db
from system.backend import services
from system.backend import services_legacy as legacy
from system.backend.models import CreateJobRequest, ValidationBatchMarkRequest, ValidationExportRequest
from system.backend.preview_fast import load_media_indexes


@pytest.fixture
def cameras(tmp_path):
    paths = []
    for name in ("site/camera-a", "site/camera-b", "site/unprocessed"):
        parent = tmp_path / name
        parent.mkdir(parents=True)
        path = parent / "IMAG0001.JPG"
        path.write_bytes(b"fixture")
        paths.append(path)
    return tmp_path, paths


def seed(path, species, validated=False):
    database = detection_db.get_db_path(str(path.parent))
    detection_db.init_db(database)
    detection_db.upsert_detection(database, path.stem, path.name, {"物种名称": species})
    detection_db.upsert_validation_bulk(database, [(path.name, validated)])


def test_root_job_writes_legacy_results_beside_each_source(cameras):
    root, (first, second, missing) = cameras
    legacy._save_detection_data_batch([
        (first, {"物种名称": "fox"}),
        (second, {"物种名称": "cat"}),
    ], root)

    assert not (root / "detections.db").exists()
    for path, species in ((first, "fox"), (second, "cat")):
        # Read using the old software's existing database API and keys.
        database = detection_db.get_db_path(str(path.parent))
        assert detection_db.get_detection(database, path.stem)["物种名称"] == species
    assert not (missing.parent / "detections.db").exists()


@pytest.mark.parametrize("preview", [legacy.preview_media_items, services.preview_media_items])
def test_root_preview_isolates_duplicate_names_and_ignores_stale_parent(cameras, preview):
    root, (first, second, missing) = cameras
    seed(first, "fox", True)
    seed(second, "cat", False)
    seed(root / first.name, "stale-root", True)

    items = {item.path: item for item in preview(str(root))}
    assert items[str(first)].species == ["fox"]
    assert items[str(first)].validated is True
    assert items[str(second)].species == ["cat"]
    assert items[str(second)].validated is False
    assert items[str(missing)].species == []
    assert items[str(missing)].validated is None


def test_single_preview_reload_and_export_lookup_use_source_directory(cameras, monkeypatch):
    root, (first, second, missing) = cameras
    seed(first, "fox", True)
    seed(second, "cat", False)
    seed(root / first.name, "stale-root", True)
    monkeypatch.setattr(services, "_build_metadata_item", services._build_fast_metadata_item)

    for path, species, validated in (
        (first, ["fox"], True), (second, ["cat"], False), (missing, [], None),
    ):
        item = services.preview_media_item(str(path), str(root))
        assert item.species == species
        assert item.validated is validated
        item = services._reload_validation_item(path, root)
        assert item.species == species
        assert item.validated is validated
        data = legacy._load_detection_data_for_path(path, [root])
        assert data.get("物种名称") == (species[0] if species else None)


@pytest.mark.parametrize("model_path", [None, "fixture.neri.json"])
def test_validation_saves_and_unmarks_only_the_selected_camera(cameras, monkeypatch, model_path):
    root, (first, second, missing) = cameras
    seed(first, "fox", False)
    seed(second, "cat", True)
    seed(root / first.name, "stale-root", True)
    # Disable optional upload enqueue; this test only exercises local storage.
    import system.training
    monkeypatch.setattr(system.training, "get_queue", lambda: SimpleNamespace(enqueue=lambda *a, **kw: None))
    from system.backend import validation_fast
    monkeypatch.setattr(validation_fast, "schedule_validation_feedback_batch", lambda *a, **kw: None)

    marked = services.mark_validation_items(ValidationBatchMarkRequest(
        input_path=str(root), file_paths=[str(first), str(missing)], action="correct",
        classification_model_path=model_path,
    ))
    assert marked[0].species == ["fox"]
    assert marked[1].species == []
    services.mark_validation_items(ValidationBatchMarkRequest(
        input_path=str(root), file_paths=[str(first)], action="unverified",
    ))
    items = {item.path: item for item in services.preview_media_items(str(root))}
    assert items[str(first)].validated is None
    assert items[str(second)].species == ["cat"]
    assert items[str(second)].validated is True
    assert detection_db.get_detection(str(root / "detections.db"), first.stem)["物种名称"] == "stale-root"


def test_bulk_indexes_query_once_per_source_directory(cameras):
    root, (first, second, missing) = cameras
    seed(first, "fox")
    seed(second, "cat")
    calls = []

    def candidates(roots, *, recursive):
        calls.append((roots, recursive))
        return legacy._candidate_detection_dbs_for_roots(roots, recursive=recursive)

    detections, _ = load_media_indexes([first, second, missing], candidates)
    assert detections[str(first)]["物种名称"] == "fox"
    assert detections[str(second)]["物种名称"] == "cat"
    assert str(missing) not in detections
    assert calls == [([path.parent], False) for path in (first, second, missing)]


def test_same_stem_different_extension_never_uses_another_media_result(tmp_path):
    image = tmp_path / "IMAG0001.JPG"
    video = tmp_path / "IMAG0001.AVI"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    seed(video, "cat")
    detections, _ = load_media_indexes([image, video], legacy._candidate_detection_dbs_for_roots)
    assert str(image) not in detections
    assert detections[str(video)]["物种名称"] == "cat"


def test_tracked_video_persists_to_its_camera_and_reads_its_temporary_output(cameras, monkeypatch):
    root, (first, _, _) = cameras
    video = first.with_suffix(".AVI")
    video.write_bytes(b"video")
    staging = root / "video-staging"
    staging.mkdir()
    monkeypatch.setattr(legacy, "_video_processing_temp_dir", lambda: staging)
    monkeypatch.setattr(legacy, "_selected_species_class_ids", lambda *a: None)
    destinations = []

    def detect(path, output, **kwargs):
        destinations.append(kwargs["extra_db_dir"])
        seed(Path(output) / Path(path).name, "fox")
        return {"status": "success"}

    item = legacy._detect_video_track(
        SimpleNamespace(detect_video_species=detect), video,
        legacy._build_fast_metadata_item(video),
        CreateJobRequest(input_dir=str(root)), root,
    )
    assert item.error is None
    assert item.species == ["fox"]
    assert destinations == [str(video.parent)]
    assert not (root / "detections.db").exists()
    assert detection_db.get_detection(str(video.parent / "detections.db"), video.stem)["物种名称"] == "fox"


def test_export_respects_directory_scope_and_separate_camera_events(cameras, monkeypatch):
    root, (first, second, missing) = cameras
    legacy._persist_validation_updates([
        (first, {"物种名称": "fox", "最低置信度": "人工校验"}, True),
        (second, {"物种名称": "fox", "最低置信度": "人工校验"}, True),
    ], root)
    monkeypatch.setattr(legacy, "_build_export_metadata", lambda path: {
        "文件名": path.name, "拍摄日期对象": datetime(2024, 7, 20, 12, 0),
    })
    from system.data_processor import DataProcessor
    exported = []

    def capture(rows, *args, **kwargs):
        exported[:] = rows
        return True

    monkeypatch.setattr(DataProcessor, "export_to_excel", capture)
    response = legacy.export_validation_data(ValidationExportRequest(input_path=str(root)))
    assert response.exported_count == 3
    by_path = {row["_source_path"]: row for row in exported}
    assert by_path[str(first)]["独立探测首只"] == "1"
    assert by_path[str(second)]["独立探测首只"] == "1"
    assert "物种名称" not in by_path[str(missing)]
    response = legacy.export_validation_data(ValidationExportRequest(input_path=str(second.parent)))
    assert response.exported_count == 1
    assert exported[0]["_source_path"] == str(second)
