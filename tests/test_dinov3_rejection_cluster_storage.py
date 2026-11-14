"""Real source-local detection database roundtrip across selected folders."""
from pathlib import Path

from system.backend import services
from system.backend.models import CreateJobRequest, DetectionItem
from system.dinov3.processor import finalize_photo_job
from tests.test_dinov3_rejection_clustering import _build_finalizer_fixture


def test_selected_folders_share_one_clustering_snapshot_in_source_databases(tmp_path):
    detector, items, _, _ = _build_finalizer_fixture(tmp_path, include_accepted=False)
    request = CreateJobRequest(input_paths=[str(tmp_path / name) for name in ("CamNorth", "CamSouth")])
    root, files = services._legacy._resolve_job_inputs(request)
    assert set(files) == {Path(item.path) for item in items}
    updated = finalize_photo_job(
        detector, items, root, job_id="two-folders",
        save_batch_fn=services._save_detection_data_batch,
        apply_data_fn=services._apply_detection_data,
    )
    run_ids, clusters = set(), {}
    for item in updated:
        path = Path(item.path)
        stored = services._load_detection_data_for_path(path, [path.parent], recursive=False)
        restored = services._apply_detection_data(
            DetectionItem(path=item.path, filename=item.filename, file_type="image"), stored)
        assert restored.detection_boxes == item.detection_boxes
        metadata = restored.detection_boxes[0]["rejection_cluster"]
        assert metadata["status"] == "clustered"
        run_ids.add(metadata["run_id"])
        clusters.setdefault(metadata["cluster_id"], set()).add(path.parent)
    assert len(run_ids) == 1
    assert len(clusters) == 2
    assert all(len(source_directories) == 2 for source_directories in clusters.values())
