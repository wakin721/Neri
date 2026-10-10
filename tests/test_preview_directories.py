from pathlib import Path

import pytest

from system.backend.preview_directories import preview_directories


def touch(root: Path, relative: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_directory_index_omits_ancestors_and_non_media_folders(tmp_path):
    touch(tmp_path, 'root.jpg')
    touch(tmp_path, '2021/site/overview.jpg')
    first = touch(tmp_path, '2021/site/2/photo.JPG').parent
    second = touch(tmp_path, '2021/site/10/video.MP4').parent
    touch(tmp_path, '2021/site/2/cache/detections.db')
    touch(tmp_path, 'reports/readme.txt')
    (tmp_path / 'empty').mkdir()
    assert set(preview_directories(str(tmp_path))) == {str(first), str(second)}
    assert preview_directories(str(first)) == []


def test_directory_index_refreshes_after_media_move(tmp_path):
    original = touch(tmp_path, 'camera/photo.jpg')
    assert preview_directories(str(tmp_path)) == [str(original.parent)]
    nested = touch(tmp_path, 'camera/new/photo.jpg')
    assert preview_directories(str(tmp_path)) == [str(nested.parent)]
    nested.unlink()
    assert preview_directories(str(tmp_path)) == [str(original.parent)]


def test_directory_index_rejects_missing_root(tmp_path):
    with pytest.raises(ValueError, match='输入文件夹不存在'):
        preview_directories(str(tmp_path / 'missing'))


def test_unreadable_child_does_not_hide_other_cameras(tmp_path, monkeypatch):
    import system.backend.preview_directories as module

    accessible = touch(tmp_path, 'good/photo.jpg').parent
    blocked = touch(tmp_path, 'blocked/photo.jpg').parent
    scandir = module.os.scandir

    def guarded(path):
        if Path(path) == blocked:
            raise PermissionError('blocked')
        return scandir(path)

    monkeypatch.setattr(module.os, 'scandir', guarded)
    assert preview_directories(str(tmp_path)) == [str(accessible)]
