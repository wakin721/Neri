import cv2
import numpy as np

from system.dinov3 import registry_examples


def test_example_frame_cache_decodes_one_still_once(monkeypatch, tmp_path):
    source = tmp_path / 'source.jpg'
    cv2.imwrite(str(source), np.zeros((100, 100, 3), dtype=np.uint8))
    calls = []
    real_imread = registry_examples.cv2.imread

    def traced_imread(path):
        calls.append(path)
        return real_imread(path)

    monkeypatch.setattr(registry_examples.cv2, 'imread', traced_imread)

    with registry_examples.example_frame_cache():
        assert registry_examples.persist_feedback_observation_example(
            tmp_path / 'feedback.sqlite3',
            'one',
            source_path=source,
            bbox=(0, 0, 40, 40),
        )
        assert registry_examples.persist_feedback_observation_example(
            tmp_path / 'feedback.sqlite3',
            'two',
            source_path=source,
            bbox=(40, 40, 90, 90),
        )

    assert len(calls) == 1


def test_example_frame_cache_reads_distinct_stills_separately(monkeypatch, tmp_path):
    first = tmp_path / 'first.jpg'
    second = tmp_path / 'second.jpg'
    cv2.imwrite(str(first), np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.imwrite(str(second), np.zeros((100, 100, 3), dtype=np.uint8))
    calls = []
    real_imread = registry_examples.cv2.imread

    def traced_imread(path):
        calls.append(path)
        return real_imread(path)

    monkeypatch.setattr(registry_examples.cv2, 'imread', traced_imread)

    with registry_examples.example_frame_cache():
        registry_examples.persist_feedback_observation_example(
            tmp_path / 'feedback.sqlite3',
            'one',
            source_path=first,
            bbox=(0, 0, 40, 40),
        )
        registry_examples.persist_feedback_observation_example(
            tmp_path / 'feedback.sqlite3',
            'two',
            source_path=second,
            bbox=(0, 0, 40, 40),
        )

    assert len(calls) == 2


def test_example_frame_cache_never_caches_video_path_even_if_imread_succeeds(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / 'clip.mp4'
    source.write_bytes(b'video-placeholder')
    calls = []
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def fake_imread(path):
        calls.append(path)
        return frame.copy()

    monkeypatch.setattr(registry_examples.cv2, 'imread', fake_imread)

    with registry_examples.example_frame_cache():
        registry_examples.persist_feedback_observation_example(
            tmp_path / 'feedback.sqlite3',
            'video-one',
            source_path=source,
            bbox=(0, 0, 40, 40),
            frame_index=1,
        )
        registry_examples.persist_feedback_observation_example(
            tmp_path / 'feedback.sqlite3',
            'video-two',
            source_path=source,
            bbox=(0, 0, 40, 40),
            frame_index=2,
        )

    assert len(calls) == 2
