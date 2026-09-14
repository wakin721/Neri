from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from system.backend.dinov3_persistence_fast import persist_runtime_observations


def _runtime_observation(
    identifier,
    result_index,
    *,
    accepted=True,
    source='checkpoint',
):
    return SimpleNamespace(
        result_index=result_index,
        box_index=0,
        observation_id=identifier,
        bbox=(1.0, 2.0, 20.0, 30.0),
        species='Known' if accepted else 'Unknown',
        best_known_species='Known',
        accepted=accepted,
        source=source,
        registry_id=None,
        known_score=0.9 if accepted else 0.4,
        threshold=0.8,
        embedding=np.eye(768, dtype=np.float32)[result_index],
    )


class _Feedback:
    def __init__(self, path):
        self.path = path
        self.batches = []
        self.singles = []

    def persist_observations(self, observations):
        batch = tuple(observations)
        self.batches.append(batch)
        return tuple(observation.id for observation in batch)

    def persist_observation(self, observation):
        self.singles.append(observation)
        return observation.id


class _Registry:
    def __init__(self):
        self.unknowns = []

    def record_unknown(self, embedding, **kwargs):
        self.unknowns.append(kwargs)

    def record_observation(self, *args, **kwargs):
        raise AssertionError('unexpected registry-id path')


def test_runtime_persistence_batches_feedback_observations_once(tmp_path):
    feedback = _Feedback(tmp_path / 'feedback.sqlite3')
    registry = _Registry()
    observations = [
        _runtime_observation('obs-1', 0, accepted=True),
        _runtime_observation('obs-2', 1, accepted=False, source='overlay'),
    ]
    detector = SimpleNamespace(
        dinov3_feedback=feedback,
        dinov3_registry=registry,
        drain_dinov3_observations=lambda: tuple(observations),
    )
    paths = [tmp_path / 'a.jpg', tmp_path / 'b.jpg']
    items = [
        SimpleNamespace(date_taken='2026-09-09T03:04:05'),
        SimpleNamespace(date_taken='2026-09-09T04:05:06'),
    ]

    persist_runtime_observations(detector, paths, items, tmp_path)

    assert len(feedback.batches) == 1
    assert feedback.singles == []
    assert [item.id for item in feedback.batches[0]] == ['obs-1', 'obs-2']
    assert feedback.batches[0][0].captured_at == datetime(2026, 9, 9, 3, 4, 5)
    assert feedback.batches[0][0].source_path == str(paths[0])
    assert feedback.batches[0][1].accepted is False
    assert len(registry.unknowns) == 1


def test_batch_write_failure_falls_back_per_observation(tmp_path):
    feedback = _Feedback(tmp_path / 'feedback.sqlite3')

    def fail_batch(_observations):
        raise RuntimeError('batch failure')

    feedback.persist_observations = fail_batch
    observations = [
        _runtime_observation('obs-1', 0),
        _runtime_observation('obs-2', 1),
    ]
    detector = SimpleNamespace(
        dinov3_feedback=feedback,
        dinov3_registry=None,
        drain_dinov3_observations=lambda: tuple(observations),
    )
    paths = [tmp_path / 'a.jpg', tmp_path / 'b.jpg']
    items = [SimpleNamespace(date_taken=None), SimpleNamespace(date_taken=None)]

    persist_runtime_observations(detector, paths, items, tmp_path)

    assert [item.id for item in feedback.singles] == ['obs-1', 'obs-2']
