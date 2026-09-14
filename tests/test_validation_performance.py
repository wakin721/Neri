from pathlib import Path
from types import SimpleNamespace

from system.backend import validation_fast


class _Feedback:
    def __init__(self, by_path):
        self.by_path = by_path
        self.lookup_paths = []
        self.records = []
        self.recomputed = []
        self.closed = 0

    def observations_for_source_path(self, path):
        path = Path(path)
        self.lookup_paths.append(path)
        return list(self.by_path.get(path, []))

    def record_feedback(
        self,
        observation_id,
        *,
        operation_id,
        action,
        confirmed_species,
    ):
        self.records.append(
            (observation_id, operation_id, action, confirmed_species)
        )
        return SimpleNamespace(affected={'豹猫'})

    def recompute_species(self, species, feature_center):
        self.recomputed.append(species)

    def close(self):
        self.closed += 1


def _observation(identifier):
    return SimpleNamespace(id=identifier)


def test_batch_feedback_opens_store_once_and_skips_ambiguous_files(
    monkeypatch,
    tmp_path,
):
    first = tmp_path / 'a.jpg'
    empty = tmp_path / 'b.jpg'
    ambiguous = tmp_path / 'c.jpg'
    feedback = _Feedback(
        {
            first: [_observation('a1')],
            empty: [],
            ambiguous: [_observation('c1'), _observation('c2')],
        }
    )
    from system.backend import dinov3_feedback_service

    opens = []
    monkeypatch.setattr(
        dinov3_feedback_service,
        '_open_feedback_state',
        lambda path: (opens.append(path) or feedback, object()),
    )
    monkeypatch.setattr(
        dinov3_feedback_service,
        '_affected_learning_species',
        lambda record: set(record.affected),
    )

    applied = validation_fast.apply_validation_feedback_batch(
        'model.json',
        [first, empty, ambiguous],
        operation_id='op-1',
        action='correct',
        confirmed_species=None,
        assign_registry_species=False,
    )

    assert applied == 1
    assert opens == ['model.json']
    assert feedback.lookup_paths == [first, empty, ambiguous]
    assert feedback.records == [('a1', 'op-1', 'correct', None)]
    assert feedback.recomputed == ['豹猫']
    assert feedback.closed == 1


def test_batch_registry_assignment_reuses_open_feedback_store(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / 'camel.jpg'
    feedback = _Feedback({path: [_observation('obs-camel')]})
    from system.backend import dinov3_feedback_service

    monkeypatch.setattr(
        dinov3_feedback_service,
        '_open_feedback_state',
        lambda _path: (feedback, object()),
    )
    assignments = []
    monkeypatch.setattr(
        dinov3_feedback_service,
        '_assign_registry_species',
        lambda fb, model, observation, **kwargs: assignments.append(
            (fb, model, observation.id, kwargs)
        ),
    )

    applied = validation_fast.apply_validation_feedback_batch(
        'model.json',
        [path],
        operation_id='op-2',
        action='update',
        confirmed_species='骆驼',
        assign_registry_species=True,
    )

    assert applied == 1
    assert len(assignments) == 1
    assert assignments[0][0] is feedback
    assert assignments[0][2] == 'obs-camel'
    assert assignments[0][3]['confirmed_species'] == '骆驼'
    assert feedback.records == []
    assert feedback.closed == 1


class _Request:
    def __init__(
        self,
        paths,
        *,
        model='model.json',
        action='correct',
        species=None,
        operation_id='op-3',
    ):
        self.file_paths = [str(path) for path in paths]
        self.classification_model_path = model
        self.action = action
        self.species_name = species
        self.feedback_operation_id = operation_id

    def model_copy(self, update):
        clone = _Request(
            [],
            model=self.classification_model_path,
            action=self.action,
            species=self.species_name,
            operation_id=self.feedback_operation_id,
        )
        clone.file_paths = list(self.file_paths)
        for key, value in update.items():
            setattr(clone, key, value)
        return clone


def test_mark_wrapper_suppresses_legacy_per_file_feedback_and_runs_one_batch(
    monkeypatch,
    tmp_path,
):
    first = tmp_path / 'a.jpg'
    second = tmp_path / 'b.jpg'
    original_requests = []
    services = SimpleNamespace(
        mark_validation_items=lambda request: (
            original_requests.append(request) or ['saved']
        )
    )
    calls = []
    monkeypatch.setattr(
        validation_fast,
        'apply_validation_feedback_batch',
        lambda model, paths, **kwargs: (
            calls.append((model, list(paths), kwargs)) or 2
        ),
    )
    monkeypatch.setattr(
        validation_fast,
        '_checkpoint_species_for_model',
        lambda _path: {'豹猫'},
    )

    wrapped = validation_fast.make_mark_validation_items(services)
    result = wrapped(_Request([first, second], action='correct'))

    assert result == ['saved']
    assert len(original_requests) == 1
    assert original_requests[0].classification_model_path is None
    assert len(calls) == 1
    assert calls[0][0] == 'model.json'
    assert calls[0][1] == [first.resolve(), second.resolve()]
