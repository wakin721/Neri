import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
        input_path,
        paths,
        *,
        model='model.json',
        action='correct',
        species=None,
        operation_id='op-3',
    ):
        self.input_path = str(input_path)
        self.file_paths = [str(path) for path in paths]
        self.classification_model_path = model
        self.action = action
        self.species_name = species
        self.feedback_operation_id = operation_id
        self.species_type = None


class _Item:
    def __init__(self, path, validated=None):
        self.path = str(path)
        self.validated = validated

    def model_copy(self, update):
        return _Item(self.path, validated=update.get('validated', self.validated))


def test_fast_mark_preserves_persist_feedback_queue_order_without_legacy_loop(
    monkeypatch,
    tmp_path,
):
    first = tmp_path / 'a.jpg'
    second = tmp_path / 'b.jpg'
    first.write_bytes(b'x')
    second.write_bytes(b'x')
    events = []

    def legacy(_request):
        events.append('legacy')
        raise AssertionError('legacy DINO marking path must not execute')

    services = SimpleNamespace(
        mark_validation_items=legacy,
        _preview_detection_db_roots=lambda input_path, output, paths: [input_path],
        _unique_existing_dirs=lambda roots: list(roots),
        _load_detection_index=lambda roots, recursive=False, filenames=None: {},
        _build_validation_update=lambda request, path, data: (
            {'物种名称': '豹猫'},
            True,
        ),
        _build_fast_metadata_item=lambda path: _Item(path),
        _apply_detection_data=lambda item, data: item,
        _persist_validation_updates=lambda updates, input_path: events.append(
            'persist'
        ),
        _update_species_database=lambda name, kind: events.append('species-db'),
    )
    calls = []
    monkeypatch.setattr(
        validation_fast,
        'apply_validation_feedback_batch',
        lambda model, paths, **kwargs: (
            events.append('feedback')
            or calls.append((model, list(paths), kwargs))
            or 2
        ),
    )

    training = ModuleType('system.training')

    class _Queue:
        def enqueue(self, *args, **kwargs):
            events.append('queue')

    training.get_queue = lambda: _Queue()
    monkeypatch.setitem(sys.modules, 'system.training', training)

    wrapped = validation_fast.make_mark_validation_items(services)
    result = wrapped(_Request(tmp_path, [first, second], action='correct'))

    assert len(result) == 2
    assert events == ['persist', 'feedback', 'queue', 'queue']
    assert len(calls) == 1
    assert calls[0][0] == 'model.json'
    assert calls[0][1] == [first.resolve(), second.resolve()]
