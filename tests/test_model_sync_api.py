import asyncio
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace
from unittest.mock import patch

from fastapi import APIRouter, FastAPI

from system.model_sync.manager import ModelSyncStatus


class FakeManager:
    def __init__(self):
        self.current = ModelSyncStatus(state='idle')
        self.start_calls = 0

    def status(self):
        return self.current

    def start(self):
        self.start_calls += 1
        if self.current.state not in {'checking', 'downloading'}:
            self.current = replace(self.current, state='checking', run_id='run-1')
        return self.current


class ModelSyncApiTests(unittest.TestCase):
    def test_router_exposes_status_and_manual_start_handlers(self):
        from system.model_sync.api import model_sync_status_payload, start_model_sync_payload
        manager = FakeManager()
        self.assertEqual(model_sync_status_payload(manager)['state'], 'idle')
        started = start_model_sync_payload(manager)
        self.assertEqual(started['state'], 'checking')
        self.assertEqual(started['run_id'], 'run-1')
        self.assertEqual(manager.start_calls, 1)

    def test_startup_wraps_existing_lifespan_in_required_order(self):
        from system.model_sync.bootstrap import install_model_sync
        events = []
        manager = FakeManager()
        app = FastAPI()

        @asynccontextmanager
        async def original(_app):
            events.append('training-start')
            try:
                yield
            finally:
                events.append('training-stop')

        app.router.lifespan_context = original
        install_model_sync(app, manager_provider=lambda: manager, migrate=lambda: events.append('migrate'))

        async def exercise():
            async with app.router.lifespan_context(app):
                events.append('serving')

        asyncio.run(exercise())
        self.assertEqual(events, ['migrate', 'training-start', 'serving', 'training-stop'])
        self.assertEqual(manager.start_calls, 1)

    def test_startup_failures_do_not_prevent_existing_app_lifespan(self):
        from system.model_sync.bootstrap import install_model_sync
        events = []
        app = FastAPI()

        @asynccontextmanager
        async def original(_app):
            events.append('core')
            yield

        class BrokenManager(FakeManager):
            def start(self):
                raise RuntimeError('offline')

        app.router.lifespan_context = original
        install_model_sync(
            app,
            manager_provider=lambda: BrokenManager(),
            migrate=lambda: (_ for _ in ()).throw(OSError('readonly')),
        )

        async def exercise():
            async with app.router.lifespan_context(app):
                events.append('serving')

        asyncio.run(exercise())
        self.assertEqual(events, ['core', 'serving'])

    def test_get_manager_is_process_singleton(self):
        import system.model_sync as model_sync
        model_sync._manager = None
        fake_layout = object()
        fake_client = object()
        fake_manager = object()
        with patch.object(model_sync, 'get_model_layout', return_value=fake_layout), \
             patch.object(model_sync, 'ModelDistributionClient', return_value=fake_client), \
             patch.object(model_sync, 'ModelSyncManager', return_value=fake_manager) as manager_cls:
            first = model_sync.get_model_sync_manager()
            second = model_sync.get_model_sync_manager()
        self.assertIs(first, fake_manager)
        self.assertIs(second, fake_manager)
        manager_cls.assert_called_once_with(fake_layout, fake_client)
        model_sync._manager = None

    def test_wire_registers_router_and_installs_lifespan(self):
        import system.model_sync.integration as integration
        app = FastAPI()
        fake_manager = object()
        with patch.object(integration, 'get_model_sync_manager', return_value=fake_manager), \
             patch.object(integration, 'create_model_sync_router') as router_factory, \
             patch.object(integration, 'install_model_sync') as installer:
            router_factory.return_value = APIRouter()
            result = integration.wire_model_sync(app)
        self.assertIs(result, app)
        router_factory.assert_called_once()
        installer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
