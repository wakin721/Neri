"""Production ASGI entrypoint for the Neri model distribution service."""

from .app import create_app

app = create_app()
