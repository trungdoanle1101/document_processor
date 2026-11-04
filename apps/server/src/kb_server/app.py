from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from kb_server.api import health
from kb_server.api.errors import register_error_handlers
from kb_server.settings import Settings
from kb_server.wiring import Services, build_services, close_services


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        if services is not None:
            app.state.services = services
            yield
            return
        built = build_services(Settings())
        app.state.services = built
        try:
            yield
        finally:
            close_services(built)

    app = FastAPI(title="knowledge base", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(health.router)
    return app


app = create_app()
