from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from kb_server.api import auth, collections, groups, health, keys, users
from kb_server.api.errors import register_error_handlers
from kb_server.settings import Settings
from kb_server.wiring import Services, build_services, close_services

API_V1 = "/api/v1"


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
    app.include_router(auth.router, prefix=API_V1)
    app.include_router(users.router, prefix=API_V1)
    app.include_router(users.me_router, prefix=API_V1)
    app.include_router(keys.router, prefix=API_V1)
    app.include_router(keys.admin_router, prefix=API_V1)
    app.include_router(groups.router, prefix=API_V1)
    app.include_router(collections.router, prefix=API_V1)
    return app


app = create_app()
