from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from kb_server.api import health
from kb_server.api.errors import register_error_handlers
from kb_server.settings import Settings
from kb_server.wiring import build_services, close_services


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    services = build_services(Settings())
    app.state.services = services
    try:
        yield
    finally:
        close_services(services)


app = FastAPI(title="knowledge base", lifespan=lifespan)
register_error_handlers(app)
app.include_router(health.router)
