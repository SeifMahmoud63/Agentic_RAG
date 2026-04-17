from fastapi import FastAPI
from contextlib import asynccontextmanager
from routes import base, data
from helpers import config
import logging

logger = logging.getLogger("uvicorn.error")
from helpers import redis

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("=== Server Starting ===")

    from VectorDatabase.MetadataStore import init_db
    init_db()

    from VectorDatabase.QdrantDb import warm_up
    warm_up()

    logger.info("=== Server Ready ===")
    yield
    logger.info("=== Server Shutting Down ===")


app = FastAPI(lifespan=lifespan)

@app.on_event("startup")
async def startup_event():
    redis.init_cache()

app.include_router(base.base_router)
app.include_router(data.data_router)
