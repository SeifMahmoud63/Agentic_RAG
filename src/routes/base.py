from fastapi import FastAPI , APIRouter,Depends
from helpers import config


base_router=APIRouter(
    prefix="/api/v1",
    tags=["api_V1"]
)

@base_router.get("/")
async def welcome(settings:config.Settings=Depends(config.get_settings)):
    return{
        "APP_VERSION":settings.APP_VERSION,
        "APP_NAME":settings.APP_NAME
    }

