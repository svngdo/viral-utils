from fastapi import FastAPI

from src.exceptions import register_exception_handlers
from src.systems.router import router as systems_router

app = FastAPI()
app.include_router(systems_router)
register_exception_handlers(app)
