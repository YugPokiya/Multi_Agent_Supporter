from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.decisions import router as decisions_router
from app.api.routes import router
from app.config import settings

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(router)
app.include_router(decisions_router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
