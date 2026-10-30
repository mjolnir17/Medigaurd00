"""Mediguard API.  Start with:  python -m uvicorn mediguard.api.main:app --port 8000"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import store
from ..jobs.manager import JobManager
from .routes import datasets, models, runs


@asynccontextmanager
async def lifespan(app):
    store.ensure_demo()
    app.state.jobs = JobManager()
    yield


app = FastAPI(title="Mediguard API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
for router in (models.router, datasets.router, runs.router):
    app.include_router(router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
