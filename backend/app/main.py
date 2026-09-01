"""
FastAPI application entrypoint.

Run with:  uvicorn app.main:app --reload --port 8000   (from the backend/ dir)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.database import init_db
from app.routes import aptitude, health, sessions

settings = get_settings()

app = FastAPI(
    title="Placement Interview Agent API",
    description="Backend for the AI-driven placement interview simulator (TCS NQT / Infosys / Wipro).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(aptitude.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {"message": "Placement Interview Agent API. See /docs for API documentation."}
