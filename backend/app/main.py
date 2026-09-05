"""
FastAPI application entrypoint.

Run with:  uvicorn app.main:app --reload --port 8000   (from the backend/ dir)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.database import init_db
from app.routes import aptitude, coding, final_evaluation, health, hr, sessions, technical

settings = get_settings()

app = FastAPI(
    title="Placement Interview Agent API",
    description="Backend for the AI-driven placement interview simulator (TCS NQT / Infosys / Wipro).",
    version="0.1.0",
)

_DEV_LOCALHOST_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1):\d+"
# Vercel gives every deployment (production and preview alike) its own
# unique subdomain of the form <project>-<hash>-<team>.vercel.app, so a
# static CORS_ORIGINS entry goes stale on the next deploy. Match any
# deployment of this specific project instead of hardcoding one URL.
_VERCEL_PROJECT_ORIGIN_REGEX = r"https://ai-interview-agent(-[a-z0-9]+)*\.vercel\.app"

_origin_regex_parts = [_VERCEL_PROJECT_ORIGIN_REGEX]
if settings.app_env == "development":
    # Vite auto-increments past its default port (5173) whenever it's
    # already taken, so accept any localhost/127.0.0.1 port in development
    # instead of playing whack-a-mole with CORS_ORIGINS every time that happens.
    _origin_regex_parts.append(_DEV_LOCALHOST_ORIGIN_REGEX)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=r"^(" + "|".join(_origin_regex_parts) + r")$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(aptitude.router)
app.include_router(coding.router)
app.include_router(technical.router)
app.include_router(hr.router)
app.include_router(final_evaluation.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {"message": "Placement Interview Agent API. See /docs for API documentation."}
