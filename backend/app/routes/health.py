from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    """Basic liveness check used by the frontend and deployment tooling."""
    return {
        "status": "ok",
        "service": "placement-interview-agent-backend",
        "time": datetime.now(timezone.utc).isoformat(),
    }
