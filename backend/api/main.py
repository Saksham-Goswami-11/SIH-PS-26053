"""
KAVACH-2.5D — FastAPI Application

Main entry point for the WebSocket streaming server.

Endpoints:
  GET  /             — API info
  GET  /health       — Health check with engine tier info
  WS   /ws/stream_map — Live LiDAR stream (Interface 3 payloads)

Start with:
  uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from backend.api.websocket_manager import manager
from backend.telemetry.serializer import get_serialization_mode

# ─── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("kavach.api")

# ─── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="KAVACH-2.5D Streaming Server",
    description="Foveated 2.5D Elevation & Semantic Mapping — WebSocket Stream",
    version="1.0.0",
)

# CORS — allow frontend from any origin (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── HTTP Endpoints ──────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "KAVACH-2.5D",
        "description": "Foveated 2.5D Elevation & Semantic Mapping Engine",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "stream": "ws://localhost:8000/ws/stream_map",
        },
    }


@app.get("/health")
async def health():
    try:
        import kavach_engine
        engine_tier = kavach_engine.get_active_tier()
        engine_info = kavach_engine.get_engine_info()
    except ImportError:
        engine_tier = "PYTHON_FALLBACK"
        engine_info = "C++ engine not built — using Python fallback"

    return {
        "status": "healthy",
        "engine": {
            "tier": engine_tier,
            "info": engine_info,
        },
        "serialization": get_serialization_mode(),
        "connected_clients": manager.client_count,
        "total_frames_processed": manager.pipeline.metrics.total_frames,
    }


# ─── WebSocket Endpoint ──────────────────────────────────────────────

@app.websocket("/ws/stream_map")
async def websocket_stream(websocket: WebSocket):
    """
    Live LiDAR stream endpoint.

    Clients connect here to receive real-time processed 2.5D grid data.
    Payload format matches Interface 3 from the PRD.
    """
    await manager.stream_to_client(websocket)
