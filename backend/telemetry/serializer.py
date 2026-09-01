"""
KAVACH-2.5D — Payload Serializer

Builds Interface 3 WebSocket payloads from engine output.
Supports JSON (Phase 1) and MessagePack (Phase 2).
"""

from __future__ import annotations

import json
import time
import os
from typing import Any, Union

# MessagePack support (optional — Phase 2)
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

USE_MSGPACK = os.environ.get("KAVACH_MSGPACK", "0") == "1"


def build_payload(
    frame_id: int,
    engine_result: dict,
    fps: float,
    latency_ms: float,
    active_engine: str,
) -> dict:
    """
    Build the full Interface 3 payload.

    Args:
        frame_id: Sequential frame counter.
        engine_result: Dict from kavach_engine.process_frame() with keys:
            "grid_data", "threats", "telemetry"
        fps: Current rolling-average FPS.
        latency_ms: Current rolling-average latency in ms.
        active_engine: Engine tier string ("CUDA_TIER_1" or "NUMBA_TIER_2").

    Returns:
        dict matching Interface 3 schema.
    """
    telemetry = engine_result.get("telemetry", {})

    return {
        "header": {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "active_engine": active_engine,
        },
        "telemetry": {
            "fps": fps,
            "latency_ms": latency_ms,
            "raw_points_count": telemetry.get("raw_points_count", 0),
            "compressed_cells_count": telemetry.get("compressed_cells_count", 0),
            "memory_saved_percent": telemetry.get("memory_saved_percent", 0.0),
        },
        "grid_data": engine_result.get("grid_data", []),
        "threats": engine_result.get("threats", []),
    }


def serialize(payload: dict) -> Union[bytes, str]:
    """
    Serialize a payload to JSON string or MessagePack bytes.

    Uses MessagePack if KAVACH_MSGPACK=1 env var is set and msgpack is installed.
    Otherwise uses JSON.
    """
    if USE_MSGPACK and HAS_MSGPACK:
        return msgpack.packb(payload, use_bin_type=True)
    else:
        return json.dumps(payload, separators=(",", ":"))


def get_serialization_mode() -> str:
    """Return the active serialization mode."""
    if USE_MSGPACK and HAS_MSGPACK:
        return "msgpack"
    return "json"
