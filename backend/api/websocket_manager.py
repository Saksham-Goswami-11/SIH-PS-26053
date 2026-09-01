"""
KAVACH-2.5D — WebSocket Stream Manager

Handles real-time streaming of processed 2.5D LiDAR grids to connected clients.
Supports multiple simultaneous clients with independent streaming loops.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from backend.data.mock_generator import MockLidarStream
from backend.telemetry.metrics import MetricsTracker
from backend.telemetry.serializer import build_payload, serialize

logger = logging.getLogger("kavach.ws")

TARGET_FPS = int(os.environ.get("KAVACH_FPS", "20"))
FRAME_INTERVAL = 1.0 / TARGET_FPS


class StreamPipeline:
    """
    Manages the C++ engine instance and frame generation.
    """

    def __init__(self):
        self.lidar_stream = MockLidarStream()
        self.metrics = MetricsTracker(window_size=30)
        self._engine = None
        self._active_tier = "UNKNOWN"
        self._init_engine()

    def _init_engine(self):
        try:
            import kavach_engine
            self._engine = kavach_engine
            self._active_tier = kavach_engine.get_active_tier()
            logger.info(f"Engine initialized: {kavach_engine.get_engine_info()}")
        except ImportError:
            logger.warning("kavach_engine module not found — using fallback")
            self._engine = None
            self._active_tier = "PYTHON_FALLBACK"

    def get_next_payload(self) -> dict:
        """Process one frame through the C++ engine and build Interface 3 payload."""
        frame_id, points = self.lidar_stream.next_frame()

        if self._engine is not None:
            result = self._engine.process_frame(points)
        else:
            # Minimal fallback
            result = {
                "grid_data": [],
                "threats": [],
                "telemetry": {
                    "raw_points_count": points.shape[0],
                    "compressed_cells_count": 0,
                    "memory_saved_percent": 0.0,
                    "latency_ms": 0.0,
                },
            }

        telemetry = result.get("telemetry", {})
        metrics = self.metrics.record_frame(
            engine_latency_ms=telemetry.get("latency_ms", 0.0),
            raw_count=telemetry.get("raw_points_count", 0),
            compressed_count=telemetry.get("compressed_cells_count", 0),
        )

        return build_payload(
            frame_id=frame_id,
            engine_result=result,
            fps=metrics.fps,
            latency_ms=metrics.latency_ms,
            active_engine=self._active_tier,
        )

    @property
    def active_tier(self) -> str:
        return self._active_tier


class ConnectionManager:
    """
    Tracks connected clients and coordinates streaming.
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.pipeline = StreamPipeline()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected ({len(self.active_connections)} active)")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected ({len(self.active_connections)} remaining)")

    async def stream_to_client(self, websocket: WebSocket):
        """
        Direct async stream loop for a connected WebSocket client.
        Pushes frames at target FPS until client disconnects.
        """
        await self.connect(websocket)
        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                t_start = time.monotonic()

                # Generate and serialize payload
                payload = self.pipeline.get_next_payload()
                message = serialize(payload)

                if isinstance(message, bytes):
                    await websocket.send_bytes(message)
                else:
                    await websocket.send_text(message)

                # Maintain target FPS
                elapsed = time.monotonic() - t_start
                sleep_time = FRAME_INTERVAL - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(0.001)  # yield control to event loop

        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.debug(f"Stream error: {e}")
        finally:
            self.disconnect(websocket)

    @property
    def client_count(self) -> int:
        return len(self.active_connections)


# Module singleton
manager = ConnectionManager()
