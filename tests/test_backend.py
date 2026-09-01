"""
Unit and integration tests for KAVACH-2.5D FastAPI + WebSocket server and C++ engine.
"""

import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

import kavach_engine
from backend.api.main import app
from backend.data.mock_generator import generate_frame
from backend.telemetry.serializer import build_payload, serialize


def test_engine_bindings():
    """Verify C++ engine processes synthetic points correctly."""
    pts = generate_frame(50000, seed=123)
    res = kavach_engine.process_frame(pts)

    assert "grid_data" in res
    assert "threats" in res
    assert "telemetry" in res

    assert res["telemetry"]["raw_points_count"] == 50000
    assert res["telemetry"]["compressed_cells_count"] == len(res["grid_data"])
    assert res["telemetry"]["latency_ms"] >= 0.0


def test_engine_numpy_binding():
    """Verify process_frame_numpy returns (M, 6) ndarray."""
    pts = generate_frame(10000, seed=456)
    grid_arr = kavach_engine.process_frame_numpy(pts)

    assert isinstance(grid_arr, np.ndarray)
    assert grid_arr.ndim == 2
    assert grid_arr.shape[1] == 6


def test_api_root():
    """Verify GET / endpoint."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "KAVACH-2.5D"
    assert "endpoints" in data


def test_api_health():
    """Verify GET /health endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "engine" in data
    assert data["engine"]["tier"] in ["CUDA_TIER_1", "NUMBA_TIER_2", "PYTHON_FALLBACK"]


def test_serializer_interface3_compliance():
    """Verify serializer creates payload conforming to Interface 3 of PRD."""
    pts = generate_frame(1000, seed=789)
    result = kavach_engine.process_frame(pts)

    payload = build_payload(
        frame_id=1042,
        engine_result=result,
        fps=28.4,
        latency_ms=11.2,
        active_engine=kavach_engine.get_active_tier(),
    )

    # Check structure
    assert "header" in payload
    assert payload["header"]["frame_id"] == 1042
    assert "timestamp" in payload["header"]
    assert "active_engine" in payload["header"]

    assert "telemetry" in payload
    assert payload["telemetry"]["fps"] == 28.4
    assert payload["telemetry"]["latency_ms"] == 11.2
    assert payload["telemetry"]["raw_points_count"] == 1000

    assert "grid_data" in payload
    assert "threats" in payload

    # Check serialization
    json_str = serialize(payload)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["header"]["frame_id"] == 1042


@pytest.mark.anyio
async def test_streaming_pipeline():
    """Verify stream pipeline produces valid Interface 3 payloads."""
    from backend.api.websocket_manager import StreamPipeline

    pipeline = StreamPipeline()
    payload = pipeline.get_next_payload()

    assert "header" in payload
    assert "telemetry" in payload
    assert "grid_data" in payload
    assert "threats" in payload

    assert payload["header"]["frame_id"] >= 0
    assert payload["telemetry"]["raw_points_count"] > 0
    assert isinstance(payload["grid_data"], list)
    assert isinstance(payload["threats"], list)

