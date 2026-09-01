# KAVACH-2.5D
## Core Mathematical Engine — Product Requirements Document (PRD)
**Module Owner:** Paarth (Lead Backend)
**Files:** `backend/engine/grid_polar.py`, `backend/core/hardware_detector.py`
**Version:** 1.0 — Final

---

## 1. Executive Summary

Autonomous defense UGVs (Unmanned Ground Vehicles) operate under severe compute, memory, and bandwidth constraints. Raw 3D point clouds (10⁵–10⁶ points/frame) cause VRAM exhaustion and high latency, while flat 2D occupancy grids discard vertical features critical for detecting negative obstacles (potholes, trenches) and low curbs.

**KAVACH-2.5D** solves this with a **Foveated 2.5D Elevation & Semantic Mapping Engine** that applies variable spatial resolution based on distance from the sensor:

| Zone | Range | Bin Size | Purpose |
|---|---|---|---|
| Near-field | 0–10 m | 5 cm | Tactical safety, fine obstacle detail |
| Mid-field | 10–50 m | 20 cm | Balanced situational resolution |
| Far-field | 50–100 m | 50 cm | Coarse macro awareness |

**Target outcome:** 85–92% reduction in memory footprint vs. raw point clouds, with microsecond-to-low-millisecond projection latency, enabling real-time (15–30 FPS) streaming to a tactical HUD.

---

## 2. System Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Rohan               │     │  Paarth               │     │  Saksham              │
│  SemanticKITTI /      │────▶│  Polar Variable-       │────▶│  FastAPI WebSocket    │
│  Synthetic Inference  │     │  Resolution Engine     │     │  Streaming Server     │
│  [X, Y, Z, Label]      │     │  (Numba)               │     │                      │
└─────────────────────┘     └──────────────────────┘     └───────────┬──────────┘
                                                                       │
                                                                       ▼
                                                     ┌──────────────────────────────┐
                                                     │  Sarthak & Medhansh            │
                                                     │  Next.js + Deck.gl Tactical HUD │
                                                     │  (2.5D Canvas & Telemetry)      │
                                                     └──────────────────────────────┘
```

---

## 3. Team Responsibility Matrix

| Member | Module | Core Deliverables | Integration Dependency |
|---|---|---|---|
| **Paarth** (Lead Backend) | Core Engine & Mathematical Projection | Polar transformation, adaptive spatial binning, Z-compression, ΔZ negative-obstacle detection, dual-tier hardware routing | Reads `numpy.ndarray (N,4)` from Data Loader; outputs processed cells to WebSocket server |
| Rohan | AI / DL Inference Pipeline | SemanticKITTI ingest, PointNet++/MinkowskiEngine inference, CUDA tensor execution → `[X, Y, Z, Label]` | Drops inference array directly into Paarth's engine pipeline |
| Saksham | Middleware & Network Streaming | FastAPI backend, WebSocket router (`/ws/stream_map`), framing loop (15–30 FPS), JSON/MsgPack serialization | Consumes Paarth's output; streams to frontend |
| Sarthak & Medhansh | Frontend UI & Visualization | Next.js app, Deck.gl 2.5D layer, tactical HUD, live telemetry graphs, threat ticker | Connects to `ws://localhost:8000/ws/stream_map` |
| Disha | Product, Analytics & Pitch Deck | Metric verification (memory % drop, FPS benchmarks), UX workflow, SIH pitch deck, presentation scripts | Audits live metrics from Paarth's and Saksham's output logs |

---

## 4. Paarth's Subsystem: Core Mathematical Engine

### 4.1 Pipeline Workflow

```
Raw Point Cloud [X, Y, Z, Label]  (Shape: N × 4)
            │
            ▼
Coordinate Conversion:  (X, Y) → (r, θ)
            │
            ▼
Adaptive Resolution Binning  (r < 10 m | 10–50 m | ≥ 50 m)
            │
            ▼
Spatial Hashing / Polar Cell Index Assignment
            │
            ▼
Cell-Level Aggregation:  Max(Z), Min(Z), Majority(Label)
            │
            ▼
Negative Obstacle Compute:  ΔZ = Max(Z) − Min(Z)
            │
            ▼
Output Grid:  [X_center, Y_center, Max_Z, ΔZ, Label, r]
```

### 4.2 Mathematical Specifications

**Polar coordinate conversion**

r = √(X² + Y²),  θ = atan2(Y, X)

**Resolution step function**

Δs(r) =
- 0.05 m, if r < 10 m
- 0.20 m, if 10 m ≤ r < 50 m
- 0.50 m, if r ≥ 50 m

*(An angular resolution Δθ(r) should be defined analogously — see Open Question 8.1.)*

**Cell spatial indexing**

Bin_r = ⌊r / Δs(r)⌋,  Bin_θ = ⌊θ / Δθ(r)⌋

**2.5D cell compression**

For every subset of points Pₖ = {(Xᵢ, Yᵢ, Zᵢ, Lᵢ)} mapped to grid cell k:

- Z_elev = max(Zᵢ)
- ΔZ = max(Zᵢ) − min(Zᵢ)
- L_cell = mode(Lᵢ)
- Cell spatial center: (X_k, Y_k)

---

## 5. Universal Data Contracts

All components must strictly conform to these interfaces for zero-friction integration.

### Interface 1 — Input to Paarth's Engine (from Rohan / Mock Generator)

- **Type:** `numpy.ndarray`, contiguous, `float32` for coordinates, `int32` for labels
- **Shape:** `(N, 4)`, N ≈ 100,000 points

| Column | Field | Type | Notes |
|---|---|---|---|
| 0 | X | float32 | meters |
| 1 | Y | float32 | meters |
| 2 | Z | float32 | meters |
| 3 | Label_ID | int32 | 0=Road, 1=Static Obstacle, 2=Dynamic Object, 3=Pothole/Trench |

### Interface 2 — Output from Paarth's Engine to Saksham's WebSocket Server

- **Type:** `numpy.ndarray`, shape `(M, 6)`, M ≪ N (typically M ∈ [4,000, 12,000])

| Column | Field | Type | Notes |
|---|---|---|---|
| 0 | Center_X | float32 | meters |
| 1 | Center_Y | float32 | meters |
| 2 | Max_Z | float32 | elevation |
| 3 | Delta_Z | float32 | vertical hazard delta |
| 4 | Label | int32 | dominant class in cell |
| 5 | Radius_r | float32 | distance from sensor |

### Interface 3 — WebSocket Payload to Frontend (Saksham → Sarthak/Medhansh)

- **Endpoint:** `ws://localhost:8000/ws/stream_map`
- **Protocol:** JSON (Phase 1) → MessagePack (Phase 2)
- **Frequency:** 15–30 Hz

```json
{
  "header": {
    "frame_id": 1042,
    "timestamp": 1725211596.34,
    "active_engine": "CUDA_TIER_1"
  },
  "telemetry": {
    "fps": 28.4,
    "latency_ms": 11.2,
    "raw_points_count": 100000,
    "compressed_cells_count": 6420,
    "memory_saved_percent": 91.8
  },
  "grid_data": [
    [1.25, 0.45, 0.12, 0.04, 0, 1.32],
    [4.10, -2.30, 1.85, 1.80, 1, 4.70],
    [12.20, 8.40, -0.65, 0.85, 3, 14.81]
  ],
  "threats": [
    {
      "type": "NEGATIVE_OBSTACLE",
      "distance_m": 14.81,
      "coordinates": [12.20, 8.40],
      "depth_m": 0.85
    }
  ]
}
```

---

## 6. Directory Structure & File Mapping

```
kavach-2.5d/
├── backend/
│   ├── api/
│   │   ├── main.py                  # FastAPI instantiation & startup lifecycle
│   │   └── websocket_manager.py     # Connection manager & streaming loop
│   ├── core/
│   │   ├── hardware_detector.py     # Auto-detects CUDA availability vs CPU fallback
│   │   └── config.py                # Resolution boundaries, thresholds, ports
│   ├── engine/
│   │   ├── grid_polar.py            # [PAARTH] Numba JIT variable-resolution engine
│   │   └── cuda_kernels.py          # [PAARTH/ROHAN] CuPy/CUDA accelerated kernels
│   ├── data/
│   │   ├── mock_generator.py        # Synthetic Lidar point cloud stream
│   │   └── dataset_loader.py        # [ROHAN] SemanticKITTI .bin parser
│   └── telemetry/
│       ├── metrics.py               # Memory reduction %, FPS, latency calculator
│       └── serializer.py            # JSON / MessagePack packer
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DeckCanvas2D.tsx     # Deck.gl 2.5D column/polygon visualizer
│   │   │   ├── TelemetryPanel.tsx   # Live metric meters & engine status badge
│   │   │   └── ThreatTicker.tsx     # Threat log feed
│   │   ├── hooks/
│   │   │   └── useLidarStream.ts    # Low-overhead WebSocket client
│   │   └── pages/
│   │       └── index.tsx            # HUD Command Center layout
│   └── package.json
└── requirements.txt
```

---

## 7. Phased Execution & Integration Plan

**Phase 1 — Scaffold & Mock**
- Paarth writes the Numba polar engine against a synthetic data array.
- Saksham streams generated output over the FastAPI WebSocket.
- Sarthak/Medhansh render the 2.5D grid on Deck.gl.

**Phase 2 — Drop-in Integration (Rohan's Pipeline)**
- Rohan finishes the model inference script.
- `mock_generator.py` is replaced with `dataset_loader.py` + PointNet++.
- Paarth's engine consumes the real inference array with zero code changes (contract-driven design).

**Phase 3 — Hardware-Agnostic Benchmarking**
- Validate Tier-1 (CUDA) execution on Rohan's RTX 4090.
- Force-disable GPU to validate Tier-2 (Numba/CPU) graceful fallback.
- Record performance metrics for Disha's pitch deck.

---

## 8. Success Metrics & Acceptance Criteria

| Metric | Target | Validation Method |
|---|---|---|
| Memory footprint reduction | 85–92% vs. raw point cloud | Compare byte size of `(N,4)` input vs. `(M,6)` output |
| Projection latency (CPU/Numba) | < 20 ms per frame (N=100,000) | `timeit` benchmark on `mock_generator.py` output |
| Projection latency (CUDA) | < 5 ms per frame (target) | Benchmark on RTX 4090 in Phase 3 |
| Compressed cell count (M) | < 10,000 (from N=100,000) | Assert on output shape |
| Streaming frame rate | 15–30 FPS sustained | Measured at Saksham's WebSocket layer |
| Negative obstacle detection | ΔZ correctly flags potholes/trenches ≥ configurable depth threshold | Unit test against labeled SemanticKITTI Pothole/Trench points (Label=3) |

---

## 9. Open Questions / Risks (Recommend Resolving Before/During Phase 1)

1. **Angular resolution Δθ(r) is undefined.** The doc specifies `Bin_θ = ⌊θ / Δθ(r)⌋` but never defines Δθ(r) the way it defines Δs(r). This needs an explicit formula (e.g., fixed angular step per zone, or arc-length-equivalent to Δs(r)) before binning logic can be implemented.
2. **Empty-cell / sparse-region handling.** What happens to bins with zero points — omitted from output, or emitted with a null/sentinel value? Downstream (Saksham, frontend) needs to agree on this.
3. **Tie-breaking for `mode(Lᵢ)`.** If a cell has an equal count of two labels, define a deterministic priority order (e.g., hazard labels like Pothole/Trench should probably win ties for safety).
4. **Negative-obstacle threshold.** ΔZ alone flags any vertical variance in a cell — a real trench vs. a curb vs. sensor noise all produce nonzero ΔZ. A minimum depth threshold and/or noise floor should be specified.
5. **Coordinate frame assumptions.** Confirm sensor is at origin (0,0) each frame, and whether X/Y are already ego-centric or need transformation from a moving-vehicle frame.
6. **Hardware fallback trigger logic.** `hardware_detector.py` needs explicit criteria for "Tier-1 vs Tier-2" beyond CUDA availability (e.g., VRAM threshold, driver version checks, forced override flag for Phase 3 testing).
7. **Numba `fastmath=True` precision trade-off.** Flag for awareness — fastmath relaxes IEEE compliance for speed; worth confirming this doesn't affect ΔZ precision near hazard thresholds.

---

## 10. Immediate Next Steps for Paarth

1. **Environment setup**
   ```bash
   pip install fastapi uvicorn numba numpy websockets msgpack torch
   ```
2. **Implement** `backend/engine/grid_polar.py` using Numba JIT decorators (`@jit(nopython=True, fastmath=True)`).
3. **Resolve open question #1** (define Δθ(r)) before finalizing the binning function signature.
4. **Validate** using `backend/data/mock_generator.py`:
   - Confirm input shape `(100000, 4)` compresses to `(M, 6)` with `M < 10,000`.
   - Confirm execution time `< 20 ms`.
5. **Hand off** the processed NumPy array interface to Saksham for WebSocket serialization, confirming Interface 2 is met exactly (dtype, column order, shape).

---
*End of PRD.*