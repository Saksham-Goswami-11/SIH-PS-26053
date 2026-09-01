"""
KAVACH-2.5D — Synthetic LiDAR Point Cloud Generator

Simulates a 64-channel spinning LiDAR (e.g. Ouster OS1 / Velodyne HDL-64):
  - Concentric ground scan rings (Label 0: Road)
  - Static vertical obstacles (Label 1: Walls, trees, curbs)
  - Dynamic vehicles/pedestrians with trajectories (Label 2)
  - Negative obstacles / potholes with negative elevation (Label 3)

Generates (N, 4) float32 arrays: [X, Y, Z, Label_ID]
Naturally compresses to ~4,000 – 9,000 cells (85–92% compression).
"""

import numpy as np
from typing import Optional

DEFAULT_N = 100_000


def generate_frame(
    n: int = DEFAULT_N,
    seed: Optional[int] = None,
    frame_id: int = 0,
) -> np.ndarray:
    """
    Generate a realistic LiDAR scan of n points.
    """
    rng = np.random.default_rng(seed if seed is not None else (frame_id * 17 + 42) & 0xFFFFFFFF)

    points = np.zeros((n, 4), dtype=np.float32)
    idx = 0

    # Sensor parameters
    sensor_height = 1.8  # meters above ground
    n_channels = 64
    n_azimuth = n // n_channels

    # Elevation pitch angles from -25 deg to +15 deg
    elevations = np.linspace(-25.0 * np.pi / 180.0, 15.0 * np.pi / 180.0, n_channels)
    azimuths = np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False)

    # ─── 1. Ground Scan Rings (Channels looking down, pitch < -1 deg) ─────
    down_channels = elevations[elevations < -0.02]
    up_channels = elevations[elevations >= -0.02]

    for pitch in down_channels:
        # Distance to flat ground: d = sensor_height / -sin(pitch)
        ground_dist = sensor_height / -np.sin(pitch)
        if ground_dist > 85.0:
            continue

        # Add slight road noise and azimuth modulation
        r = ground_dist + rng.normal(0.0, 0.02, n_azimuth)
        x = (r * np.cos(azimuths)).astype(np.float32)
        y = (r * np.sin(azimuths)).astype(np.float32)
        z = rng.normal(0.0, 0.02, n_azimuth).astype(np.float32)

        count = min(n_azimuth, n - idx)
        if count <= 0:
            break
        points[idx:idx + count, 0] = x[:count]
        points[idx:idx + count, 1] = y[:count]
        points[idx:idx + count, 2] = z[:count]
        points[idx:idx + count, 3] = 0.0  # LABEL_ROAD
        idx += count

    # ─── 2. Static Obstacles (Buildings, walls, barriers) ─────────────────
    obstacles = [
        {"cx": 15.0, "cy": 12.0, "w": 4.0, "h": 2.5, "label": 1},
        {"cx": -18.0, "cy": 25.0, "w": 6.0, "h": 3.0, "label": 1},
        {"cx": 8.0, "cy": -20.0, "w": 3.0, "h": 1.8, "label": 1},
        {"cx": -10.0, "cy": -15.0, "w": 5.0, "h": 2.0, "label": 1},
    ]

    for obs in obstacles:
        pts_count = int(n * 0.04)
        count = min(pts_count, n - idx)
        if count <= 0:
            break
        ox = rng.normal(obs["cx"], obs["w"] * 0.3, count).astype(np.float32)
        oy = rng.normal(obs["cy"], obs["w"] * 0.3, count).astype(np.float32)
        oz = rng.uniform(0.0, obs["h"], count).astype(np.float32)

        points[idx:idx + count, 0] = ox
        points[idx:idx + count, 1] = oy
        points[idx:idx + count, 2] = oz
        points[idx:idx + count, 3] = float(obs["label"])
        idx += count

    # ─── 3. Dynamic Objects (Vehicles moving) ─────────────────────────────
    vehicles = [
        {"x0": 12.0, "y0": 5.0, "vx": 0.3, "vy": 0.5, "h": 1.6},
        {"x0": -8.0, "y0": 30.0, "vx": -0.4, "vy": -0.2, "h": 2.2},
    ]

    for v in vehicles:
        pts_count = int(n * 0.03)
        count = min(pts_count, n - idx)
        if count <= 0:
            break
        # Position with temporal motion
        curr_x = v["x0"] + (frame_id * v["vx"]) % 40.0 - 20.0
        curr_y = v["y0"] + (frame_id * v["vy"]) % 40.0 - 20.0

        vx = rng.normal(curr_x, 1.0, count).astype(np.float32)
        vy = rng.normal(curr_y, 1.0, count).astype(np.float32)
        vz = rng.uniform(0.0, v["h"], count).astype(np.float32)

        points[idx:idx + count, 0] = vx
        points[idx:idx + count, 1] = vy
        points[idx:idx + count, 2] = vz
        points[idx:idx + count, 3] = 2.0  # LABEL_DYNAMIC_OBJECT
        idx += count

    # ─── 4. Negative Obstacles (Potholes / Trenches) ───────────────────────
    potholes = [
        {"cx": 6.5, "cy": 8.0, "depth": 0.45, "radius": 0.8},
        {"cx": -12.0, "cy": 14.0, "depth": 0.60, "radius": 1.2},
    ]

    for p in potholes:
        pts_count = int(n * 0.02)
        count = min(pts_count, n - idx)
        if count <= 0:
            break
        px = rng.normal(p["cx"], p["radius"] * 0.4, count).astype(np.float32)
        py = rng.normal(p["cy"], p["radius"] * 0.4, count).astype(np.float32)
        # Half points at ground, half points at depression floor
        half = count // 2
        pz_top = rng.normal(0.0, 0.02, half).astype(np.float32)
        pz_bot = rng.normal(-p["depth"], 0.04, count - half).astype(np.float32)
        pz = np.concatenate([pz_top, pz_bot])

        points[idx:idx + count, 0] = px
        points[idx:idx + count, 1] = py
        points[idx:idx + count, 2] = pz
        points[idx:idx + count, 3] = 3.0  # LABEL_POTHOLE_TRENCH
        idx += count

    # ─── 5. Fill remaining with background ground/environment ─────────────
    remaining = n - idx
    if remaining > 0:
        r_rem = rng.uniform(5.0, 75.0, remaining).astype(np.float32)
        th_rem = rng.uniform(-np.pi, np.pi, remaining).astype(np.float32)
        points[idx:, 0] = r_rem * np.cos(th_rem)
        points[idx:, 1] = r_rem * np.sin(th_rem)
        points[idx:, 2] = rng.normal(0.0, 0.05, remaining).astype(np.float32)
        points[idx:, 3] = 0.0

    return points


class MockLidarStream:
    def __init__(self, n: int = DEFAULT_N):
        self.n = n
        self.frame_id = 0

    def next_frame(self):
        frame = generate_frame(n=self.n, frame_id=self.frame_id)
        fid = self.frame_id
        self.frame_id += 1
        return fid, frame
