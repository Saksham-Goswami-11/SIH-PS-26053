"""
KAVACH-2.5D — Telemetry Metrics Calculator

Tracks per-frame and rolling-average performance metrics:
  - FPS (frames per second)
  - Processing latency (ms)
  - Memory savings (%)
  - Compression ratio (raw vs. compressed cell count)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameMetrics:
    """Metrics for a single processed frame."""
    fps: float = 0.0
    latency_ms: float = 0.0
    raw_points_count: int = 0
    compressed_cells_count: int = 0
    memory_saved_percent: float = 0.0


class MetricsTracker:
    """
    Rolling-window metrics tracker.

    Maintains a sliding window of recent frame times to compute
    smooth FPS and latency averages.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._frame_times: deque[float] = deque(maxlen=window_size)
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._last_frame_time: Optional[float] = None
        self._total_frames: int = 0

    def record_frame(self, engine_latency_ms: float, raw_count: int, compressed_count: int) -> FrameMetrics:
        """
        Record metrics for a completed frame.

        Args:
            engine_latency_ms: Time spent in the C++ engine (from engine telemetry).
            raw_count: Number of input points (N).
            compressed_count: Number of output cells (M).

        Returns:
            FrameMetrics with computed values.
        """
        now = time.monotonic()

        # Compute inter-frame time for FPS
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            self._frame_times.append(dt)
        self._last_frame_time = now

        self._latencies.append(engine_latency_ms)
        self._total_frames += 1

        # Compute rolling averages
        avg_fps = 0.0
        if self._frame_times:
            avg_dt = sum(self._frame_times) / len(self._frame_times)
            avg_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

        # Memory savings: compare raw float32 (N*4*4 bytes) vs compressed (M*6*4 bytes)
        raw_bytes = raw_count * 4 * 4  # N points × 4 columns × 4 bytes
        compressed_bytes = compressed_count * 6 * 4  # M cells × 6 columns × 4 bytes
        memory_saved = (1.0 - compressed_bytes / raw_bytes) * 100.0 if raw_bytes > 0 else 0.0

        return FrameMetrics(
            fps=round(avg_fps, 1),
            latency_ms=round(avg_latency, 2),
            raw_points_count=raw_count,
            compressed_cells_count=compressed_count,
            memory_saved_percent=round(memory_saved, 1),
        )

    @property
    def total_frames(self) -> int:
        return self._total_frames
