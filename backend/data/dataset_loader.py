"""
KAVACH-2.5D — SemanticKITTI Dataset Loader (Stub)

Placeholder for Rohan's Phase 2 inference pipeline.
When implemented, this will:
  1. Parse SemanticKITTI .bin point cloud files
  2. Run PointNet++ / MinkowskiEngine inference
  3. Output (N, 4) arrays matching Interface 1: [X, Y, Z, Label_ID]

Currently returns mock data for integration testing.
"""

import numpy as np
from pathlib import Path
from typing import Optional
from backend.data.mock_generator import generate_frame


def load_bin_file(filepath: str) -> np.ndarray:
    """
    Load a SemanticKITTI .bin point cloud file.

    Args:
        filepath: Path to .bin file (N*4 float32 values: x, y, z, reflectance)

    Returns:
        np.ndarray of shape (N, 4) — [X, Y, Z, Label_ID]
        NOTE: Label_ID is set to 0 (Road) for all points until inference pipeline is ready.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {filepath}")

    # SemanticKITTI format: float32 x N*4 (x, y, z, reflectance)
    raw = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

    # Replace reflectance column with label (default: 0 = Road)
    # Real labels will come from inference in Phase 2
    points = np.zeros((raw.shape[0], 4), dtype=np.float32)
    points[:, :3] = raw[:, :3]  # X, Y, Z
    points[:, 3] = 0.0          # Label = Road (placeholder)

    return points


def load_label_file(filepath: str) -> np.ndarray:
    """
    Load a SemanticKITTI .label file.

    Args:
        filepath: Path to .label file (uint32 per point: lower 16 bits = semantic label)

    Returns:
        np.ndarray of shape (N,) with int32 label IDs
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {filepath}")

    raw_labels = np.fromfile(str(path), dtype=np.uint32)
    # Lower 16 bits = semantic label, upper 16 = instance ID
    semantic = (raw_labels & 0xFFFF).astype(np.int32)
    return semantic


class DatasetStream:
    """
    Stream point clouds from a SemanticKITTI sequence directory.

    Expected directory structure:
        sequence_dir/
        ├── velodyne/
        │   ├── 000000.bin
        │   ├── 000001.bin
        │   └── ...
        └── labels/       (optional — Phase 2)
            ├── 000000.label
            └── ...

    Falls back to mock data if sequence_dir is None or doesn't exist.
    """

    def __init__(self, sequence_dir: Optional[str] = None):
        self.use_mock = True
        self.frame_id = 0
        self.bin_files = []

        if sequence_dir:
            velodyne_dir = Path(sequence_dir) / "velodyne"
            if velodyne_dir.exists():
                self.bin_files = sorted(velodyne_dir.glob("*.bin"))
                if self.bin_files:
                    self.use_mock = False
                    self.labels_dir = Path(sequence_dir) / "labels"

    def next_frame(self):
        """
        Get the next frame as (frame_id, np.ndarray of shape (N, 4)).
        """
        if self.use_mock:
            frame = generate_frame(frame_id=self.frame_id)
        else:
            idx = self.frame_id % len(self.bin_files)
            frame = load_bin_file(str(self.bin_files[idx]))

            # Try to load labels if available
            if self.labels_dir.exists():
                label_file = self.labels_dir / self.bin_files[idx].name.replace(".bin", ".label")
                if label_file.exists():
                    labels = load_label_file(str(label_file))
                    if len(labels) == frame.shape[0]:
                        frame[:, 3] = labels.astype(np.float32)

        fid = self.frame_id
        self.frame_id += 1
        return fid, frame
