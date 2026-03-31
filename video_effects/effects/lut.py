from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class LUT3D:
    size: int
    table: np.ndarray   # (N, N, N, 3) float32, values in [0, 1]
    full: np.ndarray     # (256, 256, 256, 3) uint8 — pre-expanded for O(1) lookup


def parse_cube_file(path: str | Path) -> LUT3D:
    """Parse an Adobe .cube 3D LUT file and pre-expand to a 256^3 lookup table."""
    path = Path(path)
    size: int | None = None
    data_lines: list[str] = []

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("TITLE"):
            continue
        if line.startswith("DOMAIN_MIN") or line.startswith("DOMAIN_MAX"):
            continue
        if line.startswith("LUT_1D_SIZE"):
            raise ValueError(f"1D LUTs not supported: {path}")
        if line.startswith("LUT_3D_SIZE"):
            size = int(line.split()[1])
            continue
        data_lines.append(line)

    if size is None:
        raise ValueError(f"Missing LUT_3D_SIZE in {path}")

    expected = size ** 3
    if len(data_lines) != expected:
        raise ValueError(
            f"Expected {expected} data lines for size {size}, got {len(data_lines)} in {path}"
        )

    flat = np.array(
        [[float(v) for v in line.split()] for line in data_lines],
        dtype=np.float32,
    )
    # .cube spec: R varies fastest, then G, then B (B is outermost loop)
    table = flat.reshape(size, size, size, 3)  # [B, G, R, 3]
    full = _expand_to_256(table, size)
    return LUT3D(size=size, table=table, full=full)


def _expand_to_256(table: np.ndarray, size: int) -> np.ndarray:
    """Pre-compute a full 256^3 lookup table via trilinear interpolation.

    Trades ~48 MB RAM for O(1) per-pixel application at runtime.
    """
    idx = np.arange(256, dtype=np.float32) * ((size - 1) / 255.0)

    i0 = np.clip(idx.astype(np.int32), 0, size - 2)
    i1 = i0 + 1
    frac = idx - i0.astype(np.float32)

    # Interpolate along R (axis 2)
    t = table[np.ix_(np.arange(size), np.arange(size), i0)] * (1 - frac)[np.newaxis, np.newaxis, :, np.newaxis] + \
        table[np.ix_(np.arange(size), np.arange(size), i1)] * frac[np.newaxis, np.newaxis, :, np.newaxis]
    # t shape: (size, size, 256, 3)

    # Interpolate along G (axis 1)
    g0 = i0[:size]; g1 = i1[:size]  # noqa: E702
    t2 = t[:, i0, :, :] * (1 - frac)[np.newaxis, :, np.newaxis, np.newaxis] + \
         t[:, i1, :, :] * frac[np.newaxis, :, np.newaxis, np.newaxis]
    # t2 shape: (size, 256, 256, 3)

    # Interpolate along B (axis 0)
    t3 = t2[i0, :, :, :] * (1 - frac)[:, np.newaxis, np.newaxis, np.newaxis] + \
         t2[i1, :, :, :] * frac[:, np.newaxis, np.newaxis, np.newaxis]
    # t3 shape: (256, 256, 256, 3)

    return np.clip(t3 * 255.0, 0, 255).astype(np.uint8)


def apply_lut3d(
    frame: np.ndarray, lut: LUT3D, intensity: float = 1.0,
) -> np.ndarray:
    """Apply a 3D LUT to an RGB uint8 frame. Uses pre-expanded table for speed."""
    if intensity <= 0.0:
        return frame

    # O(1) per-pixel: single fancy-index into the 256^3 table
    graded = lut.full[frame[..., 2], frame[..., 1], frame[..., 0]]

    if intensity >= 1.0:
        return graded
    return cv2.addWeighted(frame, 1.0 - intensity, graded, intensity, 0)
