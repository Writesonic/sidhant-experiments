"""Thin wrapper around cv_experiments face detection.

Adapts get_face_data_seek() and smooth_data() from
cv_experiments/zoom_bounce.py for use in the video effects pipeline.
"""

import sys
from pathlib import Path

import numpy as np

# Add cv_experiments to path for imports
_CV_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "cv_experiments"
if str(_CV_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_CV_EXPERIMENTS_DIR))


def detect_faces(
    video_path: str,
    active_ranges: list[tuple[int, int]],
    total_frames: int,
    stride: int = 3,
) -> list[tuple[float, float, float, float]]:
    """Run face detection on active frame ranges.

    Returns list of (cx, cy, fw, fh) for every frame in the video.
    Uses get_face_data_seek from cv_experiments/zoom_bounce.py.

    Args:
        video_path: Path to video file.
        active_ranges: List of (start_frame, end_frame) ranges to detect in.
        total_frames: Total number of frames in the video.
        stride: Detect every N frames (interpolate the rest).

    Returns:
        List of (center_x, center_y, face_width, face_height) per frame.
    """
    from zoom_bounce import get_face_data_seek

    data, fps, (w, h) = get_face_data_seek(
        video_path, active_ranges, total_frames, stride=stride
    )
    return data


def smooth_data(data: list, alpha: float = 0.1) -> np.ndarray:
    """Exponential moving average filter for tracking data.

    Adapted from cv_experiments/zoom_bounce.py smooth_data().

    Args:
        data: List of (cx, cy, fw, fh) tuples.
        alpha: Smoothing factor (0-1). Lower = smoother.

    Returns:
        Smoothed numpy array of same shape.
    """
    a = np.array(data, dtype=np.float64)
    o = np.empty_like(a)
    o[0] = a[0]
    inv = 1.0 - alpha
    for i in range(1, len(a)):
        o[i] = alpha * a[i] + inv * o[i - 1]
    return o.astype(np.int32)
