import numpy as np
import cv2

from video_effects.effects.base import BaseEffect, EffectContext
from video_effects.schemas.effects import EffectCue, VideoInfo


class ZoomEffect(BaseEffect):
    """Zoom effect with face-tracked, center, or point tracking."""

    def __init__(self):
        self._cues: list[EffectCue] = []
        self._face_data: list[tuple[float, float, float, float]] | None = None
        self._video_info: VideoInfo | None = None

    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue]) -> None:
        self._cues = effect_cues
        self._video_info = video_info

        # Pre-compute face tracking for face-tracked zoom cues
        face_cues = [
            c for c in effect_cues
            if c.zoom_params and c.zoom_params.tracking == "face"
        ]
        if face_cues:
            self._setup_face_tracking(video_info, face_cues)

    def _setup_face_tracking(
        self, video_info: VideoInfo, face_cues: list[EffectCue]
    ) -> None:
        """Run face detection on active ranges."""
        from video_effects.helpers.face_tracking import detect_faces

        active_ranges = [
            (
                int(c.start_time * video_info.fps),
                int(c.end_time * video_info.fps),
            )
            for c in face_cues
        ]
        # face_data will be populated by the helper
        self._face_data = None  # Will be set when we have video path access

    def apply_frame(
        self, frame: np.ndarray, timestamp: float, context: EffectContext
    ) -> np.ndarray:
        active_cues = self.get_active_cues(timestamp)
        if not active_cues:
            return frame

        h, w = frame.shape[:2]
        result = frame

        for cue in active_cues:
            params = cue.zoom_params
            if params is None:
                continue

            z = params.zoom_level

            # Compute easing intensity based on position within the cue
            duration = cue.end_time - cue.start_time
            if duration > 0:
                progress = (timestamp - cue.start_time) / duration
                p = self._ease(progress, params.easing)
            else:
                p = 1.0

            # Interpolate zoom: 1.0 -> target -> 1.0
            # Ramp up in first 20%, hold, ramp down in last 20%
            if progress < 0.15:
                intensity = progress / 0.15
            elif progress > 0.85:
                intensity = (1.0 - progress) / 0.15
            else:
                intensity = 1.0

            current_zoom = 1.0 + (z - 1.0) * intensity

            if params.tracking == "face" and self._face_data is not None:
                fx, fy, _, _ = self._face_data[context.frame_index]
                tx = self._lerp(w / 2, fx, intensity)
                ty = self._lerp(h / 2, fy, intensity)
            elif params.tracking == "center":
                tx, ty = w / 2, h / 2
            else:
                # point tracking — default to center
                tx, ty = w / 2, h / 2

            # Build affine matrix: zoom centered on (tx, ty)
            sx = w / 2 - tx * current_zoom
            sy = h / 2 - ty * current_zoom
            M = np.float32([[current_zoom, 0, sx], [0, current_zoom, sy]])

            result = cv2.warpAffine(
                result, M, (w, h), borderMode=cv2.BORDER_REPLICATE
            )

        return result

    def _ease(self, t: float, easing: str) -> float:
        """Apply easing function to progress value."""
        t = max(0.0, min(1.0, t))
        if easing == "snap":
            # Quick ease-in, hold, quick ease-out
            if t < 0.1:
                return t / 0.1
            elif t > 0.9:
                return (1.0 - t) / 0.1
            return 1.0
        elif easing == "overshoot":
            # Slight overshoot then settle
            if t < 0.5:
                s = t * 2
                return s * s * (2.7 * s - 1.7)
            return 1.0
        else:  # smooth
            # Smooth ease-in-out (cubic)
            if t < 0.5:
                return 4 * t * t * t
            return 1 - pow(-2 * t + 2, 3) / 2
