from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from video_effects.effects.base import BaseEffect, EffectContext
from video_effects.effects.lut import LUT3D, apply_lut3d, parse_cube_file
from video_effects.schemas.effects import EffectCue, VideoInfo

logger = logging.getLogger(__name__)

_LUT_DIR = Path(__file__).resolve().parent.parent / "data" / "luts"

LUT_PRESETS: dict[str, str] = {
    "warm":     "negative_new/kodak_portra_400.cube",
    "cool":     "fujixtransiii/fuji_xtrans_iii_classic_chrome.cube",
    "dramatic": "colorslide/kodak_kodachrome_64.cube",
    "vibrant":  "colorslide/fuji_velvia_50.cube",
    "vintage":  "instant_consumer/polaroid_time_zero_expired.cube",
    "film":     "colorslide/fuji_provia_100f.cube",
    "bw":       "negative_new/kodak_tri-x_400.cube",
    "sepia":    "fujixtransiii/fuji_xtrans_iii_sepia.cube",
}


class ColorEffect(BaseEffect):
    """LUT-based color grading effect with film stock presets."""

    def __init__(self):
        super().__init__()
        self._video_info: VideoInfo | None = None
        self._lut_cache: dict[str, LUT3D] = {}

    def setup(
        self,
        video_info: VideoInfo,
        effect_cues: list[EffectCue],
        *,
        cache_dir: str | None = None,
        video_path: str | None = None,
    ) -> None:
        self._cues = effect_cues
        self._video_info = video_info

        presets_needed = {
            cue.color_params.preset
            for cue in effect_cues
            if cue.color_params and cue.color_params.preset != "custom"
        }

        for preset in presets_needed:
            rel_path = LUT_PRESETS.get(preset)
            if not rel_path:
                continue
            lut_path = _LUT_DIR / rel_path
            try:
                self._lut_cache[preset] = parse_cube_file(lut_path)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Failed to load LUT for '%s': %s", preset, exc)

    def apply_frame(
        self, frame: np.ndarray, timestamp: float, context: EffectContext,
    ) -> np.ndarray:
        active_cues = self.get_active_cues(timestamp)
        if not active_cues:
            return frame

        result = frame.copy()

        for cue in active_cues:
            params = cue.color_params
            if params is None:
                continue

            if params.preset == "custom":
                adjustments = (
                    np.array(
                        [params.r_adjust, params.g_adjust, params.b_adjust],
                        dtype=np.float32,
                    )
                    * params.intensity
                )
                result = np.clip(
                    result.astype(np.float32) + adjustments, 0, 255,
                ).astype(np.uint8)
                continue

            lut = self._lut_cache.get(params.preset)
            if lut is None:
                continue

            result = apply_lut3d(result, lut, params.intensity)

        return result
