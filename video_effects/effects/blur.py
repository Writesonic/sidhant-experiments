import logging

import numpy as np
import cv2

from video_effects.effects.base import BaseEffect, EffectContext
from video_effects.schemas.effects import EffectCue, VideoInfo

logger = logging.getLogger(__name__)


class BlurEffect(BaseEffect):
    """Blur effect: gaussian, face_pixelate, background, radial."""

    def __init__(self):
        super().__init__()
        self._segmenter = None
        self._face_detector = None
        self._face_detector_backend: str | None = None
        self._video_info: VideoInfo | None = None

    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue],
              *, cache_dir: str | None = None, video_path: str | None = None) -> None:
        self._cues = effect_cues
        self._video_info = video_info

        # Pre-init background segmentation if needed
        bg_cues = [
            c for c in effect_cues
            if c.blur_params and c.blur_params.blur_type == "background"
        ]
        if bg_cues:
            self._setup_segmentation()
        face_cues = [
            c for c in effect_cues
            if c.blur_params and c.blur_params.blur_type == "face_pixelate"
        ]
        if face_cues:
            self._setup_face_detection()

    def _get_mediapipe_solutions(self):
        """Return MediaPipe legacy solutions module if available."""
        try:
            import mediapipe as mp
        except Exception:
            return None
        return getattr(mp, "solutions", None)

    def _setup_segmentation(self) -> None:
        """Initialize MediaPipe selfie segmentation for background blur."""
        solutions = self._get_mediapipe_solutions()
        if solutions is None or not hasattr(solutions, "selfie_segmentation"):
            logger.warning(
                "MediaPipe selfie segmentation unavailable (mp.solutions missing); "
                "background blur will be skipped."
            )
            self._segmenter = None
            return

        self._segmenter = solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1  # landscape model (faster)
        )

    def _setup_face_detection(self) -> None:
        """Initialize face detector with MediaPipe (preferred) or OpenCV fallback."""
        solutions = self._get_mediapipe_solutions()
        if solutions is not None and hasattr(solutions, "face_detection"):
            self._face_detector = solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5,
            )
            self._face_detector_backend = "mediapipe"
            return

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            logger.warning(
                "No face detector available (MediaPipe solutions missing and "
                "OpenCV Haar cascade unavailable); face_pixelate will be skipped."
            )
            self._face_detector = None
            self._face_detector_backend = None
            return

        self._face_detector = cascade
        self._face_detector_backend = "opencv"
        logger.info("Using OpenCV Haar cascade fallback for face_pixelate")

    def apply_frame(
        self, frame: np.ndarray, timestamp: float, context: EffectContext
    ) -> np.ndarray:
        active_cues = self.get_active_cues(timestamp)
        if not active_cues:
            return frame

        result = frame.copy()

        for cue in active_cues:
            params = cue.blur_params
            if params is None:
                continue

            if params.blur_type == "gaussian":
                result = self._apply_gaussian(result, params.radius, params.target_region)
            elif params.blur_type == "face_pixelate":
                result = self._apply_face_pixelate(result, params.radius)
            elif params.blur_type == "background":
                result = self._apply_background_blur(result, params.radius)
            elif params.blur_type == "radial":
                result = self._apply_radial_blur(result, params.radius)

        return result

    def _apply_gaussian(self, frame: np.ndarray, radius: float, region) -> np.ndarray:
        """Apply Gaussian blur to a target region."""
        h, w = frame.shape[:2]
        x1 = int(region.x * w)
        y1 = int(region.y * h)
        x2 = int((region.x + region.width) * w)
        y2 = int((region.y + region.height) * h)

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return frame

        k = int(radius) * 2 + 1  # Must be odd
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return frame

    def _apply_face_pixelate(self, frame: np.ndarray, radius: float) -> np.ndarray:
        """Detect face and pixelate the region."""
        if self._face_detector is None:
            self._setup_face_detection()
            if self._face_detector is None:
                return frame

        h, w = frame.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []

        if self._face_detector_backend == "mediapipe":
            # MediaPipe face detector expects RGB input.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_detector.process(rgb)
            detections = getattr(results, "detections", None) or []
            for detection in detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                boxes.append((x1, y1, bw, bh))
        elif self._face_detector_backend == "opencv":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self._face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            boxes.extend((int(x), int(y), int(bw), int(bh)) for x, y, bw, bh in detected)
        else:
            return frame

        for x1, y1, bw, bh in boxes:
            x2, y2 = min(w, x1 + bw), min(h, y1 + bh)

            if x2 <= x1 or y2 <= y1:
                continue

            # Pixelate: downscale then upscale
            roi = frame[y1:y2, x1:x2]
            pixel_size = max(2, int(radius / 3))
            small = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
            frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        return frame

    def _apply_background_blur(self, frame: np.ndarray, radius: float) -> np.ndarray:
        """Blur background using selfie segmentation mask."""
        if self._segmenter is None:
            return frame

        results = self._segmenter.process(frame)
        mask = results.segmentation_mask  # float32 [0, 1]

        # Refine mask edges with bilateral filter
        mask = cv2.bilateralFilter(mask, 9, 75, 75)

        # Threshold and smooth
        mask = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        k = int(radius) * 2 + 1
        blurred = cv2.GaussianBlur(frame, (k, k), 0)

        # Composite: person pixels from original, background from blurred
        mask_3ch = mask[:, :, np.newaxis]
        result = (frame * mask_3ch + blurred * (1.0 - mask_3ch)).astype(np.uint8)
        return result

    def _apply_radial_blur(self, frame: np.ndarray, radius: float) -> np.ndarray:
        """Radial zoom blur from center (adapted from zoom_bounce._apply_zoom_blur_cpu)."""
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        strength = min(radius / 50.0, 1.0)  # Normalize radius to 0-1 strength
        n_samples = 6
        base_zoom = 1.0
        spread = 0.05 * strength * base_zoom

        accum = np.zeros_like(frame, dtype=np.float32)
        for i in range(n_samples):
            t = (i / max(n_samples - 1, 1)) * 2.0 - 1.0
            dz = t * spread
            sz = base_zoom + dz
            M = np.float32([
                [sz, 0, cx * (1.0 - sz)],
                [0, sz, cy * (1.0 - sz)],
            ])
            sample = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            accum += sample.astype(np.float32)

        accum /= n_samples
        orig_f = frame.astype(np.float32)
        blended = orig_f + (accum - orig_f) * strength
        np.clip(blended, 0, 255, out=blended)
        return blended.astype(np.uint8)
