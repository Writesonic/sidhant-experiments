import json
import os

from video_effects.core import BaseCapability
from video_effects.helpers.face_tracking import detect_faces, smooth_data, _probe_decoded_size
from video_effects.skills.face_detection.schemas import DetectFacesRequest, DetectFacesResponse


class DetectFacesCapability(BaseCapability[DetectFacesRequest, DetectFacesResponse]):
    async def execute(self, request: DetectFacesRequest) -> DetectFacesResponse:
        video_path = request.video_path
        video_info = request.video_info
        cache_dir = request.cache_dir
        fps = video_info.get("fps", 30)
        duration = video_info.get("duration", 0)
        total_frames = video_info.get("total_frames", 0) or int(duration * fps)
        os.makedirs(cache_dir, exist_ok=True)
        face_data_path = os.path.join(cache_dir, "face_tracking_zoom.json")
        if os.path.exists(face_data_path):
            try:
                with open(face_data_path) as f:
                    raw = json.load(f)
                cached_data = raw.get("face_data", raw) if isinstance(raw, dict) else raw
                cached_dims = raw.get("dimensions") if isinstance(raw, dict) else None
                decoded_w, decoded_h = _probe_decoded_size(video_path)
                if (cached_dims
                        and cached_dims.get("width") == decoded_w
                        and cached_dims.get("height") == decoded_h
                        and len(cached_data) >= total_frames):
                    self.logger.info("Face detection cache valid: %d frames", len(cached_data))
                    return DetectFacesResponse(
                        face_data_path=face_data_path,
                        frames_detected=len(cached_data),
                        from_cache=True,
                    )
            except (json.JSONDecodeError, KeyError):
                self.logger.warning("Invalid face cache, re-detecting")
        self.heartbeat_sync("Starting full-video face detection")
        face_data = detect_faces(
            video_path=video_path,
            active_ranges=[(0, total_frames - 1)],
            total_frames=total_frames,
        )
        self.heartbeat_sync("Smoothing face data")
        smoothed = smooth_data(face_data)
        smoothed_list = [tuple(int(v) for v in row) for row in smoothed]
        decoded_w, decoded_h = _probe_decoded_size(video_path)
        cache_payload = {
            "face_data": smoothed_list,
            "dimensions": {"width": decoded_w, "height": decoded_h},
        }
        with open(face_data_path, "w") as f:
            json.dump(cache_payload, f)
        self.logger.info("Face detection complete: %d frames written to %s", len(smoothed_list), face_data_path)
        return DetectFacesResponse(
            face_data_path=face_data_path,
            frames_detected=len(smoothed_list),
            from_cache=False,
        )
