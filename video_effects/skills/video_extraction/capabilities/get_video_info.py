import json
import subprocess

from video_effects.core import BaseCapability
from video_effects.schemas.effects import VideoInfo
from video_effects.skills.video_extraction.schemas import GetVideoInfoRequest, GetVideoInfoResponse


class GetVideoInfoCapability(BaseCapability[GetVideoInfoRequest, GetVideoInfoResponse]):
    async def execute(self, request: GetVideoInfoRequest) -> GetVideoInfoResponse:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            request.video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe = json.loads(result.stdout)

        video_stream = next(
            (s for s in probe["streams"] if s["codec_type"] == "video"), None
        )
        audio_stream = next(
            (s for s in probe["streams"] if s["codec_type"] == "audio"), None
        )

        if video_stream is None:
            raise ValueError(f"No video stream found in {request.video_path}")

        fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

        duration = float(probe["format"].get("duration", 0))
        width = int(video_stream["width"])
        height = int(video_stream["height"])

        info = VideoInfo(
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            codec=video_stream.get("codec_name", ""),
            total_frames=int(duration * fps),
            audio_codec=audio_stream.get("codec_name", "") if audio_stream else "",
            has_audio=audio_stream is not None,
            color_space=video_stream.get("color_space", ""),
            color_transfer=video_stream.get("color_transfer", ""),
            color_primaries=video_stream.get("color_primaries", ""),
            pix_fmt=video_stream.get("pix_fmt", ""),
        )
        return GetVideoInfoResponse(video_info=info)
