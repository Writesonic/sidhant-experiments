import os
import shutil
import subprocess

from video_effects.core import BaseCapability
from video_effects.skills.composition.schemas import ComposeFinalRequest, ComposeFinalResponse


class ComposeFinalCapability(BaseCapability[ComposeFinalRequest, ComposeFinalResponse]):
    async def execute(self, request: ComposeFinalRequest) -> ComposeFinalResponse:
        os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
        if request.has_audio and request.audio_path and os.path.exists(request.audio_path):
            cmd = [
                "ffmpeg", "-y",
                "-i", request.processed_video,
                "-i", request.audio_path,
                "-c:v", "copy", "-c:a", "copy",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", request.output_path,
            ]
            self.logger.info(f"Composing final video (mux with audio): {request.output_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", request.processed_video,
                    "-i", request.audio_path,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-shortest", request.output_path,
                ]
                self.logger.info("Stream copy failed, re-encoding audio to AAC")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg composition failed: {result.stderr}")
        else:
            self.logger.info(f"No audio, copying processed video to: {request.output_path}")
            shutil.copy2(request.processed_video, request.output_path)
        self.logger.info(f"Final video written to {request.output_path}")
        return ComposeFinalResponse(output_video=request.output_path)
