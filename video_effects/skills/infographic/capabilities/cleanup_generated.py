from video_effects.core import BaseCapability
from video_effects.helpers.remotion import _get_remotion_dir
from video_effects.skills.infographic._shared import _rebuild_registry
from video_effects.skills.infographic.schemas import CleanupGeneratedRequest, CleanupGeneratedResponse

class CleanupGeneratedCapability(BaseCapability[CleanupGeneratedRequest, CleanupGeneratedResponse]):
    async def execute(self, request):
        remotion_dir = _get_remotion_dir()
        generated_dir = remotion_dir / "src" / "components" / "generated"
        cleaned = 0
        if generated_dir.exists():
            for f in generated_dir.iterdir():
                if f.name != ".gitignore":
                    f.unlink()
                    cleaned += 1
        generated_dir.mkdir(parents=True, exist_ok=True)
        _rebuild_registry(generated_dir)
        self.logger.info("Cleaned %d generated files", cleaned)
        return CleanupGeneratedResponse(cleaned=cleaned)
