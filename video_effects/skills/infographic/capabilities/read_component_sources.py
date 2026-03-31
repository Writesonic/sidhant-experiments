from video_effects.core import BaseCapability
from video_effects.helpers.remotion import _get_remotion_dir
from video_effects.skills.infographic.schemas import ReadComponentSourcesRequest, ReadComponentSourcesResponse


class ReadComponentSourcesCapability(BaseCapability[ReadComponentSourcesRequest, ReadComponentSourcesResponse]):
    async def execute(self, request):
        generated_dir = _get_remotion_dir() / "src" / "components" / "generated"
        sources = {}
        if generated_dir.exists():
            for f in generated_dir.iterdir():
                if f.suffix == ".tsx":
                    sources[f.stem] = f.read_text()
        return ReadComponentSourcesResponse(sources=sources)
