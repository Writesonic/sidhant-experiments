from pydantic import BaseModel


class ProgrammerBrainstormRequest(BaseModel):
    spatial_context: dict
    transcript: str = ""
    segments: list[dict] = []
    style_config: dict | None = None
    video_info: dict = {}
    effects: list[dict] = []


class ProgrammerBrainstormResponse(BaseModel):
    reasoning: str = ""
    components: list[dict] = []


class ProgrammerCritiqueRequest(BaseModel):
    proposals: list[dict]
    spatial_context: dict = {}
    transcript: str = ""
    video_info: dict = {}
    max_specs: int = 6


class ProgrammerCritiqueResponse(BaseModel):
    components: list[dict] = []


class ProgrammerGenerateCodeRequest(BaseModel):
    spec: dict
    style_config: dict | None = None
    video_info: dict = {}
    attempt: int = 1
    previous_errors: list[str] = []
    previous_code: str = ""


class ProgrammerGenerateCodeResponse(BaseModel):
    component_id: str
    tsx_code: str
    export_name: str
    props: dict = {}


class PlaceLibraryTemplatesRequest(BaseModel):
    pinned_templates: list
    spatial_context: dict = {}
    transcript: str = ""
    segments: list[dict] = []
    existing_components: list[dict] = []
    style_config: dict | None = None
    video_info: dict = {}


class PlaceLibraryTemplatesResponse(BaseModel):
    placements: list[dict] = []
