from pydantic import BaseModel

class CleanupGeneratedRequest(BaseModel):
    pass  # no params needed

class CleanupGeneratedResponse(BaseModel):
    cleaned: int = 0

class PlanCategoryRequest(BaseModel):
    category: str
    prompt_filename: str
    spatial_context: dict
    transcript: str = ""
    segments: list[dict] = []
    style_config: dict | None = None
    video_fps: int = 30

class PlanCategoryResponse(BaseModel):
    infographics: list[dict] = []
    reasoning: str = ""

class GenerateInfographicCodeRequest(BaseModel):
    spec: dict
    style_config: dict | None = None
    video_info: dict = {}
    attempt: int = 1
    previous_errors: list[str] = []
    previous_code: str = ""

class GenerateInfographicCodeResponse(BaseModel):
    component_id: str
    tsx_code: str
    export_name: str
    props: dict = {}

class ValidateInfographicRequest(BaseModel):
    component_id: str
    tsx_code: str
    export_name: str
    props: dict | None = None

class ValidateInfographicResponse(BaseModel):
    valid: bool
    errors: list[str] = []
    preview_path: str = ""

class BuildGeneratedRegistryRequest(BaseModel):
    generated_components: list[dict]
    video_fps: int = 30

class BuildGeneratedRegistryResponse(BaseModel):
    components: list[dict] = []
    registry_path: str = ""

class MaterializeLibraryTemplatesRequest(BaseModel):
    template_ids: list[str] = []

class MaterializeLibraryTemplatesResponse(BaseModel):
    materialized: list[str] = []
