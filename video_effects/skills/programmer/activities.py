from video_effects.infrastructure import run_capability_sync
from video_effects.skills.registry import register_activity
from video_effects.skills.programmer.capabilities.brainstorm import ProgrammerBrainstormCapability
from video_effects.skills.programmer.capabilities.critique import ProgrammerCritiqueCapability
from video_effects.skills.programmer.capabilities.generate_code import ProgrammerGenerateCodeCapability
from video_effects.skills.programmer.capabilities.place_library_templates import PlaceLibraryTemplatesCapability
from video_effects.skills.programmer.schemas import (
    ProgrammerBrainstormRequest,
    ProgrammerCritiqueRequest,
    ProgrammerGenerateCodeRequest,
    PlaceLibraryTemplatesRequest,
)


@register_activity(name="vfx_programmer_brainstorm", description="Creative brainstorm: propose visual components")
def vfx_programmer_brainstorm(input_data: dict) -> dict:
    request = ProgrammerBrainstormRequest(**input_data)
    response = run_capability_sync(ProgrammerBrainstormCapability, request)
    return response.model_dump()


@register_activity(name="vfx_programmer_critique", description="Self-critique and filter brainstorm proposals")
def vfx_programmer_critique(input_data: dict) -> dict:
    request = ProgrammerCritiqueRequest(**input_data)
    response = run_capability_sync(ProgrammerCritiqueCapability, request)
    return response.model_dump()


@register_activity(name="vfx_programmer_generate_code", description="Generate TSX code for a component")
def vfx_programmer_generate_code(input_data: dict) -> dict:
    request = ProgrammerGenerateCodeRequest(**input_data)
    response = run_capability_sync(ProgrammerGenerateCodeCapability, request)
    return response.model_dump()


@register_activity(name="vfx_place_library_templates", description="LLM places library templates at content-relevant moments")
def vfx_place_library_templates(input_data: dict) -> dict:
    request = PlaceLibraryTemplatesRequest(**input_data)
    response = run_capability_sync(PlaceLibraryTemplatesCapability, request)
    return response.model_dump()
