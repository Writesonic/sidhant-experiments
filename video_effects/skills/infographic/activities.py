from video_effects.infrastructure import run_capability_sync
from video_effects.skills.registry import register_activity
from video_effects.skills.infographic.capabilities.cleanup_generated import CleanupGeneratedCapability
from video_effects.skills.infographic.capabilities.plan_category import PlanCategoryCapability
from video_effects.skills.infographic.capabilities.generate_infographic_code import GenerateInfographicCodeCapability
from video_effects.skills.infographic.capabilities.validate_infographic import ValidateInfographicCapability
from video_effects.skills.infographic.capabilities.build_generated_registry import BuildGeneratedRegistryCapability
from video_effects.skills.infographic.capabilities.materialize_library_templates import MaterializeLibraryTemplatesCapability
from video_effects.skills.infographic.schemas import (
    CleanupGeneratedRequest,
    PlanCategoryRequest,
    GenerateInfographicCodeRequest,
    ValidateInfographicRequest,
    BuildGeneratedRegistryRequest,
    MaterializeLibraryTemplatesRequest,
)


@register_activity(name="vfx_cleanup_generated", description="Remove generated infographic components")
def vfx_cleanup_generated(input_data: dict) -> dict:
    request = CleanupGeneratedRequest()
    response = run_capability_sync(CleanupGeneratedCapability, request)
    return response.model_dump()

@register_activity(name="vfx_plan_infographics", description="Plan infographic components via LLM")
def vfx_plan_infographics(input_data: dict) -> dict:
    request = PlanCategoryRequest(category="infographics", prompt_filename="plan_infographics.md", **input_data)
    response = run_capability_sync(PlanCategoryCapability, request)
    return response.model_dump()

@register_activity(name="vfx_plan_diagrams", description="Plan diagram components via LLM")
def vfx_plan_diagrams(input_data: dict) -> dict:
    request = PlanCategoryRequest(category="diagrams", prompt_filename="plan_diagrams.md", **input_data)
    response = run_capability_sync(PlanCategoryCapability, request)
    return response.model_dump()

@register_activity(name="vfx_plan_timelines", description="Plan timeline components via LLM")
def vfx_plan_timelines(input_data: dict) -> dict:
    request = PlanCategoryRequest(category="timelines", prompt_filename="plan_timelines.md", **input_data)
    response = run_capability_sync(PlanCategoryCapability, request)
    return response.model_dump()

@register_activity(name="vfx_plan_quotes", description="Plan quote components via LLM")
def vfx_plan_quotes(input_data: dict) -> dict:
    request = PlanCategoryRequest(category="quotes", prompt_filename="plan_quotes.md", **input_data)
    response = run_capability_sync(PlanCategoryCapability, request)
    return response.model_dump()

@register_activity(name="vfx_plan_code_blocks", description="Plan code block components via LLM")
def vfx_plan_code_blocks(input_data: dict) -> dict:
    request = PlanCategoryRequest(category="code_blocks", prompt_filename="plan_code_blocks.md", **input_data)
    response = run_capability_sync(PlanCategoryCapability, request)
    return response.model_dump()

@register_activity(name="vfx_plan_comparisons", description="Plan comparison components via LLM")
def vfx_plan_comparisons(input_data: dict) -> dict:
    request = PlanCategoryRequest(category="comparisons", prompt_filename="plan_comparisons.md", **input_data)
    response = run_capability_sync(PlanCategoryCapability, request)
    return response.model_dump()

@register_activity(name="vfx_generate_infographic_code", description="Generate TSX code for infographic")
def vfx_generate_infographic_code(input_data: dict) -> dict:
    request = GenerateInfographicCodeRequest(**input_data)
    response = run_capability_sync(GenerateInfographicCodeCapability, request)
    return response.model_dump()

@register_activity(name="vfx_validate_infographic", description="Type-check and test-render infographic")
def vfx_validate_infographic(input_data: dict) -> dict:
    request = ValidateInfographicRequest(**input_data)
    response = run_capability_sync(ValidateInfographicCapability, request)
    return response.model_dump()

@register_activity(name="vfx_build_generated_registry", description="Build final generated component registry")
def vfx_build_generated_registry(input_data: dict) -> dict:
    request = BuildGeneratedRegistryRequest(**input_data)
    response = run_capability_sync(BuildGeneratedRegistryCapability, request)
    return response.model_dump()

@register_activity(name="vfx_materialize_library_templates", description="Write library templates to generated/")
def vfx_materialize_library_templates(input_data: dict) -> dict:
    request = MaterializeLibraryTemplatesRequest(**input_data)
    response = run_capability_sync(MaterializeLibraryTemplatesCapability, request)
    return response.model_dump()
