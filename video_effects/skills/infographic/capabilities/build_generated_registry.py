from video_effects.core import BaseCapability
from video_effects.helpers.remotion import _get_remotion_dir
from video_effects.skills.infographic.schemas import BuildGeneratedRegistryRequest, BuildGeneratedRegistryResponse

class BuildGeneratedRegistryCapability(BaseCapability[BuildGeneratedRegistryRequest, BuildGeneratedRegistryResponse]):
    async def execute(self, request):
        if not request.generated_components:
            return BuildGeneratedRegistryResponse()
        fps = request.video_fps
        remotion_dir = _get_remotion_dir()
        generated_dir = remotion_dir / "src" / "components" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        imports = []
        entries = []
        for comp in request.generated_components:
            cid = comp["component_id"]
            ename = comp["export_name"]
            imports.append(f'import {{ {ename} }} from "./{cid}";')
            entries.append(f'  "{cid}": {ename} as React.FC<any>,')
        registry_code = (
            'import React from "react";\n'
            + "\n".join(imports) + "\n"
            "\n"
            "type ComponentMap = { [key: string]: React.FC<any> };\n"
            "\n"
            "export const GeneratedRegistry: ComponentMap = {\n"
            + "\n".join(entries) + "\n"
            "};\n"
        )
        registry_path = generated_dir / "_registry.ts"
        registry_path.write_text(registry_code)
        self.logger.info("Wrote generated registry with %d components", len(request.generated_components))
        remotion_components = []
        for comp in request.generated_components:
            spec = comp["spec"]
            start_frame = round(spec["start_time"] * fps)
            end_frame = round(spec["end_time"] * fps)
            duration_frames = max(1, end_frame - start_frame)
            remotion_components.append({
                "template": comp["component_id"], "startFrame": start_frame, "durationInFrames": duration_frames,
                "props": comp["props"],
                "bounds": spec.get("bounds", {"x": 0.1, "y": 0.1, "w": 0.35, "h": 0.3}),
                "zIndex": 10, "anchor": spec.get("anchor", "static"),
            })
        return BuildGeneratedRegistryResponse(components=remotion_components, registry_path=str(registry_path))
