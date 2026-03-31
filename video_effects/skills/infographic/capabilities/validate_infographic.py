import subprocess
from video_effects.core import BaseCapability
from video_effects.helpers.remotion import render_still, _get_remotion_dir
from video_effects.skills.infographic._shared import REGISTRY_LOCK, _rebuild_registry
from video_effects.skills.infographic.schemas import ValidateInfographicRequest, ValidateInfographicResponse

class ValidateInfographicCapability(BaseCapability[ValidateInfographicRequest, ValidateInfographicResponse]):
    async def execute(self, request):
        remotion_dir = _get_remotion_dir()
        generated_dir = remotion_dir / "src" / "components" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        with REGISTRY_LOCK:
            component_path = generated_dir / f"{request.component_id}.tsx"
            component_path.write_text(request.tsx_code)
            _rebuild_registry(generated_dir, ensure_component=(request.component_id, request.export_name))
            errors: list[str] = []
            self.heartbeat_sync(f"Type-checking {request.component_id}")
            try:
                tsc_result = subprocess.run(
                    ["npx", "tsc", "--noEmit", "--pretty", "false"],
                    cwd=str(remotion_dir), capture_output=True, text=True, timeout=60,
                )
                if tsc_result.returncode != 0:
                    stderr = tsc_result.stdout + tsc_result.stderr
                    component_file = f"{request.component_id}.tsx"
                    for line in stderr.split("\n"):
                        if component_file in line:
                            errors.append(line.strip())
            except subprocess.TimeoutExpired:
                errors.append("TypeScript type-check timed out after 60 seconds")
            except FileNotFoundError:
                errors.append("npx/tsc not found")
            if errors:
                self.logger.warning("Type-check failed for %s: %d errors", request.component_id, len(errors))
                component_path.unlink(missing_ok=True)
                _rebuild_registry(generated_dir)
                return ValidateInfographicResponse(valid=False, errors=errors)
            self.heartbeat_sync(f"Test-rendering {request.component_id}")
            preview_path = str(generated_dir / f"{request.component_id}_preview.png")
            try:
                component_props = request.props or {}
                render_props = {**component_props, "position": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.3}}
                test_plan = {
                    "components": [{"template": request.component_id, "startFrame": 0, "durationInFrames": 90,
                                    "props": render_props, "bounds": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.3}, "zIndex": 1}],
                    "colorPalette": [], "includeBaseVideo": False,
                }
                render_still(composition_id="MotionOverlay", frame=30, props=test_plan, output_path=preview_path)
            except Exception as e:
                err_msg = str(e)[:500]
                errors.append(f"Render test failed: {err_msg}")
                component_path.unlink(missing_ok=True)
                _rebuild_registry(generated_dir)
                return ValidateInfographicResponse(valid=False, errors=errors)
        self.logger.info("Validation passed for %s", request.component_id)
        return ValidateInfographicResponse(valid=True, errors=[], preview_path=preview_path)
