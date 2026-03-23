import re
from video_effects.core import BaseCapability
from video_effects.schemas import template_library
from video_effects.helpers.remotion import _get_remotion_dir
from video_effects.skills.infographic._shared import REGISTRY_LOCK, _rebuild_registry
from video_effects.skills.infographic.schemas import MaterializeLibraryTemplatesRequest, MaterializeLibraryTemplatesResponse

_LIBRARY_IMPORT_PREAMBLE = """\
import React, { useMemo, useCallback, useRef, useEffect, useState } from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring, AbsoluteFill, Sequence, Img } from "remotion";
import { useFaceAwareLayout } from "../../lib/spatial";
import { useStyle } from "../../lib/styles";
"""

class MaterializeLibraryTemplatesCapability(BaseCapability[MaterializeLibraryTemplatesRequest, MaterializeLibraryTemplatesResponse]):
    async def execute(self, request):
        if not request.template_ids:
            return MaterializeLibraryTemplatesResponse()
        remotion_dir = _get_remotion_dir()
        generated_dir = remotion_dir / "src" / "components" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        materialized = []
        with REGISTRY_LOCK:
            export_map: dict[str, str] = {}
            for tid in request.template_ids:
                tpl = template_library.get_template(tid)
                if tpl is None:
                    self.logger.warning("Library template '%s' not found, skipping", tid)
                    continue
                component_path = generated_dir / f"{tid}.tsx"
                if not component_path.exists():
                    tsx_code = tpl.tsx_code
                    tsx_code_clean = re.sub(r"^import\s+.*?['\";]\s*$", "", tsx_code, flags=re.MULTILINE)
                    full_code = _LIBRARY_IMPORT_PREAMBLE + "\n" + tsx_code_clean
                    component_path.write_text(full_code)
                    self.logger.info("Materialized library template: %s", tid)
                export_map[tid] = tpl.export_name
                materialized.append(tid)
            _rebuild_registry(generated_dir, ensure_components=export_map)
        return MaterializeLibraryTemplatesResponse(materialized=materialized)
