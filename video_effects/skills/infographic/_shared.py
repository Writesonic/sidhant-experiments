import threading
from pathlib import Path

from video_effects.helpers.prompts import build_style_guide, build_spatial_user_message, derive_export_name

REGISTRY_LOCK = threading.Lock()


def _rebuild_registry(
    generated_dir: Path,
    ensure_component: tuple[str, str] | None = None,
    ensure_components: dict[str, str] | None = None,
) -> None:
    registry_path = generated_dir / "_registry.ts"
    components: dict[str, str] = {}
    for tsx_file in generated_dir.glob("*.tsx"):
        if tsx_file.name.startswith("_"):
            continue
        cid = tsx_file.stem
        components[cid] = derive_export_name(cid)
    if ensure_component:
        cid, ename = ensure_component
        components[cid] = ename
    if ensure_components:
        components.update(ensure_components)
    if not components:
        registry_path.write_text(
            'import React from "react";\n\n'
            "type ComponentMap = { [key: string]: React.FC<any> };\n\n"
            "export const GeneratedRegistry: ComponentMap = {};\n"
        )
        return
    imports = []
    entries = []
    for cid in sorted(components):
        ename = components[cid]
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
    registry_path.write_text(registry_code)
