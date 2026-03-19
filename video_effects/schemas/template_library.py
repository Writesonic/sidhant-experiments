"""Template library for user-created reusable motion graphics components."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from video_effects.config import settings
from video_effects.schemas.mg_templates import MGTemplateSpec, PropSpec, SpatialHint


class LibraryTemplate(BaseModel):
    id: str
    display_name: str
    description: str
    tsx_code: str
    export_name: str
    props: list[PropSpec] = []
    spatial: SpatialHint = Field(default_factory=SpatialHint)
    duration_range: tuple[float, float] = (1.0, 10.0)
    tags: list[str] = []
    created_at: str = ""
    preview_image: str | None = None


class TemplateLibrary(BaseModel):
    templates: dict[str, LibraryTemplate] = {}
    version: int = 1


_lock = threading.Lock()


def _library_path() -> Path:
    return Path(settings.TEMPLATE_LIBRARY_PATH)


def load() -> TemplateLibrary:
    path = _library_path()
    if not path.exists():
        return TemplateLibrary()
    return TemplateLibrary.model_validate_json(path.read_text())


def save(lib: TemplateLibrary) -> None:
    path = _library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lib.model_dump_json(indent=2))


def list_templates() -> list[LibraryTemplate]:
    with _lock:
        return list(load().templates.values())


def get_template(template_id: str) -> LibraryTemplate | None:
    with _lock:
        return load().templates.get(template_id)


def save_template(tpl: LibraryTemplate) -> LibraryTemplate:
    with _lock:
        if not tpl.created_at:
            tpl.created_at = datetime.now(timezone.utc).isoformat()
        lib = load()
        lib.templates[tpl.id] = tpl
        save(lib)
        return tpl


def delete_template(template_id: str) -> bool:
    with _lock:
        lib = load()
        if template_id not in lib.templates:
            return False
        del lib.templates[template_id]
        save(lib)
        return True


def as_mg_template_spec(tpl: LibraryTemplate) -> MGTemplateSpec:
    return MGTemplateSpec(
        name=tpl.id,
        display_name=tpl.display_name,
        description=tpl.description,
        props=tpl.props,
        duration_range=tpl.duration_range,
        spatial=tpl.spatial,
    )


def list_as_mg_specs() -> list[MGTemplateSpec]:
    return [as_mg_template_spec(t) for t in list_templates()]


def get_as_mg_spec(template_id: str) -> MGTemplateSpec | None:
    tpl = get_template(template_id)
    return as_mg_template_spec(tpl) if tpl else None
