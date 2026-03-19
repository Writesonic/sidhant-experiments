const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export { API_BASE };

export async function startWorkflow(params: {
  video_path: string;
  enable_programmer?: boolean;
  enable_mg?: boolean;
  style?: string;
  dev_mode?: boolean;
  enable_subtitles?: boolean;
  pinned_templates?: string[];
}): Promise<{ workflow_id: string }> {
  const res = await fetch(`${API_BASE}/api/workflows`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export type EffectType = "zoom" | "blur" | "color_change" | "whip" | "speed_ramp" | "vignette";

export type WorkflowStage =
  | "init" | "analyzing" | "timeline_approval" | "processing"
  | "mg_preview" | "mg_approval" | "rendering" | "done" | "error";

export interface TimelineEffect {
  effect_type: EffectType;
  start_time: number;
  end_time: number;
  confidence: number;
  verbal_cue: string;
  zoom_params?: { tracking: string; zoom_level: number; easing: string; action: string };
  whip_params?: { direction: string; intensity: number };
  speed_ramp_params?: { speed: number; easing: string };
  color_params?: { preset: string; intensity: number };
  blur_params?: { blur_type: string; radius: number };
  vignette_params?: { strength: number; radius: number };
}

export interface WorkflowStatus {
  stage: WorkflowStage;
  timeline?: { effects: TimelineEffect[]; conflicts_resolved: number; total_duration?: number };
  mg_plan?: Record<string, unknown>;
  video_info?: { fps: number; width: number; height: number; duration: number; total_frames: number };
  video_paths?: { base_video: string; face_data: string; zoom_state: string };
  result?: { output_video: string; effects_applied: number; motion_graphics_applied: number; transcript_length?: number; phases?: number };
  error?: string;
}

export async function getWorkflowStatus(id: string): Promise<WorkflowStatus> {
  const res = await fetch(`${API_BASE}/api/workflows/${id}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function signalWorkflow(
  id: string,
  signal: "approve_timeline" | "approve_mg",
  args: [boolean, string]
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/workflows/${id}/signal`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ signal, args }),
  });
  if (!res.ok) throw new Error(await res.text());
}

export function fileUrl(path: string): string {
  return `${API_BASE}/api/files?path=${encodeURIComponent(path)}`;
}

// ── Template Library ──

export interface LibraryTemplateMeta {
  id: string;
  display_name: string;
  description: string;
  tags: string[];
  created_at: string;
  export_name: string;
  duration_range: [number, number];
  tsx_code: string;
}

export interface LibraryTemplate extends LibraryTemplateMeta {
  tsx_code: string;
  props: { name: string; type: string; required: boolean; default?: unknown; description?: string }[];
  spatial: { typical_y_range: [number, number]; typical_x_range: [number, number]; edge_aligned: boolean };
  preview_image: string | null;
}

export interface CreateTemplateData {
  id: string;
  display_name: string;
  description: string;
  tsx_code: string;
  export_name: string;
  tags?: string[];
}

export interface GenerateCodeRequest {
  prompt: string;
  previous_code?: string;
  conversation?: { role: string; content: string }[];
  errors?: string[];
}

export async function listTemplates(): Promise<LibraryTemplateMeta[]> {
  const res = await fetch(`${API_BASE}/api/templates`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getTemplate(id: string): Promise<LibraryTemplate> {
  const res = await fetch(`${API_BASE}/api/templates/${id}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function saveTemplate(data: CreateTemplateData): Promise<LibraryTemplate> {
  const res = await fetch(`${API_BASE}/api/templates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateTemplate(id: string, data: Partial<CreateTemplateData>): Promise<LibraryTemplate> {
  const res = await fetch(`${API_BASE}/api/templates/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteTemplate(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/templates/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
}

export async function generateTemplateCode(req: GenerateCodeRequest): Promise<{ code: string; summary: string }> {
  const res = await fetch(`${API_BASE}/api/templates/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
