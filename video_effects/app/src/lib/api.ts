const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export { API_BASE };

export async function startWorkflow(params: {
  video_path: string;
  enable_programmer?: boolean;
  enable_mg?: boolean;
  style?: string;
  dev_mode?: boolean;
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
