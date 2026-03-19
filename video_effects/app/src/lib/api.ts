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

export interface TimelineEffect {
  effect_type: string;
  start_time: number;
  end_time: number;
  confidence: number;
  verbal_cue: string;
}

export interface WorkflowStatus {
  stage: string;
  timeline?: { effects: TimelineEffect[]; conflicts_resolved: number };
  mg_plan?: Record<string, unknown>;
  video_info?: Record<string, number>;
  video_paths?: { base_video: string; face_data: string; zoom_state: string };
  result?: Record<string, unknown>;
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
