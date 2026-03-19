"use client";

import { useState } from "react";
import { signalWorkflow, fileUrl } from "@/lib/api";
import type { CompositionPlan, ComponentSpec } from "@remotion-project/types";
import { VideoPlayer } from "./VideoPlayer";
import { FeedbackDialog } from "./FeedbackDialog";

interface ComponentCard {
  index: number;
  template: string;
  timeRange: string;
  propsSummary: string;
}

function buildCards(components: ComponentSpec[], fps: number): ComponentCard[] {
  return components
    .map((c, i) => {
      if (c.template === "subtitles") return null;
      const startS = c.startFrame / fps;
      const endS = (c.startFrame + c.durationInFrames) / fps;
      const props = c.props ?? {};
      let summary = (props.text as string) ?? (props.name as string) ?? "";
      if (summary.length > 30) summary = summary.slice(0, 27) + "...";
      return {
        index: i,
        template: c.template,
        timeRange: `${startS.toFixed(1)}s-${endS.toFixed(1)}s`,
        propsSummary: summary,
      };
    })
    .filter((c): c is ComponentCard => c !== null);
}

interface Props {
  workflowId: string;
  mgPlan: Record<string, unknown>;
  videoInfo: Record<string, number>;
  videoPaths: { base_video: string; face_data: string; zoom_state: string };
}

export function MgApproval({ workflowId, mgPlan, videoInfo, videoPaths }: Props) {
  const [sending, setSending] = useState(false);
  const [feedbackTarget, setFeedbackTarget] = useState<{
    type: "all" | "component";
    index?: number;
    template?: string;
  } | null>(null);

  const fps = videoInfo.fps ?? 30;
  const width = videoInfo.width ?? 1920;
  const height = videoInfo.height ?? 1080;
  const totalFrames =
    videoInfo.total_frames ?? Math.round((videoInfo.duration ?? 10) * fps);

  const components = (mgPlan.components as ComponentSpec[]) ?? [];
  const cards = buildCards(components, fps);

  const plan: CompositionPlan = {
    components,
    colorPalette: (mgPlan.colorPalette as string[]) ?? [],
    includeBaseVideo: true,
    baseVideoPath: fileUrl(videoPaths.base_video),
    faceDataPath: videoPaths.face_data ? fileUrl(videoPaths.face_data) : undefined,
    zoomStatePath: videoPaths.zoom_state ? fileUrl(videoPaths.zoom_state) : undefined,
    styleConfig: mgPlan.styleConfig as CompositionPlan["styleConfig"],
  };

  async function handleApprove() {
    setSending(true);
    await signalWorkflow(workflowId, "approve_mg", [true, ""]);
  }

  async function handleReject(feedback: string) {
    if (feedbackTarget?.type === "component" && feedbackTarget.index !== undefined) {
      feedback = `[component:${feedbackTarget.index}] ${feedback}`;
    }
    setFeedbackTarget(null);
    setSending(true);
    await signalWorkflow(workflowId, "approve_mg", [false, feedback]);
  }

  async function handleRemove(index: number) {
    setSending(true);
    await signalWorkflow(workflowId, "approve_mg", [
      false,
      `[component:${index}] Remove this component entirely`,
    ]);
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Motion Graphics Preview</h2>

      <div className="rounded-lg overflow-hidden bg-black">
        <VideoPlayer
          plan={plan}
          width={width}
          height={height}
          fps={fps}
          durationInFrames={totalFrames}
        />
      </div>

      <div className="space-y-3">
        <h3 className="text-lg font-semibold">
          Components ({cards.length})
        </h3>
        {cards.map((card) => (
          <div
            key={card.index}
            className="flex items-center justify-between bg-neutral-900 border border-neutral-800 rounded-lg px-4 py-3"
          >
            <div className="space-y-0.5">
              <div className="font-mono text-sm">{card.template}</div>
              <div className="text-xs text-neutral-400">
                {card.timeRange}
                {card.propsSummary && ` — ${card.propsSummary}`}
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() =>
                  setFeedbackTarget({
                    type: "component",
                    index: card.index,
                    template: card.template,
                  })
                }
                disabled={sending}
                className="px-3 py-1 text-xs bg-neutral-700 hover:bg-neutral-600 disabled:bg-neutral-800 rounded transition-colors"
              >
                Edit
              </button>
              <button
                onClick={() => handleRemove(card.index)}
                disabled={sending}
                className="px-3 py-1 text-xs bg-red-900/50 hover:bg-red-800/50 disabled:bg-neutral-800 text-red-300 rounded transition-colors"
              >
                Remove
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="flex gap-3">
        <button
          onClick={handleApprove}
          disabled={sending}
          className="px-6 py-2 bg-green-600 hover:bg-green-500 disabled:bg-neutral-700 rounded text-sm font-medium transition-colors"
        >
          Approve All
        </button>
        <button
          onClick={() => setFeedbackTarget({ type: "all" })}
          disabled={sending}
          className="px-6 py-2 bg-red-600 hover:bg-red-500 disabled:bg-neutral-700 rounded text-sm font-medium transition-colors"
        >
          Reject All
        </button>
      </div>

      {feedbackTarget && (
        <FeedbackDialog
          title={
            feedbackTarget.type === "component"
              ? `Edit: ${feedbackTarget.template}`
              : "MG Plan Feedback"
          }
          onSubmit={handleReject}
          onCancel={() => setFeedbackTarget(null)}
        />
      )}
    </div>
  );
}
