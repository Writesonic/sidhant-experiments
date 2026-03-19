"use client";

import { useState } from "react";
import { signalWorkflow, fileUrl } from "@/lib/api";
import type { CompositionPlan, ComponentSpec, NormalizedRect } from "@remotion-project/types";
import { VideoPlayer } from "./VideoPlayer";
import { FeedbackDialog } from "./FeedbackDialog";
import { Badge, Card, ActionBar } from "@/components/ui";

const TEMPLATE_BORDERS: Record<string, string> = {
  animated_title: "border-l-blue-500",
  lower_third: "border-l-amber-500",
  listicle: "border-l-emerald-500",
  data_animation: "border-l-purple-500",
  generated: "border-l-cyan-500",
};

const TEMPLATE_FILLS: Record<string, string> = {
  animated_title: "bg-blue-500/60",
  lower_third: "bg-amber-500/60",
  listicle: "bg-emerald-500/60",
  data_animation: "bg-purple-500/60",
  generated: "bg-cyan-500/60",
};

interface ComponentCard {
  index: number;
  template: string;
  timeRange: string;
  propsSummary: string;
  bounds: NormalizedRect;
}

function buildCards(components: ComponentSpec[], fps: number): ComponentCard[] {
  return components
    .map((c, i) => {
      if (c.template === "subtitles") return null;
      const startS = c.startFrame / fps;
      const endS = (c.startFrame + c.durationInFrames) / fps;
      const props = c.props ?? {};

      let summary = "";
      if (props.text) {
        const txt = String(props.text);
        summary = txt.length > 30 ? `"${txt.slice(0, 27)}..."` : `"${txt}"`;
      } else if (props.name) {
        summary = `name: ${String(props.name)}`;
      }

      return {
        index: i,
        template: c.template,
        timeRange: `${startS.toFixed(1)}s – ${endS.toFixed(1)}s`,
        propsSummary: summary,
        bounds: c.bounds,
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

  const borderClass = (template: string) =>
    TEMPLATE_BORDERS[template] ?? "border-l-neutral-600";

  const fillClass = (template: string) =>
    TEMPLATE_FILLS[template] ?? "bg-neutral-500/60";

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Motion Graphics Preview</h2>

      <Card className="p-0 overflow-hidden rounded-xl">
        <VideoPlayer
          plan={plan}
          width={width}
          height={height}
          fps={fps}
          durationInFrames={totalFrames}
        />
      </Card>

      {plan.colorPalette.length > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-neutral-500">Palette</span>
          <div className="flex gap-1">
            {plan.colorPalette.map((color, i) => (
              <div
                key={i}
                className="w-5 h-5 rounded-full border border-neutral-700"
                style={{ backgroundColor: color }}
              />
            ))}
          </div>
        </div>
      )}

      <div className="space-y-3">
        <h3 className="text-lg font-semibold">
          Components ({cards.length})
        </h3>
        {cards.map((card) => (
          <Card
            key={card.index}
            className={`relative border-l-2 ${borderClass(card.template)} px-4 py-3`}
          >
            <div className="flex items-center justify-between">
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <Badge label={card.template} type={card.template} />
                  <span className="text-sm font-mono text-neutral-300">
                    {card.timeRange}
                  </span>
                </div>
                {card.propsSummary && (
                  <div className="text-xs text-neutral-400">{card.propsSummary}</div>
                )}
              </div>

              <div className="flex items-center gap-3">
                {card.bounds && (
                  <div
                    className="relative bg-neutral-800 rounded"
                    style={{ width: 60, height: 34 }}
                  >
                    <div
                      className={`absolute rounded-sm ${fillClass(card.template)}`}
                      style={{
                        left: `${card.bounds.x * 100}%`,
                        top: `${card.bounds.y * 100}%`,
                        width: `${card.bounds.w * 100}%`,
                        height: `${card.bounds.h * 100}%`,
                      }}
                    />
                  </div>
                )}
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
                    className="px-3 py-1 text-xs border border-neutral-700 text-neutral-400 hover:bg-neutral-800 rounded transition-colors disabled:opacity-40"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleRemove(card.index)}
                    disabled={sending}
                    className="px-3 py-1 text-xs border border-red-500/30 text-red-400 hover:bg-red-500/10 rounded transition-colors disabled:opacity-40"
                  >
                    Remove
                  </button>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <ActionBar
        onApprove={handleApprove}
        onReject={() => setFeedbackTarget({ type: "all" })}
        approveLabel="Approve All"
        rejectLabel="Request Changes"
        disabled={sending}
      />

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
