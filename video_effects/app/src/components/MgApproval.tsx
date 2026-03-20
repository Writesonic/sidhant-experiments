"use client";

import { useState } from "react";
import { signalWorkflow, fileUrl } from "@/lib/api";
import type { CompositionPlan, ComponentSpec, NormalizedRect } from "@remotion-project/types";
import { VideoPlayer } from "./VideoPlayer";
import { FeedbackDialog } from "./FeedbackDialog";
import { Badge, Card, ActionBar } from "@/components/ui";

const TEMPLATE_BORDERS: Record<string, string> = {
  animated_title: "border-l-fx-zoom",
  lower_third: "border-l-fx-color-change",
  listicle: "border-l-fx-speed-ramp",
  data_animation: "border-l-fx-blur",
  generated: "border-l-fx-vignette",
};

const TEMPLATE_FILLS: Record<string, string> = {
  animated_title: "bg-fx-zoom/60",
  lower_third: "bg-fx-color-change/60",
  listicle: "bg-fx-speed-ramp/60",
  data_animation: "bg-fx-blur/60",
  generated: "bg-fx-vignette/60",
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
  const [confirmRemove, setConfirmRemove] = useState<number | null>(null);

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
    setConfirmRemove(null);
    await signalWorkflow(workflowId, "approve_mg", [
      false,
      `[component:${index}] Remove this component entirely`,
    ]);
  }

  const borderClass = (template: string) =>
    TEMPLATE_BORDERS[template] ?? "border-l-text-dim";

  const fillClass = (template: string) =>
    TEMPLATE_FILLS[template] ?? "bg-text-muted/60";

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-display font-bold">Motion Graphics Preview</h2>

      <Card className="p-0 overflow-hidden">
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
          <span className="text-xs text-text-muted">Palette</span>
          <div className="flex gap-1">
            {plan.colorPalette.map((color, i) => (
              <div
                key={i}
                className="w-5 h-5 border border-border-card"
                style={{ backgroundColor: color }}
              />
            ))}
          </div>
        </div>
      )}

      <div className="space-y-3">
        <h3 className="text-lg font-display font-bold">
          Components ({cards.length})
        </h3>
        {cards.length === 0 ? (
          <Card className="p-8 text-center">
            <p className="text-text-dim text-sm">No non-subtitle components in this plan</p>
          </Card>
        ) : (
          cards.map((card) => (
            <Card
              key={card.index}
              className={`relative border-l-2 ${borderClass(card.template)} px-4 py-3`}
            >
              <div className="flex items-center justify-between">
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2">
                    <Badge label={card.template} type={card.template} />
                    <span className="text-sm font-mono text-text-secondary">
                      {card.timeRange}
                    </span>
                  </div>
                  {card.propsSummary && (
                    <div className="text-xs text-text-dim">{card.propsSummary}</div>
                  )}
                </div>

                <div className="flex items-center gap-3">
                  {card.bounds && (
                    <div
                      className="relative bg-surface"
                      style={{ width: 80, height: 45 }}
                    >
                      <div
                        className={`absolute ${fillClass(card.template)}`}
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
                      className="px-4 py-2.5 text-xs border border-border-card text-text-dim hover:bg-surface-warm transition-colors disabled:opacity-40"
                    >
                      Suggest Edit
                    </button>
                    {confirmRemove === card.index ? (
                      <div className="flex gap-1">
                        <button
                          onClick={() => handleRemove(card.index)}
                          disabled={sending}
                          className="px-3 py-2.5 text-xs bg-negative/20 border border-negative-border text-negative font-medium transition-colors disabled:opacity-40"
                        >
                          Confirm
                        </button>
                        <button
                          onClick={() => setConfirmRemove(null)}
                          className="px-3 py-2.5 text-xs border border-border-card text-text-dim transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setConfirmRemove(card.index)}
                        disabled={sending}
                        className="px-4 py-2.5 text-xs border border-negative-border text-negative hover:bg-negative-fill transition-colors disabled:opacity-40"
                      >
                        Remove
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </Card>
          ))
        )}
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
              ? `Suggest Edit: ${feedbackTarget.template}`
              : "MG Plan Feedback"
          }
          onSubmit={handleReject}
          onCancel={() => setFeedbackTarget(null)}
        />
      )}
    </div>
  );
}
