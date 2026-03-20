"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { signalWorkflow, fileUrl, type TimelineEffect, type EffectType } from "@/lib/api";
import { FeedbackDialog } from "./FeedbackDialog";
import { Badge, Card, ActionBar } from "./ui";

const TIMELINE_COLORS: Record<string, string> = {
  zoom: "bg-fx-zoom/30",
  blur: "bg-fx-blur/30",
  color_change: "bg-fx-color-change/30",
  whip: "bg-fx-whip/30",
  speed_ramp: "bg-fx-speed-ramp/30",
  vignette: "bg-fx-vignette/30",
};

function formatTime(s: number) {
  return `${s.toFixed(1)}s`;
}

function effectParamSummary(e: TimelineEffect): string | null {
  switch (e.effect_type) {
    case "zoom":
      if (!e.zoom_params) return null;
      return `${e.zoom_params.zoom_level}x ${e.zoom_params.tracking} ${e.zoom_params.easing}`;
    case "whip":
      if (!e.whip_params) return null;
      return `${e.whip_params.direction} ${e.whip_params.intensity}x`;
    case "speed_ramp":
      if (!e.speed_ramp_params) return null;
      return `${e.speed_ramp_params.speed}x ${e.speed_ramp_params.easing}`;
    case "color_change":
      if (!e.color_params) return null;
      return `${e.color_params.preset} ${e.color_params.intensity}%`;
    case "blur":
      if (!e.blur_params) return null;
      return `${e.blur_params.blur_type} r${e.blur_params.radius}`;
    case "vignette":
      if (!e.vignette_params) return null;
      return `s${e.vignette_params.strength} r${e.vignette_params.radius}`;
    default:
      return null;
  }
}

interface Props {
  workflowId: string;
  timeline: { effects: TimelineEffect[]; conflicts_resolved: number; total_duration?: number };
  baseVideoPath?: string;
}

export function TimelineApproval({ workflowId, timeline, baseVideoPath }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [sending, setSending] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [playheadPosition, setPlayheadPosition] = useState(0);

  const effects = timeline.effects ?? [];

  const totalDuration =
    timeline.total_duration ??
    (effects.length > 0 ? Math.max(...effects.map((e) => e.end_time)) : 1);

  const safeDuration = totalDuration > 0 ? totalDuration : 1;

  const seekTo = useCallback((seconds: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = seconds;
      videoRef.current.play();
    }
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    function onTimeUpdate() {
      const v = videoRef.current;
      if (!v || !v.duration) return;
      setPlayheadPosition(v.currentTime / v.duration);
    }

    video.addEventListener("timeupdate", onTimeUpdate);
    return () => video.removeEventListener("timeupdate", onTimeUpdate);
  }, [baseVideoPath]);

  async function handleApprove() {
    setSending(true);
    await signalWorkflow(workflowId, "approve_timeline", [true, ""]);
  }

  async function handleReject(feedback: string) {
    setShowFeedback(false);
    setSending(true);
    await signalWorkflow(workflowId, "approve_timeline", [false, feedback]);
  }

  function handleEffectClick(index: number) {
    setSelectedIndex(index);
    seekTo(effects[index].start_time);
  }

  const timeLabels = [0, 0.25, 0.5, 0.75, 1].map((pct) => ({
    pct,
    label: formatTime(pct * safeDuration),
  }));

  return (
    <div className="space-y-6">
      {baseVideoPath && (
        <Card className="p-0 overflow-hidden">
          <video
            ref={videoRef}
            src={fileUrl(baseVideoPath)}
            controls
            className="w-full bg-black max-h-[400px]"
          />
        </Card>
      )}

      <div className="relative h-16 bg-surface overflow-hidden">
        {effects.map((e, i) => {
          const left = (e.start_time / safeDuration) * 100;
          const width = ((e.end_time - e.start_time) / safeDuration) * 100;
          const color = TIMELINE_COLORS[e.effect_type] ?? "bg-text-muted/30";
          return (
            <button
              key={i}
              title={`${e.effect_type} ${formatTime(e.start_time)} - ${formatTime(e.end_time)}`}
              onClick={() => handleEffectClick(i)}
              className={`absolute top-0 h-full cursor-pointer transition-opacity hover:opacity-80 ${color} ${selectedIndex === i ? "ring-1 ring-accent/60" : ""}`}
              style={{ left: `${left}%`, width: `${Math.max(width, 0.5)}%` }}
            />
          );
        })}

        <div
          className="absolute top-0 h-full w-0.5 bg-white/80 pointer-events-none z-10"
          style={{ left: `${playheadPosition * 100}%` }}
        />

        {timeLabels.map(({ pct, label }) => (
          <span
            key={pct}
            className="absolute bottom-0.5 text-[9px] text-text-ghost pointer-events-none -translate-x-1/2"
            style={{ left: `${Math.min(Math.max(pct * 100, 2), 98)}%` }}
          >
            {label}
          </span>
        ))}
      </div>

      <div className="space-y-1">
        {effects.map((e, i) => {
          const params = effectParamSummary(e);
          const confidencePct = Math.round(e.confidence * 100);
          const isSelected = selectedIndex === i;

          return (
            <div
              key={i}
              onClick={() => handleEffectClick(i)}
              className={`flex items-center gap-3 px-3 py-2 cursor-pointer transition-colors hover:bg-surface-warm ${isSelected ? "border-l-2 border-l-accent bg-accent-bg" : "border-l-2 border-l-transparent"}`}
            >
              <div className="flex flex-col gap-0.5 min-w-[110px] shrink-0">
                <Badge label={e.effect_type} type={e.effect_type} />
                <span className="font-mono text-xs text-text-muted">
                  {formatTime(e.start_time)} - {formatTime(e.end_time)}
                </span>
              </div>

              <div className="flex-1 min-w-0">
                <p className="text-sm text-text truncate">{e.verbal_cue}</p>
                <div className="mt-1 h-1.5 bg-border-card w-full relative group">
                  <div
                    className="h-1.5 bg-accent/60"
                    style={{ width: `${confidencePct}%` }}
                  />
                  <span className="absolute -top-5 right-0 text-[10px] text-text-muted opacity-0 group-hover:opacity-100 transition-opacity">
                    {confidencePct}%
                  </span>
                </div>
              </div>

              {params && (
                <span className="font-mono text-xs text-text-muted shrink-0 text-right max-w-[160px] truncate">
                  {params}
                </span>
              )}
            </div>
          );
        })}
      </div>

      <div className="border-t border-border pt-4 flex items-center gap-3">
        <span className="text-sm text-text-muted">
          {effects.length} effect{effects.length !== 1 ? "s" : ""}
        </span>
        <span className="text-text-ghost">|</span>
        <span className="text-sm text-text-muted">
          {timeline.conflicts_resolved} conflict{timeline.conflicts_resolved !== 1 ? "s" : ""} resolved
        </span>
      </div>

      <ActionBar
        onApprove={handleApprove}
        onReject={() => setShowFeedback(true)}
        disabled={sending}
      />

      {showFeedback && (
        <FeedbackDialog
          title="Timeline Feedback"
          onSubmit={handleReject}
          onCancel={() => setShowFeedback(false)}
        />
      )}
    </div>
  );
}
