"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { signalWorkflow, fileUrl, type TimelineEffect, type EffectType } from "@/lib/api";
import { FeedbackDialog } from "./FeedbackDialog";
import { Badge, Card, ActionBar } from "./ui";

const TIMELINE_COLORS: Record<string, string> = {
  zoom: "bg-blue-500/30",
  blur: "bg-purple-500/30",
  color_change: "bg-amber-500/30",
  whip: "bg-rose-500/30",
  speed_ramp: "bg-emerald-500/30",
  vignette: "bg-slate-500/30",
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
    <div className="space-y-4">
      {/* 1. Video player */}
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

      {/* 2. Timeline bar */}
      <div className="relative h-12 bg-neutral-900 rounded-lg overflow-hidden">
        {effects.map((e, i) => {
          const left = (e.start_time / safeDuration) * 100;
          const width = ((e.end_time - e.start_time) / safeDuration) * 100;
          const color = TIMELINE_COLORS[e.effect_type] ?? "bg-neutral-500/30";
          return (
            <button
              key={i}
              title={`${e.effect_type} ${formatTime(e.start_time)} - ${formatTime(e.end_time)}`}
              onClick={() => handleEffectClick(i)}
              className={`absolute top-0 h-full cursor-pointer transition-opacity hover:opacity-80 ${color} ${selectedIndex === i ? "ring-1 ring-white/40" : ""}`}
              style={{ left: `${left}%`, width: `${Math.max(width, 0.5)}%` }}
            />
          );
        })}

        {/* Playhead */}
        <div
          className="absolute top-0 h-full w-0.5 bg-white/80 pointer-events-none z-10"
          style={{ left: `${playheadPosition * 100}%` }}
        />

        {/* Time labels */}
        {timeLabels.map(({ pct, label }) => (
          <span
            key={pct}
            className="absolute bottom-0.5 text-[9px] text-neutral-500 pointer-events-none -translate-x-1/2"
            style={{ left: `${Math.min(Math.max(pct * 100, 2), 98)}%` }}
          >
            {label}
          </span>
        ))}
      </div>

      {/* 3. Effect list */}
      <div className="max-h-[300px] overflow-y-auto space-y-1">
        {effects.map((e, i) => {
          const params = effectParamSummary(e);
          const confidencePct = Math.round(e.confidence * 100);
          const isSelected = selectedIndex === i;

          return (
            <div
              key={i}
              onClick={() => handleEffectClick(i)}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors hover:bg-neutral-800/60 ${isSelected ? "border-l-2 border-l-blue-400 bg-neutral-800/40" : "border-l-2 border-l-transparent"}`}
            >
              {/* Left: badge + time range */}
              <div className="flex flex-col gap-0.5 min-w-[110px] shrink-0">
                <Badge label={e.effect_type} type={e.effect_type} />
                <span className="font-mono text-xs text-neutral-500">
                  {formatTime(e.start_time)} - {formatTime(e.end_time)}
                </span>
              </div>

              {/* Center: verbal cue + confidence */}
              <div className="flex-1 min-w-0">
                <p className="text-sm text-neutral-200 truncate">{e.verbal_cue}</p>
                <div className="mt-1 h-1 rounded-full bg-neutral-800 w-full">
                  <div
                    className="h-1 rounded-full bg-blue-500/60"
                    style={{ width: `${confidencePct}%` }}
                  />
                </div>
              </div>

              {/* Right: param summary */}
              {params && (
                <span className="font-mono text-xs text-neutral-500 shrink-0 text-right max-w-[160px] truncate">
                  {params}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Summary */}
      <p className="text-sm text-neutral-500">
        {effects.length} effect{effects.length !== 1 ? "s" : ""} | {timeline.conflicts_resolved} conflict{timeline.conflicts_resolved !== 1 ? "s" : ""} resolved
      </p>

      {/* 4. ActionBar */}
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
