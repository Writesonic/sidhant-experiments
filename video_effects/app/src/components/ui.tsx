"use client";

import React from "react";
import type { EffectType, WorkflowStage } from "@/lib/api";

const EFFECT_COLORS: Record<EffectType, { bg: string; border: string; text: string }> = {
  zoom: { bg: "bg-fx-zoom/15", border: "border-fx-zoom/40", text: "text-fx-zoom" },
  blur: { bg: "bg-fx-blur/15", border: "border-fx-blur/40", text: "text-fx-blur" },
  color_change: { bg: "bg-fx-color-change/15", border: "border-fx-color-change/40", text: "text-fx-color-change" },
  whip: { bg: "bg-fx-whip/15", border: "border-fx-whip/40", text: "text-fx-whip" },
  speed_ramp: { bg: "bg-fx-speed-ramp/15", border: "border-fx-speed-ramp/40", text: "text-fx-speed-ramp" },
  vignette: { bg: "bg-fx-vignette/15", border: "border-fx-vignette/40", text: "text-fx-vignette" },
};

const NEUTRAL = { bg: "bg-text-muted/15", border: "border-text-muted/40", text: "text-text-muted" };

export function Badge({ label, type }: { label: string; type?: EffectType | string }) {
  const c = (type && EFFECT_COLORS[type as EffectType]) || NEUTRAL;
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 text-[11px] tracking-[0.05em] font-medium border ${c.bg} ${c.border} ${c.text}`}>
      {label}
    </span>
  );
}

const STAGES = ["Analyze", "Timeline", "Process", "Graphics", "Render", "Done"] as const;

const STAGE_MAP: Record<WorkflowStage, number> = {
  init: 0,
  analyzing: 0,
  timeline_approval: 1,
  processing: 2,
  mg_preview: 3,
  mg_approval: 3,
  rendering: 4,
  done: 5,
  error: 0,
};

export function StageIndicator({ currentStage }: { currentStage: WorkflowStage }) {
  const currentIdx = STAGE_MAP[currentStage] ?? 0;

  return (
    <div className="flex items-center gap-0 w-full">
      {STAGES.map((name, i) => {
        const completed = i < currentIdx;
        const active = i === currentIdx;
        return (
          <React.Fragment key={name}>
            <div className="flex flex-col items-center gap-1">
              <div className="relative flex items-center justify-center w-6 h-6">
                {completed ? (
                  <div className="w-6 h-6 bg-accent/20 border border-accent/60 flex items-center justify-center">
                    <svg className="w-3 h-3 text-accent" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M2 6l3 3 5-5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                ) : active ? (
                  <div className="w-6 h-6 bg-accent/20 border border-accent/60 flex items-center justify-center">
                    <div className="w-2 h-2 bg-accent animate-pulse" />
                  </div>
                ) : (
                  <div className="w-6 h-6 bg-surface border border-border-card" />
                )}
              </div>
              <span className={`text-[10px] uppercase tracking-[0.15em] font-mono leading-none ${active ? "text-accent font-medium" : completed ? "text-accent/80" : "text-text-dim"}`}>
                {name}
              </span>
            </div>
            {i < STAGES.length - 1 && (
              <div className={`flex-1 h-px mx-1 mb-4 ${i < currentIdx ? "bg-accent/40" : "bg-border-card"}`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

export function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-surface border border-border-card transition-all duration-200 ${className ?? ""}`}>
      {children}
    </div>
  );
}

export function ActionBar({
  onApprove,
  onReject,
  approveLabel = "Approve",
  rejectLabel = "Reject",
  disabled = false,
}: {
  onApprove: () => void;
  onReject: () => void;
  approveLabel?: string;
  rejectLabel?: string;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <button
        onClick={onApprove}
        disabled={disabled}
        className="px-5 py-2 bg-accent hover:bg-accent/90 text-bg font-medium text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {approveLabel}
      </button>
      <button
        onClick={onReject}
        disabled={disabled}
        className="px-5 py-2 border border-negative-border text-negative hover:bg-negative-fill font-medium text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {rejectLabel}
      </button>
    </div>
  );
}

export function Stat({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <span className="text-2xl font-bold text-text">{value}</span>
      <span className="text-xs text-text-muted">{label}</span>
    </div>
  );
}

export function ActivityIndicator({ stage, description }: { stage: string; description?: string }) {
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="flex items-center gap-1.5">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-2 h-2 bg-accent animate-pulse"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
      <span className="text-sm font-medium text-text">{stage}</span>
      {description && <span className="text-xs text-text-muted">{description}</span>}
    </div>
  );
}
