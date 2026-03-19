"use client";

import React from "react";
import type { EffectType, WorkflowStage } from "@/lib/api";

const EFFECT_COLORS: Record<EffectType, { bg: string; border: string; text: string }> = {
  zoom: { bg: "bg-blue-500/15", border: "border-blue-500/40", text: "text-blue-300" },
  blur: { bg: "bg-purple-500/15", border: "border-purple-500/40", text: "text-purple-300" },
  color_change: { bg: "bg-amber-500/15", border: "border-amber-500/40", text: "text-amber-300" },
  whip: { bg: "bg-rose-500/15", border: "border-rose-500/40", text: "text-rose-300" },
  speed_ramp: { bg: "bg-emerald-500/15", border: "border-emerald-500/40", text: "text-emerald-300" },
  vignette: { bg: "bg-slate-500/15", border: "border-slate-500/40", text: "text-slate-300" },
};

const NEUTRAL = { bg: "bg-neutral-500/15", border: "border-neutral-500/40", text: "text-neutral-300" };

export function Badge({ label, type }: { label: string; type?: EffectType | string }) {
  const c = (type && EFFECT_COLORS[type as EffectType]) || NEUTRAL;
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium border ${c.bg} ${c.border} ${c.text}`}>
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
                  <div className="w-6 h-6 rounded-full bg-emerald-500/20 border border-emerald-500/60 flex items-center justify-center">
                    <svg className="w-3 h-3 text-emerald-400" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M2 6l3 3 5-5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                ) : active ? (
                  <div className="w-6 h-6 rounded-full bg-blue-500/20 border border-blue-400/60 flex items-center justify-center">
                    <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
                  </div>
                ) : (
                  <div className="w-6 h-6 rounded-full bg-neutral-800 border border-neutral-700" />
                )}
              </div>
              <span className={`text-[10px] leading-none ${active ? "text-blue-300 font-medium" : completed ? "text-emerald-400/80" : "text-neutral-600"}`}>
                {name}
              </span>
            </div>
            {i < STAGES.length - 1 && (
              <div className={`flex-1 h-px mx-1 mb-4 ${i < currentIdx ? "bg-emerald-500/40" : "bg-neutral-700"}`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

export function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-neutral-900/80 border border-neutral-800 rounded-xl ${className ?? ""}`}>
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
        className="px-5 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {approveLabel}
      </button>
      <button
        onClick={onReject}
        disabled={disabled}
        className="px-5 py-2 rounded-lg border border-red-500/40 text-red-400 hover:bg-red-500/10 font-medium text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {rejectLabel}
      </button>
    </div>
  );
}

export function Stat({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <span className="text-2xl font-bold text-white">{value}</span>
      <span className="text-xs text-neutral-500">{label}</span>
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
            className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
      <span className="text-sm font-medium text-neutral-200">{stage}</span>
      {description && <span className="text-xs text-neutral-500">{description}</span>}
    </div>
  );
}
