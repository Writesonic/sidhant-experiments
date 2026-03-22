"use client";

import type { WorkflowStep } from "@/lib/api";

const APPROVAL_KEYS = new Set(["timeline_approval", "mg_approval"]);

function StepIcon({ status }: { status: WorkflowStep["status"] }) {
  if (status === "done") {
    return (
      <div className="w-5 h-5 bg-accent/20 border border-accent/60 flex items-center justify-center shrink-0">
        <svg className="w-2.5 h-2.5 text-accent" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M2 6l3 3 5-5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    );
  }
  if (status === "active") {
    return (
      <div className="w-5 h-5 bg-accent/20 border border-accent/60 flex items-center justify-center shrink-0">
        <div className="w-1.5 h-1.5 bg-accent animate-pulse" />
      </div>
    );
  }
  return (
    <div className="w-5 h-5 bg-surface border border-border-card shrink-0" />
  );
}

function ApprovalBanner({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 mb-3 bg-accent-fill border border-accent-dim text-accent text-xs font-medium">
      <span className="text-sm">&#9889;</span>
      <span>{label} coming up &mdash; don&apos;t go away</span>
    </div>
  );
}

export function StepLog({ steps }: { steps: WorkflowStep[] }) {
  const visible = steps.filter((s) => s.status !== "skipped");
  const hasActive = visible.some((s) => s.status === "active");
  const nextApproval = visible.find(
    (s) => s.status === "pending" && APPROVAL_KEYS.has(s.key),
  );

  return (
    <div>
      {hasActive && nextApproval && (
        <ApprovalBanner label={nextApproval.label} />
      )}
      <div className="flex flex-col">
        {visible.map((step, i) => {
          const isLast = i === visible.length - 1;
          const isApproval = APPROVAL_KEYS.has(step.key);
          return (
            <div key={step.key} className="flex gap-3">
              <div className="flex flex-col items-center">
                <StepIcon status={step.status} />
                {!isLast && (
                  <div className="w-px flex-1 min-h-3 bg-border-card" />
                )}
              </div>
              <div className={`pb-3 text-xs leading-5 ${
                step.status === "active"
                  ? "text-text font-medium"
                  : step.status === "done"
                    ? "text-text-muted"
                    : "text-text-dim"
              }`}>
                {step.label}
                {isApproval && step.status === "pending" && (
                  <span className="ml-1.5 text-accent">&#9889;</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
