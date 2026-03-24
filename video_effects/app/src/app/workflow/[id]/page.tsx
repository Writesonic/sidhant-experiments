"use client";

import { use } from "react";
import Link from "next/link";
import { useWorkflow } from "@/hooks/useWorkflow";
import { TimelineApproval } from "@/components/TimelineApproval";
import { MgApproval } from "@/components/MgApproval";
import { fileUrl } from "@/lib/api";
import { StageIndicator, ActivityIndicator, Stat, Card } from "@/components/ui";
import { StepLog } from "@/components/StepLog";

export default function WorkflowPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data, error } = useWorkflow(id);

  if (error && !data) {
    return (
      <div className="space-y-6">
        <Card className="border-negative-border p-6">
          <p className="text-negative text-sm">Failed to connect: {error}</p>
          <Link
            href="/"
            className="inline-block mt-6 px-4 py-2.5 border border-border-card text-text-dim hover:text-text hover:border-accent/40 text-sm font-medium transition-colors"
          >
            Try Again
          </Link>
        </Card>
      </div>
    );
  }

  if (!data) {
    return (
      <Card className="p-8">
        <ActivityIndicator stage="Connecting" description="Reaching workflow server..." />
      </Card>
    );
  }

  const { stage } = data;

  const stepsCard = data.steps?.length ? (
    <Card className="p-4 animate-slide-up">
      <StepLog steps={data.steps} />
    </Card>
  ) : null;

  if (stage === "init" || stage === "analyzing") {
    return (
      <div className="space-y-6">
        <StageIndicator currentStage={stage} />
        {stepsCard || (
          <Card className="p-8 animate-slide-up">
            <ActivityIndicator stage="Analyzing" description="Extracting video metadata, transcribing audio..." />
          </Card>
        )}
      </div>
    );
  }

  if (stage === "timeline_approval" && data.timeline) {
    return (
      <div className="space-y-6">
        <StageIndicator currentStage={stage} />
        <div className="animate-slide-up">
          <TimelineApproval
            workflowId={id}
            timeline={data.timeline}
            baseVideoPath={data.video_paths?.base_video}
          />
        </div>
        {stepsCard}
      </div>
    );
  }

  if (stage === "processing" || stage === "mg_preview") {
    return (
      <div className="space-y-6">
        <StageIndicator currentStage={stage} />
        <div className="animate-slide-up space-y-6">
          {data.video_paths?.base_video && (
            <Card className="p-0 overflow-hidden relative">
              <video
                src={fileUrl(data.video_paths.base_video)}
                controls
                muted
                className="w-full max-h-[400px] bg-black"
              />
            </Card>
          )}
          {stepsCard || (
            <Card className="p-8">
              <ActivityIndicator stage="Processing" description="Applying effects and generating motion graphics..." />
            </Card>
          )}
        </div>
      </div>
    );
  }

  if (stage === "mg_approval" && data.mg_plan && data.video_info && data.video_paths) {
    return (
      <div className="space-y-6">
        <StageIndicator currentStage={stage} />
        <div className="animate-slide-up">
          <MgApproval
            workflowId={id}
            mgPlan={data.mg_plan}
            videoInfo={data.video_info}
            videoPaths={data.video_paths}
          />
        </div>
        {stepsCard}
      </div>
    );
  }

  if (stage === "rendering") {
    return (
      <div className="space-y-6">
        <StageIndicator currentStage={stage} />
        <div className="animate-slide-up space-y-6">
          {data.video_paths?.base_video && (
            <Card className="p-0 overflow-hidden">
              <video
                src={fileUrl(data.video_paths.base_video)}
                controls
                muted
                className="w-full max-h-[400px] bg-black"
              />
            </Card>
          )}
          {stepsCard || (
            <Card className="p-8">
              <ActivityIndicator stage="Rendering" description="Encoding final video with all overlays..." />
            </Card>
          )}
        </div>
      </div>
    );
  }

  if (stage === "done" && data.result) {
    const r = data.result;
    return (
      <div className="space-y-6">
        <StageIndicator currentStage={stage} />
        <div className="animate-slide-up space-y-6">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <Stat value={String(r.effects_applied)} label="Effects applied" />
            <Stat value={String(r.motion_graphics_applied)} label="Motion graphics" />
            {r.transcript_length != null && (
              <Stat value={String(r.transcript_length)} label="Transcript length" />
            )}
            {r.phases != null && (
              <Stat value={String(r.phases)} label="Phases" />
            )}
          </div>
          <Card className="p-5 space-y-4">
            <video
              src={fileUrl(String(r.output_video))}
              controls
              className="w-full max-h-[500px] bg-black"
            />
            <a
              href={fileUrl(String(r.output_video))}
              download
              className="inline-block px-4 py-2.5 bg-accent hover:bg-accent/90 text-bg text-sm font-semibold transition-colors"
            >
              Download
            </a>
          </Card>
          <Link
            href="/"
            className="inline-block mt-2 px-4 py-2.5 border border-border-card text-text-dim hover:text-text hover:border-accent/40 text-sm font-medium transition-colors"
          >
            Start New
          </Link>
        </div>
      </div>
    );
  }

  if (stage === "error") {
    return (
      <div className="space-y-6">
        <StageIndicator currentStage={stage} />
        <div className="animate-slide-up">
          <Card className="border-negative-border p-6">
            <pre className="overflow-x-auto max-h-60 overflow-y-auto"><code className="text-sm text-negative">{data.error ?? "Unknown error"}</code></pre>
            <Link
              href="/"
              className="inline-block mt-6 px-4 py-2.5 border border-border-card text-text-dim hover:text-text hover:border-accent/40 text-sm font-medium transition-colors"
            >
              Try Again
            </Link>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <StageIndicator currentStage={stage} />
      <Card className="p-8 animate-slide-up">
        <ActivityIndicator
          stage={stage.charAt(0).toUpperCase() + stage.slice(1)}
          description="Processing..."
        />
      </Card>
    </div>
  );
}
