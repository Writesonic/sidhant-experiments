"use client";

import { use } from "react";
import Link from "next/link";
import { useWorkflow } from "@/hooks/useWorkflow";
import { TimelineApproval } from "@/components/TimelineApproval";
import { MgApproval } from "@/components/MgApproval";
import { fileUrl } from "@/lib/api";
import { StageIndicator, ActivityIndicator, Stat, Card } from "@/components/ui";

export default function WorkflowPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data, error } = useWorkflow(id);

  if (error && !data) {
    return (
      <Card className="border-negative-border p-6">
        <p className="text-negative text-sm">Failed to connect: {error}</p>
        <Link href="/" className="text-sm text-text-dim hover:text-text mt-4 inline-block">
          Try Again
        </Link>
      </Card>
    );
  }

  if (!data) {
    return <ActivityIndicator stage="Connecting" description="Reaching workflow server..." />;
  }

  const { stage } = data;

  if (stage === "init" || stage === "analyzing") {
    return (
      <>
        <StageIndicator currentStage={stage} />
        <div className="animate-slide-up">
          <ActivityIndicator stage="Analyzing" description="Extracting video metadata, transcribing audio..." />
        </div>
      </>
    );
  }

  if (stage === "timeline_approval" && data.timeline) {
    return (
      <>
        <StageIndicator currentStage={stage} />
        <div className="animate-slide-up">
          <TimelineApproval
            workflowId={id}
            timeline={data.timeline}
            baseVideoPath={data.video_paths?.base_video}
          />
        </div>
      </>
    );
  }

  if (stage === "processing" || stage === "mg_preview") {
    const desc = stage === "processing"
      ? "Applying effects and generating motion graphics..."
      : "Building motion graphics composition...";
    const label = stage === "processing" ? "Processing" : "Preparing preview";
    return (
      <>
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
          <ActivityIndicator stage={label} description={desc} />
        </div>
      </>
    );
  }

  if (stage === "mg_approval" && data.mg_plan && data.video_info && data.video_paths) {
    return (
      <>
        <StageIndicator currentStage={stage} />
        <div className="animate-slide-up">
          <MgApproval
            workflowId={id}
            mgPlan={data.mg_plan}
            videoInfo={data.video_info}
            videoPaths={data.video_paths}
          />
        </div>
      </>
    );
  }

  if (stage === "rendering") {
    return (
      <>
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
          <ActivityIndicator stage="Rendering" description="Encoding final video with all overlays..." />
        </div>
      </>
    );
  }

  if (stage === "done" && data.result) {
    const r = data.result;
    return (
      <>
        <StageIndicator currentStage={stage} />
        <div className="animate-slide-up space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <Stat value={String(r.effects_applied)} label="Effects applied" />
            <Stat value={String(r.motion_graphics_applied)} label="Motion graphics" />
            {r.transcript_length != null && (
              <Stat value={String(r.transcript_length)} label="Transcript length" />
            )}
            {r.phases != null && (
              <Stat value={String(r.phases)} label="Phases" />
            )}
          </div>
          <Card>
            <p className="text-[10px] uppercase tracking-[0.2em] text-text-ghost mb-2">Output path</p>
            <code className="text-sm text-text break-all">{String(r.output_video)}</code>
          </Card>
          <Link href="/" className="text-sm text-text-dim hover:text-text inline-block">
            Start New
          </Link>
        </div>
      </>
    );
  }

  if (stage === "error") {
    return (
      <>
        <StageIndicator currentStage={stage} />
        <div className="animate-slide-up">
          <Card className="border-negative-border p-6">
            <pre className="overflow-x-auto"><code className="text-sm text-negative">{data.error ?? "Unknown error"}</code></pre>
            <Link href="/" className="text-sm text-text-dim hover:text-text mt-4 inline-block">
              Try Again
            </Link>
          </Card>
        </div>
      </>
    );
  }

  return (
    <>
      <StageIndicator currentStage={stage} />
      <div className="animate-slide-up">
        <ActivityIndicator
          stage={stage.charAt(0).toUpperCase() + stage.slice(1)}
          description="Processing..."
        />
      </div>
    </>
  );
}
