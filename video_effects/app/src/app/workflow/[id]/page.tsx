"use client";

import { use } from "react";
import { useWorkflow } from "@/hooks/useWorkflow";
import { TimelineApproval } from "@/components/TimelineApproval";
import { MgApproval } from "@/components/MgApproval";

function Spinner({ text }: { text: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-24">
      <div className="w-8 h-8 border-2 border-neutral-600 border-t-blue-500 rounded-full animate-spin" />
      <p className="text-neutral-400 text-sm">{text}</p>
    </div>
  );
}

export default function WorkflowPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data, error } = useWorkflow(id);

  if (error) {
    return (
      <div className="py-16 text-center">
        <p className="text-red-400">Failed to load workflow: {error}</p>
      </div>
    );
  }

  if (!data) {
    return <Spinner text="Connecting to workflow..." />;
  }

  const { stage } = data;

  if (stage === "init" || stage === "analyzing") {
    return <Spinner text="Analyzing video..." />;
  }

  if (stage === "timeline_approval" && data.timeline) {
    return (
      <TimelineApproval
        workflowId={id}
        timeline={data.timeline}
        baseVideoPath={data.video_paths?.base_video}
      />
    );
  }

  if (stage === "processing") {
    return <Spinner text="Processing effects and generating components..." />;
  }

  if (stage === "mg_preview") {
    return <Spinner text="Preparing motion graphics preview..." />;
  }

  if (stage === "mg_approval" && data.mg_plan && data.video_info && data.video_paths) {
    return (
      <MgApproval
        workflowId={id}
        mgPlan={data.mg_plan}
        videoInfo={data.video_info}
        videoPaths={data.video_paths}
      />
    );
  }

  if (stage === "rendering") {
    return <Spinner text="Rendering final video..." />;
  }

  if (stage === "done" && data.result) {
    const r = data.result;
    return (
      <div className="max-w-lg mx-auto py-16 space-y-4">
        <h2 className="text-2xl font-bold text-green-400">Done</h2>
        <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 space-y-2 text-sm">
          <p>
            <span className="text-neutral-400">Output:</span>{" "}
            <code className="text-neutral-200">{String(r.output_video)}</code>
          </p>
          <p>
            <span className="text-neutral-400">Effects:</span>{" "}
            {String(r.effects_applied)}
          </p>
          {Number(r.motion_graphics_applied) > 0 && (
            <p>
              <span className="text-neutral-400">Motion graphics:</span>{" "}
              {String(r.motion_graphics_applied)} components
            </p>
          )}
        </div>
      </div>
    );
  }

  if (stage === "error") {
    return (
      <div className="max-w-lg mx-auto py-16 space-y-4">
        <h2 className="text-2xl font-bold text-red-400">Error</h2>
        <p className="text-neutral-300 text-sm">
          {data.error ?? "Unknown error"}
        </p>
      </div>
    );
  }

  return <Spinner text={`Stage: ${stage}...`} />;
}
