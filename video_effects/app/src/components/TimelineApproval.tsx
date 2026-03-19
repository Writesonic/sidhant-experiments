"use client";

import { useRef, useState } from "react";
import { signalWorkflow, fileUrl, type TimelineEffect } from "@/lib/api";
import { FeedbackDialog } from "./FeedbackDialog";

interface Props {
  workflowId: string;
  timeline: { effects: TimelineEffect[]; conflicts_resolved: number };
  baseVideoPath?: string;
}

export function TimelineApproval({ workflowId, timeline, baseVideoPath }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [sending, setSending] = useState(false);

  const effects = timeline.effects ?? [];

  function seekTo(seconds: number) {
    if (videoRef.current) {
      videoRef.current.currentTime = seconds;
      videoRef.current.play();
    }
  }

  async function handleApprove() {
    setSending(true);
    await signalWorkflow(workflowId, "approve_timeline", [true, ""]);
  }

  async function handleReject(feedback: string) {
    setShowFeedback(false);
    setSending(true);
    await signalWorkflow(workflowId, "approve_timeline", [false, feedback]);
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Effects Timeline</h2>

      {baseVideoPath && (
        <video
          ref={videoRef}
          src={fileUrl(baseVideoPath)}
          controls
          className="w-full rounded-lg bg-black max-h-[400px]"
        />
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-neutral-400 border-b border-neutral-800">
              <th className="py-2 pr-4">#</th>
              <th className="py-2 pr-4">Type</th>
              <th className="py-2 pr-4">Start</th>
              <th className="py-2 pr-4">End</th>
              <th className="py-2 pr-4">Conf</th>
              <th className="py-2">Cue</th>
            </tr>
          </thead>
          <tbody>
            {effects.map((e, i) => (
              <tr
                key={i}
                onClick={() => seekTo(e.start_time)}
                className="border-b border-neutral-800/50 hover:bg-neutral-800/50 cursor-pointer transition-colors"
              >
                <td className="py-2 pr-4 text-neutral-500">{i + 1}</td>
                <td className="py-2 pr-4 font-mono">{e.effect_type}</td>
                <td className="py-2 pr-4">{e.start_time.toFixed(1)}s</td>
                <td className="py-2 pr-4">{e.end_time.toFixed(1)}s</td>
                <td className="py-2 pr-4">{(e.confidence * 100).toFixed(0)}%</td>
                <td className="py-2 text-neutral-300">{e.verbal_cue}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-sm text-neutral-500">
        {effects.length} effects | {timeline.conflicts_resolved} conflicts resolved
      </p>

      <div className="flex gap-3">
        <button
          onClick={handleApprove}
          disabled={sending}
          className="px-6 py-2 bg-green-600 hover:bg-green-500 disabled:bg-neutral-700 rounded text-sm font-medium transition-colors"
        >
          Approve
        </button>
        <button
          onClick={() => setShowFeedback(true)}
          disabled={sending}
          className="px-6 py-2 bg-red-600 hover:bg-red-500 disabled:bg-neutral-700 rounded text-sm font-medium transition-colors"
        >
          Reject
        </button>
      </div>

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
