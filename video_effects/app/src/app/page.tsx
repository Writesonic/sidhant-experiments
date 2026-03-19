"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { startWorkflow } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [videoPath, setVideoPath] = useState("");
  const [programmer, setProgrammer] = useState(false);
  const [mg, setMg] = useState(false);
  const [style, setStyle] = useState("");
  const [devMode, setDevMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const { workflow_id } = await startWorkflow({
        video_path: videoPath,
        enable_programmer: programmer,
        enable_mg: mg || programmer,
        style: style || undefined,
        dev_mode: devMode,
      });
      router.push(`/workflow/${workflow_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setLoading(false);
    }
  }

  return (
    <div className="max-w-lg mx-auto mt-16">
      <h1 className="text-3xl font-bold mb-8">VFX Studio</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm text-neutral-400 mb-1">
            Video path
          </label>
          <input
            type="text"
            value={videoPath}
            onChange={(e) => setVideoPath(e.target.value)}
            placeholder="/path/to/video.mp4"
            required
            className="w-full bg-neutral-900 border border-neutral-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
          />
        </div>

        <div className="flex gap-6 text-sm">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={programmer}
              onChange={(e) => setProgrammer(e.target.checked)}
              className="accent-blue-500"
            />
            Programmer
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={mg}
              onChange={(e) => setMg(e.target.checked)}
              className="accent-blue-500"
            />
            Motion Graphics
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={devMode}
              onChange={(e) => setDevMode(e.target.checked)}
              className="accent-blue-500"
            />
            Dev mode
          </label>
        </div>

        <div>
          <label className="block text-sm text-neutral-400 mb-1">
            Style (optional)
          </label>
          <input
            type="text"
            value={style}
            onChange={(e) => setStyle(e.target.value)}
            placeholder="e.g. bold, minimal"
            className="w-full bg-neutral-900 border border-neutral-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
          />
        </div>

        {error && (
          <p className="text-red-400 text-sm">{error}</p>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-neutral-700 rounded px-4 py-2 text-sm font-medium transition-colors"
        >
          {loading ? "Starting..." : "Start Workflow"}
        </button>
      </form>
    </div>
  );
}
