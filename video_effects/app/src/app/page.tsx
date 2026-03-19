"use client";

import { FormEvent, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { startWorkflow, listTemplates, type LibraryTemplateMeta } from "@/lib/api";
import { Card } from "@/components/ui";

export default function Home() {
  const router = useRouter();
  const [videoPath, setVideoPath] = useState("");
  const [programmer, setProgrammer] = useState(false);
  const [mg, setMg] = useState(false);
  const [style, setStyle] = useState("");
  const [devMode, setDevMode] = useState(false);
  const [subtitles, setSubtitles] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [libraryTemplates, setLibraryTemplates] = useState<LibraryTemplateMeta[]>([]);
  const [pinnedTemplates, setPinnedTemplates] = useState<Set<string>>(new Set());
  const [showPinning, setShowPinning] = useState(false);

  useEffect(() => {
    listTemplates().then(setLibraryTemplates).catch(() => {});
  }, []);

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
        enable_subtitles: subtitles,
        pinned_templates: Array.from(pinnedTemplates),
      });
      router.push(`/workflow/${workflow_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setLoading(false);
    }
  }

  return (
    <div className="min-h-[80vh] flex items-center justify-center animate-slide-up">
      <Card className="p-8 w-full max-w-lg">
        <div className="mb-6">
          <h1 className="text-4xl font-bold tracking-tight">VFX Studio</h1>
          <p className="text-neutral-500 text-sm mt-1">
            Professional video effects workflow
          </p>
        </div>

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
              className="w-full bg-neutral-900 border border-neutral-700 rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500"
            />
          </div>

          <div>
            <span className="block text-sm text-neutral-400 mb-2">Options</span>
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
                  checked={subtitles}
                  onChange={(e) => setSubtitles(e.target.checked)}
                  className="accent-blue-500"
                />
                Subtitles
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
              className="w-full bg-neutral-900 border border-neutral-700 rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500"
            />
          </div>

          {libraryTemplates.length > 0 && (
            <div>
              <button
                type="button"
                onClick={() => setShowPinning(!showPinning)}
                className="text-sm text-neutral-400 hover:text-neutral-200 transition-colors"
              >
                {showPinning ? "Hide" : "Pin"} library templates ({pinnedTemplates.size}/{libraryTemplates.length})
              </button>
              {showPinning && (
                <div className="mt-2 flex flex-wrap gap-2">
                  {libraryTemplates.map((t) => {
                    const pinned = pinnedTemplates.has(t.id);
                    return (
                      <button
                        key={t.id}
                        type="button"
                        onClick={() => {
                          setPinnedTemplates((prev) => {
                            const next = new Set(prev);
                            if (pinned) next.delete(t.id);
                            else next.add(t.id);
                            return next;
                          });
                        }}
                        className={`px-3 py-1 rounded-md text-xs font-medium border transition-colors ${
                          pinned
                            ? "bg-blue-500/20 border-blue-500/50 text-blue-300"
                            : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:text-neutral-200"
                        }`}
                      >
                        {t.display_name}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-neutral-700 rounded-lg h-11 px-4 text-base font-semibold transition-colors"
          >
            {loading ? "Starting..." : "Start Workflow"}
          </button>
        </form>
      </Card>
    </div>
  );
}
