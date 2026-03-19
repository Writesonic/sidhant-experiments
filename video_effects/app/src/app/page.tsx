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
          <h1 className="text-4xl font-display font-[800] tracking-[-0.02em]">Sidhant's Epic Video Effects Studio</h1>
          <p className="text-text-secondary text-sm mt-1">
            Professional video effects workflow
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-[10px] uppercase tracking-[0.2em] text-text-ghost mb-1">
              Video path
            </label>
            <input
              type="text"
              value={videoPath}
              onChange={(e) => setVideoPath(e.target.value)}
              placeholder="/path/to/video.mp4"
              required
              className="w-full bg-surface border border-border-card h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
            />
          </div>

          <div>
            <span className="block text-[10px] uppercase tracking-[0.2em] text-text-ghost mb-2">Options</span>
            <div className="flex gap-6 text-sm">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={programmer}
                  onChange={(e) => setProgrammer(e.target.checked)}
                  className="accent-accent"
                />
                Programmer
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={mg}
                  onChange={(e) => setMg(e.target.checked)}
                  className="accent-accent"
                />
                Motion Graphics
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={subtitles}
                  onChange={(e) => setSubtitles(e.target.checked)}
                  className="accent-accent"
                />
                Subtitles
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={devMode}
                  onChange={(e) => setDevMode(e.target.checked)}
                  className="accent-accent"
                />
                Dev mode
              </label>
            </div>
          </div>

          <div>
            <label className="block text-[10px] uppercase tracking-[0.2em] text-text-ghost mb-1">
              Style (optional)
            </label>
            <input
              type="text"
              value={style}
              onChange={(e) => setStyle(e.target.value)}
              placeholder="e.g. bold, minimal"
              className="w-full bg-surface border border-border-card h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
            />
          </div>

          {libraryTemplates.length > 0 && (
            <div>
              <button
                type="button"
                onClick={() => setShowPinning(!showPinning)}
                className="text-sm text-text-dim hover:text-text transition-colors"
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
                        className={`px-3 py-1 text-xs font-medium border transition-colors ${
                          pinned
                            ? "bg-accent-fill border-accent-dim text-accent"
                            : "bg-surface border-border-card text-text-dim hover:text-text"
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
            <div className="bg-negative-fill border border-negative-border p-3 text-negative text-sm">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-accent hover:bg-accent/90 disabled:bg-border-card text-bg h-11 px-4 text-base font-semibold transition-colors"
          >
            {loading ? "Starting..." : "Start Workflow"}
          </button>
        </form>
      </Card>
    </div>
  );
}
