"use client";

import { FormEvent, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { startWorkflow, uploadVideo, listTemplates, type LibraryTemplateMeta } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState("");
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
      if (!videoFile) return;
      setUploadStatus("Uploading… 0%");
      const { path } = await uploadVideo(videoFile, (pct) =>
        setUploadStatus(`Uploading… ${pct}%`),
      );
      setUploadStatus("Starting…");
      const { workflow_id } = await startWorkflow({
        video_path: path,
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
      setUploadStatus("");
      setLoading(false);
    }
  }

  return (
    <div className="max-w-xl mx-auto pt-12 animate-slide-up">
      <div className="mb-8">
        <h1 className="text-[clamp(28px,4vw,42px)] font-display font-[800] tracking-[-0.02em] leading-[1.1]">
          VFX Studio
        </h1>
        <p className="text-text-secondary text-sm mt-2">
          Professional video effects workflow
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label className="block text-[10px] uppercase tracking-[0.2em] text-text-ghost mb-2">
            Video file
          </label>
          <label className="flex items-center gap-3 w-full bg-surface border border-border-card h-11 px-3 text-sm cursor-pointer hover:border-accent/40 transition-colors">
            <span className="text-text-ghost shrink-0">Choose file</span>
            <span className="text-text truncate">
              {videoFile ? `${videoFile.name} (${(videoFile.size / 1024 / 1024).toFixed(1)} MB)` : "No file selected"}
            </span>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
              className="hidden"
            />
          </label>
        </div>

        <div>
          <span className="block text-[10px] uppercase tracking-[0.2em] text-text-ghost mb-2">Options</span>
          <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
            <label className="flex items-center gap-2 min-h-[44px]">
              <input
                type="checkbox"
                checked={programmer}
                onChange={(e) => setProgrammer(e.target.checked)}
                className="accent-accent"
              />
              Programmer
            </label>
            <label className="flex items-center gap-2 min-h-[44px]">
              <input
                type="checkbox"
                checked={mg}
                onChange={(e) => setMg(e.target.checked)}
                className="accent-accent"
              />
              Motion Graphics
            </label>
            <label className="flex items-center gap-2 min-h-[44px]">
              <input
                type="checkbox"
                checked={subtitles}
                onChange={(e) => setSubtitles(e.target.checked)}
                className="accent-accent"
              />
              Subtitles
            </label>
            <label className="flex items-center gap-2 min-h-[44px]">
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
          <label className="block text-[10px] uppercase tracking-[0.2em] text-text-ghost mb-2">
            Style (optional)
          </label>
          <input
            type="text"
            value={style}
            onChange={(e) => setStyle(e.target.value)}
            placeholder="e.g. bold, minimal"
            className="w-full bg-surface border border-border-card h-11 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
          />
        </div>

        {libraryTemplates.length > 0 && (
          <div>
            <button
              type="button"
              onClick={() => setShowPinning(!showPinning)}
              className="px-4 py-2 text-sm border border-border-card text-text-dim hover:text-text hover:border-accent/40 transition-colors"
            >
              {showPinning ? "Hide" : "Pin"} library templates ({pinnedTemplates.size}/{libraryTemplates.length})
            </button>
            {showPinning && (
              <div className="mt-3 flex flex-wrap gap-2">
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
                      className={`px-4 py-2 text-xs font-medium border transition-colors ${
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
          <div className="bg-negative-fill border border-negative-border p-4 text-negative text-sm">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading || !videoFile}
          className="w-full bg-accent hover:bg-accent/90 active:scale-[0.98] disabled:bg-border-card text-bg h-11 px-4 text-base font-semibold transition-all flex items-center justify-center gap-2"
        >
          {loading && (
            <span className="flex items-center gap-1">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="w-1.5 h-1.5 bg-bg animate-pulse"
                  style={{ animationDelay: `${i * 150}ms` }}
                />
              ))}
            </span>
          )}
          {loading ? (uploadStatus || "Starting...") : "Start Workflow"}
        </button>
      </form>
    </div>
  );
}
