"use client";

import { useMemo } from "react";
import Link from "next/link";
import { Player } from "@remotion/player";
import { DynamicCodeComp } from "@/components/DynamicCodeComp";
import { Card } from "@/components/ui";
import type { UseTemplateEditorReturn } from "@/hooks/useTemplateEditor";

interface TemplateEditorProps {
  editor: UseTemplateEditorReturn;
  onSave: () => void;
  saveLabel: string;
  emptyPreviewText: string;
  promptPlaceholder: string;
  generateLabel: string;
  codePlaceholder?: string;
}

function ThreeDotPulse() {
  return (
    <span className="inline-flex items-center gap-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-1.5 h-1.5 bg-bg rounded-full animate-pulse"
          style={{ animationDelay: `${i * 150}ms` }}
        />
      ))}
    </span>
  );
}

export function EditorSkeleton() {
  return (
    <div className="animate-slide-up">
      <Link href="/templates" className="text-sm text-text-dim hover:text-text transition-colors">
        &larr; Gallery
      </Link>
      <div className="mt-6 flex flex-col gap-4">
        <div className="h-8 bg-border-card w-1/3 animate-pulse" />
        <div className="h-12 bg-surface border border-border-card animate-pulse" />
        <div className="grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-4">
          <div className="h-80 bg-bg border border-border-card animate-pulse" />
          <div className="aspect-video bg-surface border border-border-card animate-pulse" />
        </div>
      </div>
    </div>
  );
}

export function TemplateEditor({
  editor,
  onSave,
  saveLabel,
  emptyPreviewText,
  promptPlaceholder,
  generateLabel,
  codePlaceholder,
}: TemplateEditorProps) {
  const {
    code, setCode,
    prompt, setPrompt,
    chat,
    compileError,
    generating,
    saving,
    saveError,
    displayName, setDisplayName,
    description, setDescription,
    tags, setTags,
    chatEndRef,
    handleGenerate,
  } = editor;

  const PreviewComponent = useMemo(() => {
    const Comp: React.FC<{ code: string }> = (props) => <DynamicCodeComp code={props.code} />;
    return Comp;
  }, []);

  return (
    <div className="flex flex-col gap-3">
      {/* Prompt bar — full width */}
      <div className="flex gap-2">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
          placeholder={promptPlaceholder}
          className="flex-1 bg-surface border border-border-card h-12 px-4 text-sm focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
        />
        <button
          onClick={handleGenerate}
          disabled={generating || !prompt.trim()}
          className="px-5 h-12 bg-accent hover:bg-accent/90 active:scale-[0.98] disabled:bg-border-card text-bg text-sm font-semibold transition-all"
        >
          {generating ? <ThreeDotPulse /> : generateLabel}
        </button>
      </div>

      {/* Main area — code left, preview + chat right */}
      <div className="grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-3">
        {/* Code editor */}
        <div className="order-2 lg:order-1 flex flex-col gap-2">
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            placeholder={codePlaceholder}
            spellCheck={false}
            className="w-full min-h-64 lg:min-h-96 bg-bg border border-border-card p-3 font-mono text-xs text-text resize-y focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
          />
          {compileError && (
            <div className="bg-negative-fill border border-negative-border p-2 text-negative text-xs font-mono whitespace-pre-wrap">
              {compileError}
            </div>
          )}
        </div>

        {/* Preview + Chat */}
        <div className="order-1 lg:order-2 flex flex-col gap-3 lg:sticky lg:top-6 lg:self-start">
          <div className="overflow-hidden bg-black aspect-video border border-border-card">
            {code.trim() ? (
              <Player
                component={PreviewComponent}
                inputProps={{ code }}
                durationInFrames={150}
                fps={30}
                compositionWidth={1920}
                compositionHeight={1080}
                style={{ width: "100%" }}
                controls
                loop
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-text-ghost text-sm">
                {emptyPreviewText}
              </div>
            )}
          </div>

          {chat.length > 0 && (
            <div className="bg-surface border border-border-card p-3 max-h-48 overflow-y-auto">
              <div className="space-y-1.5">
                {chat.map((msg, i) => (
                  <div
                    key={i}
                    className={`text-sm ${msg.role === "user" ? "text-accent" : "text-text-secondary"}`}
                  >
                    <span className="text-[10px] uppercase tracking-[0.15em] text-text-muted mr-2">
                      {msg.role === "user" ? "You" : "AI"}
                    </span>
                    {msg.content}
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Save bar — full width, compact */}
      {code.trim() && (
        <div className="bg-surface border border-border-card p-3 flex flex-col sm:flex-row sm:items-center gap-2">
          <input
            type="text"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            placeholder="Display name"
            className="bg-bg border border-border-card h-9 px-3 text-sm sm:w-48 focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
          />
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Description"
            className="flex-1 bg-bg border border-border-card h-9 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
          />
          <input
            type="text"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="Tags (comma-separated)"
            className="bg-bg border border-border-card h-9 px-3 text-sm sm:w-48 focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
          />
          {saveError && (
            <div className="text-negative text-xs">{saveError}</div>
          )}
          <button
            onClick={onSave}
            disabled={saving || !displayName.trim() || !code.trim()}
            className="px-6 bg-accent hover:bg-accent/90 active:scale-[0.98] disabled:bg-border-card text-bg h-9 text-sm font-semibold transition-all shrink-0"
          >
            {saving ? "Saving..." : saveLabel}
          </button>
        </div>
      )}
    </div>
  );
}
