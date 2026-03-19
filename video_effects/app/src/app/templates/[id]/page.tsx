"use client";

import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
import { Player } from "@remotion/player";
import {
  getTemplate,
  updateTemplate,
  generateTemplateCode,
  type LibraryTemplate,
} from "@/lib/api";
import { DynamicCodeComp } from "@/components/DynamicCodeComp";
import { compileComponent } from "@/lib/compiler";
import { Card } from "@/components/ui";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export default function TemplateDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [template, setTemplate] = useState<LibraryTemplate | null>(null);
  const [code, setCode] = useState("");
  const [prompt, setPrompt] = useState("");
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [compileError, setCompileError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [displayName, setDisplayName] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState("");

  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getTemplate(id).then((t) => {
      setTemplate(t);
      setCode(t.tsx_code);
      setDisplayName(t.display_name);
      setDescription(t.description);
      setTags(t.tags.join(", "));
    });
  }, [id]);

  // Recompile on code change
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      if (!code.trim()) {
        setCompileError(null);
        return;
      }
      const { error } = compileComponent(code);
      setCompileError(error);
    }, 500);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [code]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim() || generating) return;
    setGenerating(true);

    const userMsg = prompt.trim();
    setChat((prev) => [...prev, { role: "user", content: userMsg }]);
    setPrompt("");

    try {
      const result = await generateTemplateCode({
        prompt: userMsg,
        previous_code: code || undefined,
        conversation: chat.map((m) => ({ role: m.role, content: m.content })),
        errors: compileError ? [compileError] : undefined,
      });
      setCode(result.code);
      setChat((prev) => [...prev, { role: "assistant", content: result.summary }]);
    } catch (err) {
      setChat((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${err instanceof Error ? err.message : String(err)}` },
      ]);
    } finally {
      setGenerating(false);
    }
  }, [prompt, code, chat, compileError, generating]);

  const handleUpdate = useCallback(async () => {
    if (!template) return;
    setSaving(true);

    const exportMatch = code.match(/export\s+const\s+(\w+)/);
    const exportName = exportMatch ? exportMatch[1] : template.export_name;

    try {
      await updateTemplate(id, {
        display_name: displayName.trim(),
        description: description.trim(),
        tsx_code: code,
        export_name: exportName,
        tags: tags.split(",").map((t) => t.trim()).filter(Boolean),
      });
      router.push("/templates");
    } catch (err) {
      alert(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }, [id, template, displayName, description, tags, code, router]);

  const PreviewComponent = useMemo(() => {
    const Comp: React.FC<{ code: string }> = (props) => <DynamicCodeComp code={props.code} />;
    return Comp;
  }, []);

  if (!template) {
    return <p className="text-text-muted text-sm">Loading...</p>;
  }

  return (
    <div className="animate-slide-up">
      <h1 className="text-2xl font-display font-bold tracking-tight mb-6">
        Edit: {template.display_name}
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left column */}
        <div className="flex flex-col gap-4">
          {/* Chat */}
          <Card className="p-4 max-h-48 overflow-y-auto">
            {chat.length === 0 ? (
              <p className="text-text-muted text-sm">
                Use prompts to modify the component...
              </p>
            ) : (
              <div className="space-y-2">
                {chat.map((msg, i) => (
                  <div
                    key={i}
                    className={`text-sm ${msg.role === "user" ? "text-accent" : "text-text-secondary"}`}
                  >
                    <span className="font-medium text-text-muted mr-2">
                      {msg.role === "user" ? "You:" : "AI:"}
                    </span>
                    {msg.content}
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
            )}
          </Card>

          {/* Prompt */}
          <div className="flex gap-2">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
              placeholder="Edit: e.g. add a glow effect..."
              className="flex-1 bg-surface border border-border-card h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
            />
            <button
              onClick={handleGenerate}
              disabled={generating || !prompt.trim()}
              className="px-4 h-10 bg-accent hover:bg-accent/90 disabled:bg-border-card text-bg text-sm font-semibold transition-colors"
            >
              {generating ? "..." : "Edit"}
            </button>
          </div>

          {/* Code editor */}
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            spellCheck={false}
            className="w-full h-72 bg-bg border border-border-card p-3 font-mono text-xs text-text resize-y focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
          />

          {compileError && (
            <div className="bg-negative-fill border border-negative-border p-3 text-negative text-xs font-mono whitespace-pre-wrap">
              {compileError}
            </div>
          )}

          {/* Update form */}
          <Card className="p-4 space-y-3">
            <h3 className="text-sm font-semibold text-text-secondary">Update Template</h3>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Display name"
              className="w-full bg-surface border border-border-card h-9 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
            />
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Description"
              className="w-full bg-surface border border-border-card h-9 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
            />
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="Tags (comma-separated)"
              className="w-full bg-surface border border-border-card h-9 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent-dim focus:border-accent"
            />
            <button
              onClick={handleUpdate}
              disabled={saving || !displayName.trim() || !code.trim()}
              className="w-full bg-accent hover:bg-accent/90 disabled:bg-border-card text-bg h-10 text-sm font-semibold transition-colors"
            >
              {saving ? "Saving..." : "Update"}
            </button>
          </Card>
        </div>

        {/* Right column: Preview */}
        <div className="flex flex-col gap-4">
          <Card className="p-4">
            <h3 className="text-sm font-semibold text-text-secondary mb-3">Live Preview</h3>
            <div className="overflow-hidden bg-black aspect-video">
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
                  No code to preview
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
