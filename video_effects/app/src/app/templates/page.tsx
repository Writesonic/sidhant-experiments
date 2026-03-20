"use client";

import { useEffect, useState, useMemo, useRef, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Player } from "@remotion/player";
import { listTemplates, deleteTemplate, type LibraryTemplateMeta } from "@/lib/api";
import { DynamicCodeComp } from "@/components/DynamicCodeComp";
import { Card, Badge } from "@/components/ui";

function LazyTemplatePreview({ code }: { code: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const PreviewComp = useMemo(() => {
    const Comp: React.FC<{ code: string }> = (props) => (
      <DynamicCodeComp code={props.code} />
    );
    return Comp;
  }, []);

  return (
    <div ref={containerRef} className="overflow-hidden bg-black aspect-video">
      {visible && (
        <Player
          component={PreviewComp}
          inputProps={{ code }}
          durationInFrames={150}
          fps={30}
          compositionWidth={1920}
          compositionHeight={1080}
          style={{ width: "100%" }}
          autoPlay
          loop
        />
      )}
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="bg-surface border border-border-card p-4 flex flex-col gap-3 animate-pulse">
      <div className="aspect-video bg-border-card" />
      <div className="h-4 bg-border-card w-2/3" />
      <div className="h-3 bg-border-card w-full" />
      <div className="h-3 bg-border-card w-1/2" />
    </div>
  );
}

export default function TemplatesPage() {
  const router = useRouter();
  const [templates, setTemplates] = useState<LibraryTemplateMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setTemplates(await listTemplates());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function handleDelete(id: string) {
    await deleteTemplate(id);
    setConfirmDelete(null);
    setTemplates((prev) => prev.filter((t) => t.id !== id));
  }

  return (
    <div className="animate-slide-up">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-display font-bold tracking-tight">Component Gallery</h1>
          <p className="text-text-secondary text-sm mt-1">
            Reusable motion graphics templates
          </p>
        </div>
        <Link
          href="/templates/create"
          className="px-5 py-2.5 bg-accent hover:bg-accent/90 active:scale-[0.98] text-bg font-semibold text-sm transition-all"
        >
          Create New
        </Link>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      ) : templates.length === 0 ? (
        <Card className="p-12 text-center">
          <p className="text-text-dim mb-1">No templates yet</p>
          <p className="text-text-ghost text-xs mb-4">Create your first motion graphics component</p>
          <Link
            href="/templates/create"
            className="text-accent hover:text-accent/80 text-sm font-medium"
          >
            Create your first template
          </Link>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {templates.map((t) => (
            <Card
              key={t.id}
              interactive
              onClick={() => router.push(`/templates/${t.id}`)}
              className="p-4 flex flex-col gap-3"
            >
              <LazyTemplatePreview code={t.tsx_code} />
              <div>
                <h3 className="font-semibold text-text">{t.display_name}</h3>
                <p className="text-text-dim text-sm mt-1 line-clamp-2">
                  {t.description}
                </p>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                {t.tags.map((tag) => (
                  <Badge key={tag} label={tag} />
                ))}
                {t.created_at && (
                  <span className="text-xs text-text-ghost ml-auto">
                    {new Date(t.created_at).toLocaleDateString()}
                  </span>
                )}
              </div>
              <div className="flex gap-2 mt-auto pt-1" onClick={(e) => e.stopPropagation()}>
                {confirmDelete === t.id ? (
                  <div className="flex gap-1">
                    <button
                      onClick={() => handleDelete(t.id)}
                      className="px-3 py-2.5 text-xs bg-negative/20 border border-negative-border text-negative font-medium transition-colors"
                    >
                      Confirm
                    </button>
                    <button
                      onClick={() => setConfirmDelete(null)}
                      className="px-3 py-2.5 text-xs border border-border-card text-text-dim transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setConfirmDelete(t.id)}
                    className="px-4 py-2.5 border border-negative-border text-negative hover:bg-negative-fill text-xs font-medium transition-colors"
                  >
                    Delete
                  </button>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
