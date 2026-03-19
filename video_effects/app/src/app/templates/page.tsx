"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import { Player } from "@remotion/player";
import { listTemplates, deleteTemplate, type LibraryTemplateMeta } from "@/lib/api";
import { DynamicCodeComp } from "@/components/DynamicCodeComp";
import { Card, Badge } from "@/components/ui";

function TemplatePreview({ code }: { code: string }) {
  const PreviewComp = useMemo(() => {
    const Comp: React.FC<{ code: string }> = (props) => (
      <DynamicCodeComp code={props.code} />
    );
    return Comp;
  }, []);

  return (
    <div className="rounded-lg overflow-hidden bg-black aspect-video">
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
    </div>
  );
}

export default function TemplatesPage() {
  const [templates, setTemplates] = useState<LibraryTemplateMeta[]>([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try {
      setTemplates(await listTemplates());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function handleDelete(id: string) {
    await deleteTemplate(id);
    setTemplates((prev) => prev.filter((t) => t.id !== id));
  }

  return (
    <div className="animate-slide-up">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Component Gallery</h1>
          <p className="text-neutral-500 text-sm mt-1">
            Reusable motion graphics templates
          </p>
        </div>
        <Link
          href="/templates/create"
          className="px-5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm transition-colors"
        >
          Create New
        </Link>
      </div>

      {loading ? (
        <p className="text-neutral-500 text-sm">Loading...</p>
      ) : templates.length === 0 ? (
        <Card className="p-12 text-center">
          <p className="text-neutral-400 mb-4">No templates yet</p>
          <Link
            href="/templates/create"
            className="text-blue-400 hover:text-blue-300 text-sm font-medium"
          >
            Create your first template
          </Link>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {templates.map((t) => (
            <Card key={t.id} className="p-4 flex flex-col gap-3">
              <TemplatePreview code={t.tsx_code} />
              <div>
                <h3 className="font-semibold text-white">{t.display_name}</h3>
                <p className="text-neutral-400 text-sm mt-1 line-clamp-2">
                  {t.description}
                </p>
              </div>
              {t.tags.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {t.tags.map((tag) => (
                    <Badge key={tag} label={tag} />
                  ))}
                </div>
              )}
              <div className="text-xs text-neutral-600">
                {t.created_at ? new Date(t.created_at).toLocaleDateString() : ""}
              </div>
              <div className="flex gap-2 mt-auto pt-1">
                <Link
                  href={`/templates/${t.id}`}
                  className="px-3 py-1.5 rounded-md border border-neutral-700 text-neutral-300 hover:bg-neutral-800 text-xs font-medium transition-colors"
                >
                  Edit
                </Link>
                <button
                  onClick={() => handleDelete(t.id)}
                  className="px-3 py-1.5 rounded-md border border-red-500/30 text-red-400 hover:bg-red-500/10 text-xs font-medium transition-colors"
                >
                  Delete
                </button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
