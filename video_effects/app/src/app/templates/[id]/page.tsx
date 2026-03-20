"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { getTemplate, updateTemplate, type LibraryTemplate } from "@/lib/api";
import { useTemplateEditor } from "@/hooks/useTemplateEditor";
import { TemplateEditor, EditorSkeleton } from "@/components/TemplateEditor";

export default function TemplateDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [template, setTemplate] = useState<LibraryTemplate | null>(null);
  const editor = useTemplateEditor();

  useEffect(() => {
    getTemplate(id).then((t) => {
      setTemplate(t);
      editor.setCode(t.tsx_code);
      editor.setDisplayName(t.display_name);
      editor.setDescription(t.description);
      editor.setTags(t.tags.join(", "));
    });
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  const onSave = useCallback(() => {
    if (!template) return;
    const { code, displayName, description, tags } = editor;

    const exportMatch = code.match(/export\s+const\s+(\w+)/);
    const exportName = exportMatch ? exportMatch[1] : template.export_name;

    editor.handleSave(async () => {
      await updateTemplate(id, {
        display_name: displayName.trim(),
        description: description.trim(),
        tsx_code: code,
        export_name: exportName,
        tags: tags.split(",").map((t) => t.trim()).filter(Boolean),
      });
      router.push("/templates");
    });
  }, [editor, template, id, router]);

  if (!template) return <EditorSkeleton />;

  return (
    <div className="animate-slide-up">
      <Link href="/templates" className="text-sm text-text-dim hover:text-text transition-colors">
        &larr; Gallery
      </Link>
      <h1 className="text-2xl font-display font-bold tracking-tight mt-3 mb-6">
        Edit: {template.display_name}
      </h1>
      <TemplateEditor
        editor={editor}
        onSave={onSave}
        saveLabel="Update"
        emptyPreviewText="No code to preview"
        promptPlaceholder="Edit: e.g. add a glow effect..."
        generateLabel="Edit"
      />
    </div>
  );
}
