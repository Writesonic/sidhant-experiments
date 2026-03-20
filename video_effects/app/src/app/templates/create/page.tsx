"use client";

import { useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { saveTemplate } from "@/lib/api";
import { useTemplateEditor } from "@/hooks/useTemplateEditor";
import { TemplateEditor } from "@/components/TemplateEditor";

export default function CreateTemplatePage() {
  const router = useRouter();
  const editor = useTemplateEditor();

  const onSave = useCallback(() => {
    const { code, displayName, description, tags } = editor;
    if (!displayName.trim() || !code.trim()) return;

    const id = displayName
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_|_$/g, "");

    const exportMatch = code.match(/export\s+const\s+(\w+)/);
    const exportName = exportMatch ? exportMatch[1] : "CustomComponent";

    editor.handleSave(async () => {
      await saveTemplate({
        id,
        display_name: displayName.trim(),
        description: description.trim(),
        tsx_code: code,
        export_name: exportName,
        tags: tags.split(",").map((t) => t.trim()).filter(Boolean),
      });
      router.push("/templates");
    });
  }, [editor, router]);

  return (
    <div className="animate-slide-up">
      <Link href="/templates" className="text-sm text-text-dim hover:text-text transition-colors">
        &larr; Gallery
      </Link>
      <h1 className="text-2xl font-display font-bold tracking-tight mt-3 mb-6">Create Template</h1>
      <TemplateEditor
        editor={editor}
        onSave={onSave}
        saveLabel="Save to Library"
        emptyPreviewText="Describe a component to begin"
        promptPlaceholder="Describe your component..."
        generateLabel="Generate"
        codePlaceholder="// Generated code will appear here, or paste your own TSX..."
      />
    </div>
  );
}
