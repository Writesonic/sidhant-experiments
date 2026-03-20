import { useState, useCallback, useRef, useEffect } from "react";
import { generateTemplateCode } from "@/lib/api";
import { compileComponent } from "@/lib/compiler";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface UseTemplateEditorOptions {
  initialCode?: string;
  initialDisplayName?: string;
  initialDescription?: string;
  initialTags?: string;
}

export function useTemplateEditor(options: UseTemplateEditorOptions = {}) {
  const [code, setCode] = useState(options.initialCode ?? "");
  const [prompt, setPrompt] = useState("");
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [compileError, setCompileError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [displayName, setDisplayName] = useState(options.initialDisplayName ?? "");
  const [description, setDescription] = useState(options.initialDescription ?? "");
  const [tags, setTags] = useState(options.initialTags ?? "");

  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  const chatEndRef = useRef<HTMLDivElement>(null);

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
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
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

  const handleSave = useCallback(
    async (saveFn: () => Promise<void>) => {
      setSaving(true);
      setSaveError(null);
      try {
        await saveFn();
      } catch (err) {
        setSaveError(err instanceof Error ? err.message : String(err));
      } finally {
        setSaving(false);
      }
    },
    [],
  );

  return {
    code, setCode,
    prompt, setPrompt,
    chat, setChat,
    compileError,
    generating,
    saving,
    saveError,
    displayName, setDisplayName,
    description, setDescription,
    tags, setTags,
    chatEndRef,
    handleGenerate,
    handleSave,
  };
}

export type UseTemplateEditorReturn = ReturnType<typeof useTemplateEditor>;
