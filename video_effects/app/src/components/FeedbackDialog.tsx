"use client";

import { FormEvent, useState } from "react";

interface Props {
  title: string;
  onSubmit: (feedback: string) => void;
  onCancel: () => void;
}

export function FeedbackDialog({ title, onSubmit, onCancel }: Props) {
  const [text, setText] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (text.trim()) onSubmit(text.trim());
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 animate-fade-in">
      <form
        onSubmit={handleSubmit}
        className="bg-surface border border-border-card p-6 w-full max-w-md space-y-4 animate-slide-up"
      >
        <h3 className="text-lg font-semibold">{title}</h3>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="What should change?"
          rows={4}
          autoFocus
          className="w-full bg-bg border border-border-card px-3 py-2 text-sm focus:outline-none focus:border-accent resize-none"
        />
        <div className="flex gap-3 justify-end">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 text-sm text-text-dim hover:text-text"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!text.trim()}
            className="px-4 py-2 text-sm bg-accent hover:bg-accent/90 text-bg disabled:bg-border-card font-medium"
          >
            Send Feedback
          </button>
        </div>
      </form>
    </div>
  );
}
