"use client";

import { FormEvent, useState, KeyboardEvent } from "react";

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

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && text.trim()) {
      e.preventDefault();
      onSubmit(text.trim());
    }
  }

  return (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 animate-fade-in"
      onClick={(e) => { if (e.target === e.currentTarget) onCancel(); }}
    >
      <form
        onSubmit={handleSubmit}
        className="bg-surface border border-border-card p-6 w-full max-w-md space-y-4 animate-slide-up"
      >
        <div>
          <h3 className="text-lg font-semibold">{title}</h3>
          <p className="text-xs text-text-muted mt-1">
            Describe what should change. This will trigger a revision.
          </p>
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="What should change?"
          rows={5}
          autoFocus
          className="w-full bg-bg border border-border-card px-3 py-2 text-sm focus:outline-none focus:border-accent resize-none"
        />
        <div className="flex gap-3 justify-end items-center">
          <span className="text-[10px] text-text-ghost mr-auto">
            {"\u2318"}+Enter to send
          </span>
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2.5 text-sm text-text-dim hover:text-text active:text-text transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!text.trim()}
            className="px-4 py-2.5 text-sm bg-accent hover:bg-accent/90 active:scale-[0.98] text-bg disabled:bg-border-card font-medium transition-all"
          >
            Send Feedback
          </button>
        </div>
      </form>
    </div>
  );
}
