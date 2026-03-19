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
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <form
        onSubmit={handleSubmit}
        className="bg-neutral-900 border border-neutral-700 rounded-lg p-6 w-full max-w-md space-y-4"
      >
        <h3 className="text-lg font-semibold">{title}</h3>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="What should change?"
          rows={4}
          autoFocus
          className="w-full bg-neutral-800 border border-neutral-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 resize-none"
        />
        <div className="flex gap-3 justify-end">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 text-sm text-neutral-400 hover:text-neutral-200"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!text.trim()}
            className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 disabled:bg-neutral-700 rounded font-medium"
          >
            Send Feedback
          </button>
        </div>
      </form>
    </div>
  );
}
