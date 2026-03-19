"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getWorkflowStatus, type WorkflowStatus } from "@/lib/api";

export function useWorkflow(id: string | null) {
  const [data, setData] = useState<WorkflowStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const prevStage = useRef<string | null>(null);

  const poll = useCallback(async () => {
    if (!id) return;
    try {
      const status = await getWorkflowStatus(id);
      setData(status);
      setError(null);
      prevStage.current = status.stage;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [id]);

  useEffect(() => {
    if (!id) return;
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, [id, poll]);

  return { data, error, refetch: poll };
}
