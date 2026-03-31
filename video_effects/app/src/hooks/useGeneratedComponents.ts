"use client";

import { useEffect, useMemo, useRef } from "react";
import { compileComponent } from "@/lib/compiler";
import { REMOTION_EXTRA_GLOBALS } from "@/lib/remotion-globals";
import { ComponentRegistry } from "@remotion-project/components";

export function useGeneratedComponents(
  componentSources: Record<string, string> | undefined,
): { ready: boolean; errors: Record<string, string> } {
  const injectedKeys = useRef<string[]>([]);

  const result = useMemo(() => {
    // Clean up previous injections
    for (const key of injectedKeys.current) {
      delete (ComponentRegistry as Record<string, unknown>)[key];
    }
    injectedKeys.current = [];

    if (!componentSources || Object.keys(componentSources).length === 0) {
      return { ready: true, errors: {} };
    }

    const errors: Record<string, string> = {};

    for (const [templateId, code] of Object.entries(componentSources)) {
      const { Component, error } = compileComponent(code, REMOTION_EXTRA_GLOBALS);
      if (Component) {
        (ComponentRegistry as Record<string, unknown>)[templateId] = Component;
        injectedKeys.current.push(templateId);
        console.log(`[useGeneratedComponents] compiled: ${templateId}`);
      } else {
        errors[templateId] = error ?? "Unknown compilation error";
        console.error(`[useGeneratedComponents] failed: ${templateId}`, error);
      }
    }

    return { ready: true, errors };
  }, [componentSources]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      for (const key of injectedKeys.current) {
        delete (ComponentRegistry as Record<string, unknown>)[key];
      }
    };
  }, []);

  return result;
}
