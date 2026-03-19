import React from "react";
import * as Babel from "@babel/standalone";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  AbsoluteFill,
  Sequence,
  Img,
} from "remotion";

const AVAILABLE_GLOBALS: Record<string, unknown> = {
  React,
  useState: React.useState,
  useEffect: React.useEffect,
  useMemo: React.useMemo,
  useRef: React.useRef,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  AbsoluteFill,
  Sequence,
  Img,
};

function stripImports(code: string): string {
  return code.replace(/^import\s+.*?['";]\s*$/gm, "");
}

function extractExportName(code: string): string | null {
  const match = code.match(
    /export\s+const\s+(\w+)\s*[=:]/
  );
  return match ? match[1] : null;
}

export function compileComponent(code: string): {
  Component: React.FC<any> | null;
  error: string | null;
} {
  try {
    const stripped = stripImports(code);
    const exportName = extractExportName(stripped);
    if (!exportName) {
      return { Component: null, error: "No exported component found. Use: export const ComponentName: React.FC<any> = ..." };
    }

    // Remove the export keyword so the variable is assignable
    const withoutExport = stripped.replace(/export\s+const\s+/, "const ");

    const transformed = Babel.transform(withoutExport, {
      presets: ["react", "typescript"],
      filename: "component.tsx",
    });

    if (!transformed.code) {
      return { Component: null, error: "Babel transform produced no output" };
    }

    const globalKeys = Object.keys(AVAILABLE_GLOBALS);
    const globalValues = globalKeys.map((k) => AVAILABLE_GLOBALS[k]);

    // Wrap in function that returns the component
    const wrappedCode = `${transformed.code}\nreturn ${exportName};`;
    const factory = new Function(...globalKeys, wrappedCode);
    const Component = factory(...globalValues);

    if (typeof Component !== "function") {
      return { Component: null, error: `${exportName} is not a valid React component` };
    }

    return { Component, error: null };
  } catch (e) {
    return { Component: null, error: e instanceof Error ? e.message : String(e) };
  }
}
