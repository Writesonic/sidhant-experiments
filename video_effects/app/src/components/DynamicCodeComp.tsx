import React, { useMemo } from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { compileComponent } from "@/lib/compiler";

interface DynamicCodeCompProps {
  code: string;
}

function ErrorOverlay({ message }: { message: string }) {
  return (
    <AbsoluteFill
      style={{
        backgroundColor: "rgba(220, 38, 38, 0.15)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 40,
      }}
    >
      <div
        style={{
          background: "rgba(0,0,0,0.85)",
          borderRadius: 12,
          padding: 24,
          maxWidth: 600,
          fontFamily: "monospace",
          fontSize: 14,
          color: "#f87171",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          border: "1px solid rgba(248, 113, 113, 0.3)",
        }}
      >
        {message}
      </div>
    </AbsoluteFill>
  );
}

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: string | null }
> {
  state = { error: null as string | null };

  static getDerivedStateFromError(error: Error) {
    return { error: error.message };
  }

  render() {
    if (this.state.error) {
      return <ErrorOverlay message={`Runtime error: ${this.state.error}`} />;
    }
    return this.props.children;
  }
}

export const DynamicCodeComp: React.FC<DynamicCodeCompProps> = ({ code }) => {
  const frame = useCurrentFrame();

  const { Component, error } = useMemo(() => {
    if (!code || !code.trim()) {
      return { Component: null, error: "No code provided" };
    }
    return compileComponent(code);
  }, [code]);

  if (error) {
    return <ErrorOverlay message={error} />;
  }

  if (!Component) {
    return <ErrorOverlay message="Failed to compile component" />;
  }

  return (
    <ErrorBoundary key={code}>
      <AbsoluteFill>
        <Component />
      </AbsoluteFill>
    </ErrorBoundary>
  );
};
