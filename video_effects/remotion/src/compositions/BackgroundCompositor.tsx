import React from "react";
import { AbsoluteFill } from "remotion";
import { MaskedVideo } from "../components/MaskedVideo";
import { BackgroundRegistry, type BackgroundType } from "../components/backgrounds";

export interface BackgroundCompositorProps extends Record<string, unknown> {
  originalSrc: string;
  maskSrc: string;
  backgroundType: BackgroundType;
  backgroundConfig?: Record<string, any>;
}

export const BackgroundCompositor: React.FC<BackgroundCompositorProps> = ({
  originalSrc,
  maskSrc,
  backgroundType,
  backgroundConfig = {},
}) => {
  const BackgroundComponent = BackgroundRegistry[backgroundType];

  if (!BackgroundComponent) {
    throw new Error(
      `Unknown background type "${backgroundType}". ` +
      `Available: ${Object.keys(BackgroundRegistry).join(", ")}`
    );
  }

  return (
    <AbsoluteFill>
      <BackgroundComponent {...backgroundConfig} />
      {originalSrc && maskSrc && (
        <MaskedVideo originalSrc={originalSrc} maskSrc={maskSrc} />
      )}
    </AbsoluteFill>
  );
};
