"use client";

import type { PlayerRef } from "@remotion/player";
import { Player } from "@remotion/player";
import { DynamicComposition } from "@remotion-project/DynamicComposition";
import type { CompositionPlan } from "@remotion-project/types";

// Player expects Record<string, unknown> — cast the typed component
const Composition = DynamicComposition as unknown as React.FC<Record<string, unknown>>;

interface Props {
  plan: CompositionPlan;
  width: number;
  height: number;
  fps: number;
  durationInFrames: number;
}

export function VideoPlayer({ plan, width, height, fps, durationInFrames }: Props) {
  return (
    <Player
      component={Composition}
      inputProps={plan as unknown as Record<string, unknown>}
      durationInFrames={durationInFrames}
      fps={fps}
      compositionWidth={width}
      compositionHeight={height}
      controls
      style={{ width: "100%" }}
    />
  );
}
