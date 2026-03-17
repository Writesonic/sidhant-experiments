import React from "react";
import { Particles } from "./Particles";
import { Aurora } from "./Aurora";
import { MeshGradient } from "./MeshGradient";
import { GridPattern } from "./GridPattern";

export const BackgroundRegistry: Record<string, React.FC<any>> = {
  particles: Particles,
  aurora: Aurora,
  meshGradient: MeshGradient,
  gridPattern: GridPattern,
};

export type BackgroundType = keyof typeof BackgroundRegistry;

export { Particles, Aurora, MeshGradient, GridPattern };
