import React from "react";
import {
  useFaceAwareLayout,
  useFaceAvoidance,
  useZoomCompensation,
  computeOverlap,
} from "@remotion-project/lib/spatial";
import { useStyle } from "@remotion-project/lib/styles";
import {
  SPRING_GENTLE,
  SPRING_BOUNCY,
  SPRING_SNAPPY,
  SPRING_SMOOTH,
  SPRING_ELASTIC,
  SPRING_WOBBLY,
  gentleSpring,
  bouncySpring,
  snappySpring,
  smoothSpring,
  elasticSpring,
  wobblySpring,
} from "@remotion-project/lib/easing";
import {
  polarToCartesian,
  describeArc,
  generateTicks,
  linearScale,
  colorWithOpacity,
  lerpColor,
} from "@remotion-project/lib/infographic-utils";

export const REMOTION_EXTRA_GLOBALS: Record<string, unknown> = {
  useCallback: React.useCallback,
  useFaceAwareLayout,
  useFaceAvoidance,
  useZoomCompensation,
  computeOverlap,
  useStyle,
  SPRING_GENTLE,
  SPRING_BOUNCY,
  SPRING_SNAPPY,
  SPRING_SMOOTH,
  SPRING_ELASTIC,
  SPRING_WOBBLY,
  gentleSpring,
  bouncySpring,
  snappySpring,
  smoothSpring,
  elasticSpring,
  wobblySpring,
  polarToCartesian,
  describeArc,
  generateTicks,
  linearScale,
  colorWithOpacity,
  lerpColor,
};
