export interface FaceFrame {
  cx: number;
  cy: number;
  fw: number;
  fh: number;
}

export interface NormalizedRect {
  x: number;
  y: number;
  w: number;
  h: number;
  label?: string;
}

export interface ComponentSpec {
  template: string;
  startFrame: number;
  durationInFrames: number;
  props: Record<string, unknown>;
  bounds: NormalizedRect;
  zIndex: number;
}

export interface FontWeights {
  heading: string;
  body: string;
  emphasis: string;
  marker: string;
}

export interface StyleConfig {
  font_family: string;
  font_import: string;
  font_weights_to_load: string[];
  palette: string[];
  text_shadow: string;
  font_weights: FontWeights;
}

export interface CompositionPlan {
  components: ComponentSpec[];
  colorPalette: string[];
  includeBaseVideo: boolean;
  baseVideoPath?: string;
  faceDataPath?: string;
  styleConfig?: StyleConfig;
  /** Passed from Python to size the composition dynamically. */
  durationInFrames?: number;
  fps?: number;
  width?: number;
  height?: number;
}

export interface AnimatedTitleProps {
  text: string;
  style: "fade" | "slide-in" | "typewriter" | "bounce";
  position: NormalizedRect;
  fontSize?: number;
  color?: string;
  fontWeight?: string;
}

export interface LowerThirdProps {
  name: string;
  title?: string;
  accentColor?: string;
  style?: "slide" | "fade";
  position: NormalizedRect;
  fontSize?: number;
  color?: string;
}

export interface ListicleProps {
  items: string[];
  style?: "pop" | "slide";
  listStyle?: "numbered" | "bullet" | "none";
  position: NormalizedRect;
  staggerDelay?: number;
  fontSize?: number;
  color?: string;
  accentColor?: string;
}

export interface DataAnimationProps {
  style: "counter" | "stat-callout" | "bar";
  value: number;
  label: string;
  position: NormalizedRect;
  startValue?: number;
  suffix?: string;
  prefix?: string;
  delta?: number;
  items?: { label: string; value: number }[];
  fontSize?: number;
  color?: string;
  accentColor?: string;
}
