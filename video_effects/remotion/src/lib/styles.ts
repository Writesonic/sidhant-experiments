import React, { createContext, useContext } from "react";
import type { StyleConfig } from "../types";

const DEFAULT_STYLE: StyleConfig = {
  font_family: "sans-serif",
  font_import: "",
  font_weights_to_load: ["400", "700"],
  palette: ["#FFFFFF", "#000000", "#FFD700"],
  text_shadow: "0 2px 8px rgba(0,0,0,0.6)",
  font_weights: {
    heading: "700",
    body: "400",
    emphasis: "800",
    marker: "700",
  },
};

const StyleContext = createContext<StyleConfig>(DEFAULT_STYLE);

export const StyleProvider: React.FC<{
  styleConfig?: StyleConfig;
  children: React.ReactNode;
}> = ({ styleConfig, children }) => {
  const value: StyleConfig = styleConfig
    ? {
        ...DEFAULT_STYLE,
        ...styleConfig,
        font_weights: { ...DEFAULT_STYLE.font_weights, ...styleConfig.font_weights },
        palette:
          styleConfig.palette && styleConfig.palette.length >= 3
            ? styleConfig.palette
            : DEFAULT_STYLE.palette,
      }
    : DEFAULT_STYLE;
  return React.createElement(StyleContext.Provider, { value }, children);
};

export function useStyle(): StyleConfig {
  return useContext(StyleContext);
}
