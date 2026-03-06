import React from "react";
import { AnimatedTitle } from "./AnimatedTitle";
import { LowerThird } from "./LowerThird";
import { Listicle } from "./Listicle";
import { DataAnimation } from "./DataAnimation";

type ComponentMap = {
  [key: string]: React.FC<any>;
};

export const ComponentRegistry: ComponentMap = {
  animated_title: AnimatedTitle as React.FC<any>,
  lower_third: LowerThird as React.FC<any>,
  listicle: Listicle as React.FC<any>,
  data_animation: DataAnimation as React.FC<any>,
};
