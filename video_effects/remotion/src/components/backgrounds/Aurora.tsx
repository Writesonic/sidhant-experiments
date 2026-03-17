import React, { useEffect, useRef } from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";

interface AuroraProps {
  colors?: string[];
  speed?: number;
  bands?: number;
  backgroundColor?: string;
}

export const Aurora: React.FC<AuroraProps> = ({
  colors = ["#00ff88", "#0088ff", "#8800ff", "#ff0088"],
  speed = 0.3,
  bands = 5,
  backgroundColor = "#0a0a1a",
}) => {
  const frame = useCurrentFrame();
  const { width, height, fps } = useVideoConfig();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const t = (frame / fps) * speed;

    // Dark background
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Draw aurora bands using sine waves + gradient fills
    for (let b = 0; b < bands; b++) {
      const color = colors[b % colors.length];
      const phaseOffset = (b / bands) * Math.PI * 2;
      const yCenter = height * 0.25 + (b / bands) * height * 0.35;

      ctx.beginPath();
      ctx.moveTo(0, height);

      for (let x = 0; x <= width; x += 4) {
        const xNorm = x / width;
        const wave1 = Math.sin(xNorm * 3 + t * 2 + phaseOffset) * 60;
        const wave2 = Math.sin(xNorm * 5 + t * 1.3 + phaseOffset * 0.7) * 30;
        const wave3 = Math.sin(xNorm * 1.5 + t * 0.7 + phaseOffset * 1.5) * 80;
        const y = yCenter + wave1 + wave2 + wave3;
        ctx.lineTo(x, y);
      }

      ctx.lineTo(width, height);
      ctx.closePath();

      // Gradient fill for each band
      const gradient = ctx.createLinearGradient(0, yCenter - 100, 0, yCenter + 200);
      gradient.addColorStop(0, "transparent");
      gradient.addColorStop(0.3, color + "40");
      gradient.addColorStop(0.5, color + "20");
      gradient.addColorStop(1, "transparent");
      ctx.fillStyle = gradient;
      ctx.fill();
    }
  }, [frame, width, height, fps, colors, speed, bands, backgroundColor]);

  return (
    <AbsoluteFill>
      <canvas ref={canvasRef} width={width} height={height} style={{ width: "100%", height: "100%" }} />
    </AbsoluteFill>
  );
};
