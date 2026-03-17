import React, { useEffect, useRef } from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";

interface GridPatternProps {
  spacing?: number;
  dotRadius?: number;
  speed?: number;
  color?: string;
  backgroundColor?: string;
}

export const GridPattern: React.FC<GridPatternProps> = ({
  spacing = 40,
  dotRadius = 2,
  speed = 0.3,
  color = "#6366f1",
  backgroundColor = "#0f0f23",
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

    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Grid with subtle perspective shift
    const centerX = width / 2;
    const centerY = height / 2;
    const perspectiveShift = Math.sin(t * 0.5) * 0.15;

    const cols = Math.ceil(width / spacing) + 2;
    const rows = Math.ceil(height / spacing) + 2;

    for (let row = -1; row < rows; row++) {
      for (let col = -1; col < cols; col++) {
        const baseX = col * spacing;
        const baseY = row * spacing;

        // Apply subtle wave displacement
        const dx = Math.sin(baseY * 0.01 + t * 2) * 5;
        const dy = Math.cos(baseX * 0.01 + t * 1.5) * 5;

        const x = baseX + dx;
        const y = baseY + dy;

        // Distance-based opacity (brighter near center)
        const distX = (x - centerX) / width;
        const distY = (y - centerY) / height;
        const dist = Math.sqrt(distX * distX + distY * distY);
        const alpha = Math.max(0.1, 1 - dist * 1.5);

        // Perspective scale
        const scale = 1 + perspectiveShift * (1 - dist);
        const r = dotRadius * scale;

        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = color + Math.round(alpha * 255).toString(16).padStart(2, "0");
        ctx.fill();
      }
    }

    // Draw faint grid lines
    ctx.strokeStyle = color + "15";
    ctx.lineWidth = 0.5;
    for (let col = 0; col < cols; col++) {
      const x = col * spacing + Math.sin(t * 2) * 3;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let row = 0; row < rows; row++) {
      const y = row * spacing + Math.cos(t * 1.5) * 3;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }, [frame, width, height, fps, spacing, dotRadius, speed, color, backgroundColor]);

  return (
    <AbsoluteFill>
      <canvas ref={canvasRef} width={width} height={height} style={{ width: "100%", height: "100%" }} />
    </AbsoluteFill>
  );
};
