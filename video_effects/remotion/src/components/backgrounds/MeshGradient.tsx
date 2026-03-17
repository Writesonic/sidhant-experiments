import React, { useEffect, useRef } from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";

interface MeshGradientProps {
  colors?: string[];
  speed?: number;
  backgroundColor?: string;
}

export const MeshGradient: React.FC<MeshGradientProps> = ({
  colors = ["#ff6b6b", "#feca57", "#48dbfb", "#ff9ff3", "#54a0ff"],
  speed = 0.4,
  backgroundColor = "#1a1a2e",
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

    // Animated gradient blobs
    const points = colors.map((color, i) => {
      const angle = (i / colors.length) * Math.PI * 2 + t * 0.5;
      const radiusX = width * 0.25;
      const radiusY = height * 0.25;
      const cx = width / 2 + Math.cos(angle + i * 1.3) * radiusX;
      const cy = height / 2 + Math.sin(angle * 0.8 + i * 0.9) * radiusY;
      const blobRadius = Math.min(width, height) * (0.3 + 0.1 * Math.sin(t + i * 2));
      return { cx, cy, radius: blobRadius, color };
    });

    // Draw each blob as a radial gradient with soft edges
    ctx.globalCompositeOperation = "screen";
    for (const point of points) {
      const gradient = ctx.createRadialGradient(
        point.cx, point.cy, 0,
        point.cx, point.cy, point.radius
      );
      gradient.addColorStop(0, point.color + "80");
      gradient.addColorStop(0.5, point.color + "30");
      gradient.addColorStop(1, "transparent");

      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
    }
    ctx.globalCompositeOperation = "source-over";
  }, [frame, width, height, fps, colors, speed, backgroundColor]);

  return (
    <AbsoluteFill>
      <canvas ref={canvasRef} width={width} height={height} style={{ width: "100%", height: "100%" }} />
    </AbsoluteFill>
  );
};
