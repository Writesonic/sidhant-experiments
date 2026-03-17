import React, { useEffect, useRef } from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";

interface ParticlesProps {
  count?: number;
  speed?: number;
  connectionDistance?: number;
  colors?: string[];
  backgroundColor?: string;
}

export const Particles: React.FC<ParticlesProps> = ({
  count = 80,
  speed = 0.5,
  connectionDistance = 150,
  colors = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"],
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

    const t = frame / fps;

    // Deterministic particle positions seeded by index
    const particles = Array.from({ length: count }, (_, i) => {
      const seed1 = Math.sin(i * 127.1 + 311.7) * 43758.5453;
      const seed2 = Math.sin(i * 269.5 + 183.3) * 43758.5453;
      const vxSeed = Math.sin(i * 419.2 + 71.9) * 43758.5453;
      const vySeed = Math.sin(i * 563.7 + 29.1) * 43758.5453;

      const baseX = (seed1 - Math.floor(seed1)) * width;
      const baseY = (seed2 - Math.floor(seed2)) * height;
      const vx = ((vxSeed - Math.floor(vxSeed)) - 0.5) * speed * 60;
      const vy = ((vySeed - Math.floor(vySeed)) - 0.5) * speed * 60;

      return {
        x: ((baseX + vx * t) % width + width) % width,
        y: ((baseY + vy * t) % height + height) % height,
        color: colors[i % colors.length],
        radius: 2 + (seed1 - Math.floor(seed1)) * 3,
      };
    });

    // Clear
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Draw connections
    ctx.lineWidth = 0.5;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < connectionDistance) {
          const alpha = 1 - dist / connectionDistance;
          ctx.strokeStyle = `rgba(139, 92, 246, ${alpha * 0.3})`;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw particles
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();
    }
  }, [frame, width, height, fps, count, speed, connectionDistance, colors, backgroundColor]);

  return (
    <AbsoluteFill>
      <canvas ref={canvasRef} width={width} height={height} style={{ width: "100%", height: "100%" }} />
    </AbsoluteFill>
  );
};
