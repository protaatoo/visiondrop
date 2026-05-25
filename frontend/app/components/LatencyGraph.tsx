"use client";
import { useEffect, useRef } from "react";

interface Props {
  history: number[]; // last N inference times in ms
}

const MAX_POINTS = 30;
const W = 300;
const H = 60;
const PAD = 6;

export default function LatencyGraph({ history }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, W, H);

    if (history.length < 2) return;

    const points = history.slice(-MAX_POINTS);
    const min = Math.min(...points);
    const max = Math.max(...points) || 1;
    const range = max - min || 1;

    const stepX = (W - PAD * 2) / (MAX_POINTS - 1);

    // Grid lines
    ctx.strokeStyle = "#1e1e2e";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = PAD + ((H - PAD * 2) / 3) * i;
      ctx.beginPath();
      ctx.moveTo(PAD, y);
      ctx.lineTo(W - PAD, y);
      ctx.stroke();
    }

    // Fill under line
    ctx.beginPath();
    points.forEach((val, i) => {
      const x = PAD + i * stepX;
      const y = PAD + (1 - (val - min) / range) * (H - PAD * 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    const lastX = PAD + (points.length - 1) * stepX;
    ctx.lineTo(lastX, H - PAD);
    ctx.lineTo(PAD, H - PAD);
    ctx.closePath();
    ctx.fillStyle = "#00ff8818";
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    points.forEach((val, i) => {
      const x = PAD + i * stepX;
      const y = PAD + (1 - (val - min) / range) * (H - PAD * 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Last point dot
    const last = points[points.length - 1];
    const lx = PAD + (points.length - 1) * stepX;
    const ly = PAD + (1 - (last - min) / range) * (H - PAD * 2);
    ctx.beginPath();
    ctx.arc(lx, ly, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#00ff88";
    ctx.fill();

  }, [history]);

  const last = history[history.length - 1];
  const avg = history.length
    ? Math.round(history.slice(-MAX_POINTS).reduce((a, b) => a + b, 0) / Math.min(history.length, MAX_POINTS))
    : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
          latency (last {MAX_POINTS} frames)
        </span>
        <div style={{ display: "flex", gap: "16px" }}>
          <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
            now <span style={{ color: "var(--accent)" }}>{last?.toFixed(0)}ms</span>
          </span>
          <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
            avg <span style={{ color: "var(--accent)" }}>{avg}ms</span>
          </span>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        style={{
          borderRadius: "6px",
          border: "1px solid var(--border)",
          background: "var(--surface)",
          width: "100%",
          height: `${H}px`,
        }}
      />
    </div>
  );
}
