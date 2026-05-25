"use client";
import { useEffect, useRef } from "react";

interface Detection {
  label: string;
  confidence: number;
  box: [number, number, number, number];
}

interface Props {
  imageUrl: string;
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
}

export default function DetectionCanvas({ imageUrl, detections, imageWidth, imageHeight }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      // Fit image inside a max 800px wide container
      const maxW = Math.min(800, img.naturalWidth);
      const scale = maxW / img.naturalWidth;
      canvas.width = img.naturalWidth * scale;
      canvas.height = img.naturalHeight * scale;

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Scale factor from original image coords → canvas coords
      const sx = canvas.width / imageWidth;
      const sy = canvas.height / imageHeight;

      detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.box;
        const cx1 = x1 * sx, cy1 = y1 * sy;
        const cw = (x2 - x1) * sx, ch = (y2 - y1) * sy;

        // Box
        ctx.strokeStyle = "#00ff88";
        ctx.lineWidth = 2;
        ctx.strokeRect(cx1, cy1, cw, ch);

        // Label background
        const label = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 13px 'JetBrains Mono', monospace";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "#00ff88";
        ctx.fillRect(cx1, cy1 - 22, tw + 10, 22);

        // Label text
        ctx.fillStyle = "#0a0a0f";
        ctx.fillText(label, cx1 + 5, cy1 - 6);
      });
    };
  }, [imageUrl, detections, imageWidth, imageHeight]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        borderRadius: "10px",
        maxWidth: "100%",
        display: "block",
        border: "1px solid var(--border)",
      }}
    />
  );
}
