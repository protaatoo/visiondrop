"use client";
import { useEffect, useRef, useState, useCallback } from "react";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "";
const FRAME_INTERVAL_MS = 100; // send a frame every 100ms = ~10fps

interface Detection {
  label: string;
  confidence: number;
  box: [number, number, number, number];
}

interface Props {
  model: "yolov8n" | "yolov8s" | "yolov8n_ppe";
  confThreshold: number;
  onStats: (ms: number, count: number) => void;
}

export default function WebcamStream({ model, confThreshold, onStats }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const detectionsRef = useRef<Detection[]>([]);

  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const fpsCounterRef = useRef(0);

  // Draw bounding boxes on overlay canvas
  const drawDetections = useCallback((detections: Detection[]) => {
    const overlay = overlayRef.current;
    const video = videoRef.current;
    if (!overlay || !video) return;

    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;

    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.box;
      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = "bold 13px 'JetBrains Mono', monospace";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "#00ff88";
      ctx.fillRect(x1, y1 - 22, tw + 10, 22);
      ctx.fillStyle = "#0a0a0f";
      ctx.fillText(label, x1 + 5, y1 - 6);
    });
  }, []);

  const startStream = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch {
      setError("Camera access denied. Please allow camera permissions.");
      return;
    }

    const ws = new WebSocket(`${WS_URL}/ws/stream`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) return;
      detectionsRef.current = data.detections ?? [];
      drawDetections(detectionsRef.current);
      onStats(data.inference_time_ms, detectionsRef.current.length);
      fpsCounterRef.current += 1;
    };

    ws.onerror = () => setError("WebSocket error — is the backend running?");

    // Wait for WS to open before sending frames
    ws.onopen = () => {
      intervalRef.current = setInterval(() => {
        if (!canvasRef.current || !videoRef.current || ws.readyState !== WebSocket.OPEN) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        canvas.width = videoRef.current.videoWidth || 640;
        canvas.height = videoRef.current.videoHeight || 480;
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

        const frame = canvas.toDataURL("image/jpeg", 0.7);
        ws.send(JSON.stringify({ frame, model, conf: confThreshold }));
      }, FRAME_INTERVAL_MS);
    };

    setActive(true);
  }, [model, confThreshold, drawDetections, onStats]);

  const stopStream = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (wsRef.current) wsRef.current.close();
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    const overlay = overlayRef.current;
    if (overlay) overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
    setActive(false);
    setFps(0);
  }, []);

  // FPS counter — update every second
  useEffect(() => {
    const id = setInterval(() => {
      setFps(fpsCounterRef.current);
      fpsCounterRef.current = 0;
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // Cleanup on unmount
  useEffect(() => () => stopStream(), [stopStream]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      {/* Video + overlay stack */}
      <div style={{ position: "relative", display: "inline-block", lineHeight: 0 }}>
        <video
          ref={videoRef}
          muted
          playsInline
          style={{
            borderRadius: "10px",
            border: "1px solid var(--border)",
            maxWidth: "100%",
            display: active ? "block" : "none",
          }}
        />
        <canvas
          ref={overlayRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            borderRadius: "10px",
            maxWidth: "100%",
            pointerEvents: "none",
            display: active ? "block" : "none",
          }}
        />
        {/* Hidden canvas used to capture frames */}
        <canvas ref={canvasRef} style={{ display: "none" }} />

        {/* FPS badge */}
        {active && (
          <div style={{
            position: "absolute",
            top: "10px",
            right: "10px",
            background: "#0a0a0fcc",
            border: "1px solid var(--accent)",
            color: "var(--accent)",
            padding: "3px 10px",
            borderRadius: "4px",
            fontSize: "0.72rem",
            fontFamily: "inherit",
          }}>
            {fps} fps
          </div>
        )}

        {/* Placeholder when inactive */}
        {!active && (
          <div style={{
            width: "100%",
            minHeight: "300px",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: "10px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--text-muted)",
            fontSize: "0.85rem",
          }}>
            webcam inactive
          </div>
        )}
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: "10px" }}>
        {!active ? (
          <button
            onClick={startStream}
            style={{
              padding: "10px 28px",
              borderRadius: "6px",
              border: "none",
              background: "var(--accent)",
              color: "#0a0a0f",
              fontFamily: "inherit",
              fontWeight: 600,
              fontSize: "0.85rem",
              cursor: "pointer",
            }}
          >
            start webcam →
          </button>
        ) : (
          <button
            onClick={stopStream}
            style={{
              padding: "10px 28px",
              borderRadius: "6px",
              border: "1px solid var(--danger)",
              background: "transparent",
              color: "var(--danger)",
              fontFamily: "inherit",
              fontSize: "0.85rem",
              cursor: "pointer",
            }}
          >
            stop
          </button>
        )}
      </div>

      {error && (
        <div style={{
          padding: "12px 16px",
          borderRadius: "6px",
          border: "1px solid var(--danger)",
          color: "var(--danger)",
          fontSize: "0.8rem",
        }}>
          ⚠ {error}
        </div>
      )}
    </div>
  );
}
