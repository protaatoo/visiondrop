"use client";
import { useState, useCallback, useRef } from "react";
import DropZone from "./components/DropZone";
import DetectionCanvas from "./components/Canvas";
import ModelSwitcher from "./components/ModelSwitcher";
import StatsPanel from "./components/StatsPanel";
import WebcamStream from "./components/Webcam";
import LatencyGraph from "./components/LatencyGraph";

const API = process.env.NEXT_PUBLIC_API_URL || "";

interface Detection {
  label: string;
  confidence: number;
  box: [number, number, number, number];
}

interface DetectionResponse {
  model_used: string;
  inference_time_ms: number;
  image_width: number;
  image_height: number;
  detections: Detection[];
}

type Tab = "upload" | "webcam";

export default function Home() {
  const [tab, setTab] = useState<Tab>("upload");
  const [model, setModel] = useState<"yolov8n" | "yolov8s">("yolov8n");
  const [confThreshold, setConfThreshold] = useState(0.25);

  // Upload tab state
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Webcam tab state
  const [streamMs, setStreamMs] = useState<number | null>(null);
  const [streamCount, setStreamCount] = useState<number>(0);

  // Latency history — shared across both tabs
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);
  const latencyRef = useRef<number[]>([]);

  const pushLatency = (ms: number) => {
    latencyRef.current = [...latencyRef.current.slice(-59), ms];
    setLatencyHistory([...latencyRef.current]);
  };

  const handleImageSelected = useCallback((file: File, url: string) => {
    setImageFile(file);
    setPreviewUrl(url);
    setResult(null);
    setError(null);
  }, []);

  const handleDetect = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    const form = new FormData();
    form.append("file", imageFile);
    form.append("model", model);
    form.append("conf_threshold", String(confThreshold));
    try {
      const res = await fetch(`${API}/detect`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data: DetectionResponse = await res.json();
      setResult(data);
      pushLatency(data.inference_time_ms);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImageFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const tabStyle = (t: Tab) => ({
    padding: "8px 20px",
    borderRadius: "6px 6px 0 0",
    border: "1px solid var(--border)",
    borderBottom: tab === t ? "1px solid var(--bg)" : "1px solid var(--border)",
    background: tab === t ? "var(--bg)" : "var(--surface)",
    color: tab === t ? "var(--accent)" : "var(--text-muted)",
    fontFamily: "inherit",
    fontSize: "0.82rem",
    cursor: "pointer",
    marginBottom: "-1px",
    transition: "all 0.15s ease",
  });

  return (
    <main style={{
      minHeight: "100vh",
      background: "var(--bg)",
      padding: "40px 24px",
      maxWidth: "900px",
      margin: "0 auto",
    }}>
      {/* Header */}
      <div style={{ marginBottom: "36px" }}>
        <h1 style={{ fontSize: "2rem", fontWeight: 600, margin: 0, letterSpacing: "-0.02em" }}>
          <span style={{ color: "var(--accent)" }}>Vision</span>Drop
        </h1>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", margin: "6px 0 0" }}>
          YOLOv8 · ONNX Runtime · real-time object detection
        </p>
      </div>

      {/* Controls */}
      <div style={{
        display: "flex", gap: "32px", flexWrap: "wrap",
        alignItems: "flex-end", marginBottom: "28px",
      }}>
        <ModelSwitcher model={model} onChange={setModel} disabled={loading} />
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
            confidence — {(confThreshold * 100).toFixed(0)}%
          </span>
          <input
            type="range" min={0.1} max={0.9} step={0.05}
            value={confThreshold}
            onChange={(e) => setConfThreshold(Number(e.target.value))}
            disabled={loading}
            style={{ accentColor: "var(--accent)", width: "180px" }}
          />
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: "4px", borderBottom: "1px solid var(--border)", marginBottom: "24px" }}>
        <button style={tabStyle("upload")} onClick={() => setTab("upload")}>upload image</button>
        <button style={tabStyle("webcam")} onClick={() => setTab("webcam")}>live webcam</button>
      </div>

      {/* Upload Tab */}
      {tab === "upload" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {!previewUrl ? (
            <DropZone onImageSelected={handleImageSelected} disabled={loading} />
          ) : (
            <>
              <DetectionCanvas
                imageUrl={previewUrl}
                detections={result?.detections ?? []}
                imageWidth={result?.image_width ?? 640}
                imageHeight={result?.image_height ?? 640}
              />
              <div style={{ display: "flex", gap: "10px" }}>
                <button
                  onClick={handleDetect} disabled={loading}
                  style={{
                    padding: "10px 28px", borderRadius: "6px", border: "none",
                    background: loading ? "var(--border)" : "var(--accent)",
                    color: "#0a0a0f", fontFamily: "inherit", fontWeight: 600,
                    fontSize: "0.85rem", cursor: loading ? "not-allowed" : "pointer",
                    transition: "all 0.15s ease",
                  }}
                >
                  {loading ? "detecting..." : "detect →"}
                </button>
                <button
                  onClick={handleReset} disabled={loading}
                  style={{
                    padding: "10px 20px", borderRadius: "6px",
                    border: "1px solid var(--border)", background: "transparent",
                    color: "var(--text-muted)", fontFamily: "inherit", fontSize: "0.85rem",
                    cursor: loading ? "not-allowed" : "pointer",
                  }}
                >
                  reset
                </button>
              </div>
            </>
          )}

          {error && (
            <div style={{
              padding: "12px 16px", borderRadius: "6px",
              border: "1px solid var(--danger)", color: "var(--danger)", fontSize: "0.8rem",
            }}>⚠ {error}</div>
          )}

          {result && (
            <StatsPanel
              inferenceMs={result.inference_time_ms}
              modelUsed={result.model_used}
              detections={result.detections}
              imageWidth={result.image_width}
              imageHeight={result.image_height}
            />
          )}
        </div>
      )}

      {/* Webcam Tab */}
      {tab === "webcam" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <WebcamStream
            model={model}
            confThreshold={confThreshold}
            onStats={(ms, count) => {
              setStreamMs(ms);
              setStreamCount(count);
              pushLatency(ms);
            }}
          />
          {streamMs !== null && (
            <div style={{
              background: "var(--surface)", border: "1px solid var(--border)",
              borderRadius: "10px", padding: "16px 24px",
              display: "flex", gap: "32px", flexWrap: "wrap",
            }}>
              {[
                ["inference", `${streamMs} ms`],
                ["model", model],
                ["detections", String(streamCount)],
                ["device", "CPU · ONNX"],
              ].map(([label, value]) => (
                <div key={label} style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
                  <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>{label}</span>
                  <span style={{ fontSize: "1.1rem", color: "var(--accent)", fontWeight: 600 }}>{value}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Latency graph — shows up after first inference in either tab */}
      {latencyHistory.length >= 2 && (
        <div style={{ marginTop: "24px" }}>
          <LatencyGraph history={latencyHistory} />
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: "60px", color: "var(--text-muted)", fontSize: "0.7rem" }}>
        device: CPU · ONNX Runtime · YOLOv8 COCO (80 classes)
      </div>
    </main>
  );
}
