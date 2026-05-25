"use client";

interface Detection {
  label: string;
  confidence: number;
  box: [number, number, number, number];
}

interface Props {
  inferenceMs: number | null;
  modelUsed: string | null;
  detections: Detection[];
  imageWidth: number | null;
  imageHeight: number | null;
}

export default function StatsPanel({ inferenceMs, modelUsed, detections, imageWidth, imageHeight }: Props) {
  if (inferenceMs === null) return null;

  const labelCounts: Record<string, number> = {};
  detections.forEach((d) => {
    labelCounts[d.label] = (labelCounts[d.label] || 0) + 1;
  });

  const statStyle = {
    display: "flex",
    flexDirection: "column" as const,
    gap: "2px",
  };

  const labelStyle = {
    fontSize: "0.65rem",
    color: "var(--text-muted)",
    textTransform: "uppercase" as const,
    letterSpacing: "0.1em",
  };

  const valueStyle = {
    fontSize: "1.1rem",
    color: "var(--accent)",
    fontWeight: 600,
  };

  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: "10px",
      padding: "20px 24px",
      display: "flex",
      flexDirection: "column",
      gap: "20px",
    }}>
      {/* Top stats row */}
      <div style={{ display: "flex", gap: "32px", flexWrap: "wrap" }}>
        <div style={statStyle}>
          <span style={labelStyle}>inference</span>
          <span style={valueStyle}>{inferenceMs} ms</span>
        </div>
        <div style={statStyle}>
          <span style={labelStyle}>model</span>
          <span style={valueStyle}>{modelUsed}</span>
        </div>
        <div style={statStyle}>
          <span style={labelStyle}>detections</span>
          <span style={valueStyle}>{detections.length}</span>
        </div>
        <div style={statStyle}>
          <span style={labelStyle}>resolution</span>
          <span style={valueStyle}>{imageWidth}×{imageHeight}</span>
        </div>
      </div>

      {/* Detections list */}
      {detections.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          <span style={labelStyle}>detected objects</span>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "4px" }}>
            {Object.entries(labelCounts).map(([label, count]) => (
              <span
                key={label}
                style={{
                  padding: "3px 10px",
                  borderRadius: "4px",
                  border: "1px solid var(--accent)",
                  color: "var(--accent)",
                  fontSize: "0.75rem",
                  background: "var(--accent-dim)",
                }}
              >
                {label} {count > 1 ? `×${count}` : ""}
              </span>
            ))}
          </div>
        </div>
      )}

      {detections.length === 0 && (
        <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
          no objects detected above confidence threshold
        </span>
      )}
    </div>
  );
}
