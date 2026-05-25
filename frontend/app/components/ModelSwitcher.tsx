"use client";

interface Props {
  model: "yolov8n" | "yolov8s";
  onChange: (model: "yolov8n" | "yolov8s") => void;
  disabled?: boolean;
}

const MODELS = [
  { id: "yolov8n", label: "yolov8n", desc: "fast" },
  { id: "yolov8s", label: "yolov8s", desc: "accurate" },
] as const;

export default function ModelSwitcher({ model, onChange, disabled }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
        model
      </span>
      <div style={{ display: "flex", gap: "8px" }}>
        {MODELS.map((m) => (
          <button
            key={m.id}
            onClick={() => onChange(m.id)}
            disabled={disabled}
            style={{
              padding: "8px 16px",
              borderRadius: "6px",
              border: `1px solid ${model === m.id ? "var(--accent)" : "var(--border)"}`,
              background: model === m.id ? "var(--accent-dim)" : "var(--surface)",
              color: model === m.id ? "var(--accent)" : "var(--text-muted)",
              cursor: disabled ? "not-allowed" : "pointer",
              fontSize: "0.8rem",
              fontFamily: "inherit",
              transition: "all 0.15s ease",
            }}
          >
            {m.label}
            <span style={{ marginLeft: "6px", fontSize: "0.65rem", opacity: 0.6 }}>
              [{m.desc}]
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
