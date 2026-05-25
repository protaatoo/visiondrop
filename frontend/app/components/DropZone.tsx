"use client";
import { useCallback, useState } from "react";

interface Props {
  onImageSelected: (file: File, previewUrl: string) => void;
  disabled?: boolean;
}

export default function DropZone({ onImageSelected, disabled }: Props) {
  const [dragging, setDragging] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      const url = URL.createObjectURL(file);
      onImageSelected(file, url);
    },
    [onImageSelected]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "12px",
        border: `2px dashed ${dragging ? "var(--accent)" : "var(--border)"}`,
        borderRadius: "12px",
        padding: "48px 32px",
        cursor: disabled ? "not-allowed" : "pointer",
        background: dragging ? "var(--accent-dim)" : "var(--surface)",
        transition: "all 0.2s ease",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <span style={{ fontSize: "2.5rem" }}>📂</span>
      <span style={{ color: "var(--text-muted)", fontSize: "0.85rem", textAlign: "center" }}>
        drag & drop an image or <span style={{ color: "var(--accent)" }}>click to browse</span>
      </span>
      <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
        JPG, PNG, WEBP supported
      </span>
      <input
        type="file"
        accept="image/*"
        onChange={onInputChange}
        disabled={disabled}
        style={{ display: "none" }}
      />
    </label>
  );
}
