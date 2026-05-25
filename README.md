# VisionDrop

Real-time object detection platform powered by YOLOv8 + ONNX Runtime.

Upload an image or stream your webcam — get bounding boxes, confidence scores, and inference stats instantly.

## Stack

- **Model**: YOLOv8n / YOLOv8s exported to ONNX
- **Backend**: FastAPI + ONNX Runtime (async, CPU inference)
- **Frontend**: Next.js + TypeScript
- **Realtime**: WebSockets (webcam streaming)
- **Infra**: Docker + Docker Compose

## Benchmark

Tested on CPU — ONNX Runtime (`ORT_ENABLE_ALL`, 4 threads), 15 runs per image.

| Model    | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Std (ms) | FPS   |
|----------|-----------|-------------|----------|----------|----------|-------|
| yolov8n  | 32.5      | 32.0        | 30.3     | 36.8     | 1.6      | 30.8  |
| yolov8s  | 76.5      | 75.8        | 72.6     | 87.0     | 3.0      | 13.1  |

`yolov8s` is ~136% slower than `yolov8n` on CPU. For real-time webcam use, `yolov8n` comfortably holds 30 FPS.

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/visiondrop
cd visiondrop
docker compose up --build
```

Open `http://localhost:3000`

## Run without Docker

```bash
# Backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

## Project structure

```
visiondrop/
├── backend/
│   ├── main.py          # FastAPI app + WebSocket endpoint
│   ├── inference.py     # ONNX session management, pre/post processing
│   ├── utils.py         # Image decode, annotation, encoding
│   ├── schemas.py       # Pydantic models
│   ├── benchmark.py     # Benchmark script
│   └── models/          # .onnx model files
└── frontend/
    └── app/
        ├── page.tsx
        └── components/
            ├── DropZone.tsx
            ├── Canvas.tsx
            ├── Webcam.tsx
            ├── ModelSwitcher.tsx
            ├── StatsPanel.tsx
            └── LatencyGraph.tsx
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Model status + device info |
| `/detect` | POST | Image upload → JSON detections |
| `/detect/annotated` | POST | Image upload → annotated JPEG |
| `/ws/stream` | WS | Webcam frame stream → live detections |
