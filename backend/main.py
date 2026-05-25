from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import json
import base64
import numpy as np

from schemas import DetectionResponse, Detection, HealthResponse
from inference import run_inference, MODEL_PATHS
from utils import decode_image, draw_detections, image_to_jpeg_bytes

app = FastAPI(
    title="VisionDrop API",
    description="Real-time object detection using YOLOv8 + ONNX Runtime",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    available = [name for name, path in MODEL_PATHS.items() if path.exists()]
    return HealthResponse(
        status="ok",
        models_available=available,
        device="CPU (ONNX Runtime)",
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    model: str = Form(default="yolov8n"),
    conf_threshold: float = Form(default=0.25),
):
    if model not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'.")

    raw = await file.read()
    try:
        image = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    detections, inference_time_ms = run_inference(image, model_name=model, conf_threshold=conf_threshold)
    h, w = image.shape[:2]

    return DetectionResponse(
        model_used=model,
        inference_time_ms=inference_time_ms,
        image_width=w,
        image_height=h,
        detections=[Detection(**d) for d in detections],
    )


@app.post("/detect/annotated")
async def detect_annotated(
    file: UploadFile = File(...),
    model: str = Form(default="yolov8n"),
    conf_threshold: float = Form(default=0.25),
):
    if model not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'.")

    raw = await file.read()
    try:
        image = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    detections, _ = run_inference(image, model_name=model, conf_threshold=conf_threshold)
    annotated = draw_detections(image, detections)
    jpeg_bytes = image_to_jpeg_bytes(annotated)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for live webcam detection.
    Client sends JSON: { "frame": "<base64 jpeg>", "model": "yolov8n", "conf": 0.25 }
    Server replies JSON: { "detections": [...], "inference_time_ms": float }
    """
    await websocket.accept()
    try:
        while True:
            raw_text = await websocket.receive_text()
            payload = json.loads(raw_text)

            frame_b64 = payload.get("frame", "")
            model = payload.get("model", "yolov8n")
            conf = float(payload.get("conf", 0.25))

            # Strip data URL prefix if present
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]

            image_bytes = base64.b64decode(frame_b64)

            try:
                image = decode_image(image_bytes)
            except ValueError:
                await websocket.send_text(json.dumps({"error": "bad frame"}))
                continue

            detections, inference_time_ms = run_inference(
                image, model_name=model, conf_threshold=conf
            )

            await websocket.send_text(json.dumps({
                "detections": detections,
                "inference_time_ms": inference_time_ms,
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
