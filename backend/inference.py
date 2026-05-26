import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Per-model class definitions
MODEL_CLASSES = {
    "yolov8n": COCO_CLASSES,
    "yolov8s": COCO_CLASSES,
    "yolov8n_ppe": ["head", "helmet", "person"],
}

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATHS = {
    "yolov8n":     MODELS_DIR / "yolov8n.onnx",
    "yolov8s":     MODELS_DIR / "yolov8s.onnx",
    "yolov8n_ppe": MODELS_DIR / "yolov8n_ppe.onnx",
}

# Filter out models whose files don't exist
MODEL_PATHS = {k: v for k, v in MODEL_PATHS.items() if v.exists()}

_sessions: dict[str, ort.InferenceSession] = {}


def get_session(model_name: str) -> ort.InferenceSession:
    if model_name not in _sessions:
        path = MODEL_PATHS.get(model_name)
        if path is None:
            raise ValueError(f"Model '{model_name}' not found.")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        _sessions[model_name] = ort.InferenceSession(
            str(path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    return _sessions[model_name]


def preprocess(image: np.ndarray, input_size: int = 640):
    import cv2
    h, w = image.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, pad_x, pad_y


def postprocess(output, scale, pad_x, pad_y, orig_w, orig_h,
                class_names, conf_threshold=0.25, iou_threshold=0.45):
    predictions = output[0].T  # (8400, 4 + nc)
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]

    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = confidences >= conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    x1 = np.clip((x1 - pad_x) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_y) / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_x) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_y) / scale, 0, orig_h)

    keep = _nms(x1, y1, x2, y2, confidences, iou_threshold)

    results = []
    for i in keep:
        label = class_names[class_ids[i]] if class_ids[i] < len(class_names) else "unknown"
        results.append({
            "label": label,
            "confidence": round(float(confidences[i]), 4),
            "box": [round(float(x1[i]), 1), round(float(y1[i]), 1),
                    round(float(x2[i]), 1), round(float(y2[i]), 1)],
        })
    return results


def _nms(x1, y1, x2, y2, scores, iou_threshold):
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def run_inference(image: np.ndarray, model_name: str = "yolov8n",
                  conf_threshold: float = 0.25):
    session = get_session(model_name)
    class_names = MODEL_CLASSES.get(model_name, COCO_CLASSES)
    orig_h, orig_w = image.shape[:2]
    blob, scale, pad_x, pad_y = preprocess(image)
    input_name = session.get_inputs()[0].name
    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: blob})
    t1 = time.perf_counter()
    inference_time_ms = round((t1 - t0) * 1000, 2)
    detections = postprocess(
        outputs[0], scale, pad_x, pad_y, orig_w, orig_h,
        class_names, conf_threshold
    )
    return detections, inference_time_ms
