import cv2
import numpy as np


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes → BGR numpy array."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Make sure it's a valid JPG/PNG.")
    return image


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.
    Returns annotated image as BGR numpy array.
    """
    img = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = det["label"]
        conf = det["confidence"]

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 128), 2)

        # Label background
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 128), -1)

        # Label text
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Encode BGR numpy array → JPEG bytes."""
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()
