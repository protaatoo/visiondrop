"""
VisionDrop — Benchmark Script
Runs both YOLOv8 models against all images in benchmark_images/
and prints a markdown table of results.

Usage:
    python benchmark.py
    python benchmark.py --runs 20        # more runs = more accurate average
    python benchmark.py --conf 0.3
"""

import argparse
import time
import statistics
from pathlib import Path
import cv2
import numpy as np

# Make sure we can import from the backend
import sys
sys.path.insert(0, str(Path(__file__).parent))

from inference import run_inference, get_session, preprocess, MODEL_PATHS

IMAGES_DIR = Path(__file__).parent / "benchmark_images"
MODELS = ["yolov8n", "yolov8s"]


def benchmark_model(model_name: str, images: list, runs: int, conf: float):
    """Run inference N times per image, collect timing stats."""
    print(f"\n  warming up {model_name}...", end="", flush=True)

    # Warm-up: run once to load model into memory before timing
    get_session(model_name)
    _, _ = run_inference(images[0][1], model_name=model_name, conf_threshold=conf)
    print(" done")

    all_times = []
    total_detections = 0

    for img_path, image in images:
        times = []
        detections = []
        for _ in range(runs):
            dets, ms = run_inference(image, model_name=model_name, conf_threshold=conf)
            times.append(ms)
            detections = dets

        all_times.extend(times)
        total_detections += len(detections)

        avg = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0
        print(f"    {img_path.name:<20} avg={avg:.1f}ms  std=±{std:.1f}ms  detections={len(detections)}")

    return all_times, total_detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=15, help="Inference runs per image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    image_paths = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}. Add some JPG/PNG files and retry.")
        return

    print(f"\nVisionDrop Benchmark")
    print(f"{'='*50}")
    print(f"Images   : {len(image_paths)}")
    print(f"Runs/img : {args.runs}")
    print(f"Conf     : {args.conf}")
    print(f"Models   : {', '.join(MODELS)}")
    print(f"{'='*50}")

    # Load all images once
    loaded = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            loaded.append((p, img))
        else:
            print(f"  warning: could not load {p.name}")

    results = {}
    for model_name in MODELS:
        if not MODEL_PATHS[model_name].exists():
            print(f"\n  skipping {model_name} — model file not found")
            continue
        print(f"\n[{model_name}]")
        times, total_dets = benchmark_model(model_name, loaded, args.runs, args.conf)
        results[model_name] = {
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "throughput_fps": round(1000 / statistics.mean(times), 1),
            "total_detections": total_dets,
        }

    # Print markdown table
    print(f"\n\n{'='*50}")
    print("RESULTS (copy this into your README)\n")
    print("| Model    | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Std (ms) | FPS   |")
    print("|----------|-----------|-------------|----------|----------|----------|-------|")
    for model_name, r in results.items():
        print(
            f"| {model_name:<8} "
            f"| {r['mean_ms']:<9.1f} "
            f"| {r['median_ms']:<11.1f} "
            f"| {r['min_ms']:<8.1f} "
            f"| {r['max_ms']:<8.1f} "
            f"| {r['std_ms']:<8.1f} "
            f"| {r['throughput_fps']:<5} |"
        )

    if len(results) == 2:
        n = results["yolov8n"]["mean_ms"]
        s = results["yolov8s"]["mean_ms"]
        overhead = ((s - n) / n) * 100
        print(f"\nyolov8s is {overhead:.0f}% slower than yolov8n on average (CPU, ONNX Runtime)")

    print(f"\nDevice: CPU — ONNX Runtime (ORT_ENABLE_ALL, 4 threads)")


if __name__ == "__main__":
    main()
