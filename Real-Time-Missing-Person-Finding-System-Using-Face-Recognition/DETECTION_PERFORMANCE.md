# Detection Performance Evaluation (test1_dt.py)

This report explains and evaluates the face detection performance captured during a webcam run of `src/detection/test1_dt.py`. The terminal output shows YOLOv8-Face detection at 384x640 with per-image timing.

## Context
- Detector: YOLOv8-Face via `ultralytics`
- Input: Webcam at 384x640
- Device: CPU (per log)
- Tracker: Deep SORT (used in the script for track IDs)
- Logging: Console shows per-frame detection results and speed breakdowns

## Observed Output Pattern
- Example detection line:
  - `0: 384x640 2 faces, 39.9ms`
- Example speed breakdown line:
  - `Speed: 1.4ms preprocess, 39.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)`
- Typical face counts: Mostly 2 faces, occasionally 1 face
- Typical inference time (CPU, 384x640): ~33–45ms
- Occasional spikes: ~48–58ms
- Preprocess: ~1.1–2.4ms
- Postprocess: ~0.5–1.2ms

## Summary Metrics (from the provided run)
- Resolution: 384x640
- Faces per frame: commonly 2; occasionally 1
- Inference time range: ~25–58ms (most frames ~33–41ms)
- Preprocess time: ~1–2ms
- Postprocess time: ~0.5–1.2ms
- Estimated detector throughput on CPU:
  - Using typical end-to-end detection time per frame ≈ preprocess + inference + postprocess ≈ 1.5ms + 36ms + 0.7ms ≈ 38ms
  - Approx FPS ≈ 1000 / 38 ≈ 26 FPS for detector-only (real pipeline FPS will be lower due to tracking, drawing, I/O, and Python loop overhead)

## Performance Interpretation
- The inference time is the dominant cost on CPU. At ~36–40ms per frame, the detector itself can process roughly 25–28 frames per second. The overall app frame rate will be lower since tracking, rendering, and other logic also take time.
- Occasional spikes (~48–58ms) suggest transient CPU scheduling or model overhead. These are normal for live video with varying content and system background load.
- Preprocess and postprocess are small contributors (< 10% combined), meaning most improvements must target inference time.

## Recommendations
- Use GPU (CUDA) if available; expect a 2–5x speedup vs CPU, depending on the GPU and model size.
- Reduce input resolution (e.g., 320x576) to lower inference time at the cost of detection accuracy.
- Adjust YOLO model size (`yolov8n-face.pt` is already the smallest; ensure the correct face-specific model is used).
- Batch frames only if using a multi-frame detection strategy (not typical for live detection).
- Tune confidence threshold to reduce postprocess load if too many false positives occur.

## How to Reproduce
- Run the test:
  ```
  python -m src.detection.test1_dt
  ```
- Press `q` to exit. Cropped face images are saved to `.save/` with `track_id_timestamp.jpg`.
- Logs are written to `runs/test_logs/test_<timestamp>.log` (script logger). Note: The YOLO per-frame speed lines appear on console and may not be in the file log by default.

## Notes
- The reported numbers are specific to 384x640 on CPU for the sample environment. Hardware differences, background processes, and camera behavior will affect results.

