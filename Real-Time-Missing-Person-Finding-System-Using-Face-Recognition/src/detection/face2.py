


import cv2
import torch
import os
import sys
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path="yolov8n-face.pt", conf_thresh=0.3, device=None, imgsz=384, half=True):
        self.conf_thresh = conf_thresh
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.half = half and torch.cuda.is_available() and str(self.device).startswith("cuda")
        print(f"[INFO] Initializing YOLOv8-Face Detector on device: {self.device}")
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found at: {model_path}", file=sys.stderr)
            sys.exit(1)
        if str(self.device).startswith("cuda"):
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("medium")
            except Exception:
                pass
        self.model = YOLO(model_path)
        self.model.to(self.device)
        try:
            self.model.fuse()
        except Exception:
            pass
        if self.half:
            try:
                self.model.model.half()
            except Exception:
                self.half = False
        try:
            dummy = (self.imgsz, self.imgsz, 3)
            import numpy as np
            _ = self.model(np.zeros(dummy, dtype=np.uint8), verbose=False, imgsz=self.imgsz)
        except Exception:
            pass

    def detect(self, frame):
        results = self.model(frame, verbose=False, imgsz=self.imgsz)[0]
        detections = []
        boxes = results.boxes
        keypoints = results.keypoints

        for i in range(len(boxes)):
            box = boxes[i]
            conf = float(box.conf[0])

            if conf < self.conf_thresh:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            landmarks = {}
            if keypoints is not None and len(keypoints) > i:
                kpts = keypoints[i]
                if len(kpts.xy[0]) == 5:
                    lm_coords = kpts.xy[0].cpu().numpy().astype(int)
                    landmarks = {
                        'left_eye': tuple(lm_coords[0]),
                        'right_eye': tuple(lm_coords[1]),
                        'nose': tuple(lm_coords[2]),
                        'left_mouth': tuple(lm_coords[3]),
                        'right_mouth': tuple(lm_coords[4])
                    }
            
            detections.append((x1, y1, x2, y2, conf, landmarks))
                
        return detections
  


