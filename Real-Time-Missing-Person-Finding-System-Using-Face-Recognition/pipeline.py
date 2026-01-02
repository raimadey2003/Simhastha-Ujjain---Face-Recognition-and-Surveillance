#!/usr/bin/env python3
"""
Real-time Face Recognition Pipeline
====================================

A single-file, threaded pipeline that coordinates face detection, tracking,
embedding computation, recognition search, and crowd monitoring.

Usage:
    python pipeline.py --video-source 0 --config configs/default.yaml
    python pipeline.py --video-source webcam --dry-run
    python pipeline.py --video-source /path/to/video.mp4 --queue-size 50 --drop-policy drop-old

Configuration:
    - Create a YAML config file (see example below) or use CLI arguments
    - The pipeline will attempt to import existing modules from:
      * src.detection.face2 (FaceDetector)
      * src.tracking.deep_sort (Tracker)
      * src.recognition.face_recog_core (embedding/search functions)
      * src.crowd.crowd_monitor (CSRNet, load_csrnet)
    - If imports fail, placeholder implementations are used for testing

Thread Safety:
    - GPU models (detector, embedder) are called from dedicated threads
    - Each thread has its own model instance or uses thread-safe locking
    - For multi-GPU setups, assign device per thread via config

Batching & Drop Policy:
    - Embedding stage uses ThreadPoolExecutor for concurrent processing
    - Queue backpressure: 'drop-new' (default) or 'drop-old'
    - Configure batch_size and batch_wait_time in config for batching

Example Config (configs/default.yaml):
    video_source: 0
    device: cuda
    queue_size: 100
    drop_policy: drop-new
    embedding_workers: 2
    batch_size: 1
    batch_wait_time: 0.0
    log_level: INFO
    embeddings_dir: embeddings
    recognition_threshold: 0.38
    recognition_topk: 5
    crowd_enabled: true
    crowd_model_path: src/crowd/task_two_model_best.pth.tar
    crowd_threshold: 1000.0
    health_port: 8080
    save_annotated_frames: false
    annotated_frames_dir: results
"""

import argparse
import csv
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Dict, List, Optional, Tuple, Any
import yaml

try:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
except ImportError as e:
    print(f"ERROR: Required dependency missing: {e}")
    print("Install with: pip install opencv-python numpy torch pillow")
    sys.exit(1)

# ============================================================================
# Import existing modules (with fallback placeholders)
# ============================================================================

# Detector
try:
    from src.detection.face2 import FaceDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    print("[WARN] Could not import FaceDetector from src.detection.face2. Using placeholder.")
    DETECTOR_AVAILABLE = False
    class FaceDetector:
        def __init__(self, model_path="yolov8n-face.pt", conf_thresh=0.3, device=None):
            print("[PLACEHOLDER] FaceDetector initialized (no-op)")
        def detect(self, frame):
            # Placeholder: return empty detections
            return []

# Tracker
try:
    from src.tracking.deep_sort import Tracker
    TRACKER_AVAILABLE = True
except ImportError:
    print("[WARN] Could not import Tracker from src.tracking.deep_sort. Using placeholder.")
    TRACKER_AVAILABLE = False
    class Tracker:
        def __init__(self, max_age=100, n_init=3, max_iou_distance=0.7):
            print("[PLACEHOLDER] Tracker initialized (no-op)")
            self._next_id = 1
        def update(self, detections, frame):
            # Placeholder: assign sequential IDs
            outputs = []
            for det in detections:
                if len(det) >= 5:
                    x1, y1, x2, y2, conf = det[:5]
                    outputs.append({
                        'track_id': self._next_id,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf)
                    })
                    self._next_id += 1
            return outputs

# Embedding & Search
try:
    from src.recognition.face_recog_core import (
        get_embedding_pytorch, load_model, make_transform,
        load_embeddings_index, match_query, read_image, annotate_and_save
    )
    RECOGNITION_AVAILABLE = True
except ImportError:
    print("[WARN] Could not import recognition functions. Using placeholders.")
    RECOGNITION_AVAILABLE = False
    def get_embedding_pytorch(img_pil, model, device, transform):
        return np.random.rand(512).astype(np.float32)
    def load_model(device='cpu', pretrained='vggface2'):
        return None
    def make_transform(input_size=160):
        return lambda x: x
    def load_embeddings_index(embeddings_dir='embeddings'):
        return []
    def match_query(items, query_emb, topk=5, threshold=0.38):
        return []
    def read_image(path):
        try:
            import cv2
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        except:
            return Image.fromarray(np.zeros((160, 160, 3), dtype=np.uint8))
    def annotate_and_save(query_bgr, dataset_dir, matches, out_path='result_matches.jpg'):
        try:
            import cv2
            # Simple placeholder implementation
            cv2.imwrite(out_path, query_bgr)
            return out_path
        except:
            return out_path

# Crowd Monitor
try:
    from src.crowd.crowd_monitor import CSRNet, load_csrnet
    CROWD_AVAILABLE = True
except ImportError:
    print("[WARN] Could not import crowd monitor. Using placeholder.")
    CROWD_AVAILABLE = False
    class CSRNet:
        def __init__(self, load_weights=False):
            pass
        def __call__(self, x):
            return torch.zeros(1, 1, x.shape[2]//8, x.shape[3]//8)
    def load_csrnet(cfg):
        return CSRNet()


# ============================================================================
# Configuration & Data Structures
# ============================================================================

@dataclass
class Config:
    """Pipeline configuration"""
    video_source: Any = 0  # int for webcam, str for file/RTSP
    device: str = "cuda"
    queue_size: int = 100
    drop_policy: str = "drop-new"  # drop-new or drop-old
    embedding_workers: int = 2
    batch_size: int = 1
    batch_wait_time: float = 0.0
    log_level: str = "INFO"
    embeddings_dir: str = "embeddings"
    recognition_threshold: float = 0.45
    recognition_topk: int = 5
    crowd_enabled: bool = True
    crowd_model_path: str = "src/crowd/task_two_model_best.pth.tar"
    crowd_threshold: float = 1000.0
    health_port: int = 8080
    save_annotated_frames: bool = False
    annotated_frames_dir: str = "results"
    save_embeddings: bool = True  # Save embeddings to disk during detection
    embeddings_runtime_dir: str = "embeddings/runtime"  # Directory for runtime embeddings
    embeddings_index_file: str = "embeddings/runtime_index.csv"  # Index file for runtime embeddings
    detector_model_path: str = "yolov8n-face.pt"
    detector_conf_thresh: float = 0.3
    detector_imgsz: int = 384
    detector_half: bool = True
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    tracker_max_age: int = 100
    tracker_n_init: int = 3
    tracker_max_iou_distance: float = 0.7
    embedding_input_size: int = 160
    embedding_pretrained: str = "vggface2"
    restart_on_error: bool = False
    max_restarts: int = 3
    batch_recognition_enabled: bool = True  # Enable batch recognition from exported_images
    exported_images_dir: str = "exported_images"  # Folder to watch for query images
    batch_recognition_results_dir: str = "results/batch_recognition"  # Folder to save results
    batch_recognition_scan_interval: float = 2.0  # Seconds between folder scans
    batch_recognition_search_runtime: bool = True  # Search in runtime embeddings
    batch_recognition_search_precomputed: bool = True  # Search in precomputed embeddings

    @classmethod
    def from_dict(cls, d: dict):
        """Create Config from dictionary"""
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})

    @classmethod
    def from_yaml(cls, path: str):
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self):
        return asdict(self)


@dataclass
class DetectionMessage:
    """Message passed from detector to tracker/embedder"""
    frame_id: int
    frame: np.ndarray
    timestamp: float
    detections: List[Tuple]  # List of (x1, y1, x2, y2, conf, landmarks)


@dataclass
class TrackedMessage:
    """Message passed from tracker to embedder"""
    frame_id: int
    frame: np.ndarray
    timestamp: float
    tracks: List[Dict]  # List of {'track_id', 'bbox', 'confidence'}


@dataclass
class EmbeddingMessage:
    """Message passed from embedder to recognizer"""
    frame_id: int
    track_id: int
    timestamp: float
    embedding: np.ndarray
    bbox: Tuple[int, int, int, int]
    crop_path: Optional[str] = None  # Path to saved crop if enabled


@dataclass
class RecognitionResult:
    """Recognition search result"""
    frame_id: int
    track_id: int
    timestamp: float
    matches: List[Dict]  # List of {'image_file', 'cosine', 'score01'}
    bbox: Tuple[int, int, int, int]


@dataclass
class CrowdEvent:
    """Crowd density estimation event"""
    frame_id: int
    timestamp: float
    count: float
    alert: bool


# ============================================================================
# Metrics & Health
# ============================================================================

class Metrics:
    """Thread-safe metrics collector"""
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.frame_count = 0
            self.detection_count = 0
            self.track_count = 0
            self.embedding_count = 0
            self.recognition_count = 0
            self.crowd_events = 0
            self.errors = 0
            self.start_time = time.time()
            self.detection_times = deque(maxlen=100)
            self.embedding_times = deque(maxlen=100)
            self.recognition_times = deque(maxlen=100)
            self.queue_sizes = {
                'detection': deque(maxlen=100),
                'tracking': deque(maxlen=100),
                'embedding': deque(maxlen=100),
                'recognition': deque(maxlen=100),
            }

    def record_detection(self, count: int, latency_ms: float):
        with self.lock:
            self.detection_count += count
            self.detection_times.append(latency_ms)

    def record_track(self, count: int):
        with self.lock:
            self.track_count += count

    def record_embedding(self, latency_ms: float):
        with self.lock:
            self.embedding_count += 1
            self.embedding_times.append(latency_ms)

    def record_recognition(self, latency_ms: float):
        with self.lock:
            self.recognition_count += 1
            self.recognition_times.append(latency_ms)

    def record_crowd_event(self):
        with self.lock:
            self.crowd_events += 1

    def record_error(self):
        with self.lock:
            self.errors += 1

    def record_frame(self):
        with self.lock:
            self.frame_count += 1

    def update_queue_size(self, queue_name: str, size: int):
        with self.lock:
            if queue_name in self.queue_sizes:
                self.queue_sizes[queue_name].append(size)

    def get_stats(self) -> Dict:
        with self.lock:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            avg_detection = np.mean(self.detection_times) if self.detection_times else 0
            avg_embedding = np.mean(self.embedding_times) if self.embedding_times else 0
            avg_recognition = np.mean(self.recognition_times) if self.recognition_times else 0
            queue_avgs = {
                k: np.mean(v) if v else 0
                for k, v in self.queue_sizes.items()
            }
            return {
                'fps': fps,
                'frame_count': self.frame_count,
                'detection_count': self.detection_count,
                'track_count': self.track_count,
                'embedding_count': self.embedding_count,
                'recognition_count': self.recognition_count,
                'crowd_events': self.crowd_events,
                'errors': self.errors,
                'avg_detection_latency_ms': avg_detection,
                'avg_embedding_latency_ms': avg_embedding,
                'avg_recognition_latency_ms': avg_recognition,
                'queue_sizes': queue_avgs,
                'uptime_seconds': elapsed
            }


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health endpoint"""
    metrics: Metrics = None
    thread_status: Dict = None

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            stats = self.metrics.get_stats() if self.metrics else {}
            response = {
                'status': 'healthy',
                'metrics': stats,
                'threads': self.thread_status or {}
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default logging


# ============================================================================
# Queue Utilities
# ============================================================================

class DeadLetterQueue:
    """Thread-safe dead-letter queue for failed messages"""
    def __init__(self, max_size: int = 1000):
        self.items = []
        self.lock = threading.Lock()
        self.max_size = max_size

    def add(self, item: Any, reason: str = "unknown"):
        """Add a failed message to dead-letter queue"""
        with self.lock:
            if len(self.items) >= self.max_size:
                self.items.pop(0)  # Remove oldest
            self.items.append({
                'item': str(item),
                'reason': reason,
                'timestamp': time.time()
            })

    def get_all(self):
        """Get all items (for reporting)"""
        with self.lock:
            return self.items.copy()

    def clear(self):
        """Clear the queue"""
        with self.lock:
            self.items.clear()


def safe_put(queue: Queue, item: Any, drop_policy: str, logger: logging.Logger,
             dead_letter_queue: Optional[DeadLetterQueue] = None):
    """Safely put item in queue with backpressure handling"""
    try:
        queue.put_nowait(item)
        return True
    except Full:
        if drop_policy == "drop-old":
            try:
                dropped = queue.get_nowait()  # Remove oldest
                queue.put_nowait(item)  # Add new
                logger.warning(f"Queue full, dropped oldest item (policy: {drop_policy})")
                if dead_letter_queue:
                    dead_letter_queue.add(dropped, "queue_full_drop_old")
                return True
            except Empty:
                queue.put_nowait(item)
                return True
        else:  # drop-new
            logger.warning(f"Queue full, dropping new item (policy: {drop_policy})")
            if dead_letter_queue:
                dead_letter_queue.add(item, "queue_full_drop_new")
            return False


# ============================================================================
# Pipeline Threads
# ============================================================================

class VideoReaderThread(threading.Thread):
    """Thread A: Video reader & detector"""
    def __init__(self, config: Config, detection_queue: Queue, frame_queue: Queue,
                 stop_event: threading.Event, metrics: Metrics, logger: logging.Logger,
                 dead_letter_queue: Optional[DeadLetterQueue] = None):
        super().__init__(name="VideoReader", daemon=True)
        self.config = config
        self.detection_queue = detection_queue
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.metrics = metrics
        self.logger = logger
        self.dead_letter_queue = dead_letter_queue
        self.detector = None
        self.frame_id = 0
        self.restart_count = 0

    def run(self):
        try:
            self._init_detector()
            self._run_loop()
        except Exception as e:
            self.logger.error(f"VideoReader thread error: {e}", exc_info=True)
            self.metrics.record_error()
            if self.config.restart_on_error and self.restart_count < self.config.max_restarts:
                self.restart_count += 1
                self.logger.info(f"Restarting VideoReader (attempt {self.restart_count})")
                time.sleep(1)
                self.run()

    def _init_detector(self):
        device = self.config.device if torch.cuda.is_available() and self.config.device.startswith("cuda") else "cpu"
        self.detector = FaceDetector(
            model_path=self.config.detector_model_path,
            conf_thresh=self.config.detector_conf_thresh,
            device=device,
            imgsz=getattr(self.config, "detector_imgsz", 384),
            half=getattr(self.config, "detector_half", True)
        )
        self.logger.info(f"Detector initialized on {device}")

    def _run_loop(self):
        # Open video source
        if isinstance(self.config.video_source, int) or str(self.config.video_source).isdigit():
            cap = cv2.VideoCapture(int(self.config.video_source))
        else:
            cap = cv2.VideoCapture(str(self.config.video_source))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.config.video_source}")

        self.logger.info(f"Video source opened: {self.config.video_source}")

        if self.config.video_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.video_width))
        if self.config.video_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.video_height))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if isinstance(self.config.video_source, int):
                        self.logger.warning("Frame read failed, retrying...")
                        time.sleep(0.1)
                        continue
                    else:
                        self.logger.info("End of video file")
                        break

                self.frame_id += 1
                self.metrics.record_frame()
                timestamp = time.time()

                # Run detection
                t0 = time.time()
                detections = self.detector.detect(frame)
                detection_latency = (time.time() - t0) * 1000
                self.metrics.record_detection(len(detections), detection_latency)

                # Create detection message
                msg = DetectionMessage(
                    frame_id=self.frame_id,
                    frame=frame.copy(),
                    timestamp=timestamp,
                    detections=detections
                )

                # Put in queues
                safe_put(self.detection_queue, msg, self.config.drop_policy, self.logger,
                        self.dead_letter_queue)
                if self.frame_queue is not None:
                    safe_put(self.frame_queue, (self.frame_id, frame.copy(), timestamp),
                            self.config.drop_policy, self.logger,
                            self.dead_letter_queue)

                self.metrics.update_queue_size('detection', self.detection_queue.qsize())

                if self.frame_id % 30 == 0:
                    self.logger.debug(f"Frame {self.frame_id}: {len(detections)} detections")

        finally:
            cap.release()
            self.logger.info("VideoReader thread stopped")


class TrackerThread(threading.Thread):
    """Thread that tracks detections and produces track IDs"""
    def __init__(self, config: Config, detection_queue: Queue, tracking_queue: Queue,
                 stop_event: threading.Event, metrics: Metrics, logger: logging.Logger,
                 dead_letter_queue: Optional[DeadLetterQueue] = None):
        super().__init__(name="Tracker", daemon=True)
        self.config = config
        self.detection_queue = detection_queue
        self.tracking_queue = tracking_queue
        self.stop_event = stop_event
        self.metrics = metrics
        self.logger = logger
        self.dead_letter_queue = dead_letter_queue
        self.tracker = Tracker(
            max_age=config.tracker_max_age,
            n_init=config.tracker_n_init,
            max_iou_distance=config.tracker_max_iou_distance
        )
        self.restart_count = 0

    def run(self):
        try:
            self._run_loop()
        except Exception as e:
            self.logger.error(f"Tracker thread error: {e}", exc_info=True)
            self.metrics.record_error()
            if self.config.restart_on_error and self.restart_count < self.config.max_restarts:
                self.restart_count += 1
                self.logger.info(f"Restarting Tracker (attempt {self.restart_count})")
                time.sleep(1)
                self.run()

    def _run_loop(self):
        while not self.stop_event.is_set():
            try:
                msg: DetectionMessage = self.detection_queue.get(timeout=0.1)
            except Empty:
                continue

            # Convert detections format for tracker
            # Tracker expects (x1, y1, x2, y2, conf)
            detections_for_tracker = [
                (d[0], d[1], d[2], d[3], d[4]) for d in msg.detections
            ]

            # Update tracker
            tracks = self.tracker.update(detections_for_tracker, msg.frame)

            self.metrics.record_track(len(tracks))
            self.metrics.update_queue_size('tracking', self.tracking_queue.qsize())

            # Create tracked message
            tracked_msg = TrackedMessage(
                frame_id=msg.frame_id,
                frame=msg.frame,
                timestamp=msg.timestamp,
                tracks=tracks
            )

            safe_put(self.tracking_queue, tracked_msg, self.config.drop_policy, self.logger,
                    self.dead_letter_queue)

            if len(tracks) > 0:
                self.logger.debug(f"Frame {msg.frame_id}: {len(tracks)} tracks")


class EmbeddingWorker:
    """Worker for computing embeddings (used in ThreadPoolExecutor)"""
    def __init__(self, config: Config, model, transform, device, logger: logging.Logger):
        self.config = config
        self.model = model
        self.transform = transform
        self.device = device
        self.logger = logger
        self.index_lock = threading.Lock()  # Lock for thread-safe index updates
        self._ensure_embedding_dirs()

    def _ensure_embedding_dirs(self):
        """Ensure embedding directories exist"""
        if self.config.save_embeddings:
            embeddings_dir = Path(self.config.embeddings_runtime_dir)
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            # Initialize index file if it doesn't exist
            index_file = Path(self.config.embeddings_index_file)
            if not index_file.exists():
                with open(index_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['image_file', 'embedding_file', 'frame_id', 'track_id', 'timestamp'])

    def _save_embedding(self, embedding: np.ndarray, frame_id: int, track_id: int, timestamp: float) -> Optional[str]:
        """Save embedding to disk and update index"""
        if not self.config.save_embeddings:
            return None

        try:
            # Generate filename
            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S_%f')[:-3]
            emb_filename = f"{frame_id}_{track_id}_{timestamp_str}.npy"
            emb_path = Path(self.config.embeddings_runtime_dir) / emb_filename

            # Save embedding
            np.save(str(emb_path), embedding)

            # Update index file (thread-safe)
            with self.index_lock:
                index_file = Path(self.config.embeddings_index_file)
                # Read existing entries
                rows = []
                if index_file.exists():
                    with open(index_file, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        all_rows = list(reader)
                        if all_rows and all_rows[0][0] == 'image_file':
                            # Skip header
                            rows = all_rows[1:]
                        else:
                            rows = all_rows

                # Add new entry
                crop_filename = f"{frame_id}_{track_id}_{timestamp_str}.jpg"
                rows.append([crop_filename, emb_filename, str(frame_id), str(track_id), str(timestamp)])

                # Write back (keep last N entries to avoid file growing too large)
                max_entries = 10000  # Keep last 10k entries
                if len(rows) > max_entries:
                    rows = rows[-max_entries:]

                with open(index_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['image_file', 'embedding_file', 'frame_id', 'track_id', 'timestamp'])
                    writer.writerows(rows)

            self.logger.info(f"Saved embedding: {emb_filename} (frame {frame_id}, track {track_id})")
            return str(emb_path)
        except Exception as e:
            self.logger.error(f"Failed to save embedding: {e}", exc_info=True)
            return None

    def compute_embedding(self, tracked_msg: TrackedMessage, track: Dict) -> Optional[EmbeddingMessage]:
        """Compute embedding for a single tracked face"""
        try:
            x1, y1, x2, y2 = track['bbox']
            h, w = tracked_msg.frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            # Extract crop
            crop = tracked_msg.frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            # Convert to PIL
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            # Compute embedding
            t0 = time.time()
            embedding = get_embedding_pytorch(crop_pil, self.model, self.device, self.transform)
            latency = (time.time() - t0) * 1000

            # Save embedding to disk if enabled
            emb_path = None
            if self.config.save_embeddings:
                emb_path = self._save_embedding(
                    embedding, tracked_msg.frame_id, track['track_id'], tracked_msg.timestamp
                )

            # Optionally save crop
            crop_path = None
            if self.config.save_annotated_frames:
                crop_dir = Path(self.config.annotated_frames_dir) / "crops"
                crop_dir.mkdir(parents=True, exist_ok=True)
                timestamp_str = datetime.fromtimestamp(tracked_msg.timestamp).strftime('%Y%m%d_%H%M%S_%f')[:-3]
                crop_path = crop_dir / f"{tracked_msg.frame_id}_{track['track_id']}_{timestamp_str}.jpg"
                cv2.imwrite(str(crop_path), crop)

            return EmbeddingMessage(
                frame_id=tracked_msg.frame_id,
                track_id=track['track_id'],
                timestamp=tracked_msg.timestamp,
                embedding=embedding,
                bbox=track['bbox'],
                crop_path=str(crop_path) if crop_path else None
            )
        except Exception as e:
            self.logger.error(f"Embedding computation error: {e}", exc_info=True)
            return None


class EmbeddingThread(threading.Thread):
    """Thread B: Embedding workers"""
    def __init__(self, config: Config, tracking_queue: Queue, embedding_queue: Queue,
                 stop_event: threading.Event, metrics: Metrics, logger: logging.Logger,
                 dead_letter_queue: Optional[DeadLetterQueue] = None):
        super().__init__(name="Embedding", daemon=True)
        self.config = config
        self.tracking_queue = tracking_queue
        self.embedding_queue = embedding_queue
        self.stop_event = stop_event
        self.metrics = metrics
        self.logger = logger
        self.dead_letter_queue = dead_letter_queue
        self.model = None
        self.transform = None
        self.device = None
        self.executor = None
        self.restart_count = 0

    def run(self):
        try:
            self._init_embedder()
            self._run_loop()
        except Exception as e:
            self.logger.error(f"Embedding thread error: {e}", exc_info=True)
            self.metrics.record_error()
            if self.config.restart_on_error and self.restart_count < self.config.max_restarts:
                self.restart_count += 1
                self.logger.info(f"Restarting Embedding (attempt {self.restart_count})")
                time.sleep(1)
                self.run()
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)

    def _init_embedder(self):
        self.device = self.config.device if torch.cuda.is_available() and self.config.device.startswith("cuda") else "cpu"
        self.model = load_model(device=self.device, pretrained=self.config.embedding_pretrained)
        self.transform = make_transform(input_size=self.config.embedding_input_size)
        self.executor = ThreadPoolExecutor(max_workers=self.config.embedding_workers)
        self.logger.info(f"Embedder initialized on {self.device} with {self.config.embedding_workers} workers")

    def _run_loop(self):
        worker = EmbeddingWorker(self.config, self.model, self.transform, self.device, self.logger)
        batch = []
        last_batch_time = time.time()

        while not self.stop_event.is_set():
            try:
                msg: TrackedMessage = self.tracking_queue.get(timeout=0.1)
            except Empty:
                # Check if we should flush batch
                if batch and (time.time() - last_batch_time) >= self.config.batch_wait_time:
                    self._process_batch(batch, worker)
                    batch = []
                continue

            batch.append(msg)

            # Process batch if full or timeout
            if len(batch) >= self.config.batch_size:
                self._process_batch(batch, worker)
                batch = []
                last_batch_time = time.time()
            elif self.config.batch_wait_time > 0 and (time.time() - last_batch_time) >= self.config.batch_wait_time:
                self._process_batch(batch, worker)
                batch = []
                last_batch_time = time.time()

        # Flush remaining batch
        if batch:
            self._process_batch(batch, worker)

    def _process_batch(self, batch: List[TrackedMessage], worker: EmbeddingWorker):
        """Process a batch of tracked messages"""
        futures = []
        for msg in batch:
            for track in msg.tracks:
                future = self.executor.submit(worker.compute_embedding, msg, track)
                futures.append((future, msg.frame_id, track['track_id']))

        for future, frame_id, track_id in futures:
            try:
                result = future.result(timeout=5.0)
                if result:
                    self.metrics.record_embedding(0)  # Latency tracked in worker
                    if self.config.save_embeddings:
                        self.logger.debug(f"Embedding computed and saved for frame {result.frame_id}, track {result.track_id}")
                    safe_put(self.embedding_queue, result, self.config.drop_policy, self.logger,
                            self.dead_letter_queue)
                    self.metrics.update_queue_size('embedding', self.embedding_queue.qsize())
            except Exception as e:
                self.logger.error(f"Embedding future error: {e}", exc_info=True)
                self.metrics.record_error()
                if self.dead_letter_queue:
                    self.dead_letter_queue.add(f"frame_{frame_id}_track_{track_id}", f"embedding_error: {e}")


class RecognitionThread(threading.Thread):
    """Thread C: Recognition/search - matches detected faces against exported_images"""
    def __init__(self, config: Config, embedding_queue: Queue,
                 stop_event: threading.Event, metrics: Metrics, logger: logging.Logger):
        super().__init__(name="Recognition", daemon=True)
        self.config = config
        self.embedding_queue = embedding_queue
        self.stop_event = stop_event
        self.metrics = metrics
        self.logger = logger
        # Setup file handler for recognition logs
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "recognition.log"
        # Check if file handler already exists to avoid duplicates
        handler_exists = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file.absolute())
            for h in self.logger.handlers
        )
        if not handler_exists:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.items = []  # Embeddings from exported_images folder
        self.exported_images_dir = Path(config.exported_images_dir)
        self.results_dir = Path("results/recognition")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = None
        self.transform = None
        self.device = None
        self.restart_count = 0
        self.last_reload_time = 0
        self.reload_interval = 30.0  # Reload exported_images every 30 seconds

    def run(self):
        try:
            self._init_embedding_model()
            self._load_exported_images_embeddings()
            self._run_loop()
        except Exception as e:
            self.logger.error(f"Recognition thread error: {e}", exc_info=True)
            self.metrics.record_error()
            if self.config.restart_on_error and self.restart_count < self.config.max_restarts:
                self.restart_count += 1
                self.logger.info(f"Restarting Recognition (attempt {self.restart_count})")
                time.sleep(1)
                self.run()

    def _init_embedding_model(self):
        """Initialize embedding model for computing embeddings from exported_images"""
        self.device = self.config.device if torch.cuda.is_available() and self.config.device.startswith("cuda") else "cpu"
        self.embedding_model = load_model(device=self.device, pretrained=self.config.embedding_pretrained)
        self.transform = make_transform(input_size=self.config.embedding_input_size)
        self.logger.info(f"Recognition embedding model initialized on {self.device}")

    def _load_exported_images_embeddings(self):
        """Load or compute embeddings from exported_images folder"""
        self.items = []
        self.exported_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images in exported_images folder
        image_files = list(self.exported_images_dir.glob("*.jpg")) + \
                     list(self.exported_images_dir.glob("*.jpeg")) + \
                     list(self.exported_images_dir.glob("*.png"))
        
        if not image_files:
            self.logger.warning(f"No images found in {self.exported_images_dir}")
            return

        self.logger.info(f"Loading/computing embeddings for {len(image_files)} images from exported_images...")
        
        for img_path in image_files:
            try:
                # Read image and compute embedding
                img_pil = read_image(img_path)
                emb = get_embedding_pytorch(img_pil, self.embedding_model, self.device, self.transform)
                
                self.items.append({
                    'image_file': img_path.name,
                    'image_path': str(img_path),
                    'embedding': emb
                })
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")
                continue

        self.logger.info(f"Loaded {len(self.items)} embeddings from exported_images folder")
        self.last_reload_time = time.time()

    def _run_loop(self):
        """Main recognition loop"""
        while not self.stop_event.is_set():
            try:
                msg: EmbeddingMessage = self.embedding_queue.get(timeout=0.1)
            except Empty:
                # Periodically reload exported_images embeddings
                if time.time() - self.last_reload_time > self.reload_interval:
                    self._load_exported_images_embeddings()
                continue

            if len(self.items) == 0:
                self.logger.debug("No exported images loaded, skipping recognition")
                continue

            # Run search against exported_images
            t0 = time.time()
            matches = match_query(
                self.items,
                msg.embedding,
                topk=self.config.recognition_topk,
                threshold=self.config.recognition_threshold
            )
            latency = (time.time() - t0) * 1000
            self.metrics.record_recognition(latency)

            # Process results
            if matches:
                self._process_recognition_result(msg, matches)
            else:
                self.logger.debug(f"Frame {msg.frame_id}, Track {msg.track_id}: No matches found")

    def _process_recognition_result(self, msg: EmbeddingMessage, matches: List[Dict]):
        """Process recognition result: print to terminal and save annotated image"""
        # Print to terminal
        print(f"\n{'='*60}")
        print(f"MATCH FOUND! Frame {msg.frame_id}, Track {msg.track_id}")
        print(f"{'='*60}")
        print(f"Top {len(matches)} matches:")
        for i, m in enumerate(matches, 1):
            print(f"  {i}. {m['image_file']}")
            print(f"     Cosine Similarity: {m['cosine']:.4f} | Score: {m['score01']:.4f}")
        print(f"{'='*60}\n")

        # Log to logger
        self.logger.info(f"Frame {msg.frame_id}, Track {msg.track_id}: {len(matches)} matches found")
        for m in matches[:3]:
            self.logger.info(f"  Match: {m['image_file']} (cosine: {m['cosine']:.4f}, score: {m['score01']:.4f})")

        # Load crop image if available
        query_img = None
        if msg.crop_path and Path(msg.crop_path).exists():
            query_img = cv2.imread(msg.crop_path)
        else:
            # Try to find crop in results/crops directory with various patterns
            crops_dir = Path(self.config.annotated_frames_dir) / "crops"
            if crops_dir.exists():
                # Pattern 1: exact match with timestamp
                timestamp_str = datetime.fromtimestamp(msg.timestamp).strftime('%Y%m%d_%H%M%S_%f')[:-3]
                crop_path = crops_dir / f"{msg.frame_id}_{msg.track_id}_{timestamp_str}.jpg"
                if crop_path.exists():
                    query_img = cv2.imread(str(crop_path))
                
                # Pattern 2: without timestamp (fallback)
                if query_img is None:
                    crop_path = crops_dir / f"{msg.frame_id}_{msg.track_id}.jpg"
                    if crop_path.exists():
                        query_img = cv2.imread(str(crop_path))
                
                # Pattern 3: search for any file matching frame_id and track_id
                if query_img is None:
                    pattern = f"{msg.frame_id}_{msg.track_id}_*.jpg"
                    matches = list(crops_dir.glob(pattern))
                    if matches:
                        # Use the most recent one
                        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        query_img = cv2.imread(str(matches[0]))

        if query_img is None:
            self.logger.warning(f"Could not find crop image for frame {msg.frame_id}, track {msg.track_id}")
            return

        # Save annotated result using existing function
        try:
            # Prepare dataset_dir for annotate_and_save (should point to exported_images)
            result_filename = f"frame_{msg.frame_id}_track_{msg.track_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            result_path = self.results_dir / result_filename
            
            # Use the existing annotate_and_save function
            annotate_and_save(
                query_bgr=query_img,
                dataset_dir=str(self.exported_images_dir),
                matches=matches,
                out_path=str(result_path)
            )
            
            self.logger.info(f"Saved annotated recognition result: {result_filename}")
            print(f"Annotated result saved to: {result_path}\n")
        except Exception as e:
            self.logger.error(f"Error saving annotated result: {e}", exc_info=True)


class CrowdMonitorThread(threading.Thread):
    """Thread D: Crowd monitor"""
    def __init__(self, config: Config, frame_queue: Queue,
                 stop_event: threading.Event, metrics: Metrics, logger: logging.Logger):
        super().__init__(name="CrowdMonitor", daemon=True)
        self.config = config
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.metrics = metrics
        self.logger = logger
        self.model = None
        self.restart_count = 0

    def run(self):
        if not self.config.crowd_enabled:
            self.logger.info("Crowd monitoring disabled")
            return

        try:
            self._init_model()
            self._run_loop()
        except Exception as e:
            self.logger.error(f"CrowdMonitor thread error: {e}", exc_info=True)
            self.metrics.record_error()
            if self.config.restart_on_error and self.restart_count < self.config.max_restarts:
                self.restart_count += 1
                self.logger.info(f"Restarting CrowdMonitor (attempt {self.restart_count})")
                time.sleep(1)
                self.run()

    def _init_model(self):
        # Create a simple config object for load_csrnet
        class CrowdCfg:
            model_path = self.config.crowd_model_path
            use_cuda = torch.cuda.is_available() and self.config.device.startswith("cuda")
            frame_resize = (640, 360)

        cfg = CrowdCfg()
        self.model = load_csrnet(cfg)
        self.logger.info("Crowd model loaded")

    def _run_loop(self):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        while not self.stop_event.is_set():
            try:
                frame_id, frame, timestamp = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue

            # Resize and preprocess
            frame_resized = cv2.resize(frame, (640, 360))
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img_rgb).unsqueeze(0)

            device = self.config.device if torch.cuda.is_available() and self.config.device.startswith("cuda") else "cpu"
            if device == "cuda":
                img_tensor = img_tensor.cuda(non_blocking=True)

            # Inference
            with torch.no_grad():
                output = self.model(img_tensor)
            density_map = output.squeeze().cpu().numpy()
            count = float(np.sum(density_map))

            alert = count > self.config.crowd_threshold
            if alert:
                self.logger.warning(f"Frame {frame_id}: Crowd alert! Count: {count:.2f}")

            event = CrowdEvent(
                frame_id=frame_id,
                timestamp=timestamp,
                count=count,
                alert=alert
            )

            self.metrics.record_crowd_event()


class BatchRecognitionThread(threading.Thread):
    """Thread for batch recognition: processes images from exported_images folder"""
    def __init__(self, config: Config, stop_event: threading.Event, metrics: Metrics, logger: logging.Logger):
        super().__init__(name="BatchRecognition", daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.metrics = metrics
        self.logger = logger
        self.detector = None
        self.embedding_model = None
        self.transform = None
        self.device = None
        self.processed_files = set()  # Track processed files
        self.runtime_items = []
        self.precomputed_items = []
        self.restart_count = 0

    def run(self):
        if not self.config.batch_recognition_enabled:
            self.logger.info("Batch recognition disabled")
            return

        try:
            self._init_models()
            self._load_embeddings()
            self._run_loop()
        except Exception as e:
            self.logger.error(f"BatchRecognition thread error: {e}", exc_info=True)
            self.metrics.record_error()
            if self.config.restart_on_error and self.restart_count < self.config.max_restarts:
                self.restart_count += 1
                self.logger.info(f"Restarting BatchRecognition (attempt {self.restart_count})")
                time.sleep(1)
                self.run()

    def _init_models(self):
        """Initialize detector and embedding model"""
        self.device = self.config.device if torch.cuda.is_available() and self.config.device.startswith("cuda") else "cpu"
        
        # Initialize detector
        self.detector = FaceDetector(
            model_path=self.config.detector_model_path,
            conf_thresh=self.config.detector_conf_thresh,
            device=self.device,
            imgsz=getattr(self.config, "detector_imgsz", 384),
            half=getattr(self.config, "detector_half", True)
        )
        
        # Initialize embedding model
        self.embedding_model = load_model(device=self.device, pretrained=self.config.embedding_pretrained)
        self.transform = make_transform(input_size=self.config.embedding_input_size)
        
        self.logger.info(f"BatchRecognition initialized on {self.device}")

    def _load_embeddings(self):
        """Load embeddings from runtime and precomputed indices"""
        # Load runtime embeddings
        if self.config.batch_recognition_search_runtime:
            try:
                runtime_index = Path(self.config.embeddings_index_file)
                if runtime_index.exists():
                    self.runtime_items = self._load_runtime_index(runtime_index)
                    self.logger.info(f"Loaded {len(self.runtime_items)} runtime embeddings")
            except Exception as e:
                self.logger.warning(f"Could not load runtime embeddings: {e}")

        # Load precomputed embeddings
        if self.config.batch_recognition_search_precomputed:
            try:
                self.precomputed_items = load_embeddings_index(embeddings_dir=self.config.embeddings_dir)
                self.logger.info(f"Loaded {len(self.precomputed_items)} precomputed embeddings")
            except Exception as e:
                self.logger.warning(f"Could not load precomputed embeddings: {e}")

        total = len(self.runtime_items) + len(self.precomputed_items)
        if total == 0:
            self.logger.warning("No embeddings loaded for batch recognition!")

    def _load_runtime_index(self, index_file: Path) -> List[Dict]:
        """Load runtime embeddings index"""
        items = []
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    emb_file = row.get('embedding_file', '')
                    emb_path = Path(self.config.embeddings_runtime_dir) / emb_file
                    if emb_path.exists():
                        emb = np.load(str(emb_path))
                        n = np.linalg.norm(emb)
                        if n > 0:
                            emb = emb / n
                        # Use crop path or embedding file as identifier
                        image_file = row.get('image_file', emb_file.replace('.npy', '.jpg'))
                        items.append({
                            'image_file': image_file,
                            'embedding': emb,
                            'source': 'runtime',
                            'frame_id': row.get('frame_id', ''),
                            'track_id': row.get('track_id', '')
                        })
        except Exception as e:
            self.logger.error(f"Error loading runtime index: {e}")
        return items

    def _run_loop(self):
        """Main loop: scan folder and process images"""
        exported_dir = Path(self.config.exported_images_dir)
        exported_dir.mkdir(parents=True, exist_ok=True)
        
        results_dir = Path(self.config.batch_recognition_results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        while not self.stop_event.is_set():
            try:
                # Scan for new images
                image_files = list(exported_dir.glob("*.jpg")) + list(exported_dir.glob("*.jpeg")) + list(exported_dir.glob("*.png"))
                
                for img_path in image_files:
                    if img_path.name not in self.processed_files:
                        self.logger.info(f"Processing new image: {img_path.name}")
                        self._process_image(img_path, results_dir)
                        self.processed_files.add(img_path.name)
                
                # Reload embeddings periodically (in case new ones were added)
                if len(self.processed_files) % 10 == 0:  # Reload every 10 processed images
                    self._load_embeddings()

            except Exception as e:
                self.logger.error(f"Error in batch recognition loop: {e}", exc_info=True)
                self.metrics.record_error()

            time.sleep(self.config.batch_recognition_scan_interval)

    def _process_image(self, img_path: Path, results_dir: Path):
        """Process a single image: detect, embed, search, save results"""
        try:
            # Read image
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                self.logger.error(f"Could not read image: {img_path}")
                return

            # Detect faces
            detections = self.detector.detect(img_bgr)
            if not detections:
                self.logger.warning(f"No faces detected in {img_path.name}")
                # Move to lost_images if no faces found
                lost_dir = Path("lost_images")
                lost_dir.mkdir(parents=True, exist_ok=True)
                lost_path = lost_dir / img_path.name
                import shutil
                shutil.move(str(img_path), str(lost_path))
                self.logger.info(f"Moved {img_path.name} to lost_images (no faces)")
                return

            self.logger.info(f"Found {len(detections)} faces in {img_path.name}")

            # Process each detected face
            all_matches = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                
                # Extract face crop
                h, w = img_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = img_bgr[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue

                # Convert to PIL and compute embedding
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                query_emb = get_embedding_pytorch(crop_pil, self.embedding_model, self.device, self.transform)

                # Search in both indices
                matches = []
                if self.config.batch_recognition_search_runtime and self.runtime_items:
                    runtime_matches = match_query(
                        self.runtime_items,
                        query_emb,
                        topk=self.config.recognition_topk,
                        threshold=self.config.recognition_threshold
                    )
                    matches.extend(runtime_matches)

                if self.config.batch_recognition_search_precomputed and self.precomputed_items:
                    precomputed_matches = match_query(
                        self.precomputed_items,
                        query_emb,
                        topk=self.config.recognition_topk,
                        threshold=self.config.recognition_threshold
                    )
                    matches.extend(precomputed_matches)

                # Deduplicate and sort by score
                seen = set()
                unique_matches = []
                for m in matches:
                    key = m['image_file']
                    if key not in seen:
                        seen.add(key)
                        unique_matches.append(m)
                
                # Sort by cosine similarity (descending)
                unique_matches.sort(key=lambda x: x['cosine'], reverse=True)
                top_matches = unique_matches[:self.config.recognition_topk]
                
                if top_matches:
                    all_matches.append({
                        'face_index': i,
                        'bbox': (x1, y1, x2, y2),
                        'matches': top_matches
                    })
                    self.logger.info(f"Face {i+1}: Found {len(top_matches)} matches (best: {top_matches[0]['cosine']:.3f})")

            # Save results
            if all_matches:
                self._save_results(img_bgr, img_path, all_matches, results_dir)
                # Move processed image to a subfolder
                exported_dir = Path(self.config.exported_images_dir)
                processed_dir = exported_dir / "processed"
                processed_dir.mkdir(exist_ok=True)
                processed_path = processed_dir / img_path.name
                import shutil
                shutil.move(str(img_path), str(processed_path))
                self.logger.info(f"Moved {img_path.name} to processed folder")
            else:
                self.logger.warning(f"No matches found for {img_path.name}")

        except Exception as e:
            self.logger.error(f"Error processing image {img_path}: {e}", exc_info=True)
            self.metrics.record_error()

    def _save_results(self, query_img: np.ndarray, img_path: Path, all_matches: List[Dict], results_dir: Path):
        """Save recognition results: query image + top matches"""
        try:
            # For each face, create a result montage
            for face_data in all_matches:
                face_idx = face_data['face_index']
                matches = face_data['matches']
                bbox = face_data['bbox']

                # Create montage similar to annotate_and_save
                thumbs = []
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Extract face crop from query image
                x1, y1, x2, y2 = bbox
                face_crop = query_img[y1:y2, x1:x2]
                hQ, wQ = face_crop.shape[:2]
                max_h = 160
                scale = max_h / max(hQ, 1) if hQ > 0 else 1
                q_thumb = cv2.resize(face_crop, (int(wQ * scale), int(hQ * scale)))

                # Load match images
                for m in matches:
                    # Try to find image in various locations
                    img_file = m['image_file']
                    match_img = None
                    
                    # Strategy 1: Check runtime crops directory
                    crop_path = Path(self.config.annotated_frames_dir) / "crops" / img_file
                    if crop_path.exists():
                        match_img = cv2.imread(str(crop_path))
                    
                    # Strategy 2: Check if it's a runtime embedding - look up in runtime index
                    if match_img is None:
                        for item in self.runtime_items:
                            if item.get('image_file') == img_file:
                                # Try crops directory with the image_file name
                                crop_path = Path(self.config.annotated_frames_dir) / "crops" / img_file
                                if crop_path.exists():
                                    match_img = cv2.imread(str(crop_path))
                                # Also try with frame_id_track_id pattern from runtime index
                                if match_img is None and 'frame_id' in item and 'track_id' in item:
                                    alt_name = f"{item['frame_id']}_{item['track_id']}.jpg"
                                    alt_path = Path(self.config.annotated_frames_dir) / "crops" / alt_name
                                    if alt_path.exists():
                                        match_img = cv2.imread(str(alt_path))
                                break
                    
                    # Strategy 3: Check precomputed dataset directories
                    if match_img is None:
                        # Try known_faces directory
                        dataset_path = Path(self.config.embeddings_dir).parent / "data" / "known_faces" / img_file
                        if dataset_path.exists():
                            match_img = cv2.imread(str(dataset_path))
                    
                    # Strategy 4: Check embeddings directory itself (for precomputed)
                    if match_img is None:
                        emb_path = Path(self.config.embeddings_dir) / img_file
                        if emb_path.exists():
                            match_img = cv2.imread(str(emb_path))

                    if match_img is None:
                        # Create placeholder
                        match_img = np.zeros((max_h, max_h, 3), dtype=np.uint8)
                        cv2.putText(match_img, "Not found", (10, max_h//2), font, 0.5, (255, 255, 255), 1)
                    else:
                        h, w = match_img.shape[:2]
                        sc = max_h / max(h, 1) if h > 0 else 1
                        match_img = cv2.resize(match_img, (int(w * sc), int(h * sc)))
                    
                    text = f"{img_file[:20]} {m['score01']:.3f}"
                    cv2.putText(match_img, text, (5, match_img.shape[0] - 6), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    thumbs.append(match_img)

                # Create canvas
                spacer = 10
                total_w = q_thumb.shape[1] + spacer + sum(t.shape[1] + spacer for t in thumbs)
                final_h = max(q_thumb.shape[0], max((t.shape[0] for t in thumbs), default=0))
                canvas = np.zeros((final_h + 20, total_w + 20, 3), dtype=np.uint8)
                
                # Add query image
                x = 10
                canvas[10:10 + q_thumb.shape[0], x:x + q_thumb.shape[1]] = q_thumb
                cv2.putText(canvas, "Query", (x, 5), font, 0.5, (0, 255, 0), 1)
                x += q_thumb.shape[1] + spacer
                
                # Add matches
                for i, t in enumerate(thumbs):
                    canvas[10:10 + t.shape[0], x:x + t.shape[1]] = t
                    cv2.putText(canvas, f"Match {i+1}", (x, 5), font, 0.5, (0, 255, 255), 1)
                    x += t.shape[1] + spacer

                # Save result
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_filename = f"{img_path.stem}_face{face_idx}_{timestamp}.jpg"
                result_path = results_dir / result_filename
                cv2.imwrite(str(result_path), canvas)
                self.logger.info(f"Saved recognition result: {result_filename}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}", exc_info=True)


# ============================================================================
# Main Pipeline
# ============================================================================

class Pipeline:
    """Main pipeline coordinator"""
    def __init__(self, config: Config):
        self.config = config
        self.stop_event = threading.Event()
        self.metrics = Metrics()
        self.logger = self._setup_logging()
        self.queues = {}
        self.threads = []
        self.health_server = None
        self.health_thread = None
        self.dead_letter_queue = DeadLetterQueue(max_size=1000)

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("Pipeline")
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def start(self):
        """Start all pipeline threads"""
        self.logger.info("Starting pipeline...")

        # Create queues
        self.queues['detection'] = Queue(maxsize=self.config.queue_size)
        self.queues['tracking'] = Queue(maxsize=self.config.queue_size)
        self.queues['embedding'] = Queue(maxsize=self.config.queue_size)
        self.queues['recognition'] = Queue(maxsize=self.config.queue_size)
        self.queues['frame'] = Queue(maxsize=self.config.queue_size) if self.config.crowd_enabled else None

        # Create and start threads
        self.threads.append(VideoReaderThread(
            self.config, self.queues['detection'], self.queues['frame'],
            self.stop_event, self.metrics, self.logger, self.dead_letter_queue
        ))

        self.threads.append(TrackerThread(
            self.config, self.queues['detection'], self.queues['tracking'],
            self.stop_event, self.metrics, self.logger, self.dead_letter_queue
        ))

        self.threads.append(EmbeddingThread(
            self.config, self.queues['tracking'], self.queues['embedding'],
            self.stop_event, self.metrics, self.logger, self.dead_letter_queue
        ))

        self.threads.append(RecognitionThread(
            self.config, self.queues['embedding'],
            self.stop_event, self.metrics, self.logger
        ))

        if self.config.crowd_enabled:
            self.threads.append(CrowdMonitorThread(
                self.config, self.queues['frame'],
                self.stop_event, self.metrics, self.logger
            ))

        if self.config.batch_recognition_enabled:
            self.threads.append(BatchRecognitionThread(
                self.config,
                self.stop_event, self.metrics, self.logger
            ))

        # Start all threads
        for thread in self.threads:
            thread.start()
            self.logger.info(f"Started thread: {thread.name}")

        # Start health server
        self._start_health_server()

        self.logger.info("Pipeline started. All threads running.")

    def _start_health_server(self):
        """Start HTTP health endpoint"""
        def run_server():
            HealthHandler.metrics = self.metrics
            HealthHandler.thread_status = {
                t.name: 'alive' if t.is_alive() else 'dead'
                for t in self.threads
            }
            server = HTTPServer(('localhost', self.config.health_port), HealthHandler)
            self.health_server = server
            server.serve_forever()

        self.health_thread = threading.Thread(target=run_server, daemon=True)
        self.health_thread.start()
        self.logger.info(f"Health endpoint started on http://localhost:{self.config.health_port}/health")

    def stop(self):
        """Gracefully stop all threads"""
        self.logger.info("Stopping pipeline...")
        self.stop_event.set()

        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                self.logger.warning(f"Thread {thread.name} did not stop gracefully")

        # Stop health server
        if self.health_server:
            self.health_server.shutdown()

        # Print final stats
        stats = self.metrics.get_stats()
        self.logger.info("Final metrics:")
        for k, v in stats.items():
            self.logger.info(f"  {k}: {v}")

        # Report dead-letter queue
        dlq_items = self.dead_letter_queue.get_all()
        if dlq_items:
            self.logger.warning(f"Dead-letter queue contains {len(dlq_items)} failed messages")
            # Optionally save to file
            dlq_path = Path("dead_letter_queue.json")
            try:
                with open(dlq_path, 'w') as f:
                    json.dump(dlq_items[:100], f, indent=2, default=str)
                self.logger.info(f"Saved dead-letter queue to {dlq_path}")
            except Exception as e:
                self.logger.error(f"Failed to save dead-letter queue: {e}")

        self.logger.info("Pipeline stopped")

    def run(self):
        """Run pipeline until interrupted"""
        self.start()

        # Setup signal handlers
        def signal_handler(sig, frame):
            self.logger.info("Received interrupt signal")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Keep main thread alive
            while not self.stop_event.is_set():
                time.sleep(1)
                # Update thread status for health endpoint
                HealthHandler.thread_status = {
                    t.name: 'alive' if t.is_alive() else 'dead'
                    for t in self.threads
                }
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()


# ============================================================================
# CLI & Main
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Real-time Face Recognition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--video-source', type=str, default='0',
                       help='Video source: webcam index (0), file path, or RTSP URL')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--queue-size', type=int, default=None,
                       help='Queue size for inter-thread communication')
    parser.add_argument('--drop-policy', choices=['drop-new', 'drop-old'], default=None,
                       help='Backpressure policy when queue is full')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda or cpu')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=None,
                       help='Logging level')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry-run mode (webcam, minimal logging)')
    parser.add_argument('--embeddings-dir', type=str, default=None,
                       help='Directory containing embeddings index')
    parser.add_argument('--recognition-threshold', type=float, default=None,
                       help='Recognition similarity threshold')
    parser.add_argument('--recognition-topk', type=int, default=None,
                       help='Number of top matches to return')
    parser.add_argument('--crowd-enabled', action='store_true', default=None,
                       help='Enable crowd monitoring')
    parser.add_argument('--no-crowd', action='store_true',
                       help='Disable crowd monitoring')
    parser.add_argument('--health-port', type=int, default=None,
                       help='Port for health endpoint')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        config = Config.from_yaml(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = Config()
        if args.config:
            print(f"Warning: Config file {args.config} not found, using defaults")

    # Override with CLI args
    if args.video_source:
        try:
            config.video_source = int(args.video_source) if args.video_source.isdigit() else args.video_source
        except:
            config.video_source = args.video_source
    if args.queue_size is not None:
        config.queue_size = args.queue_size
    if args.drop_policy:
        config.drop_policy = args.drop_policy
    if args.device:
        config.device = args.device
    if args.log_level:
        config.log_level = args.log_level
    if args.embeddings_dir:
        config.embeddings_dir = args.embeddings_dir
    if args.recognition_threshold is not None:
        config.recognition_threshold = args.recognition_threshold
    if args.recognition_topk is not None:
        config.recognition_topk = args.recognition_topk
    if args.crowd_enabled:
        config.crowd_enabled = True
    if args.no_crowd:
        config.crowd_enabled = False
    if args.health_port is not None:
        config.health_port = args.health_port

    # Dry-run mode adjustments
    if args.dry_run:
        config.video_source = 0  # Webcam
        config.log_level = "INFO"
        config.crowd_enabled = False
        print("Running in DRY-RUN mode")

    # Create and run pipeline
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
