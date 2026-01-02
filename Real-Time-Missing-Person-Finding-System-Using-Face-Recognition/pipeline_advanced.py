
"""
Advanced Face Recognition Pipeline
==================================

Implemented as per the requirements:
1. Sequential/Parallel Pipeline.
2. 10th frame detection.
3. Cropping & Temporary Saving.
4. Embedding Calculation.
5. Recognition against MongoDB fetched images.
6. Logging.
7. Cleanup on restart.

Usage:
    python pipeline_advanced.py
"""

import os
import sys
import time
import shutil
import logging
import argparse
import threading
import subprocess
import cv2
import torch
import numpy as np
from queue import Queue, Empty
from datetime import datetime
from pathlib import Path

# Add src to path if needed (though running from root usually works)
sys.path.append(os.getcwd())

# Import existing modules
try:
    from src.detection.face2 import FaceDetector
    from src.recognition.face_recog_core import (
        load_model, 
        make_transform, 
        get_embedding_pytorch, 
        precompute_embeddings, 
        load_embeddings_index, 
        match_query,
        annotate_and_save
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import required modules. Make sure you are in the root directory. {e}")
    sys.exit(1)

# ================= Configuration =================
def parser_args():
    parser = argparse.ArgumentParser(description="Advanced Face Recognition Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or RTSP url)")
    parser.add_argument("--conf", type=float, default=0.5, help="Face detection confidence threshold")
    parser.add_argument("--threshold", type=float, default=0.5, help="Recognition cosine similarity threshold")
    parser.add_argument("--db-interval", type=int, default=60, help="Seconds between DB updates")
    return parser.parse_args()

args = parser_args()

VIDEO_SOURCE = int(args.source) if args.source.isdigit() else args.source
CONF_THRESH = args.conf
RECOGNITION_THRESHOLD = args.threshold
CHECK_DB_INTERVAL = args.db_interval

# Directories
DIRS = {
    "crops": "temp_crops",
    "embeddings": "temp_embeddings",
    "exported_images": "exported_images",
    "reference_embeddings": "reference_embeddings",
    "logs": "logs"
}



# ================= Logger =================
def setup_logger():
    os.makedirs(DIRS["logs"], exist_ok=True)
    log_file = os.path.join(DIRS["logs"], f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("Pipeline")

logger = setup_logger()

# ================= Cleanup =================
def cleanup_temp_dirs():
    """Clear cropped images, embeddings, and other temporary files."""
    logger.info("Cleaning up temporary directories...")
    for key in ["crops", "embeddings"]:
        path = DIRS[key]
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                logger.info(f"Removed {path}")
            except Exception as e:
                logger.error(f"Failed to remove {path}: {e}")
        os.makedirs(path, exist_ok=True)
    
    # Ensure other dirs exist
    for key in ["exported_images", "reference_embeddings", "logs"]:
        os.makedirs(DIRS[key], exist_ok=True)

# ================= Implementation =================

# Helper for location
def get_device_location():
    try:
        import requests
        response = requests.get('http://ip-api.com/json/', timeout=5)
        data = response.json()
        if data.get('status') == 'success':
            # Format: City, Region, Lat, Lon
            return f"{data.get('city')}, {data.get('regionName')} ({data.get('lat')}, {data.get('lon')})"
        return "Unknown Location (IP Lookup Failed)"
    except Exception as e:
        logger.warning(f"Failed to fetch location: {e}")
        return "Unknown Location"

class Pipeline:
    def __init__(self):
        self.running = True
        self.frame_queue = Queue(maxsize=10)     # Frames to process
        self.crop_queue = Queue(maxsize=50)      # Crops to embed
        self.embed_queue = Queue(maxsize=50)     # Embeddings to recognize
        
        # Models
        self.detector = None
        self.recog_model = None
        self.recog_device = None
        self.recog_transform = None
        
        # Reference Data
        self.reference_index = []
        self.reference_lock = threading.Lock()

        # ============ CONFIG =============
        # Load environment variables from .env (preferred) or fall back to .env.example
        try:
            from dotenv import load_dotenv
            _HAS_DOTENV = True
        except Exception:
            _HAS_DOTENV = False

        # Prefer a real .env; if not present, try .env.example (useful for templates)
        if _HAS_DOTENV:
            if os.path.exists('.env'):
                load_dotenv('.env')
            elif os.path.exists('.env.example'):
                load_dotenv('.env.example')

        # small helper
        def _env_str(key, default):
            v = os.getenv(key)
            return v if v is not None and v != "" else default

        MONGODB_URI = _env_str('MONGODB_URI', 'YOUR_MONGODB_URI_HERE')
        
        # Alert Manager
        from src.alerts.alert_manager import AlertManager
        self.alert_manager = AlertManager(mongo_uri=MONGODB_URI, cooldown_seconds=300)
        
        # Location
        self.location = get_device_location()
        logger.info(f"Device Location set to: {self.location}")
        
        # Shared frame for crowd monitor
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()

        # Initialize
        cleanup_temp_dirs()
        self.init_models()
    


    def init_models(self):
        logger.info("Initializing models...")
        # Detector
        # Note: FaceDetector internally might use CUDA.
        self.detector = FaceDetector(conf_thresh=CONF_THRESH)
        
        # Recognition
        self.recog_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.recog_model = load_model(device=self.recog_device)
        self.recog_transform = make_transform()
        logger.info(f"Models initialized. Recognition device: {self.recog_device}")

    def update_reference_db(self):
        """
        1. Clean old data (exported_images, reference_embeddings).
        2. Fetch images from MongoDB (using fetch_image_db.py).
        3. Detect faces in fetched images and crop them.
        4. Compute embeddings for crops.
        5. Save and index.
        """
        logger.info("Updating reference database...")
        
        # 1. Cleanup
        try:
            for d in [DIRS["exported_images"], DIRS["reference_embeddings"]]:
                if os.path.exists(d):
                    logger.info(f"Clearing old data in {d}...")
                    for filename in os.listdir(d):
                        file_path = os.path.join(d, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            logger.error(f"Failed to delete {file_path}. Reason: {e}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        # 2. Fetch
        fetch_script = os.path.join("lost_images", "fetch_image_db.py")
        if os.path.exists(fetch_script):
            try:
                logger.debug("Running fetch_image_db.py...")
                subprocess.run([sys.executable, fetch_script], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.error(f"Failed to run fetch_image_db.py: {e}")
        
        # 3. Smart Precompute (Detect -> Crop -> Embed)
        logger.info("Processing reference images (Detect -> Crop -> Embed)...")
        
        embeddings_dir = Path(DIRS["reference_embeddings"])
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = Path(DIRS["exported_images"])
        
        index_csv = embeddings_dir / 'embeddings_index.csv'
        rows = []
        
        # Import needed for smart processing
        from src.recognition.face_recog_core import read_image, get_embedding_pytorch
        from PIL import Image
        
        image_paths = sorted([p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        # Use a temporary list to avoid updating index while reading
        new_rows = []
        count_processed = 0
        
        for p in image_paths:
            try:
                # We need BGR for detection, but read_image returns PIL RGB
                # Let's read with cv2 for detection first
                img_bgr = cv2.imread(str(p))
                if img_bgr is None:
                    continue
                    
                # Detect
                detections = self.detector.detect(img_bgr)
                
                # If faces found, pick the largest/most confident
                face_crop = None
                
                if len(detections) > 0:
                    # Sort by confidence (index 4)
                    detections.sort(key=lambda x: x[4], reverse=True)
                    x1, y1, x2, y2, conf, _ = detections[0]
                    
                    h, w = img_bgr.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        face_crop = img_bgr[y1:y2, x1:x2]
                        # logger.debug(f"Face detected in {p.name}")
                else:
                    logger.warning(f"No face detected in reference image {p.name}. Skipping.")
                    continue
                
                if face_crop is None:
                    continue

                # Convert crop to PIL for embedding
                img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # Embed
                emb = get_embedding_pytorch(img_pil, self.recog_model, self.recog_device, self.recog_transform)
                if emb is None:
                    continue
                    
                emb_file = embeddings_dir / (p.stem + '.npy')
                np.save(str(emb_file), emb)
                new_rows.append([str(p.name), str(emb_file.name)])
                count_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing reference image {p}: {e}")

        # Save index
        import csv
        with open(index_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_file', 'embedding_file'])
            writer.writerows(new_rows)
            
        logger.info(f"Ref DB Update Complete. Processed {count_processed} identities.")

        # 3. Reload Index
        with self.reference_lock:
            try:
                self.reference_index = load_embeddings_index(DIRS["reference_embeddings"])
                logger.info(f"Loaded {len(self.reference_index)} reference identities.")
            except Exception as e:
                logger.error(f"Error loading reference index: {e}")

    # ================= Threads =================
    
    def thread_capture(self):
        """Captures frames from video source."""
        logger.info(f"Starting Capture Thread. Source: {VIDEO_SOURCE}")
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        
        frame_count = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame or end of stream. Retrying...")
                time.sleep(1)
                # simple reconnect logic could go here
                cap.release()
                cap = cv2.VideoCapture(VIDEO_SOURCE)
                continue
            
            frame_count += 1
            
            # Update latest frame shared variable
            with self.latest_frame_lock:
                 self.latest_frame = frame.copy()
            
            # Process every 10th frame
            if frame_count % 10 == 0:
                try:
                    # Put frame in queue (blocking if full to avoid memory overflow)
                    self.frame_queue.put((frame_count, frame.copy()), timeout=1)
                except:
                   pass # Drop frame if full
            
            # (Optional) Display feed logic can go here or in main thread
            # For this pipeline, we might just show it in the main loop or not at all.
            # Let's assume UI is handled separately or we just log.
            
        cap.release()
        logger.info("Capture Thread stopped.")

    def thread_detection(self):
        """
        Consumes frames.
        Detects faces.
        Crops and saves.
        Producers to crop_queue.
        """
        logger.info("Starting Detection Thread.")
        while self.running:
            try:
                frame_id, frame = self.frame_queue.get(timeout=1)
            except Empty:
                continue
            
            try:
                detections = self.detector.detect(frame)
                if len(detections) > 0:
                    logger.info(f"Frame {frame_id}: Detected {len(detections)} faces.")
                    
                    for i, det in enumerate(detections):
                        # Unpack (x1, y1, x2, y2, conf, landmarks)
                        x1, y1, x2, y2, conf, _ = det
                        
                        # Sanity check
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Crop
                        face_crop = frame[y1:y2, x1:x2]
                        
                        # Save temporarily
                        timestamp = datetime.now().strftime("%H%M%S%f")
                        filename = f"face_{frame_id}_{i}_{timestamp}.jpg"
                        filepath = os.path.join(DIRS["crops"], filename)
                        cv2.imwrite(filepath, face_crop)
                        
                        # Push to embedding queue
                        # We pass the image array too so we don't have to re-read it, 
                        # but we also pass filepath for reference/logging.
                        self.crop_queue.put((filepath, face_crop, frame_id))
                        
            except Exception as e:
                logger.error(f"Error in detection: {e}")
            
            self.frame_queue.task_done()

    def thread_embedding(self):
        """
        Consumes crops.
        Computes embeddings.
        Saves temporarily.
        Produces to embed_queue.
        """
        logger.info("Starting Embedding Thread.")
        while self.running:
            try:
                filepath, face_img_bgr, frame_id = self.crop_queue.get(timeout=1)
            except Empty:
                continue
            
            try:
                # Convert BGR (cv2) to RGB (PIL) for facenet
                img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
                
                # Check FaceRecogCore usage:
                # it has read_image which does cvtColor and returns PIL.
                # get_embedding_pytorch takes PIL image
                
                from PIL import Image
                img_pil = Image.fromarray(img_rgb)
                
                emb = get_embedding_pytorch(
                    img_pil, 
                    self.recog_model, 
                    self.recog_device, 
                    self.recog_transform
                )
                
                # Save embedding temporarily
                emb_filename = Path(filepath).stem + ".npy"
                emb_path = os.path.join(DIRS["embeddings"], emb_filename)
                np.save(emb_path, emb)
                
                self.embed_queue.put((emb, filepath, frame_id))
                
            except Exception as e:
                logger.error(f"Error in embedding: {e}")
                
            self.crop_queue.task_done()

    def thread_recognition(self):
        """
        Consumes embeddings.
        Matches against Reference DB.
        Logs results.
        """
        logger.info("Starting Recognition Thread.")
        
        # Log file for matches
        match_log_path = os.path.join("recognition_log.txt")
        
        while self.running:
            try:
                emb, crop_path, frame_id = self.embed_queue.get(timeout=1)
            except Empty:
                continue
            
            try:
                matches = []
                with self.reference_lock:
                    if self.reference_index:
                        # Match - Get top 5 candidates to log "near misses"
                        matches = match_query(
                            self.reference_index, 
                            emb, 
                            topk=5, 
                            threshold=0.5 # Get all top k results regardless of threshold initially
                        )
                
                if matches:
                    top_match = matches[0]
                    score = top_match['score01']
                    
                    # Log the top candidate regardless of threshold for debugging
                    logger.debug(f"Frame {frame_id}: Top match {top_match['image_file']} with score {score:.3f}")
                    
                    if score >= RECOGNITION_THRESHOLD:
                        # Extract person_id from filename (split by first '_')
                        image_file = top_match['image_file']
                        person_id = image_file.split('_')[0]

                        msg = (f"[MATCH FOUND] Frame: {frame_id} | "
                               f"Person ID: {person_id} (File: {image_file}) | "
                               f"Score: {score:.3f} | "
                               f"Source Crop: {crop_path}")
                        
                        logger.info(msg)
                        
                        # Log to file
                        with open(match_log_path, "a") as f:
                            f.write(f"{datetime.now().isoformat()} - {msg}\n")
                            
                        # Trigger Alert
                        self.alert_manager.send_alert(
                            person_id=person_id,
                            location=self.location, 
                            confidence=score,
                            image_path=image_file
                        )
                    else:
                        logger.info(f"Frame {frame_id}: Near miss - {top_match['image_file']} ({score:.3f} < {RECOGNITION_THRESHOLD})")
                        
            except Exception as e:
                logger.error(f"Error in recognition: {e}")
            
            self.embed_queue.task_done()

    def thread_db_updater(self):
        """
        Periodically updates the reference database.
        """
        logger.info("Starting DB Updater Thread.")
        while self.running:
            try:
                self.update_reference_db()
            except Exception as e:
                logger.error(f"DB Update failed: {e}")
            
            # Sleep for interval
            for _ in range(CHECK_DB_INTERVAL):
                if not self.running: break
                time.sleep(1)

    def thread_crowd(self):
        """
        Monitors crowd density every minute.
        """
        logger.info("Starting Crowd Monitor Thread.")
        
        # Import CSRNet locally to avoid global dependency if not needed
        try:
            from src.crowd.crowd_monitor import CSRNet, load_csrnet
            from torchvision import transforms
            
            # Config mock for load_csrnet
            class CrowdConfig:
                model_path = r"src/crowd/task_two_model_best.pth.tar" # Adjust path as needed
                use_cuda = torch.cuda.is_available()
            
            crowd_cfg = CrowdConfig()
            if not os.path.exists(crowd_cfg.model_path):
                 # Fallback to root if src path fails
                 crowd_cfg.model_path = r"task_two_model_best.pth.tar"
                 
            crowd_model = load_csrnet(crowd_cfg)
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            
            crowd_threshold = 100
            
            while self.running:
                # Sleep first or run first? Let's run then sleep.
                # Use a lock to get the latest frame safely
                frame_to_process = None
                with self.latest_frame_lock:
                    if self.latest_frame is not None:
                        frame_to_process = self.latest_frame.copy()
                
                if frame_to_process is not None:
                    try:
                        # Preprocess
                        # Resize for consistent inference speed/size (optional but recommended in original script)
                        frame_resized = cv2.resize(frame_to_process, (640, 360))
                        img_tensor = transform(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                        
                        if crowd_cfg.use_cuda:
                            img_tensor = img_tensor.cuda()
                            
                        with torch.no_grad():
                            output = crowd_model(img_tensor)
                            
                        density_map = output.squeeze().cpu().numpy()
                        count = np.sum(density_map)
                        
                        logger.info(f"[Crowd Monitor] Current Count: {count:.2f}")
                        
                        if count > crowd_threshold:
                            logger.warning(f"Crowd Density Exceeded: {count:.2f}")
                            self.alert_manager.send_alert(
                                person_id="CROWD_ALERT",
                                location=self.location,
                                confidence=count,
                                image_path="crowd_snapshot.jpg", # Placeholder or save actual snapshot
                                message=f"High crowd density detected: {count:.0f} people at {self.location}"
                            )
                            
                    except Exception as e:
                        logger.error(f"Error in crowd inference: {e}")
                else:
                    logger.debug("No frame available for crowd monitoring yet.")

                # Wait for 5 minutes (300 seconds)
                for _ in range(300):
                    if not self.running: break
                    time.sleep(1)
                    
        except ImportError:
            logger.error("Could not import source.crowd.crowd_monitor. Crowd thread disabled.")
        except Exception as e:
            logger.error(f"Crowd thread failed: {e}")

    def start(self):
        # Start reference DB update in main or background first?
        # Let's do it once synchronously so we have data
        self.update_reference_db()
        
        threads = [
            threading.Thread(target=self.thread_capture, name="Capture"),
            threading.Thread(target=self.thread_detection, name="Detection"),
            threading.Thread(target=self.thread_embedding, name="Embedding"),
            threading.Thread(target=self.thread_recognition, name="Recognition"),
            threading.Thread(target=self.thread_db_updater, name="DBUpdater"),
            threading.Thread(target=self.thread_crowd, name="CrowdMonitor")
        ]
        
        for t in threads:
            t.start()
            
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping pipeline...")
            self.running = False
            
        for t in threads:
            t.join()
        logger.info("Pipeline terminated.")

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.start()
