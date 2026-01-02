# Face Detection, Tracking & Recognition System

This project implements an **end-to-end real-time face detection, tracking, and recognition pipeline** using YOLOv8-Face, DeepSORT, and FaceNet (InceptionResnetV1).

---

## 1. System Overview

### Core Capabilities

* Real-time face detection with landmarks
* Persistent face ID tracking across frames
* Face crop extraction from live video
* Face embedding generation (512-D)
* Offline embedding database creation
* Face similarity search using cosine similarity

### Technology Stack

* **Detection**: YOLOv8-Face (Ultralytics)
* **Tracking**: DeepSORT (deep-sort-realtime)
* **Recognition**: FaceNet (InceptionResnetV1 – facenet-pytorch)
* **Backend**: Python, OpenCV, PyTorch

---



## 2. Environment Setup (Step-by-Step)

### Step 1: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

---

### Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install opencv-python ultralytics facenet-pytorch
pip install deep-sort-realtime scikit-learn pillow numpy
```

> ⚠️ For CUDA support, ensure compatible PyTorch + NVIDIA drivers are installed.

---

### Step 3: Download YOLOv8 Face Model

Download **YOLOv8n-Face** and place it in the project root:

```
yolov8n-face.pt
```

---

## 3. Running the System

### 3.1 Real-Time Face Detection + Tracking (Webcam)

This runs YOLOv8 face detection + DeepSORT tracking and saves face crops.

```bash
python -m src.detection.test1_dt 
```

#### What Happens Internally

1. Webcam frame captured
2. Faces detected with landmarks
3. DeepSORT assigns persistent IDs
4. Bounding boxes + IDs rendered
5. Face crops saved to `.save/`

Press **`q`** to exit.

---

## 4. Face Recognition Pipeline

### 4.1 Prepare Face Dataset

Using the real-time detected faces saved in .save/ as a dataset.

```
.save/
├── 1_20251227_215952.jpg
├── 1_20251227_215953.jpg
├── 1_20251227_215954.jpg
```

---

### 4.2 Precompute Face Embeddings

```bash
python -m src.recognition.precompute_embeddings --dataset_dir .save --precompute  --device cuda
```

#### Output

* `embeddings/*.npy`
* `embeddings/embeddings_index.csv`

---

### 4.3 Search / Identify a Face

```bash
python -m src.recognition.search_query --dataset_dir .save  --query sample3.jpg --topk 3  --show_image
```

#### Output

* Console similarity scores
* Saved result montage (`result_matches.jpg`)

---

## 5. End-to-End Logical Flow

```
Webcam Frame
   ↓
YOLOv8 Face Detection
   ↓
DeepSORT Tracking (Face ID)
   ↓
Face Crop Extraction
   ↓
FaceNet Embedding (512-D)
   ↓
Cosine Similarity Matching
```

---

## 6. Configuration Tips

* **Detection confidence**: `conf_thresh` in `face2.py`
* **Tracking stability**: `max_age`, `n_init` in `deep_sort.py`
* **Recognition threshold**: `--threshold` in `search_query.py`
* **Embedding model**: `vggface2` or `casia-webface`

---



## 7. Intended Use Cases

* Missing person identification
* Surveillance analytics
* Smart attendance systems
* Research & academic projects

---

**Author:** Srishti Majumdar
**Domain:** Computer Vision · Face Recognition
