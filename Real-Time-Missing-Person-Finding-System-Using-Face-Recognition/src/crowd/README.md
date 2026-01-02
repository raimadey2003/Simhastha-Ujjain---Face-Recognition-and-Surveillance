# Crowd Density Predictor

This repository contains Python scripts for crowd density prediction using CSRNet.

## Files

- `predict.py`: A script to predict crowd density on a single image.
- `crowd_monitor.py`: A script to monitor crowd density from multiple RTSP streams in real-time.
- `task_two_checkpoint.pth.tar`: Model checkpoint file.
- `task_two_model_best.pth.tar`: Best performing model file.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SupratimGhosh/crowd_density_predictor.git
    cd crowd_density_predictor
    ```

2.  **Install Git LFS:**

    This project uses Git LFS to handle large model files (`.pth.tar`). Make sure you have Git LFS installed on your system. If not, you can install it from [here](https://git-lfs.github.com/).

    After installing, run:

    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Install required Python packages:**

    ```bash
    pip install torch torchvision Pillow numpy opencv-python
    ```

## Usage

### Predict Crowd Density on a Single Image

To predict crowd density on an image using `predict.py`:

1.  Place your image (e.g., `crowd.jpg`) in the same directory as `predict.py`.
2.  Update the `image_path` variable in `predict.py` if your image has a different name.
3.  Run the script:

    ```bash
    python predict.py
    ```

    The script will output the estimated crowd count and save a density map as `density_map_output.png`.

### Real-time Crowd Monitoring from RTSP Streams

To monitor crowd density from RTSP streams using `crowd_monitor.py`:

1.  Update the `rtsp_streams` list in the `initialize` function within `crowd_monitor.py` with your RTSP stream URLs.
2.  Run the script:

    ```bash
    python crowd_monitor.py
    ```

    The script will open windows for each stream, displaying the live feed with estimated crowd counts. Alerts will be triggered if the crowd count exceeds the defined `crowd_threshold`.