"""
Multi-RTSP Real-time Crowd Density Monitor using CSRNet
--------------------------------------------------------
This script connects to multiple RTSP streams concurrently,
estimates crowd density in real-time, and triggers alerts
if the crowd count exceeds a threshold.
"""

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import threading
import queue
import time
from collections import deque


# ============================================
# 1️⃣ CSRNet Model Definition
# ============================================
def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M',
                              256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            for i in range(len(list(self.frontend.state_dict().items()))):
                list(self.frontend.state_dict().items())[i][1].data[:] = \
                    list(mod.state_dict().items())[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


# ============================================
# 2️⃣ Configuration
# ============================================
def initialize():
    class Config:
        # List of RTSP links
        rtsp_streams = [
            "rtsp://user:pass@192.168.1.10:554/stream1",
            "rtsp://user:pass@192.168.1.11:554/stream1",
            "rtsp://user:pass@192.168.1.12:554/stream1"
        ]

        model_path = r"task_two_model_best.pth.tar"
        use_cuda = torch.cuda.is_available()
        frame_resize = (640, 360)
        fps_target = 10
        crowd_threshold = 1000
        frame_queue_size = 10
    return Config()


# ============================================
# 3️⃣ Load Model (shared for all threads)
# ============================================
def load_csrnet(cfg):
    model = CSRNet()
    checkpoint = torch.load(cfg.model_path, map_location='cuda' if cfg.use_cuda else 'cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if cfg.use_cuda:
        model = model.cuda()
    print("[info] CSRNet model loaded successfully.")
    return model


# ============================================
# 4️⃣ Thread Worker: Each Stream Handler
# ============================================
def stream_worker(rtsp_url, model, cfg, stream_id):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[error] Could not open stream {stream_id}: {rtsp_url}")
        return

    counts_window = deque(maxlen=5)
    last_alert_time = 0
    alert_interval = 5  # seconds

    print(f"[info] Stream {stream_id} started ({rtsp_url})")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[warn] Stream {stream_id} frame read failed.")
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, cfg.frame_resize)
        img_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)

        if cfg.use_cuda:
            img_tensor = img_tensor.cuda(non_blocking=True)

        # Inference
        with torch.no_grad():
            output = model(img_tensor)
        density_map = output.squeeze().cpu().numpy()
        count = np.sum(density_map)
        counts_window.append(count)
        avg_count = np.mean(counts_window)

        # Display count
        cv2.putText(frame, f"Count: {avg_count:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Camera {stream_id}", frame)

        # Alert check
        if avg_count > cfg.crowd_threshold and (time.time() - last_alert_time) > alert_interval:
            print(f"🚨 ALERT [Camera {stream_id}] Crowd Density Exceeded: {avg_count:.2f}")
            last_alert_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"[info] Stopping Stream {stream_id}")
            break

        # Limit FPS
        time.sleep(1 / cfg.fps_target)

    cap.release()
    cv2.destroyWindow(f"Camera {stream_id}")


# ============================================
# 5️⃣ Main Execution
# ============================================
def main():
    cfg = initialize()
    model = load_csrnet(cfg)

    threads = []
    for i, rtsp_link in enumerate(cfg.rtsp_streams):
        t = threading.Thread(
            target=stream_worker,
            args=(rtsp_link, model, cfg, i + 1),
            daemon=True
        )
        threads.append(t)
        t.start()
        time.sleep(0.5)  # stagger thread startup

    print(f"[info] Monitoring {len(threads)} RTSP streams. Press 'q' to quit.\n")

    try:
        # Keep alive until user interrupts
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user. Exiting...")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
