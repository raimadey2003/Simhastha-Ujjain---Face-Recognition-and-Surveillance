# ===========================
# CSRNet Crowd Density Predictor
# ===========================

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os


# ============================================
# 1️⃣ Model Definition (Same as in training)
# ============================================
def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
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
# 2️⃣ Initialization: paths and parameters
# ============================================
def initialize():
    """
    All config variables stored here
    """
    class Config:
        # Path to pretrained checkpoint (.pth.tar)
        model_path = r".\task_two_model_best.pth.tar"

        # Path to image to predict on
        image_path = r"crowd.jpg"

        # Whether to use CUDA
        use_cuda = torch.cuda.is_available()

        # Output density map path
        output_density_path = r"density_map_output.png"

    return Config()


# ============================================
# 3️⃣ Preprocessing and Prediction
# ============================================
def predict_density():
    cfg = initialize()

    # Load model
    model = CSRNet()
    checkpoint = torch.load(cfg.model_path, map_location='cuda' if cfg.use_cuda else 'cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if cfg.use_cuda:
        model = model.cuda()

    # Image preprocessing (same as training normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and transform image
    img = Image.open(cfg.image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    if cfg.use_cuda:
        img_tensor = img_tensor.cuda()

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)

    # Convert density map to numpy
    density_map = output.squeeze().cpu().numpy()
    count = np.sum(density_map)

    print(f"Estimated Crowd Count: {count:.2f}")

    # Optional: save density map as image
    norm_map = (density_map / density_map.max()) * 255
    norm_map = norm_map.astype(np.uint8)
    cv2.imwrite(cfg.output_density_path, norm_map)
    print(f"Density map saved at: {cfg.output_density_path}")

    return count


# ============================================
# 4️⃣ Run
# ============================================
if __name__ == "__main__":
    predict_density()
