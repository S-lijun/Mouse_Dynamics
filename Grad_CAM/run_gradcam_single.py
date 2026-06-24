import os
import sys
import torch
import cv2
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ======================================================
# Project root
# ======================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

# ======================================================
# Import model
# ======================================================

from models.pretrained_googlenet_multi import PretrainedGoogLeNet_Multilabel


# ======================================================
# Paths
# ======================================================

# 换成 GoogLeNet 训练保存的 checkpoint（文件名按实际 timestamp 改）
MODEL_PATH = os.path.join(
    ROOT,
    "saved_models",
    "multilabel_P1_best_20260228_052714.pth"
)

IMAGE_PATH = os.path.join(
    ROOT,
    "Images",
    "Balabit",
    "SRP",
    "event120",
    "user15",
    "session_0205904470-3.png"
)

OUTPUT_ROOT = os.path.join(
    ROOT,
    "Grad_CAM",
    "outputs"
)


# ======================================================
# Model config
# ======================================================

NUM_USERS = 10
TARGET_USER = 1


# ======================================================
# Parse image info
# ======================================================

user_name = os.path.basename(os.path.dirname(IMAGE_PATH))
session_name = os.path.basename(IMAGE_PATH)

print("User :", user_name)
print("Session :", session_name)


# ======================================================
# Load model
# ======================================================

print("Loading model...")

model = PretrainedGoogLeNet_Multilabel(num_users=NUM_USERS)

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint)

model.eval()

print("Model loaded")


# ======================================================
# Target layer for Grad-CAM
# ======================================================

# 对应 ScratchCNN 的 stage5 最后一层 conv：GoogLeNet 用最后一个 Inception block
target_layers = [model.base.inception5b]

print("Grad-CAM target layer:", target_layers)


# ======================================================
# Load image
# ======================================================

print("Loading image:", IMAGE_PATH)

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise RuntimeError("Image not found")

img = cv2.resize(img, (224, 224))

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb_float = rgb_img.astype(np.float32) / 255.0

input_tensor = torch.tensor(rgb_float).permute(2,0,1).unsqueeze(0).float()


# ======================================================
# Grad-CAM
# ======================================================

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

targets = [ClassifierOutputTarget(TARGET_USER)]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

heatmap = grayscale_cam[0]


# ======================================================
# Overlay heatmap
# ======================================================

visualization = show_cam_on_image(
    rgb_float,
    heatmap,
    use_rgb=True
)


# ======================================================
# Save result
# ======================================================

user_output_dir = os.path.join(OUTPUT_ROOT, user_name)

os.makedirs(user_output_dir, exist_ok=True)

output_path = os.path.join(user_output_dir, session_name)

cv2.imwrite(
    output_path,
    cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
)

print("Grad-CAM saved to:", output_path)