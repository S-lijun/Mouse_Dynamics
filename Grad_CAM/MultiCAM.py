import argparse
import os
import sys

import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models.pretrained_googlenet_multi import PretrainedGoogLeNet_Multilabel

sys.path.insert(0, os.path.dirname(__file__))
from run_gradcam_single import (
    NUM_USERS,
    load_sample,
    load_tensor_dataset,
    resolve_project_path,
    resolve_sample_index,
)

# Tensor CHW layout from SRP_chunk_vxvy.rgb_to_tensor_chw (RGB order):
#   ch0 = R = Distance, ch1 = G = Vx, ch2 = B = Vy
CHANNELS = [
    {"idx": 0, "key": "R_distance", "label": "Distance (R)", "mode": "srp"},
    {"idx": 1, "key": "G_vx", "label": "Vx (G)", "mode": "ablate"},
    {"idx": 2, "key": "B_vy", "label": "Vy (B)", "mode": "ablate"},
]

DEFAULT_SRP_MODEL = os.path.join(
    ROOT, "saved_models", "multilabel_P1_best_20260629_204243.pth"
)


def infer_srp_tensor_root(tensor_root):
    normalized = tensor_root.replace("\\", "/")
    if "SRP_vxvy" in normalized:
        return normalized.replace("SRP_vxvy", "SRP")
    return None


def load_model(model_path):
    model = PretrainedGoogLeNet_Multilabel(num_users=NUM_USERS)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def ablate_channels(img, keep_idx):
    ablated = np.zeros_like(img)
    ablated[keep_idx] = img[keep_idx]
    return ablated


def channel_to_rgb_float(channel):
    gray = channel.astype(np.float32) / 255.0
    return np.stack([gray, gray, gray], axis=-1)


def save_rgb(path, image):
    if image.dtype == np.uint8:
        rgb_uint8 = image
    else:
        rgb_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))


def run_srp_distance_gradcam(model, cam, targets, srp_tensor_root, sample_index):
    """Same pipeline as run_gradcam_single.py on pure SRP tensors."""
    input_tensor, rgb_float, _, _ = load_sample(srp_tensor_root, sample_index)

    with torch.no_grad():
        target_user = targets[0].category
        score = torch.sigmoid(model(input_tensor))[0, target_user].item()

    heatmap = cam(input_tensor=input_tensor, targets=targets)[0]
    overlay = show_cam_on_image(rgb_float, heatmap, use_rgb=True)

    return rgb_float.copy(), overlay, score


def run_ablated_gradcam(model, cam, targets, img, keep_idx):
    ablated = ablate_channels(img, keep_idx)
    channel_bg = channel_to_rgb_float(ablated[keep_idx])
    input_tensor = torch.from_numpy(ablated).float().div(255).unsqueeze(0)

    with torch.no_grad():
        target_user = targets[0].category
        score = torch.sigmoid(model(input_tensor))[0, target_user].item()

    heatmap = cam(input_tensor=input_tensor, targets=targets)[0]
    overlay = show_cam_on_image(channel_bg, heatmap, use_rgb=True)

    return channel_bg, overlay, score


def main():
    parser = argparse.ArgumentParser(
        description="Channel-ablation Grad-CAM for SRP_vxvy models."
    )
    parser.add_argument(
        "--model",
        default=os.path.join(ROOT, "saved_models", "multilabel_P1_best_20260629_215456.pth"),
        help="Model for Vx / Vy stripe channels (SRP_vxvy-trained)",
    )
    parser.add_argument(
        "--distance-model",
        default=None,
        help="Model for Distance channel (SRP-trained). "
        "Default: auto-pick SRP checkpoint when --tensor-root contains SRP_vxvy",
    )
    parser.add_argument(
        "--tensor-root",
        default=os.path.join(ROOT, "ImagesTensors", "Balabit", "SRP_vxvy", "event125"),
        help="SRP_vxvy tensor folder",
    )
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--session", default=None, help="Session id prefix, e.g. session_0041905381")
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--target-user", type=int, default=None)
    parser.add_argument(
        "--srp-tensor-root",
        default=None,
        help="Pure SRP tensor folder for Distance channel. "
        "Default: auto from --tensor-root by replacing SRP_vxvy -> SRP",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(ROOT, "Grad_CAM", "outputs", "MultiCAM"),
    )
    args = parser.parse_args()

    args.model = resolve_project_path(args.model)
    args.tensor_root = resolve_project_path(args.tensor_root)
    args.output_root = resolve_project_path(args.output_root)

    srp_tensor_root = args.srp_tensor_root
    if srp_tensor_root is not None:
        srp_tensor_root = resolve_project_path(srp_tensor_root)
    else:
        inferred = infer_srp_tensor_root(args.tensor_root)
        if inferred is not None:
            srp_tensor_root = resolve_project_path(inferred)

    if args.distance_model is not None:
        distance_model_path = resolve_project_path(args.distance_model)
    elif srp_tensor_root is not None and "SRP_vxvy" in args.tensor_root.replace("\\", "/"):
        distance_model_path = DEFAULT_SRP_MODEL
    else:
        distance_model_path = args.model

    if args.sample_index is None and args.session is None:
        parser.error("Provide either --sample-index or --session")

    _, _, sessions = load_tensor_dataset(args.tensor_root)

    if args.sample_index is not None:
        sample_index = args.sample_index
    else:
        sample_index = resolve_sample_index(sessions, args.session, args.chunk_index)

    images, labels, _ = load_tensor_dataset(args.tensor_root)
    img = np.array(images[sample_index], copy=True)
    true_user = int(labels[sample_index].argmax())
    session_id = str(sessions[sample_index])
    target_user = true_user if args.target_user is None else args.target_user

    print("Tensor root     :", args.tensor_root)
    print("SRP ref root    :", srp_tensor_root)
    print("Vx/Vy model     :", args.model)
    print("Distance model  :", distance_model_path)
    print("Sample index    :", sample_index)
    print("Session id      :", session_id)
    print("True user       :", true_user)
    print("Target user     :", target_user)

    vxvy_model = load_model(args.model)
    distance_model = load_model(distance_model_path)

    vxvy_cam = GradCAM(model=vxvy_model, target_layers=[vxvy_model.base.inception5b])
    distance_cam = GradCAM(model=distance_model, target_layers=[distance_model.base.inception5b])
    targets = [ClassifierOutputTarget(target_user)]

    out_dir = os.path.join(
        args.output_root,
        f"user{true_user}",
        f"{session_id}_idx{sample_index}_target{target_user}",
    )
    os.makedirs(out_dir, exist_ok=True)

    for ch in CHANNELS:
        if ch["mode"] == "srp":
            if srp_tensor_root is None:
                raise RuntimeError("Distance channel requires a pure SRP tensor folder.")
            original, overlay, score = run_srp_distance_gradcam(
                distance_model, distance_cam, targets, srp_tensor_root, sample_index
            )
        else:
            original, overlay, score = run_ablated_gradcam(
                vxvy_model, vxvy_cam, targets, img, ch["idx"]
            )

        orig_path = os.path.join(out_dir, f"{ch['key']}_original.png")
        cam_path = os.path.join(out_dir, f"{ch['key']}_gradcam.png")

        save_rgb(orig_path, original)
        save_rgb(cam_path, overlay)

        print(f"[{ch['label']}] score={score:.4f}")
        print(f"  original -> {orig_path}")
        print(f"  gradcam  -> {cam_path}")

    print("Done. Output folder:", out_dir)


if __name__ == "__main__":
    main()
