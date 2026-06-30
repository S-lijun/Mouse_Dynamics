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

NUM_USERS = 10
H = 448
W = 448


def resolve_project_path(path):
    """Resolve paths relative to project root (same as train_multi_CNN.py)."""
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def load_tensor_dataset(tensor_root):
    img_path = os.path.join(tensor_root, "images.npy")
    lab_path = os.path.join(tensor_root, "labels.npy")
    ses_path = os.path.join(tensor_root, "sessions.npy")

    raw_labels = np.memmap(lab_path, dtype=np.uint8, mode="r")
    n = raw_labels.size // NUM_USERS

    images = np.memmap(
        img_path,
        dtype=np.uint8,
        mode="r",
        shape=(n, 3, H, W),
    )
    labels = raw_labels.reshape(n, NUM_USERS)
    sessions = np.load(ses_path, allow_pickle=True)

    return images, labels, sessions


def resolve_sample_index(sessions, session_query, chunk_index=None):
    matches = [i for i, s in enumerate(sessions) if session_query in str(s)]
    if not matches:
        raise ValueError(f"No sample found for session query: {session_query}")

    if chunk_index is None:
        return matches[0]

    if chunk_index < 0 or chunk_index >= len(matches):
        raise ValueError(
            f"chunk_index={chunk_index} out of range for session {session_query} "
            f"({len(matches)} chunks found)"
        )

    return matches[chunk_index]


def load_sample(tensor_root, sample_index):
    images, labels, sessions = load_tensor_dataset(tensor_root)

    if sample_index < 0 or sample_index >= len(images):
        raise IndexError(f"sample_index={sample_index} out of range [0, {len(images) - 1}]")

    img = np.array(images[sample_index], copy=True)
    label = labels[sample_index]
    session_id = str(sessions[sample_index])
    true_user = int(label.argmax())

    # Training uses BGR tensor / 255 directly (see Images_convert.py + train_multi_CNN.py).
    input_tensor = torch.from_numpy(img).float().div(255).unsqueeze(0)

    # Convert to RGB only for visualization overlay.
    rgb_float = cv2.cvtColor(
        img.transpose(1, 2, 0),
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32) / 255.0 

    return input_tensor, rgb_float, session_id, true_user


def main():
    parser = argparse.ArgumentParser(
        description="Run Grad-CAM for tensor-trained Protocol 1 GoogLeNet models."
    )
    parser.add_argument(
        "--model",
        default=os.path.join(ROOT, "saved_models", "multilabel_P1_best_20260629_204243.pth"),
        help="Path to .pth checkpoint",
    )
    parser.add_argument(
        "--tensor-root",
        default=os.path.join(ROOT, "ImagesTensors", "Balabit", "SRP_protocol1", "event125"),
        help="Folder containing images.npy / labels.npy / sessions.npy",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Direct index into the tensor dataset",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session id prefix, e.g. session_0041905381",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Which chunk to use when multiple rows share the same session",
    )
    parser.add_argument(
        "--target-user",
        type=int,
        default=None,
        help="User head to explain (0-9). Default: ground-truth user of the sample",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(ROOT, "Grad_CAM", "outputs", "SRP"),
        help="Directory to save Grad-CAM overlays",
    )
    args = parser.parse_args()

    args.model = resolve_project_path(args.model)
    args.tensor_root = resolve_project_path(args.tensor_root)
    args.output_root = resolve_project_path(args.output_root)

    if args.sample_index is None and args.session is None:
        parser.error("Provide either --sample-index or --session")

    _, _, sessions = load_tensor_dataset(args.tensor_root)

    if args.sample_index is not None:
        sample_index = args.sample_index
    else:
        sample_index = resolve_sample_index(sessions, args.session, args.chunk_index)

    input_tensor, rgb_float, session_id, true_user = load_sample(args.tensor_root, sample_index)
    target_user = true_user if args.target_user is None else args.target_user

    print("Tensor root :", args.tensor_root)
    print("Sample index:", sample_index)
    print("Session id  :", session_id)
    print("True user   :", true_user)
    print("Target user :", target_user)

    print("Loading model:", args.model)
    model = PretrainedGoogLeNet_Multilabel(num_users=NUM_USERS)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    target_layers = [model.base.inception5b]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_user)]

    with torch.no_grad():
        logit = torch.sigmoid(model(input_tensor))[0, target_user].item()
    print(f"Target user score: {logit:.4f}")

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    out_dir = os.path.join(args.output_root, f"user{true_user}")
    os.makedirs(out_dir, exist_ok=True)

    output_name = f"{session_id}_idx{sample_index}_target{target_user}.png"
    output_path = os.path.join(out_dir, output_name)

    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print("Grad-CAM saved to:", output_path)


if __name__ == "__main__":
    main()
