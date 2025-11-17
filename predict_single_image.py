# encoding: utf-8
"""Run the CheXNet classifier on a single chest X-ray image.

Usage (from E:\CheXNet):

    python predict_single_image.py --image_path "path/to/image.png"

This uses the original DenseNet121 CheXNet checkpoint in model.pth.tar
and outputs probabilities for each of the 14 disease classes.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from model import DenseNet121, N_CLASSES, CLASS_NAMES, CKPT_PATH


def load_model(device: torch.device) -> nn.Module:
    """Load DenseNet121 CheXNet model with pretrained weights on given device."""
    model = DenseNet121(N_CLASSES)

    checkpoint_path = Path(CKPT_PATH)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    # Strip the 'module.' prefix introduced by DataParallel in the original training
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading state_dict: {unexpected}")

    model.to(device)
    model.eval()
    return model


def build_transform() -> transforms.Compose:
    """Image preprocessing pipeline (same as original CheXNet)."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


def predict_single_image(image_path: Path, device: torch.device) -> None:
    model = load_model(device)
    transform = build_transform()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\nRunning CheXNet on image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        output = model(input_tensor)  # (1, 14), already passed through sigmoid
        probs = output.squeeze(0).cpu().numpy()

    print("\nPredicted probabilities (0–1) per disease:")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name:<20} {p:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CheXNet on a single chest X-ray image.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the chest X-ray image (PNG/JPG)")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict_single_image(Path(args.image_path), device)
