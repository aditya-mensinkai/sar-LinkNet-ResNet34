"""Qualitative-only inference for 256x256 Sentinel-1 PNG/TIFF tiles.

No ground-truth masks are supplied here, so this script never calculates or
claims Mumbai IoU, Dice, or any other quantitative accuracy.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model import build_model
from sar_dataset import IMAGENET_MEAN, IMAGENET_STD


def preprocess(path: Path) -> torch.Tensor:
    image = np.array(Image.open(path).convert("RGB"))
    if image.shape[:2] != (256, 256):
        raise ValueError(f"{path.name} is {image.shape[:2]}; expected a 256x256 tile.")
    image = image.astype(np.float32) / 255.0
    image = (image - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create qualitative oil-mask predictions; no accuracy is computed.")
    parser.add_argument("input_dir", help="Directory of 256x256 Sentinel-1 PNG/TIFF tiles")
    parser.add_argument("checkpoint", help="Path to best_model.pth")
    parser.add_argument("output_dir", help="Directory for binary PNG prediction masks")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint)
    model.eval()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff"})
    if not paths:
        raise RuntimeError(f"No PNG/TIFF tiles found in {input_dir}")

    with torch.inference_mode():
        for path in paths:
            probability = torch.sigmoid(model(preprocess(path).to(device)))[0, 0].cpu().numpy()
            mask = (probability >= args.threshold).astype(np.uint8) * 255
            Image.fromarray(mask).save(output_dir / f"{path.stem}_oil_mask.png")
    print(f"Wrote {len(paths)} qualitative prediction masks to {output_dir.resolve()}")
    print("No Mumbai accuracy metric was computed: these tiles have no ground truth.")


if __name__ == "__main__":
    main()
