"""Save a raw, unnormalised Sentinel-1 image/mask sanity-check grid."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sar_dataset import SARSentinelDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", nargs="?", default=r"C:\Users\datha\Desktop\sar\dataset\SOS")
    parser.add_argument("--output", default="sample_check.png")
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    dataset = SARSentinelDataset(args.data_root, "train", transform=None)
    count = min(args.samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    figure, axes = plt.subplots(count, 2, figsize=(7, 3 * count), squeeze=False)

    for row, index in enumerate(indices):
        image_path, mask_path = dataset.pairs[index]
        image = np.array(Image.open(image_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"Image: {image_path.name}")
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Oil mask")
        for axis in axes[row]:
            axis.axis("off")

    figure.tight_layout()
    output = Path(args.output)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {count} raw image/mask pairs to {output.resolve()}")


if __name__ == "__main__":
    main()
