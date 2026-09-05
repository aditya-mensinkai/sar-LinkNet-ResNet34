from pathlib import Path
from PIL import Image
import numpy as np

# CHANGE THIS if your path is different
DATASET = Path("dataset/SOS")

IMAGE_DIR = DATASET / "images"
MASK_DIR = DATASET / "masks"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# --------------------------------------------------
# 1. Find all images and masks
# --------------------------------------------------

images = [
    p for p in IMAGE_DIR.rglob("*")
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

masks = [
    p for p in MASK_DIR.rglob("*")
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Number of images : {len(images)}")
print(f"Number of masks  : {len(masks)}")


# --------------------------------------------------
# 2. Inspect one image
# --------------------------------------------------

if images:
    image_path = images[0]
    image = Image.open(image_path)

    print("\n" + "=" * 60)
    print("SAMPLE IMAGE")
    print("=" * 60)

    print(f"File       : {image_path}")
    print(f"Format     : {image.format}")
    print(f"Mode       : {image.mode}")
    print(f"Dimensions : {image.size}")

    arr = np.array(image)

    print(f"Shape      : {arr.shape}")
    print(f"Dtype      : {arr.dtype}")
    print(f"Min value  : {arr.min()}")
    print(f"Max value  : {arr.max()}")


# --------------------------------------------------
# 3. Inspect one mask
# --------------------------------------------------

if masks:
    mask_path = masks[0]
    mask = Image.open(mask_path)

    print("\n" + "=" * 60)
    print("SAMPLE MASK")
    print("=" * 60)

    print(f"File       : {mask_path}")
    print(f"Format     : {mask.format}")
    print(f"Mode       : {mask.mode}")
    print(f"Dimensions : {mask.size}")

    mask_arr = np.array(mask)

    print(f"Shape      : {mask_arr.shape}")
    print(f"Dtype      : {mask_arr.dtype}")
    print(f"Min value  : {mask_arr.min()}")
    print(f"Max value  : {mask_arr.max()}")

    print(f"Unique values: {np.unique(mask_arr)}")


# --------------------------------------------------
# 4. Check filename matching
# --------------------------------------------------

from collections import Counter

image_relpaths = {
    p.relative_to(IMAGE_DIR).as_posix(): p
    for p in images
}

mask_relpaths = {
    p.relative_to(MASK_DIR).as_posix(): p
    for p in masks
}

image_keys = set(image_relpaths.keys())
mask_keys = set(mask_relpaths.keys())

print("\n" + "=" * 60)
print("IMAGE / MASK MATCHING")
print("=" * 60)

print(f"Images                    : {len(image_keys)}")
print(f"Masks                     : {len(mask_keys)}")
print(f"Matching relative paths   : {len(image_keys & mask_keys)}")
print(f"Images without masks      : {len(image_keys - mask_keys)}")
print(f"Masks without images      : {len(mask_keys - image_keys)}")

if image_keys - mask_keys:
    print("\nExample missing masks:")
    for x in list(image_keys - mask_keys)[:10]:
        print(x)

if mask_keys - image_keys:
    print("\nExample missing images:")
    for x in list(mask_keys - image_keys)[:10]:
        print(x)


if images:
    image = Image.open(images[0]).convert("RGB")
    arr = np.array(image)

    print("\n" + "=" * 60)
    print("CHANNEL ANALYSIS")
    print("=" * 60)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    print("R == G:", np.array_equal(r, g))
    print("G == B:", np.array_equal(g, b))
    print("R == B:", np.array_equal(r, b))

    print("Unique R values:", len(np.unique(r)))
    print("Unique G values:", len(np.unique(g)))
    print("Unique B values:", len(np.unique(b)))
if masks:
    mask = Image.open(masks[0]).convert("RGB")
    mask_arr = np.array(mask)

    r = mask_arr[:, :, 0]
    g = mask_arr[:, :, 1]
    b = mask_arr[:, :, 2]

    print("\n" + "=" * 60)
    print("MASK CHANNEL ANALYSIS")
    print("=" * 60)

    print("R == G:", np.array_equal(r, g))
    print("G == B:", np.array_equal(g, b))
    print("R == B:", np.array_equal(r, b))