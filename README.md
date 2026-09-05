# SAR Oil-Spill Segmentation (LinkNet + ResNet34, Sentinel-1)

Binary oil-spill segmentation on the Sentinel-1-only split of the Refined Deep-SAR (SOS) dataset. Best checkpoint: `checkpoints/best_model.pth` (val IoU 0.8107, Dice 0.8954).

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
.\venv\Scripts\pip.exe install -r requirements.txt          # inference (pinned)
.\venv\Scripts\pip.exe install matplotlib                   # extra, only needed for train.py plots
```

## Files

- `sar_dataset.py` — Sentinel-1 dataset/dataloaders (ImageNet normalize, 256×256 tiles)
- `model.py` — `build_model()` (LinkNet, ResNet34 encoder, 3-channel in, 1 logit out)
- `losses.py`, `metrics.py` — DiceBCE loss, IoU/Dice/precision/recall/F1
- `train.py` — training + `evaluation_report.md` + `predictions_sample.png`
- `infer_pipeline.py` — production inference (CLI + importable), see below

## Inference

Mumbai demo (no ground truth — outputs mask/overlay/stats only, never accuracy numbers):

```powershell
.\venv\Scripts\python.exe infer_pipeline.py --checkpoint checkpoints/best_model.pth --input path\to\mumbai_tiles_or_image --output-dir outputs/mumbai_demo
```

Common variants:

```powershell
# Raw dB-scale GeoTIFF (prints min/max used for min-max to 0-255) + tuned threshold
.\venv\Scripts\python.exe infer_pipeline.py --checkpoint checkpoints/best_model.pth --input path\to\tile.tif --output-dir outputs\demo --rescale --threshold 0.4

# SOS-val smoke test (metrics reported ONLY because --gt-mask is supplied)
.\venv\Scripts\python.exe infer_pipeline.py --checkpoint checkpoints/best_model.pth --input outputs/smoke_val_input --output-dir outputs/smoke_val --gt-mask dataset/SOS/masks/val --overlap 0

# Side-by-side original | CFAR | DL comparison panel
.\venv\Scripts\python.exe infer_pipeline.py --checkpoint checkpoints/best_model.pth --input path\to\image.png --output-dir outputs\demo --cfar-mask path\to\cfar_result.png
```

Full CLI:

```
--checkpoint  default checkpoints/best_model.pth
--input       image file (PNG/JPG/TIFF/GeoTIFF) or directory of tiles (required)
--output-dir  (required)
--threshold   default 0.5
--batch-size  default 4
--rescale     min-max normalize raw values to 0-255 first (for dB GeoTIFFs)
--overlap     tile overlap in px, default 32 (0 = non-overlapping)
--cfar-mask   optional CFAR mask file (or dir matched by filename stem)
--gt-mask     optional ground-truth mask file (or dir); metrics computed ONLY when supplied
```

Importable use:

```python
from infer_pipeline import run_inference
run_inference(checkpoint="checkpoints/best_model.pth", input="path/to/mumbai_tiles_or_image", output_dir="outputs/mumbai_demo")
```

Per input image, `--output-dir` receives the following. The main output is the mask; the rest is presentation + audit trail:

- `{name}_mask.png` — **main output.** Binary oil map (0/255, white = oil), same H×W as the input. Produced from raw logits via sigmoid + `--threshold`.
- `{name}_overlay.png` — original image with predicted oil tinted red (~40% alpha). The demo image; use it to visually judge false positives the bare mask hides.
- `{name}_stats.json` — `oil_pixel_count`, `oil_pixel_fraction` (headline "how much of this scene is oil"), `threshold`, `tiling`, `rescale_applied`, `preprocessing` log, `input_shape`/`raw_shape`/`raw_dtype`, `inference_time_sec`. `gt_mask` + `metrics_vs_gt` (IoU/Dice/precision/recall/F1) appear **only** when `--gt-mask` is passed; otherwise they are absent by design.
- `{name}_comparison.png` — only with `--cfar-mask`: side-by-side original | CFAR mask | DL mask.

A one-line summary (oil fraction, time) is printed per image, along with the exact preprocessing applied — nothing is silently resized.

## Preprocessing notes (read before Mumbai)

- Model input is `[B, 3, 256, 256]`, ImageNet-normalized; outputs raw logits (sigmoid + threshold in the pipeline).
- Larger images are tiled (256px tiles, 32px overlap, reflect-pad on short edges) and stitched by averaging overlapping logits before thresholding.
- Channel handling: single-band is replicated to R=G=B; R=G=B input is kept as-is (this matches training — SOS tiles are grayscale-in-RGB uint8); RGB with differing channels is collapsed by channel-mean then replicated. **This last case is the biggest domain-shift risk**: if Mumbai input is false-colour or dual-pol VV/VH rather than single-channel grayscale, expect to retune `--threshold` and validate visually via the overlays.
- `--rescale` is required for raw dB-scale GeoTIFFs; without it, non-uint8 values are misinterpreted.
- CPU is used automatically when no GPU is present (verified; ±1 px vs GPU from float rounding).

## Verification

Full SOS validation set (839 images) through `infer_pipeline.py`'s own preprocessing reproduces `train.py`: IoU 0.8107, Dice 0.8954, precision 0.8862, recall 0.9049. A 3-image sample varies widely (IoU 0.02–1.00) — sampling noise, not a bug.
