"""Train the Sentinel-1 LinkNet baseline with memory-conscious defaults."""

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from losses import DiceBCELoss
from metrics import BinarySegmentationMetrics
from model import build_model
from sar_dataset import get_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"C:\Users\datha\Desktop\sar\dataset\SOS")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Use only the first N train samples (smoke test).")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Use only the first N validation samples (smoke test).")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--log-path", default="training_log.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_subset_loader(loader: DataLoader, limit: int | None, shuffle: bool, drop_last: bool, workers: int) -> DataLoader:
    dataset = loader.dataset if limit is None else Subset(loader.dataset, range(min(limit, len(loader.dataset))))
    return DataLoader(dataset, batch_size=loader.batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=workers, pin_memory=True)


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, report_memory=False):
    training = optimizer is not None
    model.train(training)
    metrics = BinarySegmentationMetrics()
    total_loss = 0.0
    total_items = 0
    for step, (images, masks) in enumerate(loader):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_context = torch.amp.autocast("cuda", enabled=True) if device.type == "cuda" else nullcontext()
        with amp_context:
            logits = model(images)
            loss = loss_fn(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        metrics.update(logits.detach(), masks)
        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
        if report_memory and step == 2 and device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Peak CUDA memory after 3 train steps: {peak:.3f} GB")
    return total_loss / total_items, metrics.compute()


@torch.no_grad()
def save_prediction_grid(model, loader, device, output: Path, count: int = 6) -> None:
    model.eval()
    images, masks = next(iter(loader))
    images = images[:count].to(device)
    probabilities = torch.sigmoid(model(images)).cpu()
    images, masks = images.cpu(), masks[:count]
    image_count = images.shape[0]
    figure, axes = plt.subplots(image_count, 3, figsize=(9, 3 * image_count), squeeze=False)
    # Images are normalised copies of grayscale SAR tiles; invert just for display.
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    for row in range(image_count):
        display = (images[row] * std + mean).clamp(0, 1)[0]
        panels = (display, masks[row, 0], probabilities[row, 0] >= 0.5)
        titles = ("Sentinel-1 image", "Ground truth", "Prediction (0.5 threshold)")
        for column, (panel, title) in enumerate(zip(panels, titles)):
            axes[row, column].imshow(panel, cmap="gray", vmin=0, vmax=1)
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_report(metrics: dict[str, float], best_epoch: int, best_iou: float, path: Path) -> None:
    path.write_text(
        "# Sentinel-1 validation evaluation\n\n"
        f"Best checkpoint epoch: {best_epoch}\n\n"
        f"- IoU/Jaccard: {metrics['iou']:.6f}\n"
        f"- Dice: {metrics['dice']:.6f}\n"
        f"- Precision: {metrics['precision']:.6f}\n"
        f"- Recall: {metrics['recall']:.6f}\n"
        f"- F1: {metrics['f1']:.6f}\n\n"
        "`predictions_sample.png` contains qualitative validation examples. Mumbai inference is qualitative only; no Mumbai ground truth or accuracy claim is included.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; batch size: {args.batch_size}; LR: {args.lr}")

    base_train, base_val = get_dataloaders(args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = make_subset_loader(base_train, args.max_train_samples, True, True, args.num_workers)
    val_loader = make_subset_loader(base_val, args.max_val_samples, False, False, args.num_workers)
    print(f"Training samples/batches: {len(train_loader.dataset)}/{len(train_loader)}")
    print(f"Validation samples/batches: {len(val_loader.dataset)}/{len(val_loader)}")
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Subset is too small for the selected batch size.")

    # Step-2 oil fraction is 22.87%, so the requested <5% pos_weight rule does not apply.
    loss_fn = DiceBCELoss(pos_weight=None).to(device)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path, log_path = Path(args.checkpoint), Path(args.log_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True) if log_path.parent != Path(".") else None
    is_smoke = args.max_train_samples is not None or args.max_val_samples is not None
    best_iou, best_epoch, stalled = -1.0, 0, 0

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "val_precision", "val_recall", "val_f1"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, _ = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler, report_memory=(epoch == 1))
            val_loss, val_metrics = run_epoch(model, val_loader, loss_fn, device)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "val_iou": val_metrics["iou"], "val_dice": val_metrics["dice"],
                   "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"], "val_f1": val_metrics["f1"]}
            writer.writerow(row); log_file.flush()
            print(f"Epoch {epoch:03d}/{args.epochs}: train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                  f"val_iou={val_metrics['iou']:.5f} val_dice={val_metrics['dice']:.5f}")
            if val_metrics["iou"] > best_iou:
                best_iou, best_epoch, stalled = val_metrics["iou"], epoch, 0
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_iou": best_iou,
                            "args": vars(args)}, checkpoint_path)
                print(f"Saved best checkpoint: {checkpoint_path} (IoU {best_iou:.5f})")
            else:
                stalled += 1
                if not is_smoke and stalled >= args.patience:
                    print(f"Early stopping after {args.patience} epochs without IoU improvement.")
                    break

    if device.type == "cuda":
        print(f"Peak CUDA memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.3f} GB")
    if not is_smoke:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_loss, final_metrics = run_epoch(model, val_loader, loss_fn, device)
        save_prediction_grid(model, val_loader, device, Path("predictions_sample.png"))
        write_report(final_metrics, best_epoch, best_iou, Path("evaluation_report.md"))
        print(f"Best validation IoU: {best_iou:.5f}; report and prediction grid saved.")


if __name__ == "__main__":
    main()
