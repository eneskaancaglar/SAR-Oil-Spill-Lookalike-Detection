from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from losses import BCEDiceLoss
from oil_spill_dataset import OilSpillDataset
from unet import UNet


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset_manifest.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline U-Net egitimi"
    )

    parser.add_argument("--run-name", default="baseline_unet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="0 tam epoch anlamina gelir.",
    )
    parser.add_argument(
        "--max-val-steps",
        type=int,
        default=0,
        help="0 butun validation verisi anlamina gelir.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(
    true_positive: float,
    false_positive: float,
    false_negative: float,
) -> dict[str, float]:
    epsilon = 1e-7

    dice = (
        2.0 * true_positive + epsilon
    ) / (
        2.0 * true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    iou = (
        true_positive + epsilon
    ) / (
        true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    precision = (
        true_positive + epsilon
    ) / (
        true_positive + false_positive + epsilon
    )

    recall = (
        true_positive + epsilon
    ) / (
        true_positive + false_negative + epsilon
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: BCEDiceLoss,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None,
    max_steps: int,
) -> dict[str, float]:
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    bce_loss_sum = 0.0
    dice_loss_sum = 0.0
    sample_count = 0

    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for step, batch in enumerate(loader, start=1):
            if max_steps > 0 and step > max_steps:
                break

            images = batch["image"].to(
                device,
                non_blocking=True,
            )
            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            total_loss, bce_loss, dice_loss = (
                loss_function.components(logits, masks)
            )

            if training:
                total_loss.backward()
                optimizer.step()

            batch_size = images.shape[0]

            total_loss_sum += float(total_loss.item()) * batch_size
            bce_loss_sum += float(bce_loss.item()) * batch_size
            dice_loss_sum += float(dice_loss.item()) * batch_size
            sample_count += batch_size

            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= threshold
            targets = masks >= 0.5

            true_positive += float(
                (predictions & targets).sum().item()
            )
            false_positive += float(
                (predictions & ~targets).sum().item()
            )
            false_negative += float(
                (~predictions & targets).sum().item()
            )

    if sample_count == 0:
        raise RuntimeError("Epoch icinde hic batch islenmedi.")

    metrics = calculate_metrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )

    return {
        "loss": total_loss_sum / sample_count,
        "bce_loss": bce_loss_sum / sample_count,
        "dice_loss": dice_loss_sum / sample_count,
        **metrics,
    }


def save_history_plots(
    history: list[dict[str, object]],
    figure_dir: Path,
) -> None:
    epochs = [int(item["epoch"]) for item in history]

    train_losses = [
        float(item["train"]["loss"])
        for item in history
    ]
    val_losses = [
        float(item["val"]["loss"])
        for item in history
    ]

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(1, 1, 1)

    axis.plot(epochs, train_losses, marker="o", label="Train Loss")
    axis.plot(epochs, val_losses, marker="o", label="Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Baseline U-Net Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        figure_dir / "loss_curve.png",
        dpi=160,
    )
    plt.close(figure)

    val_dice = [
        float(item["val"]["dice"])
        for item in history
    ]
    val_iou = [
        float(item["val"]["iou"])
        for item in history
    ]

    metric_figure = plt.figure(figsize=(8, 5))
    metric_axis = metric_figure.add_subplot(1, 1, 1)

    metric_axis.plot(
        epochs,
        val_dice,
        marker="o",
        label="Validation Dice",
    )
    metric_axis.plot(
        epochs,
        val_iou,
        marker="o",
        label="Validation IoU",
    )
    metric_axis.set_xlabel("Epoch")
    metric_axis.set_ylabel("Metric")
    metric_axis.set_title("Validation Segmentation Metrics")
    metric_axis.grid(True, alpha=0.3)
    metric_axis.legend()

    metric_figure.tight_layout()
    metric_figure.savefig(
        figure_dir / "validation_metrics.png",
        dpi=160,
    )
    plt.close(metric_figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA istendi ancak kullanilabilir degil."
        )

    run_dir = PROJECT_ROOT / "outputs" / args.run_name
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.run_name
    figure_dir = run_dir / "figures"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASELINE U-NET EGITIMI")
    print("=" * 70)
    print("Run:", args.run_name)
    print("Cihaz:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    train_dataset = OilSpillDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
    )
    val_dataset = OilSpillDataset(
        manifest_path=MANIFEST_PATH,
        split="val",
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    loss_function = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    history: list[dict[str, object]] = []
    best_val_dice = -1.0

    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            device=device,
            threshold=args.threshold,
            optimizer=optimizer,
            max_steps=args.max_train_steps,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            loss_function=loss_function,
            device=device,
            threshold=args.threshold,
            optimizer=None,
            max_steps=args.max_val_steps,
        )

        epoch_seconds = time.perf_counter() - epoch_start

        history_item = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "epoch_seconds": epoch_seconds,
        }
        history.append(history_item)

        print()
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"{epoch_seconds:.1f} saniye"
        )
        print(
            "Train | "
            f"Loss {train_metrics['loss']:.4f} | "
            f"Dice {train_metrics['dice']:.4f} | "
            f"IoU {train_metrics['iou']:.4f}"
        )
        print(
            "Val   | "
            f"Loss {val_metrics['loss']:.4f} | "
            f"Dice {val_metrics['dice']:.4f} | "
            f"IoU {val_metrics['iou']:.4f} | "
            f"P {val_metrics['precision']:.4f} | "
            f"R {val_metrics['recall']:.4f}"
        )

        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "args": vars(args),
        }

        torch.save(
            last_checkpoint,
            checkpoint_dir / "last.pth",
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]

            torch.save(
                last_checkpoint,
                checkpoint_dir / "best.pth",
            )

            print(
                f"Yeni en iyi model kaydedildi. "
                f"Val Dice: {best_val_dice:.4f}"
            )

        history_path = run_dir / "history.json"

        with history_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "args": vars(args),
                    "history": history,
                    "best_val_dice": best_val_dice,
                },
                file,
                indent=2,
            )

        save_history_plots(
            history=history,
            figure_dir=figure_dir,
        )

    total_seconds = time.perf_counter() - start_time

    print()
    print("=" * 70)
    print("EGITIM TAMAMLANDI")
    print("=" * 70)
    print(f"Toplam sure: {total_seconds:.1f} saniye")
    print(f"En iyi validation Dice: {best_val_dice:.4f}")
    print(f"Checkpoint: {checkpoint_dir.resolve()}")
    print(f"Sonuclar:   {run_dir.resolve()}")

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )
        print(
            "GPU maksimum bellek:",
            f"{peak_memory_mb:.2f} MB",
        )


if __name__ == "__main__":
    main()
