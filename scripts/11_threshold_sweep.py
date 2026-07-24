from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
        description="Farkli segmentation threshold degerlerini karsilastirir."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints"
            / "pilot_baseline"
            / "best.pth"
        ),
    )

    parser.add_argument(
        "--thresholds",
        default="0.30,0.40,0.50,0.60,0.70,0.80,0.90",
        help="Virgulle ayrilmis threshold listesi.",
    )

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "pilot_baseline"
            / "threshold_sweep"
        ),
    )

    return parser.parse_args()


def parse_thresholds(text: str) -> list[float]:
    thresholds = []

    for part in text.split(","):
        value = float(part.strip())

        if not 0.0 < value < 1.0:
            raise ValueError(
                f"Threshold 0 ile 1 arasinda olmali: {value}"
            )

        thresholds.append(value)

    thresholds = sorted(set(thresholds))

    if not thresholds:
        raise ValueError("En az bir threshold verilmelidir.")

    return thresholds


def create_state() -> dict[str, Any]:
    return {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
        "no_oil_image_count": 0,
        "no_oil_alarm_count": 0,
        "no_oil_false_positive_pixels": 0,
        "no_oil_total_pixels": 0,
    }


def update_state(
    state: dict[str, Any],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    has_oil: torch.Tensor,
) -> None:
    predictions = predictions.bool()
    targets = targets.bool()

    state["true_positive"] += int(
        (predictions & targets).sum().item()
    )

    state["false_positive"] += int(
        (predictions & ~targets).sum().item()
    )

    state["false_negative"] += int(
        (~predictions & targets).sum().item()
    )

    state["true_negative"] += int(
        (~predictions & ~targets).sum().item()
    )

    for index in range(predictions.shape[0]):
        if int(has_oil[index].item()) != 0:
            continue

        no_oil_prediction = predictions[index]

        predicted_positive_pixels = int(
            no_oil_prediction.sum().item()
        )

        state["no_oil_image_count"] += 1
        state["no_oil_alarm_count"] += int(
            predicted_positive_pixels > 0
        )

        state["no_oil_false_positive_pixels"] += (
            predicted_positive_pixels
        )

        state["no_oil_total_pixels"] += int(
            no_oil_prediction.numel()
        )


def finalize_state(
    state: dict[str, Any],
    threshold: float,
) -> dict[str, float | int]:
    tp = float(state["true_positive"])
    fp = float(state["false_positive"])
    fn = float(state["false_negative"])
    tn = float(state["true_negative"])

    epsilon = 1e-7

    dice = (
        2.0 * tp + epsilon
    ) / (
        2.0 * tp + fp + fn + epsilon
    )

    iou = (
        tp + epsilon
    ) / (
        tp + fp + fn + epsilon
    )

    precision = (
        tp + epsilon
    ) / (
        tp + fp + epsilon
    )

    recall = (
        tp + epsilon
    ) / (
        tp + fn + epsilon
    )

    false_positive_rate = (
        fp + epsilon
    ) / (
        fp + tn + epsilon
    )

    no_oil_image_count = int(
        state["no_oil_image_count"]
    )

    no_oil_alarm_count = int(
        state["no_oil_alarm_count"]
    )

    no_oil_total_pixels = int(
        state["no_oil_total_pixels"]
    )

    return {
        "threshold": threshold,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pixel_false_positive_rate": false_positive_rate,
        "no_oil_image_count": no_oil_image_count,
        "no_oil_alarm_count": no_oil_alarm_count,
        "no_oil_alarm_percentage": (
            no_oil_alarm_count / no_oil_image_count * 100
            if no_oil_image_count > 0
            else 0.0
        ),
        "no_oil_false_positive_pixel_percentage": (
            state["no_oil_false_positive_pixels"]
            / no_oil_total_pixels
            * 100
            if no_oil_total_pixels > 0
            else 0.0
        ),
    }


def save_csv(
    results: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    fieldnames = list(results[0].keys())

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def save_metric_plot(
    results: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    thresholds = [
        float(item["threshold"])
        for item in results
    ]

    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(1, 1, 1)

    axis.plot(
        thresholds,
        [float(item["dice"]) for item in results],
        marker="o",
        label="Dice",
    )

    axis.plot(
        thresholds,
        [float(item["iou"]) for item in results],
        marker="o",
        label="IoU",
    )

    axis.plot(
        thresholds,
        [float(item["precision"]) for item in results],
        marker="o",
        label="Precision",
    )

    axis.plot(
        thresholds,
        [float(item["recall"]) for item in results],
        marker="o",
        label="Recall",
    )

    axis.set_title("Threshold ve segmentasyon metrikleri")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Metrik")
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_alarm_plot(
    results: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    thresholds = [
        float(item["threshold"])
        for item in results
    ]

    alarm_percentages = [
        float(item["no_oil_alarm_percentage"])
        for item in results
    ]

    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(1, 1, 1)

    axis.plot(
        thresholds,
        alarm_percentages,
        marker="o",
    )

    axis.set_title(
        "Petrolsuz goruntulerde alarm orani"
    )
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Alarm verilen petrolsuz goruntu (%)")
    axis.set_ylim(0.0, 100.0)
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadi: {args.checkpoint}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 75)
    print("THRESHOLD SWEEP")
    print("=" * 75)
    print("Cihaz:", device)
    print("Checkpoint:", args.checkpoint)
    print("Thresholds:", thresholds)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    checkpoint_args = checkpoint.get("args", {})

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=int(
            checkpoint_args.get("base_channels", 16)
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    dataset = OilSpillDataset(
        manifest_path=MANIFEST_PATH,
        split="val",
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    states = {
        threshold: create_state()
        for threshold in thresholds
    }

    processed_images = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            has_oil = batch["metadata"]["has_oil"]

            logits = model(images)
            probabilities = torch.sigmoid(logits)
            targets = masks >= 0.5

            for threshold in thresholds:
                predictions = probabilities >= threshold

                update_state(
                    state=states[threshold],
                    predictions=predictions,
                    targets=targets,
                    has_oil=has_oil,
                )

            processed_images += images.shape[0]

            if processed_images % 250 == 0:
                print(
                    f"{processed_images}/{len(dataset)} "
                    "goruntu tamamlandi."
                )

    results = [
        finalize_state(
            state=states[threshold],
            threshold=threshold,
        )
        for threshold in thresholds
    ]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = args.output_dir / "threshold_results.json"
    csv_path = args.output_dir / "threshold_results.csv"
    metric_plot_path = (
        args.output_dir / "metrics_vs_threshold.png"
    )
    alarm_plot_path = (
        args.output_dir
        / "no_oil_alarm_vs_threshold.png"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    save_csv(results, csv_path)
    save_metric_plot(results, metric_plot_path)
    save_alarm_plot(results, alarm_plot_path)

    print()
    print("=" * 95)
    print("SONUCLAR")
    print("=" * 95)

    print(
        f"{'Thr':>5} | "
        f"{'Dice':>7} | "
        f"{'IoU':>7} | "
        f"{'Prec':>7} | "
        f"{'Recall':>7} | "
        f"{'No-oil alarm':>13} | "
        f"{'FP pixel':>9}"
    )

    print("-" * 95)

    for result in results:
        print(
            f"{float(result['threshold']):5.2f} | "
            f"{float(result['dice']):7.4f} | "
            f"{float(result['iou']):7.4f} | "
            f"{float(result['precision']):7.4f} | "
            f"{float(result['recall']):7.4f} | "
            f"{float(result['no_oil_alarm_percentage']):12.2f}% | "
            f"{float(result['no_oil_false_positive_pixel_percentage']):8.4f}%"
        )

    best_dice_result = max(
        results,
        key=lambda item: float(item["dice"]),
    )

    print()
    print(
        "En yuksek Dice threshold:",
        f"{float(best_dice_result['threshold']):.2f}",
    )

    print(
        "En yuksek Dice:",
        f"{float(best_dice_result['dice']):.4f}",
    )

    print()
    print("JSON:", json_path.resolve())
    print("CSV: ", csv_path.resolve())
    print("Metrik grafigi:", metric_plot_path.resolve())
    print("Alarm grafigi: ", alarm_plot_path.resolve())


if __name__ == "__main__":
    main()
