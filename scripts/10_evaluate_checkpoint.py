from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
        description="Checkpoint'i tum validation setinde degerlendirir."
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

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "pilot_baseline"
            / "full_validation_evaluation.json"
        ),
    )

    return parser.parse_args()


def create_state() -> dict[str, Any]:
    return {
        "image_count": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
        "predicted_positive_pixels": 0,
        "target_positive_pixels": 0,
        "total_pixels": 0,
        "images_with_alarm": 0,
        "image_dice_sum": 0.0,
    }


def update_state(
    state: dict[str, Any],
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    prediction = prediction.bool()
    target = target.bool()

    tp = int((prediction & target).sum().item())
    fp = int((prediction & ~target).sum().item())
    fn = int((~prediction & target).sum().item())
    tn = int((~prediction & ~target).sum().item())

    predicted_positive = int(prediction.sum().item())
    target_positive = int(target.sum().item())

    if target_positive == 0:
        image_dice = 1.0 if predicted_positive == 0 else 0.0
    else:
        image_dice = (
            2.0 * tp
        ) / (
            2.0 * tp + fp + fn + 1e-7
        )

    state["image_count"] += 1
    state["true_positive"] += tp
    state["false_positive"] += fp
    state["false_negative"] += fn
    state["true_negative"] += tn
    state["predicted_positive_pixels"] += predicted_positive
    state["target_positive_pixels"] += target_positive
    state["total_pixels"] += int(target.numel())
    state["images_with_alarm"] += int(predicted_positive > 0)
    state["image_dice_sum"] += image_dice


def finalize_state(state: dict[str, Any]) -> dict[str, float | int]:
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

    image_count = int(state["image_count"])
    total_pixels = int(state["total_pixels"])

    return {
        "image_count": image_count,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pixel_false_positive_rate": false_positive_rate,
        "mean_image_dice": (
            state["image_dice_sum"] / image_count
            if image_count > 0
            else 0.0
        ),
        "images_with_alarm": int(state["images_with_alarm"]),
        "alarm_image_percentage": (
            state["images_with_alarm"] / image_count * 100
            if image_count > 0
            else 0.0
        ),
        "predicted_oil_pixel_percentage": (
            state["predicted_positive_pixels"]
            / total_pixels
            * 100
            if total_pixels > 0
            else 0.0
        ),
        "target_oil_pixel_percentage": (
            state["target_positive_pixels"]
            / total_pixels
            * 100
            if total_pixels > 0
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadi: {args.checkpoint}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("TAM VALIDATION DEGERLENDIRMESI")
    print("=" * 70)
    print("Cihaz:", device)
    print("Checkpoint:", args.checkpoint)
    print("Threshold:", args.threshold)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    checkpoint_args = checkpoint.get("args", {})
    base_channels = int(
        checkpoint_args.get("base_channels", 16)
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
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

    states: dict[str, dict[str, Any]] = defaultdict(create_state)

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

            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= args.threshold
            targets = masks >= 0.5

            metadata = batch["metadata"]

            for index in range(images.shape[0]):
                sensor = metadata["sensor"][index]
                has_oil = int(metadata["has_oil"][index].item())

                oil_group = "oil" if has_oil else "no_oil"

                group_names = [
                    "overall",
                    sensor,
                    oil_group,
                    f"{sensor}_{oil_group}",
                ]

                for group_name in group_names:
                    update_state(
                        states[group_name],
                        predictions[index],
                        targets[index],
                    )

                processed_images += 1

                if processed_images % 250 == 0:
                    print(
                        f"{processed_images}/{len(dataset)} "
                        "goruntu tamamlandi."
                    )

    results = {
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "split": "val",
        "groups": {
            name: finalize_state(state)
            for name, state in states.items()
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print()
    print("=" * 70)
    print("GENEL SONUC")
    print("=" * 70)

    overall = results["groups"]["overall"]

    print(f"Goruntu sayisi: {overall['image_count']}")
    print(f"Dice:           {overall['dice']:.4f}")
    print(f"IoU:            {overall['iou']:.4f}")
    print(f"Precision:      {overall['precision']:.4f}")
    print(f"Recall:         {overall['recall']:.4f}")

    print()
    print("SENSOR BAZINDA")

    for sensor in ("Sentinel-1", "PALSAR"):
        if sensor not in results["groups"]:
            continue

        metrics = results["groups"][sensor]

        print(
            f"{sensor:10s} | "
            f"N={metrics['image_count']:4d} | "
            f"Dice={metrics['dice']:.4f} | "
            f"IoU={metrics['iou']:.4f} | "
            f"P={metrics['precision']:.4f} | "
            f"R={metrics['recall']:.4f}"
        )

    print()
    print("PETROLSUZ GORUNTULER")

    no_oil = results["groups"].get("no_oil")

    if no_oil:
        print(f"Goruntu sayisi:       {no_oil['image_count']}")
        print(
            "Alarm verilen goruntu: "
            f"{no_oil['images_with_alarm']}"
        )
        print(
            "Alarm goruntu orani:   "
            f"%{no_oil['alarm_image_percentage']:.2f}"
        )
        print(
            "Yanlis petrol pikseli: "
            f"%{no_oil['predicted_oil_pixel_percentage']:.4f}"
        )

    print()
    print("Rapor:", args.output.resolve())


if __name__ == "__main__":
    main()
