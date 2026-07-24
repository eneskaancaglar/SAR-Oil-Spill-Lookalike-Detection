from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


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
        description="Checkpoint tahminlerini görselleştirir."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints"
            / "smoke_baseline"
            / "best.pth"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "smoke_baseline"
            / "figures"
            / "prediction_samples.png"
        ),
    )

    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def calculate_sample_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    prediction = prediction.bool()
    target = target.bool()

    true_positive = float(
        (prediction & target).sum().item()
    )

    false_positive = float(
        (prediction & ~target).sum().item()
    )

    false_negative = float(
        (~prediction & target).sum().item()
    )

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

    return {
        "dice": dice,
        "iou": iou,
    }


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadi: {args.checkpoint}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 65)
    print("CHECKPOINT TAHMIN GORSELLESTIRME")
    print("=" * 65)
    print("Cihaz:", device)
    print("Checkpoint:", args.checkpoint)

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

    random_generator = random.Random(args.seed)

    oil_indices = []
    no_oil_indices = []

    for index, row in enumerate(dataset.rows):
        if int(row["has_oil"]) == 1:
            oil_indices.append(index)
        else:
            no_oil_indices.append(index)

    oil_sample_count = min(
        args.sample_count - 1,
        len(oil_indices),
    )

    selected_indices = random_generator.sample(
        oil_indices,
        oil_sample_count,
    )

    if no_oil_indices:
        selected_indices.append(
            random_generator.choice(no_oil_indices)
        )

    while len(selected_indices) < args.sample_count:
        candidate = random_generator.randrange(len(dataset))

        if candidate not in selected_indices:
            selected_indices.append(candidate)

    row_count = len(selected_indices)

    figure, axes = plt.subplots(
        row_count,
        4,
        figsize=(14, row_count * 3.5),
        squeeze=False,
    )

    print()
    print("Ornek sonuclari:")

    for row_number, dataset_index in enumerate(selected_indices):
        sample = dataset[dataset_index]

        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        metadata = sample["metadata"]

        with torch.no_grad():
            logits = model(image)
            probability = torch.sigmoid(logits)
            prediction = probability >= args.threshold

        metrics = calculate_sample_metrics(
            prediction=prediction,
            target=mask >= 0.5,
        )

        image_array = (
            image[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        mask_array = (
            mask[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        probability_array = (
            probability[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        prediction_array = (
            prediction[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        sample_name = metadata["sample_name"]
        sensor = metadata["sensor"]
        has_oil = metadata["has_oil"]

        print(
            f"- {sample_name} | "
            f"{sensor} | "
            f"has_oil={has_oil} | "
            f"Dice={metrics['dice']:.4f} | "
            f"IoU={metrics['iou']:.4f}"
        )

        axes[row_number, 0].imshow(
            image_array,
            cmap="gray",
        )
        axes[row_number, 0].set_title(
            f"SAR goruntusu\n{sample_name}"
        )
        axes[row_number, 0].axis("off")

        axes[row_number, 1].imshow(
            mask_array,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row_number, 1].set_title(
            f"Gercek maske\nPetrol={has_oil}"
        )
        axes[row_number, 1].axis("off")

        axes[row_number, 2].imshow(
            probability_array,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row_number, 2].set_title(
            "Tahmin olasiligi"
        )
        axes[row_number, 2].axis("off")

        axes[row_number, 3].imshow(
            prediction_array,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row_number, 3].set_title(
            f"Ikili tahmin\n"
            f"Dice={metrics['dice']:.3f}"
        )
        axes[row_number, 3].axis("off")

    figure.tight_layout()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        args.output,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print()
    print("Gorsel kaydedildi:", args.output.resolve())


if __name__ == "__main__":
    main()
