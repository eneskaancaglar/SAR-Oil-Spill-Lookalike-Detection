from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unet import UNet


DEFAULT_RESULTS = (
    ROOT
    / "outputs"
    / "external_test_finetuned"
    / "dartis_no_oil_per_image_results.csv"
)

DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "hard_negative_finetune"
    / "best_balanced.pth"
)

DARTIS_RAW = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "remaining_error_gallery"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_args = checkpoint.get(
        "args",
        {},
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=int(
            checkpoint_args.get(
                "base_channels",
                16,
            )
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def predict(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as image:
        image = image.convert("L")

        image_array = (
            np.asarray(
                image,
                dtype=np.float32,
            )
            / 255.0
        )

    tensor = (
        torch.from_numpy(
            image_array.copy()
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        probability = torch.sigmoid(
            model(tensor)
        )[0, 0]

    return (
        image_array,
        probability.cpu().numpy(),
    )


def create_gallery(
    dataframe: pd.DataFrame,
    group_name: str,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    top_k: int,
) -> None:
    subset = (
        dataframe[
            dataframe["image_set"] == group_name
        ]
        .sort_values(
            "t060_positive_ratio",
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    if subset.empty:
        raise RuntimeError(
            f"{group_name} grubu boş."
        )

    figure, axes = plt.subplots(
        len(subset),
        4,
        figsize=(15, 4 * len(subset)),
    )

    if len(subset) == 1:
        axes = np.expand_dims(
            axes,
            axis=0,
        )

    for row_index, row in subset.iterrows():
        image_path = (
            DARTIS_RAW
            / str(row["image_set"])
            / str(row["image_name"])
        )

        image, probability = predict(
            model,
            image_path,
            device,
        )

        binary = (
            probability >= threshold
        )

        overlay = np.stack(
            [image, image, image],
            axis=-1,
        )

        overlay[
            binary,
            0,
        ] = 1.0

        overlay[
            binary,
            1,
        ] *= 0.25

        overlay[
            binary,
            2,
        ] *= 0.25

        predicted_percentage = (
            binary.mean() * 100.0
        )

        axes[row_index, 0].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[row_index, 0].set_title(
            f"SAR\n{row['image_name']}"
        )

        axes[row_index, 1].imshow(
            probability,
            cmap="inferno",
            vmin=0,
            vmax=1,
        )

        axes[row_index, 1].set_title(
            "Olasılık haritası"
        )

        axes[row_index, 2].imshow(
            binary,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[row_index, 2].set_title(
            f"Binary tahmin\n%{predicted_percentage:.2f}"
        )

        axes[row_index, 3].imshow(
            overlay,
        )

        axes[row_index, 3].set_title(
            "Yanlış alarm overlay"
        )

        for axis in axes[row_index]:
            axis.axis("off")

    figure.suptitle(
        f"DARTIS {group_name.upper()} — "
        f"En büyük kalan yanlış alarmlar",
        fontsize=16,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.98)
    )

    output_path = (
        OUTPUT_DIR
        / f"remaining_false_alarms_{group_name}.png"
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    subset.to_csv(
        OUTPUT_DIR
        / f"remaining_false_alarms_{group_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"{group_name}: {output_path.resolve()}"
    )


def main() -> None:
    args = parse_args()

    if not args.results.exists():
        raise FileNotFoundError(
            f"Sonuç CSV bulunamadı: {args.results}"
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {args.checkpoint}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    dataframe = pd.read_csv(
        args.results,
        encoding="utf-8-sig",
    )

    model = load_model(
        args.checkpoint,
        device,
    )

    print("=" * 72)
    print("KALAN YANLIŞ ALARM GÖRSEL ANALİZİ")
    print("=" * 72)
    print("Cihaz:", device)
    print("Threshold:", args.threshold)

    create_gallery(
        dataframe,
        "nc",
        model,
        device,
        args.threshold,
        args.top_k,
    )

    create_gallery(
        dataframe,
        "nw",
        model,
        device,
        args.threshold,
        args.top_k,
    )


if __name__ == "__main__":
    main()
