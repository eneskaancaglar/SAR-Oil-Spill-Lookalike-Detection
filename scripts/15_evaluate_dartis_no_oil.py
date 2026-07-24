from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unet import UNet


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "metadata"
    / "dartis_no_oil_manifest.csv"
)

DEFAULT_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "full_baseline"
    / "best.pth"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dartis_stress_test"
)

MODEL_SIZE_MULTIPLE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "U-Net modelini DARTIS no-oil goruntulerinde "
            "yanlis alarm acisindan degerlendirir."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--thresholds",
        default="0.60,0.80,0.90",
    )

    parser.add_argument(
        "--visual-threshold",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 butun 2290 goruntu.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def parse_thresholds(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )

    if not values:
        raise ValueError("En az bir threshold gereklidir.")

    for value in values:
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"Threshold 0 ile 1 arasinda olmali: {value}"
            )

    return values


def threshold_key(threshold: float) -> str:
    return f"t{int(round(threshold * 100)):03d}"


def load_grayscale_tensor(
    image_path: Path,
) -> tuple[torch.Tensor, int, int]:
    with Image.open(image_path) as image:
        image = image.convert("L")
        width, height = image.size

        tensor = pil_to_tensor(image).float() / 255.0

    return tensor, width, height


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = tensor.shape

    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple

    if pad_height == 0 and pad_width == 0:
        return tensor, 0, 0

    padded = functional.pad(
        tensor,
        pad=(0, pad_width, 0, pad_height),
        mode="replicate",
    )

    return padded, pad_height, pad_width


def infer_probability(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    original_height = image_tensor.shape[-2]
    original_width = image_tensor.shape[-1]

    batch = image_tensor.unsqueeze(0).to(
        device,
        non_blocking=True,
    )

    padded_batch, _, _ = pad_to_multiple(
        batch,
        MODEL_SIZE_MULTIPLE,
    )

    with torch.no_grad():
        logits = model(padded_batch)
        probability = torch.sigmoid(logits)

    probability = probability[
        0,
        0,
        :original_height,
        :original_width,
    ]

    return probability.cpu()


def create_group_state() -> dict[str, Any]:
    return {
        "image_count": 0,
        "total_pixels": 0,
        "positive_pixels": 0,
        "any_alarm_count": 0,
        "alarm_0_01_percent_count": 0,
        "alarm_0_1_percent_count": 0,
        "alarm_1_percent_count": 0,
        "positive_ratios": [],
    }


def update_group(
    state: dict[str, Any],
    positive_pixels: int,
    total_pixels: int,
) -> None:
    ratio = positive_pixels / total_pixels

    state["image_count"] += 1
    state["total_pixels"] += total_pixels
    state["positive_pixels"] += positive_pixels
    state["positive_ratios"].append(ratio)

    state["any_alarm_count"] += int(
        positive_pixels > 0
    )

    state["alarm_0_01_percent_count"] += int(
        ratio >= 0.0001
    )

    state["alarm_0_1_percent_count"] += int(
        ratio >= 0.001
    )

    state["alarm_1_percent_count"] += int(
        ratio >= 0.01
    )


def finalize_group(
    state: dict[str, Any],
) -> dict[str, float | int]:
    image_count = int(state["image_count"])
    total_pixels = int(state["total_pixels"])

    ratios = np.asarray(
        state["positive_ratios"],
        dtype=np.float64,
    )

    def percentage(count: int) -> float:
        if image_count == 0:
            return 0.0

        return count / image_count * 100.0

    return {
        "image_count": image_count,
        "any_alarm_count": int(
            state["any_alarm_count"]
        ),
        "any_alarm_percentage": percentage(
            int(state["any_alarm_count"])
        ),
        "alarm_ge_0_01_percent_count": int(
            state["alarm_0_01_percent_count"]
        ),
        "alarm_ge_0_01_percent_percentage": percentage(
            int(state["alarm_0_01_percent_count"])
        ),
        "alarm_ge_0_1_percent_count": int(
            state["alarm_0_1_percent_count"]
        ),
        "alarm_ge_0_1_percent_percentage": percentage(
            int(state["alarm_0_1_percent_count"])
        ),
        "alarm_ge_1_percent_count": int(
            state["alarm_1_percent_count"]
        ),
        "alarm_ge_1_percent_percentage": percentage(
            int(state["alarm_1_percent_count"])
        ),
        "global_false_positive_pixel_percentage": (
            state["positive_pixels"]
            / total_pixels
            * 100.0
            if total_pixels > 0
            else 0.0
        ),
        "mean_image_false_positive_percentage": (
            float(ratios.mean() * 100.0)
            if len(ratios) > 0
            else 0.0
        ),
        "median_image_false_positive_percentage": (
            float(np.median(ratios) * 100.0)
            if len(ratios) > 0
            else 0.0
        ),
        "maximum_image_false_positive_percentage": (
            float(ratios.max() * 100.0)
            if len(ratios) > 0
            else 0.0
        ),
    }


def save_top_false_alarm_figure(
    records: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    raw_dir: Path,
    threshold: float,
    top_k: int,
    output_path: Path,
    device: torch.device,
) -> None:
    ratio_column = (
        f"{threshold_key(threshold)}_positive_ratio"
    )

    selected = sorted(
        records,
        key=lambda item: float(item[ratio_column]),
        reverse=True,
    )[:top_k]

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

    figure, axes = plt.subplots(
        nrows=len(selected),
        ncols=3,
        figsize=(12, 3.4 * len(selected)),
        squeeze=False,
    )

    for row_index, record in enumerate(selected):
        image_path = (
            raw_dir
            / str(record["image_set"])
            / str(record["image_name"])
        )

        image_tensor, _, _ = load_grayscale_tensor(
            image_path
        )

        probability = infer_probability(
            model=model,
            image_tensor=image_tensor,
            device=device,
        )

        prediction = probability >= threshold

        axes[row_index, 0].imshow(
            image_tensor[0].numpy(),
            cmap="gray",
        )
        axes[row_index, 0].set_title(
            f"{record['image_name']} | {record['image_set']}"
        )

        axes[row_index, 1].imshow(
            probability.numpy(),
            vmin=0.0,
            vmax=1.0,
        )
        axes[row_index, 1].set_title(
            "Petrol olasiligi"
        )

        axes[row_index, 2].imshow(
            prediction.numpy(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row_index, 2].set_title(
            "Yanlis alarm maskesi\n"
            f"Alan: %{float(record[ratio_column]) * 100:.4f}"
        )

        for column_index in range(3):
            axes[row_index, column_index].axis("off")

    figure.suptitle(
        "DARTIS no-oil: En buyuk yanlis alarm ornekleri\n"
        f"Checkpoint: {checkpoint_path.name}, "
        f"threshold={threshold:.2f}",
        fontsize=14,
    )

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    if args.visual_threshold not in thresholds:
        thresholds.append(args.visual_threshold)
        thresholds = sorted(set(thresholds))

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Manifest bulunamadi: {args.manifest}"
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadi: {args.checkpoint}"
        )

    dataframe = pd.read_csv(
        args.manifest,
        encoding="utf-8-sig",
    )

    if args.max_images > 0:
        dataframe = dataframe.head(args.max_images)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    group_states = {
        threshold: defaultdict(create_group_state)
        for threshold in thresholds
    }

    per_image_records: list[dict[str, Any]] = []

    missing_files: list[str] = []
    dimension_mismatches: list[dict[str, Any]] = []

    print("=" * 78)
    print("DARTIS NO-OIL BAGIMSIZ STRES TESTI")
    print("=" * 78)
    print("Cihaz:      ", device)
    print("Checkpoint: ", args.checkpoint)
    print("Goruntu:    ", len(dataframe))
    print("Thresholds: ", thresholds)
    print(
        "Not: Butun DARTIS ornekleri petrolsuzdur; "
        "pozitif tahminler false positive kabul edilir."
    )

    for processed_index, row in enumerate(
        dataframe.itertuples(index=False),
        start=1,
    ):
        image_set = str(row.image_set)
        image_name = str(row.image_name)

        image_path = (
            args.raw_dir
            / image_set
            / image_name
        )

        if not image_path.exists():
            missing_files.append(str(image_path))
            continue

        image_tensor, width, height = (
            load_grayscale_tensor(image_path)
        )

        expected_width = int(row.width)
        expected_height = int(row.height)

        if (
            width != expected_width
            or height != expected_height
        ):
            dimension_mismatches.append(
                {
                    "image_name": image_name,
                    "expected_width": expected_width,
                    "expected_height": expected_height,
                    "actual_width": width,
                    "actual_height": height,
                }
            )

        probability = infer_probability(
            model=model,
            image_tensor=image_tensor,
            device=device,
        )

        total_pixels = int(probability.numel())

        record: dict[str, Any] = {
            "image_name": image_name,
            "image_set": image_set,
            "category": str(row.category),
            "width": width,
            "height": height,
            "mean_probability": float(
                probability.mean().item()
            ),
            "maximum_probability": float(
                probability.max().item()
            ),
        }

        for threshold in thresholds:
            prediction = probability >= threshold

            positive_pixels = int(
                prediction.sum().item()
            )

            positive_ratio = (
                positive_pixels / total_pixels
            )

            key = threshold_key(threshold)

            record[f"{key}_positive_pixels"] = (
                positive_pixels
            )
            record[f"{key}_positive_ratio"] = (
                positive_ratio
            )
            record[f"{key}_any_alarm"] = int(
                positive_pixels > 0
            )

            for group_name in (
                "overall",
                image_set,
            ):
                update_group(
                    state=group_states[
                        threshold
                    ][group_name],
                    positive_pixels=positive_pixels,
                    total_pixels=total_pixels,
                )

        per_image_records.append(record)

        if processed_index % 100 == 0:
            print(
                f"{processed_index}/{len(dataframe)} "
                "goruntu tamamlandi."
            )

    if missing_files:
        raise RuntimeError(
            f"{len(missing_files)} goruntu bulunamadi. "
            f"Ilk eksik: {missing_files[0]}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_image_csv_path = (
        args.output_dir
        / "dartis_no_oil_per_image_results.csv"
    )

    summary_json_path = (
        args.output_dir
        / "dartis_no_oil_summary.json"
    )

    figure_path = (
        args.output_dir
        / "top_false_alarm_examples.png"
    )

    pd.DataFrame(per_image_records).to_csv(
        per_image_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "dataset": "DARTIS_2019_no_oil",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "processed_image_count": len(
            per_image_records
        ),
        "dimension_mismatch_count": len(
            dimension_mismatches
        ),
        "dimension_mismatches": (
            dimension_mismatches[:20]
        ),
        "thresholds": {},
    }

    for threshold in thresholds:
        summary["thresholds"][
            f"{threshold:.2f}"
        ] = {
            group_name: finalize_group(state)
            for group_name, state
            in group_states[threshold].items()
        }

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    save_top_false_alarm_figure(
        records=per_image_records,
        checkpoint=checkpoint,
        checkpoint_path=args.checkpoint,
        raw_dir=args.raw_dir,
        threshold=args.visual_threshold,
        top_k=args.top_k,
        output_path=figure_path,
        device=device,
    )

    print()
    print("=" * 100)
    print("DARTIS NO-OIL SONUCLARI")
    print("=" * 100)

    print(
        f"{'Thr':>5} | "
        f"{'Grup':>7} | "
        f"{'N':>5} | "
        f"{'Any alarm':>10} | "
        f"{'>=0.01%':>9} | "
        f"{'>=0.1%':>8} | "
        f"{'>=1%':>7} | "
        f"{'FP pixel':>9}"
    )

    print("-" * 100)

    for threshold in thresholds:
        for group_name in ("overall", "nc", "nw"):
            state = group_states[threshold].get(
                group_name
            )

            if state is None:
                continue

            metrics = finalize_group(state)

            print(
                f"{threshold:5.2f} | "
                f"{group_name:>7} | "
                f"{metrics['image_count']:5d} | "
                f"{metrics['any_alarm_percentage']:9.2f}% | "
                f"{metrics['alarm_ge_0_01_percent_percentage']:8.2f}% | "
                f"{metrics['alarm_ge_0_1_percent_percentage']:7.2f}% | "
                f"{metrics['alarm_ge_1_percent_percentage']:6.2f}% | "
                f"{metrics['global_false_positive_pixel_percentage']:8.4f}%"
            )

    print()
    print(
        "Any alarm: En az 1 pozitif piksel bulunan goruntu."
    )
    print(
        ">=0.1%: Goruntunun en az binde biri petrol tahmini."
    )
    print(
        ">=1%: Goruntunun en az yuzde biri petrol tahmini."
    )

    print()
    print("CSV:   ", per_image_csv_path.resolve())
    print("JSON:  ", summary_json_path.resolve())
    print("Gorsel:", figure_path.resolve())


if __name__ == "__main__":
    main()
