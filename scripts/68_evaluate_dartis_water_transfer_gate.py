from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "water_unet_flood_v06_robust"
    / "best.pth"
)

DEFAULT_DARTIS_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_masks_final.csv"
)

DEFAULT_MANUAL_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_manual_land_corrections.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_water_transfer_gate"
)

THRESHOLDS = [
    round(value, 2)
    for value in np.arange(
        0.30,
        0.901,
        0.05,
    )
]

PREPROCESSING_MODES = (
    "duplicate_raw",
    "duplicate_percentile",
    "raw_clahe",
    "percentile_clahe",
)

POSTPROCESSING_MODES = (
    "none",
    "conservative",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sen1Floods11 üzerinde eğitilen su modelinin DARTIS "
            "kıyı görüntülerine aktarılabilirliğini 7 manuel hizalı "
            "geliştirme maskesi üzerinde ölçer. Bu aşamada tüm DARTIS "
            "verisine maske uygulanmaz."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--dartis-manifest",
        type=Path,
        default=DEFAULT_DARTIS_MANIFEST,
    )

    parser.add_argument(
        "--manual-manifest",
        type=Path,
        default=DEFAULT_MANUAL_MANIFEST,
    )

    parser.add_argument(
        "--minimum-water-precision",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--minimum-water-iou",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--coast-buffer-pixels",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--minimum-component-pixels",
        type=int,
        default=256,
    )

    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        group_count = min(
            8,
            out_channels,
        )

        while (
            out_channels % group_count != 0
            and group_count > 1
        ):
            group_count -= 1

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                group_count,
                out_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                group_count,
                out_channels,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(tensor)


class SmallUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2

        self.enc1 = ConvBlock(
            in_channels,
            c1,
        )
        self.enc2 = ConvBlock(
            c1,
            c2,
        )
        self.enc3 = ConvBlock(
            c2,
            c3,
        )
        self.bridge = ConvBlock(
            c3,
            c4,
        )

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(
            c4,
            c3,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = ConvBlock(
            c3 + c3,
            c3,
        )

        self.up2 = nn.ConvTranspose2d(
            c3,
            c2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ConvBlock(
            c2 + c2,
            c2,
        )

        self.up1 = nn.ConvTranspose2d(
            c2,
            c1,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = ConvBlock(
            c1 + c1,
            c1,
        )

        self.output = nn.Conv2d(
            c1,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        encoder1 = self.enc1(tensor)

        encoder2 = self.enc2(
            self.pool(encoder1)
        )

        encoder3 = self.enc3(
            self.pool(encoder2)
        )

        bridge = self.bridge(
            self.pool(encoder3)
        )

        decoder3 = self.up3(bridge)
        decoder3 = self.dec3(
            torch.cat(
                [
                    decoder3,
                    encoder3,
                ],
                dim=1,
            )
        )

        decoder2 = self.up2(decoder3)
        decoder2 = self.dec2(
            torch.cat(
                [
                    decoder2,
                    encoder2,
                ],
                dim=1,
            )
        )

        decoder1 = self.up1(decoder2)
        decoder1 = self.dec1(
            torch.cat(
                [
                    decoder1,
                    encoder1,
                ],
                dim=1,
            )
        )

        return self.output(decoder1)


def read_grayscale(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.array(image)


def percentile_normalize(
    image: np.ndarray,
) -> np.ndarray:
    array = image.astype(np.float32)

    low = float(
        np.percentile(
            array,
            2,
        )
    )
    high = float(
        np.percentile(
            array,
            98,
        )
    )

    if high <= low:
        return (
            array / 255.0
        ).astype(np.float32)

    normalized = np.clip(
        (array - low)
        / (high - low),
        0.0,
        1.0,
    )

    return normalized.astype(
        np.float32
    )


def clahe_channel(
    image: np.ndarray,
) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(
        image.astype(np.uint8)
    )

    return (
        enhanced.astype(np.float32)
        / 255.0
    )


def prepare_input(
    image: np.ndarray,
    mode: str,
) -> np.ndarray:
    raw = (
        image.astype(np.float32)
        / 255.0
    )

    percentile = percentile_normalize(
        image
    )

    clahe = clahe_channel(
        image
    )

    if mode == "duplicate_raw":
        channels = [
            raw,
            raw,
        ]
    elif mode == "duplicate_percentile":
        channels = [
            percentile,
            percentile,
        ]
    elif mode == "raw_clahe":
        channels = [
            raw,
            clahe,
        ]
    elif mode == "percentile_clahe":
        channels = [
            percentile,
            clahe,
        ]
    else:
        raise ValueError(
            f"Bilinmeyen preprocessing: {mode}"
        )

    return np.stack(
        channels,
        axis=0,
    ).astype(np.float32)


def keep_border_components(
    mask: np.ndarray,
    minimum_component_pixels: int,
) -> np.ndarray:
    count, labels, statistics, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
    )

    border_labels = set(
        np.unique(
            np.concatenate(
                [
                    labels[0, :],
                    labels[-1, :],
                    labels[:, 0],
                    labels[:, -1],
                ]
            )
        ).tolist()
    )

    output = np.zeros_like(
        mask,
        dtype=bool,
    )

    for label_id in range(
        1,
        count,
    ):
        area = int(
            statistics[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if (
            label_id in border_labels
            and area
            >= minimum_component_pixels
        ):
            output[
                labels == label_id
            ] = True

    return output


def conservative_postprocess(
    prediction: np.ndarray,
    coast_buffer_pixels: int,
    minimum_component_pixels: int,
) -> np.ndarray:
    result = keep_border_components(
        prediction,
        minimum_component_pixels,
    )

    if coast_buffer_pixels > 0:
        kernel_size = (
            2 * coast_buffer_pixels
            + 1
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )

        result = cv2.erode(
            result.astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)

    return result


def apply_postprocessing(
    prediction: np.ndarray,
    mode: str,
    coast_buffer_pixels: int,
    minimum_component_pixels: int,
) -> np.ndarray:
    if mode == "none":
        return prediction.astype(bool)

    if mode == "conservative":
        return conservative_postprocess(
            prediction,
            coast_buffer_pixels,
            minimum_component_pixels,
        )

    raise ValueError(
        f"Bilinmeyen postprocessing: {mode}"
    )


def empty_counts() -> dict[str, float]:
    return {
        "true_positive": 0.0,
        "false_positive": 0.0,
        "false_negative": 0.0,
        "true_negative": 0.0,
    }


def update_counts(
    counts: dict[str, float],
    prediction: np.ndarray,
    truth_water: np.ndarray,
    valid: np.ndarray,
) -> None:
    prediction = prediction & valid
    truth_water = truth_water & valid
    truth_land = (~truth_water) & valid

    counts["true_positive"] += float(
        (
            prediction
            & truth_water
        ).sum()
    )

    counts["false_positive"] += float(
        (
            prediction
            & truth_land
        ).sum()
    )

    counts["false_negative"] += float(
        (
            (~prediction)
            & truth_water
        ).sum()
    )

    counts["true_negative"] += float(
        (
            (~prediction)
            & truth_land
        ).sum()
    )


def finalize_counts(
    counts: dict[str, float],
) -> dict[str, float]:
    epsilon = 1e-8

    true_positive = counts[
        "true_positive"
    ]
    false_positive = counts[
        "false_positive"
    ]
    false_negative = counts[
        "false_negative"
    ]
    true_negative = counts[
        "true_negative"
    ]

    precision = (
        true_positive + epsilon
    ) / (
        true_positive
        + false_positive
        + epsilon
    )

    recall = (
        true_positive + epsilon
    ) / (
        true_positive
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

    dice = (
        2.0 * true_positive
        + epsilon
    ) / (
        2.0 * true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    accuracy = (
        true_positive
        + true_negative
        + epsilon
    ) / (
        true_positive
        + true_negative
        + false_positive
        + false_negative
        + epsilon
    )

    land_leakage_rate = (
        false_positive + epsilon
    ) / (
        false_positive
        + true_negative
        + epsilon
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "water_precision": float(
            precision
        ),
        "water_recall": float(
            recall
        ),
        "accuracy": float(
            accuracy
        ),
        "land_leakage_rate": float(
            land_leakage_rate
        ),
        **{
            key: float(value)
            for key, value
            in counts.items()
        },
    }


def load_calibration_records(
    dartis_manifest_path: Path,
    manual_manifest_path: Path,
) -> pd.DataFrame:
    dartis = pd.read_csv(
        dartis_manifest_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    manual = pd.read_csv(
        manual_manifest_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if "sample_id" not in dartis.columns:
        raise RuntimeError(
            "DARTIS manifestinde sample_id yok."
        )

    if "image_path" not in dartis.columns:
        raise RuntimeError(
            "DARTIS manifestinde image_path yok."
        )

    required_manual_columns = {
        "sample_id",
        "land_mask_path",
        "safe_water_mask_path",
    }

    missing = (
        required_manual_columns
        - set(manual.columns)
    )

    if missing:
        raise RuntimeError(
            "Manuel manifestte eksik sütunlar: "
            f"{sorted(missing)}"
        )

    columns = [
        "sample_id",
        "image_path",
    ]

    if "subset" in dartis.columns:
        columns.append("subset")

    merged = manual.merge(
        dartis[columns],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    missing_images = merged[
        merged["image_path"].isna()
    ]

    if not missing_images.empty:
        raise RuntimeError(
            "DARTIS manifestinde bulunamayan manuel örnekler: "
            + ", ".join(
                missing_images[
                    "sample_id"
                ].astype(str)
            )
        )

    return merged


def infer_probability(
    model: nn.Module,
    device: torch.device,
    image: np.ndarray,
    preprocessing_mode: str,
) -> np.ndarray:
    input_array = prepare_input(
        image,
        preprocessing_mode,
    )

    tensor = torch.from_numpy(
        input_array
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(
                device.type == "cuda"
            ),
        ):
            logits = model(tensor)

        probability = torch.sigmoid(
            logits[0, 0].float()
        ).cpu().numpy()

    return probability


def save_preview(
    image: np.ndarray,
    truth_water: np.ndarray,
    valid: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    gray = image.astype(np.uint8)

    rgb = np.stack(
        [
            gray,
            gray,
            gray,
        ],
        axis=2,
    )

    overlay = rgb.copy()

    true_positive = (
        prediction
        & truth_water
        & valid
    )

    false_positive = (
        prediction
        & (~truth_water)
        & valid
    )

    false_negative = (
        (~prediction)
        & truth_water
        & valid
    )

    overlay[
        true_positive
    ] = (
        255,
        255,
        0,
    )

    overlay[
        false_positive
    ] = (
        255,
        0,
        0,
    )

    overlay[
        false_negative
    ] = (
        0,
        255,
        0,
    )

    probability_view = (
        np.clip(
            probability,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(np.uint8)

    probability_rgb = cv2.applyColorMap(
        probability_view,
        cv2.COLORMAP_TURBO,
    )

    probability_rgb = cv2.cvtColor(
        probability_rgb,
        cv2.COLOR_BGR2RGB,
    )

    truth_view = np.zeros_like(
        rgb
    )

    truth_view[
        truth_water & valid
    ] = (
        255,
        255,
        255,
    )

    prediction_view = np.zeros_like(
        rgb
    )

    prediction_view[
        prediction & valid
    ] = (
        255,
        255,
        255,
    )

    canvas = np.concatenate(
        [
            rgb,
            probability_rgb,
            truth_view,
            prediction_view,
            overlay,
        ],
        axis=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        canvas
    ).save(output_path)


def create_contact_sheet(
    preview_paths: list[Path],
    output_path: Path,
) -> None:
    if not preview_paths:
        return

    images = [
        Image.open(path).convert("RGB")
        for path in preview_paths
    ]

    target_width = 1400
    resized = []

    for image in images:
        ratio = (
            target_width
            / image.width
        )

        resized.append(
            image.resize(
                (
                    target_width,
                    max(
                        1,
                        int(
                            image.height
                            * ratio
                        ),
                    ),
                ),
                Image.Resampling.LANCZOS,
            )
        )

    total_height = sum(
        image.height
        for image in resized
    )

    sheet = Image.new(
        "RGB",
        (
            target_width,
            total_height,
        ),
        "black",
    )

    y = 0

    for image in resized:
        sheet.paste(
            image,
            (0, y),
        )
        y += image.height

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=90,
    )


def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    dartis_manifest_path = (
        resolve_path(
            args.dartis_manifest
        )
    )

    manual_manifest_path = (
        resolve_path(
            args.manual_manifest
        )
    )

    for path in (
        checkpoint_path,
        dartis_manifest_path,
        manual_manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint.get(
        "config",
        {},
    )

    base_channels = int(
        config.get(
            "base_channels",
            16,
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SmallUNet(
        in_channels=2,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    records = load_calibration_records(
        dartis_manifest_path,
        manual_manifest_path,
    )

    print("=" * 78)
    print(
        "v0.6 DARTIS WATER TRANSFER GATE"
    )
    print("=" * 78)
    print(
        "Checkpoint:",
        checkpoint_path,
    )
    print(
        "Manuel geliştirme örneği:",
        len(records),
    )
    print(
        "Cihaz:",
        device,
    )
    print()

    cached: dict[
        tuple[str, str],
        dict[str, np.ndarray],
    ] = {}

    for row in records.to_dict(
        orient="records"
    ):
        sample_id = str(
            row["sample_id"]
        )

        image_path = resolve_path(
            row["image_path"]
        )

        land_path = resolve_path(
            row["land_mask_path"]
        )

        safe_water_path = resolve_path(
            row["safe_water_mask_path"]
        )

        for path in (
            image_path,
            land_path,
            safe_water_path,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Örnek dosyası bulunamadı: {path}"
                )

        image = read_grayscale(
            image_path
        )

        land = (
            read_grayscale(
                land_path
            )
            > 127
        )

        safe_water = (
            read_grayscale(
                safe_water_path
            )
            > 127
        )

        if land.shape != safe_water.shape:
            raise RuntimeError(
                "Manuel land ve safe-water maskeleri aynı boyutta değil: "
                f"{sample_id}: land={land.shape}, water={safe_water.shape}"
            )

        if image.shape != land.shape:
            original_shape = image.shape

            target_height, target_width = land.shape

            interpolation = (
                cv2.INTER_AREA
                if (
                    image.shape[0] > target_height
                    or image.shape[1] > target_width
                )
                else cv2.INTER_CUBIC
            )

            image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=interpolation,
            )

            print(
                "Yeniden boyutlandırıldı:",
                sample_id,
                f"{original_shape} -> {image.shape}",
            )

        overlap = land & safe_water

        if overlap.any():
            raise RuntimeError(
                "Land ve safe-water maskeleri çakışıyor: "
                f"{sample_id}"
            )

        valid = land | safe_water

        if not valid.any():
            raise RuntimeError(
                f"Geçerli kalibrasyon pikseli yok: {sample_id}"
            )

        cached[
            (
                sample_id,
                "base",
            )
        ] = {
            "image": image,
            "land": land,
            "safe_water": safe_water,
            "valid": valid,
        }

        for preprocessing_mode in (
            PREPROCESSING_MODES
        ):
            probability = infer_probability(
                model=model,
                device=device,
                image=image,
                preprocessing_mode=(
                    preprocessing_mode
                ),
            )

            cached[
                (
                    sample_id,
                    preprocessing_mode,
                )
            ] = {
                "probability": probability,
            }

    sweep_rows = []
    per_sample_rows = []

    for preprocessing_mode in (
        PREPROCESSING_MODES
    ):
        for postprocessing_mode in (
            POSTPROCESSING_MODES
        ):
            for threshold in THRESHOLDS:
                aggregate = empty_counts()

                for row in records.to_dict(
                    orient="records"
                ):
                    sample_id = str(
                        row["sample_id"]
                    )

                    base = cached[
                        (
                            sample_id,
                            "base",
                        )
                    ]

                    probability = cached[
                        (
                            sample_id,
                            preprocessing_mode,
                        )
                    ]["probability"]

                    prediction = (
                        probability
                        >= threshold
                    )

                    prediction = (
                        apply_postprocessing(
                            prediction=prediction,
                            mode=postprocessing_mode,
                            coast_buffer_pixels=(
                                args.coast_buffer_pixels
                            ),
                            minimum_component_pixels=(
                                args.minimum_component_pixels
                            ),
                        )
                    )

                    sample_counts = empty_counts()

                    update_counts(
                        counts=sample_counts,
                        prediction=prediction,
                        truth_water=base[
                            "safe_water"
                        ],
                        valid=base[
                            "valid"
                        ],
                    )

                    update_counts(
                        counts=aggregate,
                        prediction=prediction,
                        truth_water=base[
                            "safe_water"
                        ],
                        valid=base[
                            "valid"
                        ],
                    )

                    sample_metrics = (
                        finalize_counts(
                            sample_counts
                        )
                    )

                    per_sample_rows.append(
                        {
                            "sample_id": sample_id,
                            "preprocessing": (
                                preprocessing_mode
                            ),
                            "postprocessing": (
                                postprocessing_mode
                            ),
                            "threshold": threshold,
                            **sample_metrics,
                        }
                    )

                aggregate_metrics = (
                    finalize_counts(
                        aggregate
                    )
                )

                sweep_rows.append(
                    {
                        "preprocessing": (
                            preprocessing_mode
                        ),
                        "postprocessing": (
                            postprocessing_mode
                        ),
                        "threshold": threshold,
                        **aggregate_metrics,
                    }
                )

    sweep = pd.DataFrame(
        sweep_rows
    )

    eligible = sweep[
        (
            sweep[
                "water_precision"
            ]
            >= args.minimum_water_precision
        )
        & (
            sweep["iou"]
            >= args.minimum_water_iou
        )
    ].copy()

    if eligible.empty:
        selected = (
            sweep.sort_values(
                [
                    "water_precision",
                    "iou",
                    "water_recall",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

        gate_passed = False
    else:
        selected = (
            eligible.sort_values(
                [
                    "iou",
                    "water_precision",
                    "water_recall",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

        gate_passed = True

    selected_preprocessing = str(
        selected["preprocessing"]
    )

    selected_postprocessing = str(
        selected["postprocessing"]
    )

    selected_threshold = float(
        selected["threshold"]
    )

    sweep_path = (
        OUTPUT_DIR
        / "transfer_sweep.csv"
    )

    per_sample_path = (
        OUTPUT_DIR
        / "transfer_per_sample.csv"
    )

    sweep.to_csv(
        sweep_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        per_sample_rows
    ).to_csv(
        per_sample_path,
        index=False,
        encoding="utf-8-sig",
    )

    preview_directory = (
        OUTPUT_DIR
        / "selected_previews"
    )

    preview_paths = []

    selected_sample_metrics = []

    for row in records.to_dict(
        orient="records"
    ):
        sample_id = str(
            row["sample_id"]
        )

        base = cached[
            (
                sample_id,
                "base",
            )
        ]

        probability = cached[
            (
                sample_id,
                selected_preprocessing,
            )
        ]["probability"]

        prediction = (
            probability
            >= selected_threshold
        )

        prediction = apply_postprocessing(
            prediction=prediction,
            mode=selected_postprocessing,
            coast_buffer_pixels=(
                args.coast_buffer_pixels
            ),
            minimum_component_pixels=(
                args.minimum_component_pixels
            ),
        )

        sample_counts = empty_counts()

        update_counts(
            counts=sample_counts,
            prediction=prediction,
            truth_water=base[
                "safe_water"
            ],
            valid=base["valid"],
        )

        metrics = finalize_counts(
            sample_counts
        )

        selected_sample_metrics.append(
            {
                "sample_id": sample_id,
                **metrics,
            }
        )

        preview_path = (
            preview_directory
            / (
                sample_id.replace(
                    ":",
                    "__",
                )
                + ".png"
            )
        )

        save_preview(
            image=base["image"],
            truth_water=base[
                "safe_water"
            ],
            valid=base["valid"],
            probability=probability,
            prediction=prediction,
            output_path=preview_path,
        )

        preview_paths.append(
            preview_path
        )

    selected_sample_path = (
        OUTPUT_DIR
        / "selected_per_sample.csv"
    )

    pd.DataFrame(
        selected_sample_metrics
    ).to_csv(
        selected_sample_path,
        index=False,
        encoding="utf-8-sig",
    )

    contact_sheet_path = (
        OUTPUT_DIR
        / "selected_contact_sheet.jpg"
    )

    create_contact_sheet(
        preview_paths,
        contact_sheet_path,
    )

    summary = {
        "stage": (
            "v06_dartis_water_transfer_gate"
        ),
        "checkpoint": relative(
            checkpoint_path
        ),
        "manual_development_count": int(
            len(records)
        ),
        "manual_development_note": (
            "Bu 7 örnek hedef-domain geliştirme/kalibrasyon "
            "kümesidir; bağımsız final test değildir."
        ),
        "minimum_water_precision": float(
            args.minimum_water_precision
        ),
        "minimum_water_iou": float(
            args.minimum_water_iou
        ),
        "gate_passed": bool(
            gate_passed
        ),
        "selected_preprocessing": (
            selected_preprocessing
        ),
        "selected_postprocessing": (
            selected_postprocessing
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "selected_metrics": {
            key: float(
                selected[key]
            )
            for key in (
                "dice",
                "iou",
                "water_precision",
                "water_recall",
                "accuracy",
                "land_leakage_rate",
            )
        },
        "sweep_path": relative(
            sweep_path
        ),
        "selected_per_sample_path": (
            relative(
                selected_sample_path
            )
        ),
        "selected_contact_sheet": (
            relative(
                contact_sheet_path
            )
        ),
        "all_dartis_masks_generated": False,
        "training_performed": False,
        "official_sen1floods11_test_used": False,
    }

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        "DARTIS WATER TRANSFER GATE SONUCU"
    )
    print("=" * 78)
    print(
        "Gate:",
        (
            "PASS"
            if gate_passed
            else "FAIL"
        ),
    )
    print(
        "Preprocessing:",
        selected_preprocessing,
    )
    print(
        "Postprocessing:",
        selected_postprocessing,
    )
    print(
        "Threshold:",
        f"{selected_threshold:.2f}",
    )
    print(
        "Water precision:",
        f"{float(selected['water_precision']):.4f}",
    )
    print(
        "Water recall:",
        f"{float(selected['water_recall']):.4f}",
    )
    print(
        "Water IoU:",
        f"{float(selected['iou']):.4f}",
    )
    print(
        "Land leakage rate:",
        f"{float(selected['land_leakage_rate']):.4f}",
    )
    print(
        "Özet:",
        summary_path.resolve(),
    )
    print(
        "Görsel:",
        contact_sheet_path.resolve(),
    )
    print()
    print(
        "Bu aşamada tüm DARTIS görüntülerine maske uygulanmadı."
    )


if __name__ == "__main__":
    main()
