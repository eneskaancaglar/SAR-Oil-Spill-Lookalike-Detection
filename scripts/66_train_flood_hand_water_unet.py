from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_sen1floods11_official_pairs.csv"
)

CHECKPOINT_ROOT = (
    ROOT
    / "checkpoints"
)

OUTPUT_ROOT = (
    ROOT
    / "outputs"
)


@dataclass
class Config:
    manifest: str
    run_name: str
    epochs: int
    batch_size: int
    crop_size: int
    learning_rate: float
    weight_decay: float
    base_channels: int
    num_workers: int
    seed: int
    max_train_batches: int
    max_val_batches: int
    device: str
    amp: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sen1Floods11'in yalnız yüksek kaliteli flood "
            "hand-labeled train/validation çiftleriyle iki kanallı "
            "SAR kara-su U-Net eğitir. Resmi test splitine dokunmaz."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="water_unet_flood_v06",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="0 = bütün train batchleri",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="0 = bütün validation batchleri",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def normalize_sar(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(
        array,
        nan=-50.0,
        posinf=1.0,
        neginf=-50.0,
    ).astype(np.float32)

    array = np.clip(
        array,
        -50.0,
        1.0,
    )

    return (array + 50.0) / 51.0


def normalize_label(array: np.ndarray) -> np.ndarray:
    label = array.astype(np.int64)

    label[label == -1] = 255

    unexpected = ~np.isin(
        label,
        np.array(
            [0, 1, 255],
            dtype=np.int64,
        ),
    )

    if unexpected.any():
        values = np.unique(
            label[unexpected]
        )

        raise ValueError(
            "Beklenmeyen flood label değerleri: "
            f"{values.tolist()}"
        )

    return label


def random_crop(
    image: np.ndarray,
    label: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = image.shape

    if (
        height < crop_size
        or width < crop_size
    ):
        pad_height = max(
            crop_size - height,
            0,
        )
        pad_width = max(
            crop_size - width,
            0,
        )

        image = np.pad(
            image,
            (
                (0, 0),
                (0, pad_height),
                (0, pad_width),
            ),
            mode="reflect",
        )

        label = np.pad(
            label,
            (
                (0, pad_height),
                (0, pad_width),
            ),
            mode="constant",
            constant_values=255,
        )

        _, height, width = image.shape

    top = random.randint(
        0,
        height - crop_size,
    )
    left = random.randint(
        0,
        width - crop_size,
    )

    return (
        image[
            :,
            top:top + crop_size,
            left:left + crop_size,
        ],
        label[
            top:top + crop_size,
            left:left + crop_size,
        ],
    )


class FloodWaterDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        training: bool,
        crop_size: int,
    ) -> None:
        self.records = dataframe.to_dict(
            orient="records"
        )
        self.training = training
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        str,
    ]:
        record = self.records[index]

        image_path = resolve_project_path(
            record["image_path"]
        )
        mask_path = resolve_project_path(
            record["mask_path"]
        )

        with rasterio.open(
            image_path
        ) as source:
            image = source.read()

        with rasterio.open(
            mask_path
        ) as source:
            label = source.read(1)

        if image.shape[0] < 2:
            raise RuntimeError(
                "Sentinel-1 görüntüsünde iki SAR bandı "
                f"bulunamadı: {image_path}"
            )

        image = normalize_sar(
            image[:2]
        )
        label = normalize_label(
            label
        )

        if image.shape[1:] != label.shape:
            raise RuntimeError(
                "Görüntü ve maske piksel boyutları farklı: "
                f"{image_path.name} {image.shape[1:]} "
                f"!= {mask_path.name} {label.shape}"
            )

        if self.training:
            image, label = random_crop(
                image,
                label,
                self.crop_size,
            )

            if random.random() < 0.5:
                image = image[:, :, ::-1]
                label = label[:, ::-1]

            if random.random() < 0.5:
                image = image[:, ::-1, :]
                label = label[::-1, :]

        image = np.ascontiguousarray(
            image,
            dtype=np.float32,
        )
        label = np.ascontiguousarray(
            label,
            dtype=np.int64,
        )

        return (
            torch.from_numpy(image),
            torch.from_numpy(label),
            str(record["sample_key"]),
        )


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
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


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
        x: torch.Tensor,
    ) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(
            self.pool(e1)
        )
        e3 = self.enc3(
            self.pool(e2)
        )
        bridge = self.bridge(
            self.pool(e3)
        )

        d3 = self.up3(bridge)
        d3 = self.dec3(
            torch.cat(
                [d3, e3],
                dim=1,
            )
        )

        d2 = self.up2(d3)
        d2 = self.dec2(
            torch.cat(
                [d2, e2],
                dim=1,
            )
        )

        d1 = self.up1(d2)
        d1 = self.dec1(
            torch.cat(
                [d1, e1],
                dim=1,
            )
        )

        return self.output(d1)


def count_training_pixels(
    dataframe: pd.DataFrame,
) -> tuple[int, int]:
    positive = 0
    negative = 0

    for row in dataframe.to_dict(
        orient="records"
    ):
        mask_path = resolve_project_path(
            row["mask_path"]
        )

        with rasterio.open(
            mask_path
        ) as source:
            label = normalize_label(
                source.read(1)
            )

        valid = label != 255

        positive += int(
            ((label == 1) & valid).sum()
        )
        negative += int(
            ((label == 0) & valid).sum()
        )

    return positive, negative


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    logits = logits.squeeze(1)

    valid = target != 255
    target_binary = (
        target == 1
    ).float()

    bce_map = F.binary_cross_entropy_with_logits(
        logits,
        target_binary,
        reduction="none",
        pos_weight=positive_weight,
    )

    bce = bce_map[
        valid
    ].mean()

    probabilities = torch.sigmoid(
        logits
    )

    valid_float = valid.float()

    intersection = (
        probabilities
        * target_binary
        * valid_float
    ).sum()

    denominator = (
        probabilities
        * valid_float
    ).sum() + (
        target_binary
        * valid_float
    ).sum()

    dice_loss = 1.0 - (
        (2.0 * intersection + 1.0)
        / (denominator + 1.0)
    )

    total = bce + dice_loss

    return total, bce, dice_loss


def empty_metric_accumulator() -> dict[str, float]:
    return {
        "intersection": 0.0,
        "union": 0.0,
        "prediction_positive": 0.0,
        "target_positive": 0.0,
        "true_positive": 0.0,
        "valid_pixels": 0.0,
        "correct_pixels": 0.0,
    }


def update_metrics(
    accumulator: dict[str, float],
    logits: torch.Tensor,
    target: torch.Tensor,
) -> None:
    prediction = (
        torch.sigmoid(
            logits.squeeze(1)
        ) >= 0.5
    )

    valid = target != 255
    target_positive = target == 1

    prediction &= valid
    target_positive &= valid

    intersection = (
        prediction
        & target_positive
    ).sum().item()

    union = (
        prediction
        | target_positive
    ).sum().item()

    true_positive = intersection

    accumulator["intersection"] += (
        intersection
    )
    accumulator["union"] += union
    accumulator[
        "prediction_positive"
    ] += prediction.sum().item()
    accumulator[
        "target_positive"
    ] += target_positive.sum().item()
    accumulator[
        "true_positive"
    ] += true_positive
    accumulator[
        "valid_pixels"
    ] += valid.sum().item()
    accumulator[
        "correct_pixels"
    ] += (
        (prediction == target_positive)
        & valid
    ).sum().item()


def finalize_metrics(
    accumulator: dict[str, float],
) -> dict[str, float]:
    epsilon = 1e-8

    intersection = accumulator[
        "intersection"
    ]
    union = accumulator["union"]

    prediction_positive = accumulator[
        "prediction_positive"
    ]
    target_positive = accumulator[
        "target_positive"
    ]
    true_positive = accumulator[
        "true_positive"
    ]

    dice = (
        2.0 * intersection + epsilon
    ) / (
        prediction_positive
        + target_positive
        + epsilon
    )

    iou = (
        intersection + epsilon
    ) / (
        union + epsilon
    )

    precision = (
        true_positive + epsilon
    ) / (
        prediction_positive + epsilon
    )

    recall = (
        true_positive + epsilon
    ) / (
        target_positive + epsilon
    )

    accuracy = (
        accumulator["correct_pixels"]
        + epsilon
    ) / (
        accumulator["valid_pixels"]
        + epsilon
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    positive_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    max_batches: int,
) -> dict[str, float]:
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    batch_count = 0

    metrics = empty_metric_accumulator()

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for batch_index, (
            images,
            labels,
            _,
        ) in enumerate(loader):
            if (
                max_batches > 0
                and batch_index >= max_batches
            ):
                break

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(images)

                (
                    loss,
                    bce,
                    dice_loss,
                ) = masked_bce_dice_loss(
                    logits,
                    labels,
                    positive_weight,
                )

            if training:
                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

            loss_sum += float(
                loss.detach().item()
            )
            bce_sum += float(
                bce.detach().item()
            )
            dice_loss_sum += float(
                dice_loss.detach().item()
            )
            batch_count += 1

            update_metrics(
                metrics,
                logits.detach(),
                labels,
            )

    if batch_count == 0:
        raise RuntimeError(
            "Hiç batch işlenmedi."
        )

    finalized = finalize_metrics(
        metrics
    )

    finalized.update(
        {
            "loss": (
                loss_sum
                / batch_count
            ),
            "bce_loss": (
                bce_sum
                / batch_count
            ),
            "dice_loss": (
                dice_loss_sum
                / batch_count
            ),
            "batch_count": float(
                batch_count
            ),
        }
    )

    return finalized


def sar_preview(
    tensor: torch.Tensor,
) -> np.ndarray:
    # VH bandını görselleştir.
    band_index = (
        1
        if tensor.shape[0] > 1
        else 0
    )

    array = tensor[
        band_index
    ].detach().cpu().numpy()

    array = np.clip(
        array,
        0.0,
        1.0,
    )

    return (
        array * 255.0
    ).astype(np.uint8)


def save_validation_previews(
    model: nn.Module,
    dataset: FloodWaterDataset,
    device: torch.device,
    output_directory: Path,
    maximum_count: int = 8,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=min(
            maximum_count,
            len(dataset),
        ),
        dtype=int,
    )

    preview_images: list[
        Image.Image
    ] = []

    with torch.no_grad():
        for preview_number, index in enumerate(
            indices,
            start=1,
        ):
            (
                image,
                label,
                sample_key,
            ) = dataset[int(index)]

            logits = model(
                image.unsqueeze(0).to(
                    device
                )
            )

            probability = torch.sigmoid(
                logits[0, 0]
            ).cpu().numpy()

            prediction = probability >= 0.5

            gray = sar_preview(image)

            rgb = np.stack(
                [gray, gray, gray],
                axis=2,
            )

            label_array = label.numpy()

            valid = label_array != 255
            truth = (
                label_array == 1
            ) & valid

            overlay = rgb.copy()

            # Ground truth: yeşil
            overlay[truth, 1] = 255

            # Prediction: kırmızı
            overlay[prediction, 0] = 255

            # Agreement: sarı
            agreement = truth & prediction
            overlay[agreement] = (
                255,
                255,
                0,
            )

            prediction_view = np.stack(
                [
                    (
                        prediction.astype(
                            np.uint8
                        )
                        * 255
                    ),
                ]
                * 3,
                axis=2,
            )

            truth_view = np.stack(
                [
                    (
                        truth.astype(
                            np.uint8
                        )
                        * 255
                    ),
                ]
                * 3,
                axis=2,
            )

            canvas = np.concatenate(
                [
                    rgb,
                    overlay,
                    truth_view,
                    prediction_view,
                ],
                axis=1,
            )

            output_path = (
                output_directory
                / (
                    f"{preview_number:02d}"
                    f"__{sample_key.replace(':', '__')}.png"
                )
            )

            image_object = Image.fromarray(
                canvas
            )

            image_object.save(
                output_path
            )

            preview_images.append(
                image_object
            )

    if preview_images:
        maximum_width = max(
            image.width
            for image in preview_images
        )

        total_height = sum(
            image.height
            for image in preview_images
        )

        contact_sheet = Image.new(
            "RGB",
            (
                maximum_width,
                total_height,
            ),
            "black",
        )

        y = 0

        for image in preview_images:
            contact_sheet.paste(
                image,
                (0, y),
            )
            y += image.height

        contact_sheet.save(
            output_directory
            / "validation_contact_sheet.jpg",
            quality=90,
        )


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    manifest_path = (
        resolve_project_path(
            args.manifest
        )
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest bulunamadı: {manifest_path}"
        )

    dataframe = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = {
        "sample_key",
        "dataset_group",
        "split",
        "image_path",
        "mask_path",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Manifestte eksik sütunlar: "
            f"{sorted(missing_columns)}"
        )

    flood = dataframe[
        dataframe[
            "dataset_group"
        ].eq("flood")
    ].copy()

    train_dataframe = flood[
        flood["split"].eq("train")
    ].copy()

    validation_dataframe = flood[
        flood[
            "split"
        ].eq("validation")
    ].copy()

    test_count = int(
        flood["split"].eq("test").sum()
    )

    external_holdout_count = int(
        flood[
            "split"
        ].eq(
            "external_holdout"
        ).sum()
    )

    if train_dataframe.empty:
        raise RuntimeError(
            "Flood train spliti boş."
        )

    if validation_dataframe.empty:
        raise RuntimeError(
            "Flood validation spliti boş."
        )

    overlap = set(
        train_dataframe[
            "sample_key"
        ]
    ) & set(
        validation_dataframe[
            "sample_key"
        ]
    )

    if overlap:
        raise RuntimeError(
            "Train-validation leakage bulundu."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = bool(
        device.type == "cuda"
        and not args.no_amp
    )

    config = Config(
        manifest=relative_to_root(
            manifest_path
        ),
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_batches=(
            args.max_train_batches
        ),
        max_val_batches=(
            args.max_val_batches
        ),
        device=str(device),
        amp=amp_enabled,
    )

    checkpoint_directory = (
        CHECKPOINT_ROOT
        / args.run_name
    )

    output_directory = (
        OUTPUT_ROOT
        / args.run_name
    )

    preview_directory = (
        output_directory
        / "validation_previews"
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "v0.6 FLOOD HAND-LABELED WATER U-NET"
    )
    print("=" * 78)
    print(
        "Manifest:",
        manifest_path,
    )
    print(
        "Train:",
        len(train_dataframe),
    )
    print(
        "Validation:",
        len(validation_dataframe),
    )
    print(
        "Resmi test ayrıldı:",
        test_count,
    )
    print(
        "External holdout ayrıldı:",
        external_holdout_count,
    )
    print(
        "Cihaz:",
        device,
    )
    print(
        "AMP:",
        amp_enabled,
    )
    print()

    positive_pixels, negative_pixels = (
        count_training_pixels(
            train_dataframe
        )
    )

    positive_weight_value = (
        negative_pixels
        / max(
            positive_pixels,
            1,
        )
    )

    positive_weight_value = float(
        np.clip(
            positive_weight_value,
            1.0,
            10.0,
        )
    )

    print(
        "Train water pixels:",
        positive_pixels,
    )
    print(
        "Train non-water pixels:",
        negative_pixels,
    )
    print(
        "Otomatik pos_weight:",
        f"{positive_weight_value:.4f}",
    )
    print()

    train_dataset = FloodWaterDataset(
        train_dataframe,
        training=True,
        crop_size=args.crop_size,
    )

    validation_dataset = (
        FloodWaterDataset(
            validation_dataframe,
            training=False,
            crop_size=args.crop_size,
        )
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    model = SmallUNet(
        in_channels=2,
        base_channels=args.base_channels,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                args.epochs,
                1,
            ),
        )
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
    )

    positive_weight = torch.tensor(
        positive_weight_value,
        dtype=torch.float32,
        device=device,
    )

    history: list[dict[str, Any]] = []
    best_validation_iou = -1.0
    best_epoch = 0
    best_checkpoint_path = (
        checkpoint_directory
        / "best.pth"
    )

    print(
        "Model parameters:",
        f"{parameter_count:,}",
    )
    print()

    training_start = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            positive_weight=positive_weight,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            max_batches=(
                args.max_train_batches
            ),
        )

        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            positive_weight=positive_weight,
            optimizer=None,
            scaler=scaler,
            amp_enabled=amp_enabled,
            max_batches=(
                args.max_val_batches
            ),
        )

        scheduler.step()

        elapsed = (
            time.time()
            - epoch_start
        )

        row = {
            "epoch": epoch,
            "learning_rate": (
                optimizer.param_groups[
                    0
                ]["lr"]
            ),
            "epoch_seconds": elapsed,
            **{
                f"train_{key}": value
                for key, value
                in train_metrics.items()
            },
            **{
                f"validation_{key}": value
                for key, value
                in validation_metrics.items()
            },
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train loss {train_metrics['loss']:.4f} "
            f"Dice {train_metrics['dice']:.4f} "
            f"IoU {train_metrics['iou']:.4f} | "
            f"Val loss {validation_metrics['loss']:.4f} "
            f"Dice {validation_metrics['dice']:.4f} "
            f"IoU {validation_metrics['iou']:.4f} "
            f"P {validation_metrics['precision']:.4f} "
            f"R {validation_metrics['recall']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if (
            validation_metrics["iou"]
            > best_validation_iou
        ):
            best_validation_iou = (
                validation_metrics["iou"]
            )
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "validation_metrics": (
                        validation_metrics
                    ),
                    "train_metrics": (
                        train_metrics
                    ),
                    "positive_weight": (
                        positive_weight_value
                    ),
                    "parameter_count": (
                        parameter_count
                    ),
                    "config": asdict(
                        config
                    ),
                },
                best_checkpoint_path,
            )

            print(
                "  Best checkpoint saved:",
                best_checkpoint_path,
            )

    total_seconds = (
        time.time()
        - training_start
    )

    history_path = (
        output_directory
        / "history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    save_validation_previews(
        model=model,
        dataset=validation_dataset,
        device=device,
        output_directory=(
            preview_directory
        ),
        maximum_count=8,
    )

    summary = {
        "stage": (
            "v06_flood_hand_labeled_water_unet"
        ),
        "run_name": args.run_name,
        "train_count": int(
            len(train_dataframe)
        ),
        "validation_count": int(
            len(validation_dataframe)
        ),
        "official_test_reserved_count": (
            test_count
        ),
        "external_holdout_reserved_count": (
            external_holdout_count
        ),
        "official_test_evaluated": False,
        "external_holdout_evaluated": False,
        "parameter_count": int(
            parameter_count
        ),
        "positive_weight": (
            positive_weight_value
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_iou": float(
            best_validation_iou
        ),
        "best_validation_metrics": (
            checkpoint[
                "validation_metrics"
            ]
        ),
        "training_seconds": float(
            total_seconds
        ),
        "checkpoint_path": (
            relative_to_root(
                best_checkpoint_path
            )
        ),
        "history_path": relative_to_root(
            history_path
        ),
        "preview_contact_sheet": (
            relative_to_root(
                preview_directory
                / "validation_contact_sheet.jpg"
            )
        ),
        "config": asdict(config),
    }

    summary_path = (
        output_directory
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

    print()
    print("=" * 78)
    print(
        "EĞİTİM TAMAMLANDI"
    )
    print("=" * 78)
    print(
        "Best epoch:",
        best_epoch,
    )
    print(
        "Best validation IoU:",
        f"{best_validation_iou:.4f}",
    )
    print(
        "Checkpoint:",
        best_checkpoint_path.resolve(),
    )
    print(
        "History:",
        history_path.resolve(),
    )
    print(
        "Preview:",
        (
            preview_directory
            / "validation_contact_sheet.jpg"
        ).resolve(),
    )
    print(
        "Summary:",
        summary_path.resolve(),
    )
    print()
    print(
        "Resmi test spliti kullanılmadı."
    )


def relative_to_root(
    path: Path,
) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
