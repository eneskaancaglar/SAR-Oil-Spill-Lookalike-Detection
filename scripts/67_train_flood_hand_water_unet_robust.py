from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "66_train_flood_hand_water_unet.py"
DEFAULT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_sen1floods11_official_pairs.csv"
)


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Gerekli 66 scripti bulunamadÄ±: {BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "water_unet_v06_base",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("66 scripti modÃ¼l olarak yÃ¼klenemedi.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


@dataclass
class Config:
    manifest: str
    run_name: str
    epochs: int
    batch_size: int
    crop_size: int
    repeat_factor: int
    water_crop_probability: float
    crop_attempts: int
    learning_rate: float
    weight_decay: float
    base_channels: int
    max_pos_weight: float
    num_workers: int
    seed: int
    device: str
    amp: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NaN validation loss sorununu dÃ¼zelten, su odaklÄ± crop "
            "Ã¶rneklemesi ve validation threshold kalibrasyonu kullanan "
            "Sen1Floods11 su segmentasyonu eÄŸitimi."
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
        default="water_unet_flood_v06_robust",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
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
        "--repeat-factor",
        type=int,
        default=3,
        help="Her train gÃ¶rÃ¼ntÃ¼sÃ¼nden epoch baÅŸÄ±na kaÃ§ farklÄ± crop Ã¼retileceÄŸi.",
    )
    parser.add_argument(
        "--water-crop-probability",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--crop-attempts",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=7e-4,
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
        "--max-pos-weight",
        type=float,
        default=3.0,
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


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(ROOT.resolve())
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def prepare_for_crop(
    image: np.ndarray,
    label: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = image.shape

    pad_height = max(crop_size - height, 0)
    pad_width = max(crop_size - width, 0)

    if pad_height or pad_width:
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

    return image, label


def crop_from_center(
    image: np.ndarray,
    label: np.ndarray,
    center_y: int,
    center_x: int,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = image.shape

    top = int(
        np.clip(
            center_y - crop_size // 2,
            0,
            height - crop_size,
        )
    )

    left = int(
        np.clip(
            center_x - crop_size // 2,
            0,
            width - crop_size,
        )
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


def balanced_random_crop(
    image: np.ndarray,
    label: np.ndarray,
    crop_size: int,
    water_crop_probability: float,
    crop_attempts: int,
) -> tuple[np.ndarray, np.ndarray]:
    image, label = prepare_for_crop(
        image,
        label,
        crop_size,
    )

    _, height, width = image.shape
    water_coordinates = np.argwhere(label == 1)

    best_pair = None
    best_score = -1.0

    for _ in range(max(crop_attempts, 1)):
        use_water_center = bool(
            len(water_coordinates) > 0
            and random.random() < water_crop_probability
        )

        if use_water_center:
            coordinate = water_coordinates[
                random.randrange(len(water_coordinates))
            ]
            center_y = int(coordinate[0])
            center_x = int(coordinate[1])
        else:
            center_y = random.randrange(height)
            center_x = random.randrange(width)

        crop_image, crop_label = crop_from_center(
            image,
            label,
            center_y,
            center_x,
            crop_size,
        )

        valid = crop_label != 255
        valid_fraction = float(valid.mean())

        if valid.any():
            water_fraction = float(
                ((crop_label == 1) & valid).sum()
                / valid.sum()
            )
        else:
            water_fraction = 0.0

        # Hem geÃ§erli piksel hem de sÄ±nÄ±f Ã§eÅŸitliliÄŸi olan crop'larÄ± Ã¶ne alÄ±r.
        class_balance_score = min(
            water_fraction,
            1.0 - water_fraction,
        )

        score = (
            valid_fraction
            + 0.35 * class_balance_score
        )

        if score > best_score:
            best_score = score
            best_pair = (
                crop_image,
                crop_label,
            )

        if (
            valid_fraction >= 0.90
            and (
                water_fraction >= 0.01
                or not use_water_center
            )
        ):
            return crop_image, crop_label

    if best_pair is None:
        raise RuntimeError("GeÃ§erli eÄŸitim crop'u Ã¼retilemedi.")

    return best_pair


class BalancedFloodDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        training: bool,
        crop_size: int,
        repeat_factor: int,
        water_crop_probability: float,
        crop_attempts: int,
    ) -> None:
        self.records = dataframe.to_dict(
            orient="records"
        )
        self.training = training
        self.crop_size = crop_size
        self.repeat_factor = (
            max(repeat_factor, 1)
            if training
            else 1
        )
        self.water_crop_probability = water_crop_probability
        self.crop_attempts = crop_attempts

    def __len__(self) -> int:
        return len(self.records) * self.repeat_factor

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        str,
    ]:
        record = self.records[
            index % len(self.records)
        ]

        image_path = resolve_path(
            record["image_path"]
        )
        mask_path = resolve_path(
            record["mask_path"]
        )

        with rasterio.open(image_path) as source:
            image = source.read()

        with rasterio.open(mask_path) as source:
            label = source.read(1)

        if image.shape[0] < 2:
            raise RuntimeError(
                f"Ä°ki SAR bandÄ± bulunamadÄ±: {image_path}"
            )

        image = BASE.normalize_sar(image[:2])
        label = BASE.normalize_label(label)

        if image.shape[1:] != label.shape:
            raise RuntimeError(
                "GÃ¶rÃ¼ntÃ¼ ve maske boyutu farklÄ±: "
                f"{image_path.name} {image.shape[1:]} != "
                f"{mask_path.name} {label.shape}"
            )

        if self.training:
            image, label = balanced_random_crop(
                image=image,
                label=label,
                crop_size=self.crop_size,
                water_crop_probability=self.water_crop_probability,
                crop_attempts=self.crop_attempts,
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


def masked_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
] | None:
    logits = logits.squeeze(1)

    valid = target != 255

    if int(valid.sum().item()) == 0:
        return None

    target_binary = (target == 1).float()

    valid_logits = logits[valid].float()
    valid_targets = target_binary[valid].float()

    bce = F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
        pos_weight=positive_weight.float(),
    )

    probabilities = torch.sigmoid(
        logits.float()
    )

    valid_float = valid.float()

    intersection = (
        probabilities
        * target_binary
        * valid_float
    ).sum()

    denominator = (
        probabilities * valid_float
    ).sum() + (
        target_binary * valid_float
    ).sum()

    dice_loss = 1.0 - (
        (2.0 * intersection + 1.0)
        / (denominator + 1.0)
    )

    loss = bce + dice_loss

    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Loss sonlu deÄŸil. "
            f"BCE={float(bce.detach())}, "
            f"Dice={float(dice_loss.detach())}"
        )

    return loss, bce, dice_loss


def empty_counts() -> dict[str, float]:
    return {
        "intersection": 0.0,
        "union": 0.0,
        "prediction_positive": 0.0,
        "target_positive": 0.0,
        "correct": 0.0,
        "valid": 0.0,
    }


def update_counts(
    counts: dict[str, float],
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> None:
    valid = target != 255

    if int(valid.sum().item()) == 0:
        return

    prediction = (
        torch.sigmoid(logits.squeeze(1).float())
        >= threshold
    )

    truth = target == 1

    prediction &= valid
    truth &= valid

    counts["intersection"] += float(
        (prediction & truth).sum().item()
    )

    counts["union"] += float(
        (prediction | truth).sum().item()
    )

    counts["prediction_positive"] += float(
        prediction.sum().item()
    )

    counts["target_positive"] += float(
        truth.sum().item()
    )

    counts["correct"] += float(
        ((prediction == truth) & valid).sum().item()
    )

    counts["valid"] += float(
        valid.sum().item()
    )


def finalize_counts(
    counts: dict[str, float],
) -> dict[str, float]:
    epsilon = 1e-8

    intersection = counts["intersection"]
    prediction_positive = counts[
        "prediction_positive"
    ]
    target_positive = counts[
        "target_positive"
    ]

    return {
        "dice": float(
            (2.0 * intersection + epsilon)
            / (
                prediction_positive
                + target_positive
                + epsilon
            )
        ),
        "iou": float(
            (intersection + epsilon)
            / (counts["union"] + epsilon)
        ),
        "precision": float(
            (intersection + epsilon)
            / (
                prediction_positive
                + epsilon
            )
        ),
        "recall": float(
            (intersection + epsilon)
            / (
                target_positive
                + epsilon
            )
        ),
        "accuracy": float(
            (counts["correct"] + epsilon)
            / (counts["valid"] + epsilon)
        ),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    positive_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None

    model.train(training)

    loss_total = 0.0
    bce_total = 0.0
    dice_loss_total = 0.0
    processed_batches = 0
    skipped_empty_batches = 0

    counts = empty_counts()

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for images, labels, _ in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            if int((labels != 255).sum().item()) == 0:
                skipped_empty_batches += 1
                continue

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(images)

            loss_result = masked_loss(
                logits=logits,
                target=labels,
                positive_weight=positive_weight,
            )

            if loss_result is None:
                skipped_empty_batches += 1
                continue

            loss, bce, dice_loss = loss_result

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                scaler.step(optimizer)
                scaler.update()

            loss_total += float(
                loss.detach().item()
            )
            bce_total += float(
                bce.detach().item()
            )
            dice_loss_total += float(
                dice_loss.detach().item()
            )
            processed_batches += 1

            update_counts(
                counts=counts,
                logits=logits.detach(),
                target=labels,
                threshold=0.5,
            )

    if processed_batches == 0:
        raise RuntimeError(
            "Epoch iÃ§inde geÃ§erli batch iÅŸlenmedi."
        )

    metrics = finalize_counts(counts)

    metrics.update(
        {
            "loss": loss_total / processed_batches,
            "bce_loss": bce_total / processed_batches,
            "dice_loss": (
                dice_loss_total / processed_batches
            ),
            "processed_batches": float(
                processed_batches
            ),
            "skipped_empty_batches": float(
                skipped_empty_batches
            ),
        }
    )

    return metrics


def threshold_sweep(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    thresholds = [
        round(value, 2)
        for value in np.arange(
            0.30,
            0.851,
            0.05,
        )
    ]

    counts_by_threshold = {
        threshold: empty_counts()
        for threshold in thresholds
    }

    skipped = 0
    model.eval()

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            if int((labels != 255).sum().item()) == 0:
                skipped += 1
                continue

            logits = model(images)

            for threshold in thresholds:
                update_counts(
                    counts=counts_by_threshold[
                        threshold
                    ],
                    logits=logits,
                    target=labels,
                    threshold=threshold,
                )

    rows = []

    for threshold in thresholds:
        metrics = finalize_counts(
            counts_by_threshold[threshold]
        )

        rows.append(
            {
                "threshold": threshold,
                **metrics,
                "skipped_empty_images": skipped,
            }
        )

    return pd.DataFrame(rows)


def preview_gray(
    image: torch.Tensor,
) -> np.ndarray:
    band_index = 1 if image.shape[0] > 1 else 0

    array = image[
        band_index
    ].cpu().numpy()

    return (
        np.clip(array, 0.0, 1.0)
        * 255.0
    ).astype(np.uint8)


def save_previews(
    model: torch.nn.Module,
    dataset: BalancedFloodDataset,
    device: torch.device,
    threshold: float,
    output_directory: Path,
    count: int = 8,
) -> Path:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=min(count, len(dataset)),
        dtype=int,
    )

    images_for_sheet: list[Image.Image] = []

    with torch.no_grad():
        for number, index in enumerate(
            indices,
            start=1,
        ):
            image, label, sample_key = dataset[
                int(index)
            ]

            logits = model(
                image.unsqueeze(0).to(device)
            )

            prediction = (
                torch.sigmoid(
                    logits[0, 0].float()
                ).cpu().numpy()
                >= threshold
            )

            truth = label.numpy() == 1
            valid = label.numpy() != 255

            prediction &= valid
            truth &= valid

            gray = preview_gray(image)
            rgb = np.stack(
                [gray, gray, gray],
                axis=2,
            )

            overlay = rgb.copy()

            false_negative = truth & (~prediction)
            false_positive = prediction & (~truth)
            agreement = truth & prediction

            overlay[false_negative] = (
                0,
                255,
                0,
            )
            overlay[false_positive] = (
                255,
                0,
                0,
            )
            overlay[agreement] = (
                255,
                255,
                0,
            )

            truth_view = np.stack(
                [
                    truth.astype(np.uint8)
                    * 255
                ]
                * 3,
                axis=2,
            )

            prediction_view = np.stack(
                [
                    prediction.astype(np.uint8)
                    * 255
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

            image_object = Image.fromarray(
                canvas
            )

            output_path = (
                output_directory
                / (
                    f"{number:02d}__"
                    f"{sample_key.replace(':', '__')}.png"
                )
            )

            image_object.save(output_path)
            images_for_sheet.append(image_object)

    contact_sheet = output_directory / (
        "validation_contact_sheet.jpg"
    )

    if images_for_sheet:
        width = max(
            image.width
            for image in images_for_sheet
        )

        height = sum(
            image.height
            for image in images_for_sheet
        )

        sheet = Image.new(
            "RGB",
            (width, height),
            "black",
        )

        y = 0

        for image in images_for_sheet:
            sheet.paste(image, (0, y))
            y += image.height

        sheet.save(
            contact_sheet,
            quality=90,
        )

    return contact_sheet


def count_pixels(
    dataframe: pd.DataFrame,
) -> tuple[int, int, int]:
    water = 0
    non_water = 0
    all_invalid_images = 0

    for row in dataframe.to_dict(
        orient="records"
    ):
        with rasterio.open(
            resolve_path(row["mask_path"])
        ) as source:
            label = BASE.normalize_label(
                source.read(1)
            )

        valid = label != 255

        if not valid.any():
            all_invalid_images += 1
            continue

        water += int(
            ((label == 1) & valid).sum()
        )
        non_water += int(
            ((label == 0) & valid).sum()
        )

    return water, non_water, all_invalid_images


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    manifest_path = resolve_path(
        args.manifest
    )

    dataframe = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    flood = dataframe[
        dataframe["dataset_group"].eq(
            "flood"
        )
    ].copy()

    train_dataframe = flood[
        flood["split"].eq("train")
    ].copy()

    validation_dataframe = flood[
        flood["split"].eq("validation")
    ].copy()

    test_count = int(
        flood["split"].eq("test").sum()
    )

    external_holdout_count = int(
        flood["split"].eq(
            "external_holdout"
        ).sum()
    )

    if train_dataframe.empty:
        raise RuntimeError("Train spliti boÅŸ.")

    if validation_dataframe.empty:
        raise RuntimeError("Validation spliti boÅŸ.")

    overlap = set(
        train_dataframe["sample_key"]
    ) & set(
        validation_dataframe["sample_key"]
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

    checkpoint_directory = (
        ROOT
        / "checkpoints"
        / args.run_name
    )

    output_directory = (
        ROOT
        / "outputs"
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

    train_water, train_non_water, train_all_invalid = (
        count_pixels(train_dataframe)
    )

    (
        validation_water,
        validation_non_water,
        validation_all_invalid,
    ) = count_pixels(validation_dataframe)

    raw_ratio = (
        train_non_water
        / max(train_water, 1)
    )

    positive_weight_value = float(
        np.clip(
            math.sqrt(raw_ratio),
            1.0,
            args.max_pos_weight,
        )
    )

    print("=" * 78)
    print("v0.6 ROBUST FLOOD WATER U-NET")
    print("=" * 78)
    print("Train gÃ¶rÃ¼ntÃ¼:", len(train_dataframe))
    print(
        "Validation gÃ¶rÃ¼ntÃ¼:",
        len(validation_dataframe),
    )
    print("Resmi test ayrÄ±ldÄ±:", test_count)
    print(
        "External holdout ayrÄ±ldÄ±:",
        external_holdout_count,
    )
    print("Cihaz:", device)
    print("AMP:", amp_enabled)
    print(
        "Train all-invalid mask:",
        train_all_invalid,
    )
    print(
        "Validation all-invalid mask:",
        validation_all_invalid,
    )
    print(
        "Ham sÄ±nÄ±f oranÄ±:",
        f"{raw_ratio:.4f}",
    )
    print(
        "KullanÄ±lan pos_weight:",
        f"{positive_weight_value:.4f}",
    )
    print()

    train_dataset = BalancedFloodDataset(
        dataframe=train_dataframe,
        training=True,
        crop_size=args.crop_size,
        repeat_factor=args.repeat_factor,
        water_crop_probability=(
            args.water_crop_probability
        ),
        crop_attempts=args.crop_attempts,
    )

    validation_dataset = BalancedFloodDataset(
        dataframe=validation_dataframe,
        training=False,
        crop_size=args.crop_size,
        repeat_factor=1,
        water_crop_probability=0.0,
        crop_attempts=1,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = BASE.SmallUNet(
        in_channels=2,
        base_channels=args.base_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
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

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    config = Config(
        manifest=relative(manifest_path),
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        repeat_factor=args.repeat_factor,
        water_crop_probability=(
            args.water_crop_probability
        ),
        crop_attempts=args.crop_attempts,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        max_pos_weight=args.max_pos_weight,
        num_workers=args.num_workers,
        seed=args.seed,
        device=str(device),
        amp=amp_enabled,
    )

    best_path = checkpoint_directory / "best.pth"
    history_rows = []
    best_iou = -1.0
    best_epoch = 0
    start_time = time.time()

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
        )

        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            positive_weight=positive_weight,
            optimizer=None,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

        scheduler.step()

        elapsed = time.time() - epoch_start

        history_rows.append(
            {
                "epoch": epoch,
                "learning_rate": (
                    optimizer.param_groups[0][
                        "lr"
                    ]
                ),
                "seconds": elapsed,
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
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train L {train_metrics['loss']:.4f} "
            f"D {train_metrics['dice']:.4f} "
            f"IoU {train_metrics['iou']:.4f} | "
            f"Val L {validation_metrics['loss']:.4f} "
            f"D {validation_metrics['dice']:.4f} "
            f"IoU {validation_metrics['iou']:.4f} "
            f"P {validation_metrics['precision']:.4f} "
            f"R {validation_metrics['recall']:.4f} | "
            f"skip {int(validation_metrics['skipped_empty_batches'])} | "
            f"{elapsed:.1f}s"
        )

        if validation_metrics["iou"] > best_iou:
            best_iou = validation_metrics["iou"]
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
                    "config": asdict(config),
                },
                best_path,
            )

            print("  Best checkpoint:", best_path)

    history_path = output_directory / "history.csv"

    pd.DataFrame(
        history_rows
    ).to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    sweep = threshold_sweep(
        model=model,
        loader=validation_loader,
        device=device,
    )

    sweep_path = (
        output_directory
        / "validation_threshold_sweep.csv"
    )

    sweep.to_csv(
        sweep_path,
        index=False,
        encoding="utf-8-sig",
    )

    best_threshold_row = sweep.loc[
        sweep["iou"].idxmax()
    ]

    precision_candidates = sweep[
        sweep["precision"] >= 0.80
    ]

    if precision_candidates.empty:
        safe_threshold_row = best_threshold_row
    else:
        safe_threshold_row = (
            precision_candidates.sort_values(
                [
                    "recall",
                    "iou",
                    "threshold",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            ).iloc[0]
        )

    selected_threshold = float(
        safe_threshold_row["threshold"]
    )

    contact_sheet = save_previews(
        model=model,
        dataset=validation_dataset,
        device=device,
        threshold=selected_threshold,
        output_directory=preview_directory,
        count=8,
    )

    summary = {
        "stage": (
            "v06_robust_flood_water_unet"
        ),
        "run_name": args.run_name,
        "train_count": int(
            len(train_dataframe)
        ),
        "validation_count": int(
            len(validation_dataframe)
        ),
        "train_samples_per_epoch": int(
            len(train_dataset)
        ),
        "official_test_reserved_count": (
            test_count
        ),
        "external_holdout_reserved_count": (
            external_holdout_count
        ),
        "official_test_evaluated": False,
        "external_holdout_evaluated": False,
        "train_all_invalid_masks": (
            train_all_invalid
        ),
        "validation_all_invalid_masks": (
            validation_all_invalid
        ),
        "parameter_count": parameter_count,
        "positive_weight": (
            positive_weight_value
        ),
        "best_epoch": best_epoch,
        "best_validation_iou_at_0_5": (
            best_iou
        ),
        "best_checkpoint_metrics": (
            checkpoint["validation_metrics"]
        ),
        "validation_best_iou_threshold": (
            float(
                best_threshold_row[
                    "threshold"
                ]
            )
        ),
        "validation_best_iou_metrics": {
            key: float(
                best_threshold_row[key]
            )
            for key in (
                "dice",
                "iou",
                "precision",
                "recall",
                "accuracy",
            )
        },
        "selected_safe_threshold": (
            selected_threshold
        ),
        "selected_safe_threshold_metrics": {
            key: float(
                safe_threshold_row[key]
            )
            for key in (
                "dice",
                "iou",
                "precision",
                "recall",
                "accuracy",
            )
        },
        "training_seconds": float(
            time.time() - start_time
        ),
        "checkpoint_path": relative(
            best_path
        ),
        "history_path": relative(
            history_path
        ),
        "threshold_sweep_path": relative(
            sweep_path
        ),
        "preview_contact_sheet": relative(
            contact_sheet
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
    print("ROBUST EÄÄ°TÄ°M TAMAMLANDI")
    print("=" * 78)
    print("Best epoch:", best_epoch)
    print(
        "Best validation IoU @0.50:",
        f"{best_iou:.4f}",
    )
    print(
        "Best-IoU threshold:",
        f"{float(best_threshold_row['threshold']):.2f}",
    )
    print(
        "SeÃ§ilen gÃ¼venli threshold:",
        f"{selected_threshold:.2f}",
    )
    print(
        "GÃ¼venli threshold metrics:",
        {
            key: round(
                float(safe_threshold_row[key]),
                4,
            )
            for key in (
                "dice",
                "iou",
                "precision",
                "recall",
            )
        },
    )
    print("Checkpoint:", best_path.resolve())
    print("Summary:", summary_path.resolve())
    print("Preview:", contact_sheet.resolve())
    print()
    print("Resmi test spliti kullanÄ±lmadÄ±.")


if __name__ == "__main__":
    main()


