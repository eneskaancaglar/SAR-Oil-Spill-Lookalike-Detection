from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = ROOT / "scripts" / "68_evaluate_dartis_water_transfer_gate.py"

DEFAULT_MANIFEST = (
    ROOT / "data" / "metadata" / "v06_dartis_water_active_training_manifest.csv"
)

DEFAULT_SPLIT_MANIFEST = (
    ROOT / "data" / "metadata" / "v06_dartis_water_locked_split.csv"
)

DEFAULT_SOURCE_CHECKPOINT = (
    ROOT / "checkpoints" / "water_unet_flood_v06_robust" / "best.pth"
)

CHECKPOINT_DIR = ROOT / "checkpoints" / "water_unet_dartis_landaware_cv_v06"
OUTPUT_DIR = ROOT / "outputs" / "water_unet_dartis_landaware_cv_v06"


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"Gerekli 68 scripti bulunamadı: {BASE_SCRIPT}")

    spec = importlib.util.spec_from_file_location(
        "dartis_water_base_v77",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("68 scripti modül olarak yüklenemedi.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kilitli 8 DARTIS test örneğine dokunmadan, kalan 24 geliştirme "
            "örneğinde 4-fold kara-odaklı cross-validation yapar. Kara zengin "
            "ve kıyı sınırı zengin crop örneklemesi ile yanlış su tahminlerini "
            "ağır cezalandırır. OOF gate geçerse 24 geliştirme örneğinin tamamıyla "
            "final modeli eğitir."
        )
    )

    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SOURCE_CHECKPOINT,
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--repeat-factor", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--land-loss-weight", type=float, default=5.0)
    parser.add_argument("--water-loss-weight", type=float, default=1.0)
    parser.add_argument("--tversky-fp-weight", type=float, default=0.80)
    parser.add_argument("--tversky-fn-weight", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260729)

    parser.add_argument("--minimum-oof-precision", type=float, default=0.95)
    parser.add_argument("--minimum-oof-iou", type=float, default=0.50)
    parser.add_argument("--maximum-oof-land-leakage", type=float, default=0.05)
    parser.add_argument(
        "--maximum-per-image-land-leakage",
        type=float,
        default=0.15,
    )
    parser.add_argument("--minimum-per-image-iou", type=float, default=0.35)

    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_percentile(image: np.ndarray) -> np.ndarray:
    array = image.astype(np.float32)
    low = float(np.percentile(array, 2))
    high = float(np.percentile(array, 98))

    if high <= low:
        return (array / 255.0).astype(np.float32)

    return np.clip(
        (array - low) / (high - low),
        0.0,
        1.0,
    ).astype(np.float32)


def load_development_manifest(
    manifest_path: Path,
    split_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Eğitim manifesti bulunamadı: {manifest_path}")

    if not split_manifest_path.exists():
        raise FileNotFoundError(
            f"Kilitli split manifesti bulunamadı: {split_manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    split = pd.read_csv(
        split_manifest_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_manifest = {
        "sample_id",
        "coastal_group",
        "annotation_image_path",
        "manual_mask_path",
        "hard_pass",
        "land_fraction",
    }

    required_split = {
        "sample_id",
        "split",
    }

    missing_manifest = required_manifest - set(manifest.columns)
    missing_split = required_split - set(split.columns)

    if missing_manifest:
        raise RuntimeError(
            f"Eğitim manifestinde eksik sütunlar: {sorted(missing_manifest)}"
        )

    if missing_split:
        raise RuntimeError(
            f"Split manifestinde eksik sütunlar: {sorted(missing_split)}"
        )

    manifest = manifest.copy()
    manifest["hard_pass"] = (
        manifest["hard_pass"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )
    manifest = manifest[manifest["hard_pass"]].copy()

    frame = manifest.merge(
        split[["sample_id", "split"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if len(frame) != 32 or frame["split"].isna().any():
        raise RuntimeError(
            "Manifest ve split birleşimi tam 32 örnek üretmedi."
        )

    locked = frame[frame["split"].eq("locked_test")].copy()
    development = frame[~frame["split"].eq("locked_test")].copy()

    if len(locked) != 8 or len(development) != 24:
        raise RuntimeError(
            f"Beklenen geliştirme/test sayısı 24/8 değil: "
            f"{len(development)}/{len(locked)}"
        )

    for column in ("annotation_image_path", "manual_mask_path"):
        development[f"resolved_{column}"] = development[column].map(
            lambda value: str(resolve_path(value))
        )

        missing = development[
            ~development[f"resolved_{column}"].map(
                lambda value: Path(value).exists()
            )
        ]

        if not missing.empty:
            raise FileNotFoundError(
                f"{column} için eksik dosya: "
                + ", ".join(missing["sample_id"].astype(str).head(5))
            )

    return (
        development.reset_index(drop=True),
        locked.reset_index(drop=True),
    )


def assign_cv_folds(
    development: pd.DataFrame,
    fold_count: int,
    seed: int,
) -> pd.DataFrame:
    if fold_count != 4:
        raise ValueError("Bu aşamada fold sayısı 4 olmalıdır.")

    parts = []

    for group_name in ("nc", "oc"):
        group = development[
            development["coastal_group"].eq(group_name)
        ].copy()

        if len(group) != 12:
            raise RuntimeError(
                f"{group_name} geliştirme grubu 12 örnek olmalı: {len(group)}"
            )

        group = group.sort_values(
            ["land_fraction", "sample_id"],
            ascending=[True, True],
        ).reset_index(drop=True)

        # Kara oranı spektrumunu her fold'a dağıtmak için sıralı round-robin.
        fold_ids = np.array([0, 1, 2, 3] * 3, dtype=np.int64)

        rng = np.random.default_rng(seed + (0 if group_name == "nc" else 1))
        offset = int(rng.integers(0, 4))
        fold_ids = (fold_ids + offset) % 4

        group["cv_fold"] = fold_ids
        parts.append(group)

    frame = pd.concat(parts, ignore_index=True)

    counts = frame.groupby(
        ["coastal_group", "cv_fold"]
    ).size().to_dict()

    expected = {
        (group, fold): 3
        for group in ("nc", "oc")
        for fold in range(4)
    }

    if counts != expected:
        raise RuntimeError(f"CV fold dağılımı geçersiz: {counts}")

    return frame


class LandAwareDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        train: bool,
        crop_size: int,
        repeat_factor: int,
        seed: int,
    ) -> None:
        self.records = frame.to_dict(orient="records")
        self.train = train
        self.crop_size = crop_size
        self.repeat_factor = repeat_factor if train else 1
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records) * self.repeat_factor

    @staticmethod
    def load_pair(
        record: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        image = np.array(
            Image.open(
                record["resolved_annotation_image_path"]
            ).convert("L")
        )

        mask = np.array(
            Image.open(
                record["resolved_manual_mask_path"]
            ).convert("L")
        )

        if image.shape != mask.shape:
            raise RuntimeError(
                f"Boyut uyuşmazlığı: {record['sample_id']} "
                f"{image.shape} != {mask.shape}"
            )

        return image, mask

    def pad_if_needed(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape

        pad_y = max(0, self.crop_size - height)
        pad_x = max(0, self.crop_size - width)

        if pad_y == 0 and pad_x == 0:
            return image, mask

        top = pad_y // 2
        bottom = pad_y - top
        left = pad_x // 2
        right = pad_x - left

        image = np.pad(
            image,
            ((top, bottom), (left, right)),
            mode="reflect",
        )

        mask = np.pad(
            mask,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=128,
        )

        return image, mask

    def choose_center(
        self,
        mask: np.ndarray,
        mode: str,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        if mode == "land":
            candidate = mask == 0
        elif mode == "boundary":
            water = (mask == 255).astype(np.uint8)
            kernel = np.ones((11, 11), dtype=np.uint8)
            candidate = cv2.morphologyEx(
                water,
                cv2.MORPH_GRADIENT,
                kernel,
            ) > 0
            candidate &= mask != 128
        else:
            candidate = mask != 128

        coordinates = np.argwhere(candidate)

        if len(coordinates) == 0:
            coordinates = np.argwhere(mask != 128)

        if len(coordinates) == 0:
            return mask.shape[0] // 2, mask.shape[1] // 2

        index = int(rng.integers(0, len(coordinates)))
        y, x = coordinates[index]
        return int(y), int(x)

    def crop_around(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        center_y: int,
        center_x: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape
        half = self.crop_size // 2

        y0 = int(np.clip(center_y - half, 0, height - self.crop_size))
        x0 = int(np.clip(center_x - half, 0, width - self.crop_size))

        return (
            image[y0:y0 + self.crop_size, x0:x0 + self.crop_size],
            mask[y0:y0 + self.crop_size, x0:x0 + self.crop_size],
        )

    def augment(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        if rng.random() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        rotation = int(rng.integers(0, 4))

        if rotation:
            image = np.rot90(image, rotation)
            mask = np.rot90(mask, rotation)

        normalized = normalize_percentile(image)

        if rng.random() < 0.75:
            gamma = float(rng.uniform(0.85, 1.15))
            normalized = np.power(
                np.clip(normalized, 0.0, 1.0),
                gamma,
            )

        if rng.random() < 0.50:
            contrast = float(rng.uniform(0.90, 1.10))
            brightness = float(rng.uniform(-0.035, 0.035))
            normalized = np.clip(
                normalized * contrast + brightness,
                0.0,
                1.0,
            )

        if rng.random() < 0.30:
            noise_std = float(rng.uniform(0.004, 0.018))
            normalized = np.clip(
                normalized
                + rng.normal(
                    0.0,
                    noise_std,
                    size=normalized.shape,
                ),
                0.0,
                1.0,
            )

        return (
            np.ascontiguousarray(normalized.astype(np.float32)),
            np.ascontiguousarray(mask.astype(np.uint8)),
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        record_index = index % len(self.records)
        record = self.records[record_index]

        image, mask = self.load_pair(record)

        if self.train:
            rng = np.random.default_rng(
                self.seed
                + self.epoch * 1_000_003
                + index * 1009
                + record_index * 7919
            )

            image, mask = self.pad_if_needed(image, mask)

            selector = index % 10

            if selector <= 4:
                mode = "land"
            elif selector <= 7:
                mode = "boundary"
            else:
                mode = "random"

            center_y, center_x = self.choose_center(mask, mode, rng)

            image, mask = self.crop_around(
                image,
                mask,
                center_y,
                center_x,
            )

            image, mask = self.augment(image, mask, rng)
        else:
            image = np.ascontiguousarray(
                normalize_percentile(image).astype(np.float32)
            )
            mask = np.ascontiguousarray(mask.astype(np.uint8))

        tensor = np.stack([image, image], axis=0).astype(np.float32)

        return (
            torch.from_numpy(tensor),
            torch.from_numpy(mask),
            str(record["sample_id"]),
        )


def land_aware_loss(
    logits: torch.Tensor,
    raw_mask: torch.Tensor,
    land_weight: float,
    water_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> torch.Tensor:
    valid = (raw_mask != 128).float()
    target = (raw_mask == 255).float()

    valid_count = valid.sum().clamp_min(1.0)

    bce_map = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )

    class_weight = (
        target * water_weight
        + (1.0 - target) * land_weight
    )

    weighted_bce = (
        bce_map
        * class_weight
        * valid
    ).sum() / valid_count

    probability = torch.sigmoid(logits)

    true_positive = (
        probability * target * valid
    ).sum()

    false_positive = (
        probability * (1.0 - target) * valid
    ).sum()

    false_negative = (
        (1.0 - probability) * target * valid
    ).sum()

    tversky = (
        true_positive + 1.0
    ) / (
        true_positive
        + fp_weight * false_positive
        + fn_weight * false_negative
        + 1.0
    )

    return 0.65 * weighted_bce + 0.35 * (1.0 - tversky)


def load_source_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    base_channels = int(
        checkpoint.get("config", {}).get("base_channels", 16)
    )

    model = BASE.SmallUNet(
        in_channels=2,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model, base_channels


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    model.eval()
    rows = []

    with torch.no_grad():
        for images, masks, sample_ids in loader:
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)[:, 0]

            probabilities = torch.sigmoid(
                logits.float()
            ).cpu().numpy()

            masks_np = masks.numpy()

            for index, sample_id in enumerate(sample_ids):
                rows.append(
                    (
                        str(sample_id),
                        probabilities[index],
                        masks_np[index],
                    )
                )

    return rows


def threshold_metrics(
    predictions: list[tuple[str, np.ndarray, np.ndarray]],
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    per_image = []

    for sample_id, probability, raw_mask in predictions:
        valid = raw_mask != 128
        water = raw_mask == 255
        land = raw_mask == 0
        predicted_water = probability >= threshold

        tp = int((predicted_water & water & valid).sum())
        fp = int((predicted_water & land & valid).sum())
        fn = int(((~predicted_water) & water & valid).sum())
        tn = int(((~predicted_water) & land & valid).sum())

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        precision = tp / (tp + fp) if tp + fp > 0 else 1.0
        recall = tp / (tp + fn) if tp + fn > 0 else 1.0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 1.0
        leakage = fp / (fp + tn) if fp + tn > 0 else 0.0

        per_image.append(
            {
                "sample_id": sample_id,
                "precision": float(precision),
                "recall": float(recall),
                "iou": float(iou),
                "land_leakage": float(leakage),
            }
        )

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 1.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 1.0
    )

    iou = (
        total_tp / (total_tp + total_fp + total_fn)
        if total_tp + total_fp + total_fn > 0
        else 1.0
    )

    dice = (
        2.0 * total_tp
        / (2.0 * total_tp + total_fp + total_fn)
        if 2.0 * total_tp + total_fp + total_fn > 0
        else 1.0
    )

    leakage = (
        total_fp / (total_fp + total_tn)
        if total_fp + total_tn > 0
        else 0.0
    )

    metrics = {
        "threshold": float(threshold),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "land_leakage": float(leakage),
        "minimum_per_image_iou": float(
            min(row["iou"] for row in per_image)
        ),
        "minimum_per_image_precision": float(
            min(row["precision"] for row in per_image)
        ),
        "minimum_per_image_recall": float(
            min(row["recall"] for row in per_image)
        ),
        "maximum_per_image_land_leakage": float(
            max(row["land_leakage"] for row in per_image)
        ),
    }

    return metrics, per_image


def choose_threshold(
    predictions: list[tuple[str, np.ndarray, np.ndarray]],
    minimum_precision: float,
    minimum_iou: float,
    maximum_leakage: float,
    maximum_per_image_leakage: float,
    minimum_per_image_iou: float,
) -> tuple[dict[str, float], list[dict[str, float | str]], bool]:
    candidates = []

    for threshold in np.arange(0.30, 0.951, 0.05):
        metrics, per_image = threshold_metrics(
            predictions,
            float(threshold),
        )

        safe = (
            metrics["precision"] >= minimum_precision
            and metrics["iou"] >= minimum_iou
            and metrics["land_leakage"] <= maximum_leakage
            and metrics["maximum_per_image_land_leakage"]
            <= maximum_per_image_leakage
            and metrics["minimum_per_image_iou"]
            >= minimum_per_image_iou
        )

        score = (
            (2.0 if safe else 0.0)
            + metrics["iou"]
            + 0.25 * metrics["precision"]
            + 0.15 * metrics["recall"]
            - 1.50 * metrics["land_leakage"]
            - 0.40 * metrics["maximum_per_image_land_leakage"]
        )

        candidates.append(
            (metrics, per_image, safe, score)
        )

    selected = max(
        candidates,
        key=lambda item: item[3],
    )

    return selected[0], selected[1], selected[2]


def train_fold(
    fold_index: int,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    Path,
    int,
    dict[str, float],
    bool,
    list[tuple[str, np.ndarray, np.ndarray]],
    list[dict[str, Any]],
]:
    model, base_channels = load_source_model(
        resolve_path(args.source_checkpoint),
        device,
    )

    train_dataset = LandAwareDataset(
        train_frame,
        train=True,
        crop_size=args.crop_size,
        repeat_factor=args.repeat_factor,
        seed=args.seed + fold_index * 100_000,
    )

    validation_dataset = LandAwareDataset(
        validation_frame,
        train=False,
        crop_size=args.crop_size,
        repeat_factor=1,
        seed=args.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed + fold_index)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    best_score = -math.inf
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_safe = False
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()

        total_loss = 0.0
        batches = 0

        for images, masks, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)[:, 0]

                loss = land_aware_loss(
                    logits,
                    masks,
                    land_weight=args.land_loss_weight,
                    water_weight=args.water_loss_weight,
                    fp_weight=args.tversky_fp_weight,
                    fn_weight=args.tversky_fn_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().item())
            batches += 1

        predictions = collect_predictions(
            model,
            validation_loader,
            device,
        )

        metrics, _, safe = choose_threshold(
            predictions,
            minimum_precision=args.minimum_oof_precision,
            minimum_iou=args.minimum_oof_iou,
            maximum_leakage=args.maximum_oof_land_leakage,
            maximum_per_image_leakage=args.maximum_per_image_land_leakage,
            minimum_per_image_iou=args.minimum_per_image_iou,
        )

        score = (
            (2.0 if safe else 0.0)
            + metrics["iou"]
            + 0.25 * metrics["precision"]
            + 0.15 * metrics["recall"]
            - 1.50 * metrics["land_leakage"]
            - 0.40 * metrics["maximum_per_image_land_leakage"]
        )

        row = {
            "fold": fold_index + 1,
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "safe": safe,
            "selection_score": score,
            **metrics,
        }
        history.append(row)

        print(
            f"Fold {fold_index + 1}/4 Epoch {epoch:02d}/{args.epochs:02d} "
            f"| Loss {row['train_loss']:.4f} "
            f"| IoU {metrics['iou']:.4f} "
            f"P {metrics['precision']:.4f} "
            f"R {metrics['recall']:.4f} "
            f"Leak {metrics['land_leakage']:.4f} "
            f"| thr {metrics['threshold']:.2f} "
            f"| safe {safe}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(metrics)
            best_safe = bool(safe)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"Fold {fold_index + 1}: early stopping "
                f"({args.patience} epoch iyileşme yok)."
            )
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError(f"Fold {fold_index + 1} checkpoint üretilemedi.")

    model.load_state_dict(best_state)

    held_out_predictions = collect_predictions(
        model,
        validation_loader,
        device,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        CHECKPOINT_DIR / f"fold_{fold_index + 1:02d}_best.pth"
    )

    torch.save(
        {
            "stage": "v06_dartis_landaware_cv_fold",
            "fold": fold_index + 1,
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
            "validation_safe": best_safe,
            "config": {
                "base_channels": base_channels,
                "in_channels": 2,
                "crop_size": args.crop_size,
                "land_loss_weight": args.land_loss_weight,
                "water_loss_weight": args.water_loss_weight,
                "tversky_fp_weight": args.tversky_fp_weight,
                "tversky_fn_weight": args.tversky_fn_weight,
                "seed": args.seed,
            },
            "locked_test_used": False,
        },
        checkpoint_path,
    )

    return (
        checkpoint_path,
        best_epoch,
        best_metrics,
        best_safe,
        held_out_predictions,
        history,
    )


def train_final_model(
    development: pd.DataFrame,
    epoch_count: int,
    selected_threshold: float,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    model, base_channels = load_source_model(
        resolve_path(args.source_checkpoint),
        device,
    )

    dataset = LandAwareDataset(
        development,
        train=True,
        crop_size=args.crop_size,
        repeat_factor=args.repeat_factor,
        seed=args.seed + 900_000,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed + 900_000)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    for epoch in range(1, epoch_count + 1):
        dataset.set_epoch(epoch)
        model.train()

        total_loss = 0.0
        batches = 0

        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)[:, 0]

                loss = land_aware_loss(
                    logits,
                    masks,
                    land_weight=args.land_loss_weight,
                    water_weight=args.water_loss_weight,
                    fp_weight=args.tversky_fp_weight,
                    fn_weight=args.tversky_fn_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().item())
            batches += 1

        print(
            f"Final Epoch {epoch:02d}/{epoch_count:02d} "
            f"| Loss {total_loss / max(batches, 1):.4f}"
        )

    checkpoint_path = CHECKPOINT_DIR / "final_best.pth"

    torch.save(
        {
            "stage": "v06_dartis_landaware_final",
            "model_state_dict": model.state_dict(),
            "selected_threshold": float(selected_threshold),
            "trained_epochs": int(epoch_count),
            "source_checkpoint": relative(
                resolve_path(args.source_checkpoint)
            ),
            "config": {
                "base_channels": base_channels,
                "in_channels": 2,
                "crop_size": args.crop_size,
                "land_loss_weight": args.land_loss_weight,
                "water_loss_weight": args.water_loss_weight,
                "tversky_fp_weight": args.tversky_fp_weight,
                "tversky_fn_weight": args.tversky_fn_weight,
                "seed": args.seed,
            },
            "locked_test_used": False,
        },
        checkpoint_path,
    )

    return checkpoint_path


def sample_hash(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        sorted(frame["sample_id"].astype(str))
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    source_checkpoint = resolve_path(args.source_checkpoint)

    if not source_checkpoint.exists():
        raise FileNotFoundError(
            f"Kaynak checkpoint bulunamadı: {source_checkpoint}"
        )

    development, locked_test = load_development_manifest(
        resolve_path(args.manifest),
        resolve_path(args.split_manifest),
    )

    cv_frame = assign_cv_folds(
        development,
        args.folds,
        args.seed,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cv_manifest_path = OUTPUT_DIR / "development_cv_folds.csv"

    cv_frame[
        [
            "sample_id",
            "coastal_group",
            "land_fraction",
            "water_fraction",
            "ignore_fraction",
            "cv_fold",
        ]
    ].to_csv(
        cv_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 78)
    print("v0.6 DARTIS LAND-AWARE 4-FOLD DEVELOPMENT CV")
    print("=" * 78)
    print("Development:", len(development))
    print("Locked test:", len(locked_test))
    print("Locked test görüntü ve maskeleri açılmayacak.")
    print("Cihaz:", device)
    print(
        "Crop dağılımı: %50 kara | %30 kıyı sınırı | %20 rastgele"
    )
    print(
        "Kara FP loss ağırlığı:",
        args.land_loss_weight,
    )
    print()

    start = time.time()

    all_oof_predictions: list[
        tuple[str, np.ndarray, np.ndarray]
    ] = []

    fold_summaries = []
    all_history = []
    best_epochs = []

    for fold_index in range(args.folds):
        train_frame = cv_frame[
            ~cv_frame["cv_fold"].eq(fold_index)
        ].copy()

        validation_frame = cv_frame[
            cv_frame["cv_fold"].eq(fold_index)
        ].copy()

        print()
        print(
            f"--- Fold {fold_index + 1}/4 "
            f"| Train {len(train_frame)} "
            f"| Validation {len(validation_frame)} ---"
        )

        (
            checkpoint_path,
            best_epoch,
            best_metrics,
            best_safe,
            held_out_predictions,
            history,
        ) = train_fold(
            fold_index,
            train_frame,
            validation_frame,
            args,
            device,
        )

        best_epochs.append(best_epoch)
        all_oof_predictions.extend(
            held_out_predictions
        )
        all_history.extend(history)

        fold_summaries.append(
            {
                "fold": fold_index + 1,
                "train_count": len(train_frame),
                "validation_count": len(validation_frame),
                "best_epoch": best_epoch,
                "best_safe": best_safe,
                "best_metrics": best_metrics,
                "checkpoint": relative(checkpoint_path),
            }
        )

    oof_metrics, oof_per_image, oof_gate = choose_threshold(
        all_oof_predictions,
        minimum_precision=args.minimum_oof_precision,
        minimum_iou=args.minimum_oof_iou,
        maximum_leakage=args.maximum_oof_land_leakage,
        maximum_per_image_leakage=args.maximum_per_image_land_leakage,
        minimum_per_image_iou=args.minimum_per_image_iou,
    )

    pd.DataFrame(all_history).to_csv(
        OUTPUT_DIR / "cv_training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(oof_per_image).to_csv(
        OUTPUT_DIR / "oof_per_image_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_checkpoint = None
    final_epochs = int(
        max(
            1,
            round(float(np.median(best_epochs))),
        )
    )

    if oof_gate:
        print()
        print(
            "OOF gate PASS. Final model 24 development örneğiyle eğitiliyor."
        )

        final_checkpoint = train_final_model(
            development=cv_frame,
            epoch_count=final_epochs,
            selected_threshold=oof_metrics["threshold"],
            args=args,
            device=device,
        )
    else:
        print()
        print(
            "OOF gate FAIL. Locked test açılmadı ve final model üretilmedi."
        )

    summary = {
        "stage": "v06_dartis_landaware_cv",
        "development_count": int(len(development)),
        "locked_test_count": int(len(locked_test)),
        "locked_test_sample_hash": sample_hash(locked_test),
        "locked_test_used": False,
        "source_checkpoint": relative(source_checkpoint),
        "fold_count": int(args.folds),
        "fold_summaries": fold_summaries,
        "oof_gate_passed": bool(oof_gate),
        "oof_metrics": oof_metrics,
        "criteria": {
            "minimum_oof_precision": args.minimum_oof_precision,
            "minimum_oof_iou": args.minimum_oof_iou,
            "maximum_oof_land_leakage": args.maximum_oof_land_leakage,
            "maximum_per_image_land_leakage": (
                args.maximum_per_image_land_leakage
            ),
            "minimum_per_image_iou": args.minimum_per_image_iou,
        },
        "final_training_epochs": (
            final_epochs if oof_gate else 0
        ),
        "final_checkpoint": (
            relative(final_checkpoint)
            if final_checkpoint is not None
            else None
        ),
        "selected_threshold": (
            oof_metrics["threshold"] if oof_gate else None
        ),
        "cv_manifest": relative(cv_manifest_path),
        "cv_training_history": relative(
            OUTPUT_DIR / "cv_training_history.csv"
        ),
        "oof_per_image_metrics": relative(
            OUTPUT_DIR / "oof_per_image_metrics.csv"
        ),
        "training_seconds": float(time.time() - start),
        "official_sen1floods11_test_used": False,
        "scientific_note": (
            "Sekiz kilitli DARTIS test örneğinin görüntü ve maskeleri bu "
            "aşamada açılmamıştır. Geliştirme kararı yalnız kalan 24 örneğin "
            "4-fold out-of-fold tahminlerinden verilmiştir."
        ),
    }

    summary_path = OUTPUT_DIR / "summary.json"

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
    print("DARTIS LAND-AWARE CV SONUCU")
    print("=" * 78)
    print("OOF gate:", "PASS" if oof_gate else "FAIL")
    print(
        "OOF P:",
        f"{oof_metrics['precision']:.4f}",
        "| R:",
        f"{oof_metrics['recall']:.4f}",
        "| IoU:",
        f"{oof_metrics['iou']:.4f}",
        "| Leak:",
        f"{oof_metrics['land_leakage']:.4f}",
    )
    print(
        "Worst leakage:",
        f"{oof_metrics['maximum_per_image_land_leakage']:.4f}",
        "| Min image IoU:",
        f"{oof_metrics['minimum_per_image_iou']:.4f}",
        "| thr:",
        f"{oof_metrics['threshold']:.2f}",
    )
    print("Locked test kullanıldı mı: False")
    print("Final checkpoint:", final_checkpoint)
    print("Özet:", summary_path.resolve())


if __name__ == "__main__":
    main()
