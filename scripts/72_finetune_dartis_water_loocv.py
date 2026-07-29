from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from copy import deepcopy
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

BASE_SCRIPT = (
    ROOT
    / "scripts"
    / "68_evaluate_dartis_water_transfer_gate.py"
)

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
    / "v06_dartis_water_loocv_adaptation"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "water_unet_dartis_loocv_v06"
)

THRESHOLDS = [
    round(value, 2)
    for value in np.arange(
        0.30,
        0.701,
        0.05,
    )
]

POSTPROCESSING_OPTIONS = (
    "none",
    "conservative",
)


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Gerekli 68 scripti bulunamadı: {BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "dartis_transfer_base_v72",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "68 scripti modül olarak yüklenemedi."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "7 manuel DARTIS kara-su örneğinde hedef-domain "
            "adaptasyonunu leave-one-out çapraz doğrulamayla ölçer. "
            "Her fold: 5 train, 1 validation, 1 bağımsız test."
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
        "--epochs",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--repeat-factor",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=512,
        help=(
            "Eğitim batchlerinde bütün görüntü ve maskelerin "
            "getirileceği ortak kare boyut."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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
        "--minimum-water-precision",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--minimum-water-iou",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--maximum-land-leakage",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--maximum-per-image-land-leakage",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--minimum-per-image-water-recall",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--minimum-per-image-water-iou",
        type=float,
        default=0.35,
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
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resize_image_to_mask(
    image: np.ndarray,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    if image.shape == mask_shape:
        return image

    target_height, target_width = mask_shape

    interpolation = (
        cv2.INTER_AREA
        if (
            image.shape[0] > target_height
            or image.shape[1] > target_width
        )
        else cv2.INTER_CUBIC
    )

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=interpolation,
    )


def prepare_records(
    dartis_manifest_path: Path,
    manual_manifest_path: Path,
) -> list[dict[str, Any]]:
    merged = BASE.load_calibration_records(
        dartis_manifest_path,
        manual_manifest_path,
    )

    records: list[dict[str, Any]] = []

    for row in merged.to_dict(
        orient="records"
    ):
        sample_id = str(row["sample_id"])

        image = BASE.read_grayscale(
            resolve_path(
                row["image_path"]
            )
        )

        land = (
            BASE.read_grayscale(
                resolve_path(
                    row["land_mask_path"]
                )
            )
            > 127
        )

        safe_water = (
            BASE.read_grayscale(
                resolve_path(
                    row[
                        "safe_water_mask_path"
                    ]
                )
            )
            > 127
        )

        if land.shape != safe_water.shape:
            raise RuntimeError(
                "Manuel kara ve su maskeleri farklı boyutta: "
                f"{sample_id}"
            )

        original_shape = image.shape

        image = resize_image_to_mask(
            image,
            land.shape,
        )

        if original_shape != image.shape:
            print(
                "Yeniden boyutlandırıldı:",
                sample_id,
                f"{original_shape} -> {image.shape}",
            )

        if (land & safe_water).any():
            raise RuntimeError(
                "Manuel kara ve su maskeleri çakışıyor: "
                f"{sample_id}"
            )

        valid = land | safe_water

        if not valid.any():
            raise RuntimeError(
                "Geçerli piksel bulunamadı: "
                f"{sample_id}"
            )

        label = np.full(
            land.shape,
            255,
            dtype=np.uint8,
        )

        label[land] = 0
        label[safe_water] = 1

        records.append(
            {
                "sample_id": sample_id,
                "image": image,
                "label": label,
                "land": land,
                "safe_water": safe_water,
                "valid": valid,
            }
        )

    return sorted(
        records,
        key=lambda item: item["sample_id"],
    )


def percentile_normalize(
    image: np.ndarray,
) -> np.ndarray:
    array = image.astype(np.float32)

    low = float(
        np.percentile(array, 2)
    )
    high = float(
        np.percentile(array, 98)
    )

    if high <= low:
        return (
            array / 255.0
        ).astype(np.float32)

    return np.clip(
        (array - low)
        / (high - low),
        0.0,
        1.0,
    ).astype(np.float32)


def augment_pair(
    image: np.ndarray,
    label: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = image.copy()
    label = label.copy()

    rotation_count = random.randrange(4)

    if rotation_count:
        image = np.rot90(
            image,
            rotation_count,
        )
        label = np.rot90(
            label,
            rotation_count,
        )

    if random.random() < 0.5:
        image = np.fliplr(image)
        label = np.fliplr(label)

    if random.random() < 0.5:
        image = np.flipud(image)
        label = np.flipud(label)

    gamma = random.uniform(
        0.80,
        1.20,
    )

    image = np.power(
        np.clip(image, 0.0, 1.0),
        gamma,
    )

    contrast = random.uniform(
        0.85,
        1.15,
    )
    brightness = random.uniform(
        -0.08,
        0.08,
    )

    image = np.clip(
        image * contrast + brightness,
        0.0,
        1.0,
    )

    if random.random() < 0.5:
        noise = np.random.normal(
            0.0,
            random.uniform(
                0.0,
                0.025,
            ),
            size=image.shape,
        ).astype(np.float32)

        image = np.clip(
            image + noise,
            0.0,
            1.0,
        )

    return (
        np.ascontiguousarray(
            image,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            label,
            dtype=np.int64,
        ),
    )


class DartisAdaptationDataset(
    Dataset
):
    def __init__(
        self,
        records: list[
            dict[str, Any]
        ],
        training: bool,
        repeat_factor: int,
        target_size: int,
    ) -> None:
        self.records = records
        self.training = training
        self.target_size = max(
            int(target_size),
            64,
        )
        self.repeat_factor = (
            max(
                repeat_factor,
                1,
            )
            if training
            else 1
        )

    def __len__(self) -> int:
        return (
            len(self.records)
            * self.repeat_factor
        )

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

        image = percentile_normalize(
            record["image"]
        )
        label = record["label"].copy()

        target_shape = (
            self.target_size,
            self.target_size,
        )

        if image.shape != target_shape:
            image_interpolation = (
                cv2.INTER_AREA
                if (
                    image.shape[0] > self.target_size
                    or image.shape[1] > self.target_size
                )
                else cv2.INTER_CUBIC
            )

            image = cv2.resize(
                image,
                (
                    self.target_size,
                    self.target_size,
                ),
                interpolation=image_interpolation,
            )

            label = cv2.resize(
                label,
                (
                    self.target_size,
                    self.target_size,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.training:
            image, label = augment_pair(
                image,
                label,
            )

        two_channel = np.stack(
            [
                image,
                image,
            ],
            axis=0,
        ).astype(np.float32)

        return (
            torch.from_numpy(
                np.ascontiguousarray(
                    two_channel
                )
            ),
            torch.from_numpy(
                np.ascontiguousarray(
                    label,
                    dtype=np.int64,
                )
            ),
            str(record["sample_id"]),
        )


def count_pixels(
    records: list[
        dict[str, Any]
    ],
) -> tuple[int, int]:
    water = 0
    land = 0

    for record in records:
        label = record["label"]

        water += int(
            (label == 1).sum()
        )

        land += int(
            (label == 0).sum()
        )

    return water, land


def masked_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    logits = logits.squeeze(1).float()
    valid = target != 255

    if int(valid.sum().item()) == 0:
        raise RuntimeError(
            "Batchte geçerli piksel yok."
        )

    truth = (
        target == 1
    ).float()

    bce = F.binary_cross_entropy_with_logits(
        logits[valid],
        truth[valid],
        pos_weight=positive_weight,
    )

    probability = torch.sigmoid(
        logits
    )
    valid_float = valid.float()

    intersection = (
        probability
        * truth
        * valid_float
    ).sum()

    denominator = (
        probability
        * valid_float
    ).sum() + (
        truth
        * valid_float
    ).sum()

    dice_loss = 1.0 - (
        (2.0 * intersection + 1.0)
        / (denominator + 1.0)
    )

    loss = bce + dice_loss

    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Loss sonlu değil."
        )

    return loss


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    positive_weight: torch.Tensor,
    amp_enabled: bool,
) -> float:
    model.train()
    losses: list[float] = []

    for images, labels, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(images)

        loss = masked_loss(
            logits,
            labels,
            positive_weight,
        )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=3.0,
        )

        scaler.step(optimizer)
        scaler.update()

        losses.append(
            float(
                loss.detach().item()
            )
        )

    return float(
        np.mean(losses)
    )


def infer_probability(
    model: torch.nn.Module,
    device: torch.device,
    record: dict[str, Any],
) -> np.ndarray:
    image = percentile_normalize(
        record["image"]
    )

    input_array = np.stack(
        [
            image,
            image,
        ],
        axis=0,
    ).astype(np.float32)

    tensor = torch.from_numpy(
        input_array
    ).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(tensor)

        probability = torch.sigmoid(
            logits[0, 0].float()
        ).cpu().numpy()

    return probability


def postprocess_prediction(
    prediction: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return prediction.astype(bool)

    if mode == "conservative":
        return BASE.apply_postprocessing(
            prediction=prediction,
            mode="conservative",
            coast_buffer_pixels=2,
            minimum_component_pixels=64,
        )

    raise ValueError(
        f"Bilinmeyen postprocess: {mode}"
    )


def evaluate_record(
    probability: np.ndarray,
    record: dict[str, Any],
    threshold: float,
    postprocessing: str,
) -> dict[str, float]:
    prediction = (
        probability >= threshold
    )

    prediction = postprocess_prediction(
        prediction,
        postprocessing,
    )

    counts = BASE.empty_counts()

    BASE.update_counts(
        counts=counts,
        prediction=prediction,
        truth_water=record[
            "safe_water"
        ],
        valid=record["valid"],
    )

    return BASE.finalize_counts(
        counts
    )


def choose_validation_configuration(
    probability: np.ndarray,
    validation_record: dict[
        str,
        Any,
    ],
) -> tuple[
    float,
    str,
    dict[str, float],
]:
    candidates: list[
        dict[str, Any]
    ] = []

    for threshold in THRESHOLDS:
        for postprocessing in (
            POSTPROCESSING_OPTIONS
        ):
            metrics = evaluate_record(
                probability=probability,
                record=validation_record,
                threshold=threshold,
                postprocessing=(
                    postprocessing
                ),
            )

            candidates.append(
                {
                    "threshold": threshold,
                    "postprocessing": (
                        postprocessing
                    ),
                    **metrics,
                }
            )

    frame = pd.DataFrame(
        candidates
    )

    safe = frame[
        frame[
            "water_precision"
        ].ge(0.90)
    ]

    if safe.empty:
        selected = (
            frame.sort_values(
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
    else:
        selected = (
            safe.sort_values(
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

    return (
        float(selected["threshold"]),
        str(
            selected[
                "postprocessing"
            ]
        ),
        {
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
    )


def save_fold_preview(
    record: dict[str, Any],
    probability: np.ndarray,
    threshold: float,
    postprocessing: str,
    output_path: Path,
) -> None:
    prediction = (
        probability >= threshold
    )

    prediction = postprocess_prediction(
        prediction,
        postprocessing,
    )

    BASE.save_preview(
        image=record["image"],
        truth_water=record[
            "safe_water"
        ],
        valid=record["valid"],
        probability=probability,
        prediction=prediction,
        output_path=output_path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    base_channels = int(
        source_checkpoint.get(
            "config",
            {},
        ).get(
            "base_channels",
            16,
        )
    )

    records = prepare_records(
        dartis_manifest_path,
        manual_manifest_path,
    )

    if len(records) < 7:
        raise RuntimeError(
            "Leave-one-out için en az 7 manuel örnek bekleniyor."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = (
        device.type == "cuda"
    )

    print("=" * 78)
    print(
        "v0.6 DARTIS WATER LOOCV ADAPTATION"
    )
    print("=" * 78)
    print(
        "Manuel örnek:",
        len(records),
    )
    print(
        "Her fold: 5 train, 1 validation, 1 test"
    )
    print(
        "Cihaz:",
        device,
    )
    print(
        "Epoch/fold:",
        args.epochs,
    )
    print(
        "Ortak train boyutu:",
        f"{args.train_size}x{args.train_size}",
    )
    print()

    fold_rows: list[
        dict[str, Any]
    ] = []
    preview_paths: list[Path] = []

    total_start = time.time()

    for test_index in range(
        len(records)
    ):
        validation_index = (
            test_index + 1
        ) % len(records)

        test_record = records[
            test_index
        ]
        validation_record = records[
            validation_index
        ]

        train_records = [
            record
            for index, record
            in enumerate(records)
            if index
            not in (
                test_index,
                validation_index,
            )
        ]

        fold_number = (
            test_index + 1
        )

        fold_seed = (
            args.seed
            + fold_number
        )

        set_seed(fold_seed)

        model = BASE.SmallUNet(
            in_channels=2,
            base_channels=base_channels,
        ).to(device)

        model.load_state_dict(
            source_checkpoint[
                "model_state_dict"
            ]
        )

        train_water, train_land = (
            count_pixels(
                train_records
            )
        )

        raw_ratio = (
            train_land
            / max(
                train_water,
                1,
            )
        )

        positive_weight_value = float(
            np.clip(
                math.sqrt(
                    raw_ratio
                ),
                1.0,
                args.max_pos_weight,
            )
        )

        positive_weight = torch.tensor(
            positive_weight_value,
            dtype=torch.float32,
            device=device,
        )

        train_dataset = (
            DartisAdaptationDataset(
                records=train_records,
                training=True,
                repeat_factor=(
                    args.repeat_factor
                ),
                target_size=(
                    args.train_size
                ),
            )
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
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=(
                args.weight_decay
            ),
        )

        scaler = (
            torch.amp.GradScaler(
                device.type,
                enabled=amp_enabled,
            )
        )

        best_validation_iou = -1.0
        best_validation_precision = -1.0
        best_epoch = 0
        best_state = None
        best_threshold = 0.5
        best_postprocessing = "none"
        best_validation_metrics: dict[
            str,
            float,
        ] = {}

        fold_start = time.time()

        for epoch in range(
            1,
            args.epochs + 1,
        ):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                positive_weight=(
                    positive_weight
                ),
                amp_enabled=(
                    amp_enabled
                ),
            )

            validation_probability = (
                infer_probability(
                    model=model,
                    device=device,
                    record=(
                        validation_record
                    ),
                )
            )

            (
                threshold,
                postprocessing,
                validation_metrics,
            ) = (
                choose_validation_configuration(
                    probability=(
                        validation_probability
                    ),
                    validation_record=(
                        validation_record
                    ),
                )
            )

            improved = bool(
                validation_metrics[
                    "iou"
                ]
                > best_validation_iou
                or (
                    math.isclose(
                        validation_metrics[
                            "iou"
                        ],
                        best_validation_iou,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                    and validation_metrics[
                        "water_precision"
                    ]
                    > best_validation_precision
                )
            )

            if improved:
                best_validation_iou = (
                    validation_metrics[
                        "iou"
                    ]
                )
                best_validation_precision = (
                    validation_metrics[
                        "water_precision"
                    ]
                )
                best_epoch = epoch
                best_state = deepcopy(
                    model.state_dict()
                )
                best_threshold = (
                    threshold
                )
                best_postprocessing = (
                    postprocessing
                )
                best_validation_metrics = (
                    validation_metrics
                )

            print(
                f"Fold {fold_number}/"
                f"{len(records)} "
                f"Epoch {epoch:02d}/"
                f"{args.epochs:02d} | "
                f"Loss {train_loss:.4f} | "
                f"Val IoU "
                f"{validation_metrics['iou']:.4f} "
                f"P "
                f"{validation_metrics['water_precision']:.4f} "
                f"R "
                f"{validation_metrics['water_recall']:.4f} | "
                f"thr {threshold:.2f} "
                f"{postprocessing}"
            )

        if best_state is None:
            raise RuntimeError(
                f"Fold {fold_number}: en iyi model seçilemedi."
            )

        model.load_state_dict(
            best_state
        )

        fold_checkpoint_path = (
            CHECKPOINT_DIR
            / (
                f"fold_{fold_number:02d}"
                f"__test_"
                f"{test_record['sample_id'].replace(':', '__')}"
                f".pth"
            )
        )

        torch.save(
            {
                "fold": fold_number,
                "test_sample_id": (
                    test_record[
                        "sample_id"
                    ]
                ),
                "validation_sample_id": (
                    validation_record[
                        "sample_id"
                    ]
                ),
                "train_sample_ids": [
                    record["sample_id"]
                    for record
                    in train_records
                ],
                "best_epoch": (
                    best_epoch
                ),
                "selected_threshold": (
                    best_threshold
                ),
                "selected_postprocessing": (
                    best_postprocessing
                ),
                "validation_metrics": (
                    best_validation_metrics
                ),
                "model_state_dict": (
                    best_state
                ),
                "source_checkpoint": (
                    relative(
                        checkpoint_path
                    )
                ),
            },
            fold_checkpoint_path,
        )

        test_probability = (
            infer_probability(
                model=model,
                device=device,
                record=test_record,
            )
        )

        test_metrics = evaluate_record(
            probability=test_probability,
            record=test_record,
            threshold=best_threshold,
            postprocessing=(
                best_postprocessing
            ),
        )

        preview_path = (
            OUTPUT_DIR
            / "fold_previews"
            / (
                f"fold_{fold_number:02d}"
                f"__{test_record['sample_id'].replace(':', '__')}"
                f".png"
            )
        )

        save_fold_preview(
            record=test_record,
            probability=test_probability,
            threshold=best_threshold,
            postprocessing=(
                best_postprocessing
            ),
            output_path=preview_path,
        )

        preview_paths.append(
            preview_path
        )

        fold_rows.append(
            {
                "fold": fold_number,
                "test_sample_id": (
                    test_record[
                        "sample_id"
                    ]
                ),
                "validation_sample_id": (
                    validation_record[
                        "sample_id"
                    ]
                ),
                "train_sample_ids": (
                    "|".join(
                        record["sample_id"]
                        for record
                        in train_records
                    )
                ),
                "best_epoch": best_epoch,
                "selected_threshold": (
                    best_threshold
                ),
                "selected_postprocessing": (
                    best_postprocessing
                ),
                "positive_weight": (
                    positive_weight_value
                ),
                "validation_iou": (
                    best_validation_metrics[
                        "iou"
                    ]
                ),
                "validation_precision": (
                    best_validation_metrics[
                        "water_precision"
                    ]
                ),
                "validation_recall": (
                    best_validation_metrics[
                        "water_recall"
                    ]
                ),
                "test_dice": (
                    test_metrics["dice"]
                ),
                "test_iou": (
                    test_metrics["iou"]
                ),
                "test_water_precision": (
                    test_metrics[
                        "water_precision"
                    ]
                ),
                "test_water_recall": (
                    test_metrics[
                        "water_recall"
                    ]
                ),
                "test_accuracy": (
                    test_metrics[
                        "accuracy"
                    ]
                ),
                "test_land_leakage_rate": (
                    test_metrics[
                        "land_leakage_rate"
                    ]
                ),
                "fold_seconds": (
                    time.time()
                    - fold_start
                ),
                "checkpoint_path": (
                    relative(
                        fold_checkpoint_path
                    )
                ),
            }
        )

        print(
            f"Fold {fold_number} TEST | "
            f"IoU {test_metrics['iou']:.4f} "
            f"P {test_metrics['water_precision']:.4f} "
            f"R {test_metrics['water_recall']:.4f} "
            f"Leak "
            f"{test_metrics['land_leakage_rate']:.4f}"
        )
        print()

    folds = pd.DataFrame(
        fold_rows
    )

    fold_results_path = (
        OUTPUT_DIR
        / "loocv_fold_results.csv"
    )

    folds.to_csv(
        fold_results_path,
        index=False,
        encoding="utf-8-sig",
    )

    contact_sheet_path = (
        OUTPUT_DIR
        / "loocv_test_contact_sheet.jpg"
    )

    BASE.create_contact_sheet(
        preview_paths,
        contact_sheet_path,
    )

    aggregate_precision = float(
        folds[
            "test_water_precision"
        ].mean()
    )
    aggregate_recall = float(
        folds[
            "test_water_recall"
        ].mean()
    )
    aggregate_iou = float(
        folds[
            "test_iou"
        ].mean()
    )
    aggregate_dice = float(
        folds[
            "test_dice"
        ].mean()
    )
    aggregate_leakage = float(
        folds[
            "test_land_leakage_rate"
        ].mean()
    )

    worst_leakage = float(
        folds[
            "test_land_leakage_rate"
        ].max()
    )
    minimum_recall = float(
        folds[
            "test_water_recall"
        ].min()
    )
    minimum_iou = float(
        folds[
            "test_iou"
        ].min()
    )

    leakage_violations = int(
        (
            folds[
                "test_land_leakage_rate"
            ]
            > args.maximum_per_image_land_leakage
        ).sum()
    )
    recall_violations = int(
        (
            folds[
                "test_water_recall"
            ]
            < args.minimum_per_image_water_recall
        ).sum()
    )
    iou_violations = int(
        (
            folds[
                "test_iou"
            ]
            < args.minimum_per_image_water_iou
        ).sum()
    )

    gate_passed = bool(
        aggregate_precision
        >= args.minimum_water_precision
        and aggregate_iou
        >= args.minimum_water_iou
        and aggregate_leakage
        <= args.maximum_land_leakage
        and leakage_violations == 0
        and recall_violations == 0
        and iou_violations == 0
    )

    failed_reasons: list[str] = []

    if (
        aggregate_precision
        < args.minimum_water_precision
    ):
        failed_reasons.append(
            "mean_precision"
        )

    if (
        aggregate_iou
        < args.minimum_water_iou
    ):
        failed_reasons.append(
            "mean_iou"
        )

    if (
        aggregate_leakage
        > args.maximum_land_leakage
    ):
        failed_reasons.append(
            "mean_land_leakage"
        )

    if leakage_violations:
        failed_reasons.append(
            "per_image_land_leakage"
        )

    if recall_violations:
        failed_reasons.append(
            "per_image_water_recall"
        )

    if iou_violations:
        failed_reasons.append(
            "per_image_water_iou"
        )

    summary = {
        "stage": (
            "v06_dartis_water_loocv_adaptation"
        ),
        "gate_passed": gate_passed,
        "failed_reasons": (
            failed_reasons
        ),
        "method": (
            "7-fold leave-one-out: "
            "5 target-domain train, "
            "1 target-domain validation, "
            "1 independent target-domain test"
        ),
        "sample_count": int(
            len(records)
        ),
        "training_config": {
            "epochs": int(args.epochs),
            "repeat_factor": int(args.repeat_factor),
            "batch_size": int(args.batch_size),
            "train_size": int(args.train_size),
            "learning_rate": float(args.learning_rate),
        },
        "criteria": {
            "minimum_mean_water_precision": float(
                args.minimum_water_precision
            ),
            "minimum_mean_water_iou": float(
                args.minimum_water_iou
            ),
            "maximum_mean_land_leakage": float(
                args.maximum_land_leakage
            ),
            "maximum_per_image_land_leakage": float(
                args.maximum_per_image_land_leakage
            ),
            "minimum_per_image_water_recall": float(
                args.minimum_per_image_water_recall
            ),
            "minimum_per_image_water_iou": float(
                args.minimum_per_image_water_iou
            ),
        },
        "held_out_test_metrics": {
            "mean_dice": (
                aggregate_dice
            ),
            "mean_iou": (
                aggregate_iou
            ),
            "mean_water_precision": (
                aggregate_precision
            ),
            "mean_water_recall": (
                aggregate_recall
            ),
            "mean_land_leakage_rate": (
                aggregate_leakage
            ),
            "worst_land_leakage_rate": (
                worst_leakage
            ),
            "minimum_water_recall": (
                minimum_recall
            ),
            "minimum_water_iou": (
                minimum_iou
            ),
        },
        "violation_counts": {
            "land_leakage": (
                leakage_violations
            ),
            "water_recall": (
                recall_violations
            ),
            "water_iou": (
                iou_violations
            ),
        },
        "fold_results": relative(
            fold_results_path
        ),
        "test_contact_sheet": relative(
            contact_sheet_path
        ),
        "source_checkpoint": relative(
            checkpoint_path
        ),
        "training_seconds": float(
            time.time()
            - total_start
        ),
        "all_dartis_masks_generated": False,
        "official_sen1floods11_test_used": False,
        "independent_final_release_test": False,
        "scientific_note": (
            "Her örnek kendi fold'unda train ve validation dışında "
            "tutularak yalnız bir kez test edilmiştir. Yedi örnek "
            "küçük olduğu için sonuç hedef-domain adaptasyon kanıtıdır, "
            "final ürün doğrulaması değildir."
        ),
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
        "DARTIS LOOCV ADAPTATION SONUCU"
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
        "Mean precision:",
        f"{aggregate_precision:.4f}",
    )
    print(
        "Mean recall:",
        f"{aggregate_recall:.4f}",
    )
    print(
        "Mean IoU:",
        f"{aggregate_iou:.4f}",
    )
    print(
        "Mean land leakage:",
        f"{aggregate_leakage:.4f}",
    )
    print(
        "Worst land leakage:",
        f"{worst_leakage:.4f}",
    )
    print(
        "Minimum per-image recall:",
        f"{minimum_recall:.4f}",
    )
    print(
        "Minimum per-image IoU:",
        f"{minimum_iou:.4f}",
    )
    print(
        "Leakage violations:",
        leakage_violations,
    )
    print(
        "Recall violations:",
        recall_violations,
    )
    print(
        "IoU violations:",
        iou_violations,
    )

    if failed_reasons:
        print(
            "Başarısız koşullar:",
            ", ".join(
                failed_reasons
            ),
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
        "Bu aşamada tüm DARTIS görüntülerine "
        "maske uygulanmadı."
    )


if __name__ == "__main__":
    main()
