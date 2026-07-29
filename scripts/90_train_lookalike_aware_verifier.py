from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import runpy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]

DATASET_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v07_verifier_binary_dataset.csv"
)

OLD_DETECTOR_SCRIPT = (
    ROOT
    / "scripts"
    / "33_run_end_to_end_oil_detector.py"
)

OLD_MODEL_DIR = (
    ROOT
    / "checkpoints"
    / "final_model_v03"
)

OLD_VERIFIER_CHECKPOINT = (
    OLD_MODEL_DIR
    / "oil_candidate_verifier_resnet18.pth"
)

OLD_CALIBRATION_CONFIG = (
    OLD_MODEL_DIR
    / "calibration_config.json"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "verifier_v07"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "verifier_v07"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best.pth"
)

CALIBRATION_CONFIG = (
    CHECKPOINT_DIR
    / "calibration_config.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history.csv"
)

INTERNAL_PREDICTIONS = (
    OUTPUT_DIR
    / "internal_validation_predictions.csv"
)

CALIBRATION_PREDICTIONS = (
    OUTPUT_DIR
    / "calibration_predictions.csv"
)

CONTACT_SHEET = (
    OUTPUT_DIR
    / "calibration_decision_contact_sheet.jpg"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.7 petrol verifier'ını sahne izolasyonlu ve sınıf/sahne "
            "dengeli örnekleme ile yeniden eğitir. Model iki sınıf öğrenir: "
            "LOOK_ALIKE ve CONFIRMED_OIL. Calibration üzerinde iki eşik "
            "seçilerek aradaki bölge UNCERTAIN olarak bırakılır."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--internal-validation-scene-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-negative-confirm-rate",
        type=float,
        default=0.0,
        help=(
            "Calibration LOOK_ALIKE örneklerinde CONFIRMED_OIL kararına "
            "izin verilen en yüksek oran. 23 negatifte 0.0 tam sıfır "
            "yanlış onay anlamına gelir."
        ),
    )

    parser.add_argument(
        "--max-positive-lookalike-rate",
        type=float,
        default=0.01,
        help=(
            "Calibration petrol örneklerinde kesin LOOK_ALIKE kararına "
            "izin verilen en yüksek oran."
        ),
    )

    parser.add_argument(
        "--minimum-confirmed-oil-recall",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--minimum-lookalike-recall",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--minimum-confirmed-oil-precision",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--minimum-lookalike-precision",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="auto",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value: str | Path) -> Path:
    path = Path(
        str(value).strip().replace(
            "\\",
            "/",
        )
    )

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


def select_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")

    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA istendi ancak kullanılamıyor."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def deterministic_bucket(
    value: str,
    modulo: int = 100,
) -> int:
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "big",
    ) % modulo


def load_manifest() -> pd.DataFrame:
    if not DATASET_MANIFEST.exists():
        raise FileNotFoundError(
            f"Verifier manifesti bulunamadı: {DATASET_MANIFEST}"
        )

    frame = pd.read_csv(
        DATASET_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "candidate_id",
        "scene_id",
        "label",
        "class_name",
        "split_role",
        "crop_path",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            "Verifier manifestinde eksik sütunlar: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()

    frame["label"] = (
        frame["label"]
        .astype(int)
    )

    frame["scene_id"] = (
        frame["scene_id"]
        .astype(str)
    )

    frame["resolved_crop_path"] = (
        frame["crop_path"].map(
            lambda value: str(
                resolve_path(value)
            )
        )
    )

    missing_files = frame[
        ~frame[
            "resolved_crop_path"
        ].map(
            lambda value: Path(
                value
            ).exists()
        )
    ]

    if not missing_files.empty:
        raise FileNotFoundError(
            "Eksik crop dosyaları bulundu: "
            + ", ".join(
                missing_files[
                    "candidate_id"
                ].astype(str).head(5)
            )
        )

    return frame.reset_index(
        drop=True
    )


def split_internal_validation(
    train_frame: pd.DataFrame,
    scene_ratio: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    validation_scenes: set[str] = set()

    for label in (0, 1):
        class_scenes = sorted(
            train_frame[
                train_frame[
                    "label"
                ].eq(label)
            ][
                "scene_id"
            ].unique()
        )

        if len(class_scenes) < 2:
            raise RuntimeError(
                f"Etiket {label} için sahne bazlı internal validation "
                "oluşturmak için yeterli sahne yok."
            )

        ranked = sorted(
            class_scenes,
            key=lambda scene_id: (
                deterministic_bucket(
                    f"internal-validation::{label}::{scene_id}",
                    modulo=10**9,
                )
            ),
        )

        validation_count = max(
            1,
            int(
                round(
                    len(ranked)
                    * scene_ratio
                )
            ),
        )

        validation_count = min(
            validation_count,
            len(ranked) - 1,
        )

        validation_scenes.update(
            ranked[
                :validation_count
            ]
        )

    internal_validation = train_frame[
        train_frame[
            "scene_id"
        ].isin(
            validation_scenes
        )
    ].copy()

    actual_train = train_frame[
        ~train_frame[
            "scene_id"
        ].isin(
            validation_scenes
        )
    ].copy()

    overlap = set(
        actual_train[
            "scene_id"
        ]
    ) & set(
        internal_validation[
            "scene_id"
        ]
    )

    if overlap:
        raise RuntimeError(
            "Internal train/validation sahne sızıntısı oluştu."
        )

    for name, subset in (
        (
            "actual_train",
            actual_train,
        ),
        (
            "internal_validation",
            internal_validation,
        ),
    ):
        labels = set(
            subset["label"].unique()
        )

        if labels != {
            0,
            1,
        }:
            raise RuntimeError(
                f"{name} iki sınıfı da içermiyor: {labels}"
            )

    return (
        actual_train.reset_index(
            drop=True
        ),
        internal_validation.reset_index(
            drop=True
        ),
    )


def verify_scene_isolation(
    train_frame: pd.DataFrame,
    internal_validation: pd.DataFrame,
    calibration_frame: pd.DataFrame,
) -> None:
    train_scenes = set(
        train_frame[
            "scene_id"
        ].astype(str)
    )

    internal_scenes = set(
        internal_validation[
            "scene_id"
        ].astype(str)
    )

    calibration_scenes = set(
        calibration_frame[
            "scene_id"
        ].astype(str)
    )

    intersections = {
        "train_internal": (
            train_scenes
            & internal_scenes
        ),
        "train_calibration": (
            train_scenes
            & calibration_scenes
        ),
        "internal_calibration": (
            internal_scenes
            & calibration_scenes
        ),
    }

    leaking = {
        name: values
        for name, values
        in intersections.items()
        if values
    }

    if leaking:
        raise RuntimeError(
            "Sahne bazlı split leakage bulundu: "
            + "; ".join(
                f"{name}={len(values)}"
                for name, values
                in leaking.items()
            )
        )


class VerifierDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Any,
        train: bool,
        seed: int,
    ) -> None:
        self.records = frame.to_dict(
            orient="records"
        )
        self.transform = transform
        self.train = train
        self.seed = seed
        self.epoch = 0

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        self.epoch = int(
            epoch
        )

    def __len__(self) -> int:
        return len(
            self.records
        )

    def augment(
        self,
        image: Image.Image,
        index: int,
    ) -> Image.Image:
        rng = random.Random(
            self.seed
            + self.epoch
            * 1_000_003
            + index
            * 1009
        )

        result = image

        if rng.random() < 0.5:
            result = ImageOps.mirror(
                result
            )

        if rng.random() < 0.5:
            result = ImageOps.flip(
                result
            )

        rotation = rng.choice(
            [
                0,
                90,
                180,
                270,
            ]
        )

        if rotation:
            result = result.rotate(
                rotation,
                expand=False,
                resample=(
                    Image.Resampling.BILINEAR
                ),
            )

        if rng.random() < 0.65:
            result = ImageEnhance.Contrast(
                result
            ).enhance(
                rng.uniform(
                    0.85,
                    1.18,
                )
            )

        if rng.random() < 0.45:
            result = ImageEnhance.Brightness(
                result
            ).enhance(
                rng.uniform(
                    0.90,
                    1.10,
                )
            )

        return result

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        str,
        str,
    ]:
        record = self.records[
            index
        ]

        image = Image.open(
            record[
                "resolved_crop_path"
            ]
        ).convert("L")

        if self.train:
            image = self.augment(
                image,
                index,
            )

        tensor = self.transform(
            image
        )

        label = torch.tensor(
            float(
                record[
                    "label"
                ]
            ),
            dtype=torch.float32,
        )

        return (
            tensor,
            label,
            str(
                record[
                    "candidate_id"
                ]
            ),
            str(
                record[
                    "scene_id"
                ]
            ),
        )


def build_scene_class_balanced_weights(
    frame: pd.DataFrame,
) -> torch.Tensor:
    class_scene_counts = (
        frame[
            [
                "label",
                "scene_id",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "label"
        )
        .size()
        .to_dict()
    )

    crop_counts = (
        frame.groupby(
            [
                "label",
                "scene_id",
            ]
        )
        .size()
        .to_dict()
    )

    weights = []

    for row in frame.itertuples():
        class_scene_count = int(
            class_scene_counts[
                int(
                    row.label
                )
            ]
        )

        crop_count = int(
            crop_counts[
                (
                    int(
                        row.label
                    ),
                    str(
                        row.scene_id
                    ),
                )
            ]
        )

        weight = (
            1.0
            / max(
                class_scene_count,
                1,
            )
            / max(
                crop_count,
                1,
            )
        )

        weights.append(
            weight
        )

    return torch.tensor(
        weights,
        dtype=torch.double,
    )


def load_old_verifier(
    device: torch.device,
) -> tuple[
    dict[str, Any],
    torch.nn.Module,
    dict[str, Any],
    Any,
]:
    for path in (
        OLD_DETECTOR_SCRIPT,
        OLD_VERIFIER_CHECKPOINT,
        OLD_CALIBRATION_CONFIG,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Gerekli eski verifier kaynağı bulunamadı: {path}"
            )

    detector = runpy.run_path(
        str(
            OLD_DETECTOR_SCRIPT
        ),
        run_name=(
            "v07_verifier_training_old_detector"
        ),
    )

    detector[
        "VERIFIER_CHECKPOINT"
    ] = OLD_VERIFIER_CHECKPOINT

    detector[
        "CALIBRATION_CONFIG"
    ] = OLD_CALIBRATION_CONFIG

    (
        model,
        verifier_config,
    ) = detector[
        "build_verifier_model"
    ](
        device
    )

    transform = detector[
        "build_verifier_transform"
    ](
        verifier_config
    )

    return (
        detector,
        model,
        verifier_config,
        transform,
    )


def model_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    logits = model(
        images
    )

    if isinstance(
        logits,
        (
            tuple,
            list,
        ),
    ):
        logits = logits[0]

    return logits.reshape(
        -1
    )


def binary_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    labels = labels.astype(
        np.int64
    )

    positive_count = int(
        (labels == 1).sum()
    )

    negative_count = int(
        (labels == 0).sum()
    )

    if (
        positive_count == 0
        or negative_count == 0
    ):
        return float(
            "nan"
        )

    order = np.argsort(
        scores,
        kind="mergesort",
    )

    ranks = np.empty_like(
        order,
        dtype=np.float64,
    )

    ranks[
        order
    ] = np.arange(
        1,
        len(scores) + 1,
        dtype=np.float64,
    )

    sorted_scores = scores[
        order
    ]

    start = 0

    while start < len(
        sorted_scores
    ):
        end = start + 1

        while (
            end
            < len(
                sorted_scores
            )
            and sorted_scores[
                end
            ]
            == sorted_scores[
                start
            ]
        ):
            end += 1

        average_rank = float(
            ranks[
                order[
                    start:end
                ]
            ].mean()
        )

        ranks[
            order[
                start:end
            ]
        ] = average_rank

        start = end

    positive_rank_sum = float(
        ranks[
            labels == 1
        ].sum()
    )

    auc = (
        positive_rank_sum
        - positive_count
        * (
            positive_count + 1
        )
        / 2.0
    ) / (
        positive_count
        * negative_count
    )

    return float(
        auc
    )


def average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    labels = labels.astype(
        np.int64
    )

    positive_count = int(
        (labels == 1).sum()
    )

    if positive_count == 0:
        return float(
            "nan"
        )

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    sorted_labels = labels[
        order
    ]

    true_positive = np.cumsum(
        sorted_labels == 1
    )

    false_positive = np.cumsum(
        sorted_labels == 0
    )

    precision = (
        true_positive
        / np.maximum(
            true_positive
            + false_positive,
            1,
        )
    )

    return float(
        precision[
            sorted_labels == 1
        ].sum()
        / positive_count
    )


def evaluate_binary(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    prediction = (
        probabilities >= 0.5
    ).astype(
        np.int64
    )

    labels_int = labels.astype(
        np.int64
    )

    tp = int(
        (
            (prediction == 1)
            & (labels_int == 1)
        ).sum()
    )

    fp = int(
        (
            (prediction == 1)
            & (labels_int == 0)
        ).sum()
    )

    fn = int(
        (
            (prediction == 0)
            & (labels_int == 1)
        ).sum()
    )

    tn = int(
        (
            (prediction == 0)
            & (labels_int == 0)
        ).sum()
    )

    positive_recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    negative_recall = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 1.0
    )

    balanced_accuracy = (
        positive_recall
        + negative_recall
    ) / 2.0

    return {
        "auc": binary_auc(
            labels_int,
            probabilities,
        ),
        "average_precision": (
            average_precision(
                labels_int,
                probabilities,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "positive_recall": float(
            positive_recall
        ),
        "negative_recall": float(
            negative_recall
        ),
        "positive_precision": float(
            precision
        ),
    }


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()

    rows = []

    for (
        images,
        labels,
        candidate_ids,
        scene_ids,
    ) in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(
                device.type == "cuda"
            ),
        ):
            logits = model_logits(
                model,
                images,
            )

        logits_np = (
            logits.float()
            .cpu()
            .numpy()
        )

        probabilities = (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        logits_np,
                        -30.0,
                        30.0,
                    )
                )
            )
        )

        for index in range(
            len(
                candidate_ids
            )
        ):
            rows.append(
                {
                    "candidate_id": str(
                        candidate_ids[
                            index
                        ]
                    ),
                    "scene_id": str(
                        scene_ids[
                            index
                        ]
                    ),
                    "label": int(
                        labels[
                            index
                        ].item()
                    ),
                    "logit": float(
                        logits_np[
                            index
                        ]
                    ),
                    "raw_probability": float(
                        probabilities[
                            index
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> float:
    logits_tensor = torch.tensor(
        logits,
        dtype=torch.float32,
        device=device,
    )

    labels_tensor = torch.tensor(
        labels,
        dtype=torch.float32,
        device=device,
    )

    class_counts = {
        0: max(
            int(
                (
                    labels
                    == 0
                ).sum()
            ),
            1,
        ),
        1: max(
            int(
                (
                    labels
                    == 1
                ).sum()
            ),
            1,
        ),
    }

    sample_weights = torch.tensor(
        [
            1.0
            / class_counts[
                int(label)
            ]
            for label in labels
        ],
        dtype=torch.float32,
        device=device,
    )

    sample_weights = (
        sample_weights
        / sample_weights.mean()
    )

    log_temperature = torch.nn.Parameter(
        torch.zeros(
            1,
            device=device,
        )
    )

    optimizer = torch.optim.LBFGS(
        [
            log_temperature
        ],
        lr=0.1,
        max_iter=100,
        line_search_fn=(
            "strong_wolfe"
        ),
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        ).clamp(
            0.05,
            20.0,
        )

        loss_map = (
            F.binary_cross_entropy_with_logits(
                logits_tensor
                / temperature,
                labels_tensor,
                reduction="none",
            )
        )

        loss = (
            loss_map
            * sample_weights
        ).mean()

        loss.backward()

        return loss

    optimizer.step(
        closure
    )

    temperature = float(
        torch.exp(
            log_temperature.detach()
        )
        .clamp(
            0.05,
            20.0,
        )
        .cpu()
        .item()
    )

    return temperature


def choose_abstention_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    max_negative_confirm_rate: float,
    max_positive_lookalike_rate: float,
) -> tuple[
    float,
    float,
    dict[str, Any],
]:
    labels = labels.astype(
        np.int64
    )

    positive_probabilities = probabilities[
        labels == 1
    ]

    negative_probabilities = probabilities[
        labels == 0
    ]

    if (
        len(
            positive_probabilities
        )
        == 0
        or len(
            negative_probabilities
        )
        == 0
    ):
        raise RuntimeError(
            "Calibration iki sınıfı da içermiyor."
        )

    candidates = np.unique(
        np.concatenate(
            [
                np.array(
                    [
                        -1e-6,
                        0.0,
                        1.0,
                        1.000001,
                    ],
                    dtype=np.float64,
                ),
                probabilities.astype(
                    np.float64
                ),
            ]
        )
    )

    high_options = []

    for threshold in candidates:
        negative_confirm_rate = float(
            (
                negative_probabilities
                >= threshold
            ).mean()
        )

        if (
            negative_confirm_rate
            > max_negative_confirm_rate
            + 1e-12
        ):
            continue

        positive_confirm_recall = float(
            (
                positive_probabilities
                >= threshold
            ).mean()
        )

        confirmed_count = int(
            (
                probabilities
                >= threshold
            ).sum()
        )

        confirmed_true_positive = int(
            (
                (
                    probabilities
                    >= threshold
                )
                & (
                    labels
                    == 1
                )
            ).sum()
        )

        confirmed_precision = (
            confirmed_true_positive
            / confirmed_count
            if confirmed_count > 0
            else 1.0
        )

        high_options.append(
            {
                "threshold": float(
                    threshold
                ),
                "positive_confirm_recall": (
                    positive_confirm_recall
                ),
                "negative_confirm_rate": (
                    negative_confirm_rate
                ),
                "confirmed_precision": float(
                    confirmed_precision
                ),
            }
        )

    if not high_options:
        raise RuntimeError(
            "Güvenli CONFIRMED_OIL eşiği bulunamadı."
        )

    selected_high = max(
        high_options,
        key=lambda item: (
            item[
                "positive_confirm_recall"
            ],
            item[
                "confirmed_precision"
            ],
            -item[
                "threshold"
            ],
        ),
    )

    high_threshold = float(
        selected_high[
            "threshold"
        ]
    )

    low_options = []

    for threshold in candidates:
        if threshold >= high_threshold:
            continue

        positive_lookalike_rate = float(
            (
                positive_probabilities
                <= threshold
            ).mean()
        )

        if (
            positive_lookalike_rate
            > max_positive_lookalike_rate
            + 1e-12
        ):
            continue

        negative_lookalike_recall = float(
            (
                negative_probabilities
                <= threshold
            ).mean()
        )

        lookalike_count = int(
            (
                probabilities
                <= threshold
            ).sum()
        )

        lookalike_true_negative = int(
            (
                (
                    probabilities
                    <= threshold
                )
                & (
                    labels
                    == 0
                )
            ).sum()
        )

        lookalike_precision = (
            lookalike_true_negative
            / lookalike_count
            if lookalike_count > 0
            else 1.0
        )

        low_options.append(
            {
                "threshold": float(
                    threshold
                ),
                "negative_lookalike_recall": (
                    negative_lookalike_recall
                ),
                "positive_lookalike_rate": (
                    positive_lookalike_rate
                ),
                "lookalike_precision": float(
                    lookalike_precision
                ),
            }
        )

    if not low_options:
        raise RuntimeError(
            "Güvenli LOOK_ALIKE eşiği bulunamadı."
        )

    selected_low = max(
        low_options,
        key=lambda item: (
            item[
                "negative_lookalike_recall"
            ],
            item[
                "lookalike_precision"
            ],
            item[
                "threshold"
            ],
        ),
    )

    low_threshold = float(
        selected_low[
            "threshold"
        ]
    )

    decisions = np.full(
        len(
            probabilities
        ),
        "UNCERTAIN",
        dtype=object,
    )

    decisions[
        probabilities
        <= low_threshold
    ] = "LOOK_ALIKE"

    decisions[
        probabilities
        >= high_threshold
    ] = "CONFIRMED_OIL"

    positive_mask = (
        labels == 1
    )

    negative_mask = (
        labels == 0
    )

    confirmed_mask = (
        decisions
        == "CONFIRMED_OIL"
    )

    lookalike_mask = (
        decisions
        == "LOOK_ALIKE"
    )

    uncertain_mask = (
        decisions
        == "UNCERTAIN"
    )

    confirmed_true_positive = int(
        (
            confirmed_mask
            & positive_mask
        ).sum()
    )

    confirmed_false_positive = int(
        (
            confirmed_mask
            & negative_mask
        ).sum()
    )

    lookalike_true_negative = int(
        (
            lookalike_mask
            & negative_mask
        ).sum()
    )

    lookalike_false_negative = int(
        (
            lookalike_mask
            & positive_mask
        ).sum()
    )

    confirmed_precision = (
        confirmed_true_positive
        / (
            confirmed_true_positive
            + confirmed_false_positive
        )
        if (
            confirmed_true_positive
            + confirmed_false_positive
        ) > 0
        else 1.0
    )

    lookalike_precision = (
        lookalike_true_negative
        / (
            lookalike_true_negative
            + lookalike_false_negative
        )
        if (
            lookalike_true_negative
            + lookalike_false_negative
        ) > 0
        else 1.0
    )

    metrics = {
        "lookalike_threshold": (
            low_threshold
        ),
        "confirmed_oil_threshold": (
            high_threshold
        ),
        "confirmed_oil_precision": float(
            confirmed_precision
        ),
        "confirmed_oil_recall": float(
            (
                confirmed_mask[
                    positive_mask
                ]
            ).mean()
        ),
        "lookalike_precision": float(
            lookalike_precision
        ),
        "lookalike_recall": float(
            (
                lookalike_mask[
                    negative_mask
                ]
            ).mean()
        ),
        "positive_uncertain_rate": float(
            (
                uncertain_mask[
                    positive_mask
                ]
            ).mean()
        ),
        "negative_uncertain_rate": float(
            (
                uncertain_mask[
                    negative_mask
                ]
            ).mean()
        ),
        "overall_uncertain_rate": float(
            uncertain_mask.mean()
        ),
        "definite_decision_rate": float(
            1.0
            - uncertain_mask.mean()
        ),
        "confirmed_oil_count": int(
            confirmed_mask.sum()
        ),
        "lookalike_count": int(
            lookalike_mask.sum()
        ),
        "uncertain_count": int(
            uncertain_mask.sum()
        ),
        "confirmed_false_positive_count": (
            confirmed_false_positive
        ),
        "lookalike_false_negative_count": (
            lookalike_false_negative
        ),
    }

    return (
        low_threshold,
        high_threshold,
        metrics,
    )


def add_decisions(
    frame: pd.DataFrame,
    temperature: float,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    result = frame.copy()

    calibrated_probability = (
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    result[
                        "logit"
                    ].to_numpy(
                        dtype=np.float64
                    )
                    / temperature,
                    -30.0,
                    30.0,
                )
            )
        )
    )

    result[
        "calibrated_probability"
    ] = calibrated_probability

    decisions = np.full(
        len(result),
        "UNCERTAIN",
        dtype=object,
    )

    decisions[
        calibrated_probability
        <= low_threshold
    ] = "LOOK_ALIKE"

    decisions[
        calibrated_probability
        >= high_threshold
    ] = "CONFIRMED_OIL"

    result[
        "decision"
    ] = decisions

    return result


def enhance_image(
    image: Image.Image,
) -> Image.Image:
    array = np.asarray(
        image.convert("L"),
        dtype=np.float32,
    )

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
        enhanced = array
    else:
        enhanced = (
            np.clip(
                (
                    array
                    - low
                )
                / (
                    high
                    - low
                ),
                0.0,
                1.0,
            )
            * 255.0
        )

    return Image.fromarray(
        enhanced.astype(
            np.uint8
        )
    )


def create_contact_sheet(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    merged = predictions.merge(
        manifest[
            [
                "candidate_id",
                "class_name",
                "crop_path",
            ]
        ],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )

    merged[
        "priority"
    ] = 0

    merged.loc[
        (
            merged["label"].eq(0)
            & merged[
                "decision"
            ].eq(
                "CONFIRMED_OIL"
            )
        ),
        "priority",
    ] = 5

    merged.loc[
        (
            merged["label"].eq(1)
            & merged[
                "decision"
            ].eq(
                "LOOK_ALIKE"
            )
        ),
        "priority",
    ] = 5

    merged.loc[
        merged[
            "decision"
        ].eq(
            "UNCERTAIN"
        ),
        "priority",
    ] = 3

    selected = (
        merged.sort_values(
            [
                "priority",
                "calibrated_probability",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(64)
        .reset_index(
            drop=True
        )
    )

    if selected.empty:
        return

    tile_width = 250
    image_height = 190
    header_height = 58
    tile_height = (
        image_height
        + header_height
    )

    columns = 4
    rows = int(
        math.ceil(
            len(selected)
            / columns
        )
    )

    sheet = Image.new(
        "RGB",
        (
            columns
            * tile_width,
            rows
            * tile_height,
        ),
        "black",
    )

    font = ImageFont.load_default()

    for index, row in enumerate(
        selected.itertuples()
    ):
        image = enhance_image(
            Image.open(
                resolve_path(
                    row.crop_path
                )
            )
        ).convert(
            "RGB"
        )

        image.thumbnail(
            (
                tile_width,
                image_height,
            ),
            Image.Resampling.LANCZOS,
        )

        tile = Image.new(
            "RGB",
            (
                tile_width,
                tile_height,
            ),
            "black",
        )

        offset_x = (
            tile_width
            - image.width
        ) // 2

        offset_y = (
            image_height
            - image.height
        ) // 2

        tile.paste(
            image,
            (
                offset_x,
                header_height
                + offset_y,
            ),
        )

        draw = ImageDraw.Draw(
            tile
        )

        draw.text(
            (5, 5),
            (
                f"GT={row.class_name} | "
                f"{row.decision}"
            )[:44],
            fill="white",
            font=font,
        )

        draw.text(
            (5, 25),
            (
                f"p={float(row.calibrated_probability):.4f} "
                f"| {row.scene_id}"
            )[:44],
            fill="white",
            font=font,
        )

        sheet.paste(
            tile,
            (
                (
                    index
                    % columns
                )
                * tile_width,
                (
                    index
                    // columns
                )
                * tile_height,
            ),
        )

    CONTACT_SHEET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        CONTACT_SHEET,
        quality=92,
    )


def main() -> None:
    args = parse_args()

    set_seed(
        args.seed
    )

    device = select_device(
        args.device
    )

    manifest = load_manifest()

    train_pool = manifest[
        manifest[
            "split_role"
        ].eq(
            "train"
        )
    ].copy()

    calibration_frame = manifest[
        manifest[
            "split_role"
        ].eq(
            "calibration"
        )
    ].copy()

    if train_pool.empty:
        raise RuntimeError(
            "Train bölümü boş."
        )

    if calibration_frame.empty:
        raise RuntimeError(
            "Calibration bölümü boş."
        )

    (
        actual_train,
        internal_validation,
    ) = split_internal_validation(
        train_pool,
        scene_ratio=(
            args.internal_validation_scene_ratio
        ),
    )

    verify_scene_isolation(
        actual_train,
        internal_validation,
        calibration_frame,
    )

    (
        detector,
        model,
        verifier_config,
        transform,
    ) = load_old_verifier(
        device
    )

    train_dataset = VerifierDataset(
        actual_train,
        transform=transform,
        train=True,
        seed=args.seed,
    )

    internal_dataset = VerifierDataset(
        internal_validation,
        transform=transform,
        train=False,
        seed=args.seed,
    )

    calibration_dataset = VerifierDataset(
        calibration_frame,
        transform=transform,
        train=False,
        seed=args.seed,
    )

    weights = (
        build_scene_class_balanced_weights(
            actual_train
        )
    )

    samples_per_epoch = max(
        int(
            args.samples_per_epoch
        ),
        int(
            args.batch_size
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    internal_loader = DataLoader(
        internal_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "v0.7 LOOK_ALIKE-AWARE VERIFIER TRAINING"
    )
    print("=" * 78)

    print(
        "Cihaz:",
        device,
    )

    print(
        "Actual train:",
        len(actual_train),
        actual_train.groupby(
            "class_name"
        ).size().to_dict(),
    )

    print(
        "Internal validation:",
        len(internal_validation),
        internal_validation.groupby(
            "class_name"
        ).size().to_dict(),
    )

    print(
        "Calibration (eğitimde kullanılmayacak):",
        len(calibration_frame),
        calibration_frame.groupby(
            "class_name"
        ).size().to_dict(),
    )

    print(
        "Sampler: sınıf + sahne dengeli"
    )

    print()

    best_score = -math.inf
    best_epoch = 0
    best_state = None
    best_internal_metrics = None
    history = []
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        model.train()

        total_loss = 0.0
        batch_count = 0

        for (
            images,
            labels,
            _,
            _,
        ) in train_loader:
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
                enabled=(
                    device.type == "cuda"
                ),
            ):
                logits = model_logits(
                    model,
                    images,
                )

                loss = (
                    F.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                    )
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            total_loss += float(
                loss.detach().item()
            )

            batch_count += 1

        internal_predictions = (
            collect_predictions(
                model,
                internal_loader,
                device,
            )
        )

        internal_metrics = (
            evaluate_binary(
                internal_predictions[
                    "label"
                ].to_numpy(
                    dtype=np.int64
                ),
                internal_predictions[
                    "raw_probability"
                ].to_numpy(
                    dtype=np.float64
                ),
            )
        )

        selection_score = float(
            0.45
            * internal_metrics[
                "auc"
            ]
            + 0.30
            * internal_metrics[
                "balanced_accuracy"
            ]
            + 0.15
            * internal_metrics[
                "negative_recall"
            ]
            + 0.10
            * internal_metrics[
                "average_precision"
            ]
        )

        history_row = {
            "epoch": epoch,
            "train_loss": float(
                total_loss
                / max(
                    batch_count,
                    1,
                )
            ),
            "selection_score": (
                selection_score
            ),
            **internal_metrics,
        }

        history.append(
            history_row
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} "
            f"| loss {history_row['train_loss']:.4f} "
            f"| AUC {internal_metrics['auc']:.4f} "
            f"| BalAcc {internal_metrics['balanced_accuracy']:.4f} "
            f"| OilR {internal_metrics['positive_recall']:.4f} "
            f"| LookR {internal_metrics['negative_recall']:.4f} "
            f"| score {selection_score:.4f}"
        )

        if selection_score > best_score:
            best_score = (
                selection_score
            )

            best_epoch = epoch

            best_internal_metrics = dict(
                internal_metrics
            )

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "Early stopping."
            )
            break

    if (
        best_state is None
        or best_internal_metrics
        is None
    ):
        raise RuntimeError(
            "En iyi verifier checkpoint seçilemedi."
        )

    model.load_state_dict(
        best_state
    )

    final_internal_predictions = (
        collect_predictions(
            model,
            internal_loader,
            device,
        )
    )

    final_internal_predictions.to_csv(
        INTERNAL_PREDICTIONS,
        index=False,
        encoding="utf-8-sig",
    )

    calibration_predictions = (
        collect_predictions(
            model,
            calibration_loader,
            device,
        )
    )

    temperature = fit_temperature(
        logits=calibration_predictions[
            "logit"
        ].to_numpy(
            dtype=np.float64
        ),
        labels=calibration_predictions[
            "label"
        ].to_numpy(
            dtype=np.int64
        ),
        device=device,
    )

    calibrated_probabilities = (
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    calibration_predictions[
                        "logit"
                    ].to_numpy(
                        dtype=np.float64
                    )
                    / temperature,
                    -30.0,
                    30.0,
                )
            )
        )
    )

    (
        lookalike_threshold,
        confirmed_oil_threshold,
        calibration_metrics,
    ) = choose_abstention_thresholds(
        labels=calibration_predictions[
            "label"
        ].to_numpy(
            dtype=np.int64
        ),
        probabilities=(
            calibrated_probabilities
        ),
        max_negative_confirm_rate=(
            args.max_negative_confirm_rate
        ),
        max_positive_lookalike_rate=(
            args.max_positive_lookalike_rate
        ),
    )

    calibration_predictions = (
        add_decisions(
            calibration_predictions,
            temperature=temperature,
            low_threshold=(
                lookalike_threshold
            ),
            high_threshold=(
                confirmed_oil_threshold
            ),
        )
    )

    calibration_predictions.to_csv(
        CALIBRATION_PREDICTIONS,
        index=False,
        encoding="utf-8-sig",
    )

    gate_checks = {
        "confirmed_oil_precision": (
            calibration_metrics[
                "confirmed_oil_precision"
            ]
            >= args.minimum_confirmed_oil_precision
        ),
        "confirmed_oil_recall": (
            calibration_metrics[
                "confirmed_oil_recall"
            ]
            >= args.minimum_confirmed_oil_recall
        ),
        "lookalike_precision": (
            calibration_metrics[
                "lookalike_precision"
            ]
            >= args.minimum_lookalike_precision
        ),
        "lookalike_recall": (
            calibration_metrics[
                "lookalike_recall"
            ]
            >= args.minimum_lookalike_recall
        ),
        "zero_or_allowed_negative_confirm_rate": (
            calibration_metrics[
                "confirmed_false_positive_count"
            ]
            <= math.floor(
                args.max_negative_confirm_rate
                * int(
                    (
                        calibration_predictions[
                            "label"
                        ].eq(0)
                    ).sum()
                )
                + 1e-9
            )
        ),
    }

    failed_checks = [
        name
        for name, passed
        in gate_checks.items()
        if not passed
    ]

    gate_passed = (
        len(
            failed_checks
        )
        == 0
    )

    torch.save(
        {
            "stage": (
                "v07_lookalike_aware_verifier"
            ),
            "model_state_dict": (
                best_state
            ),
            "best_epoch": int(
                best_epoch
            ),
            "best_internal_metrics": (
                best_internal_metrics
            ),
            "temperature": float(
                temperature
            ),
            "lookalike_threshold": float(
                lookalike_threshold
            ),
            "confirmed_oil_threshold": float(
                confirmed_oil_threshold
            ),
            "calibration_metrics": (
                calibration_metrics
            ),
            "gate_passed": bool(
                gate_passed
            ),
            "verifier_config": (
                verifier_config
            ),
            "training_config": vars(
                args
            ),
            "dataset_manifest": relative(
                DATASET_MANIFEST
            ),
            "old_verifier_initialization": relative(
                OLD_VERIFIER_CHECKPOINT
            ),
            "scene_isolation": True,
        },
        BEST_CHECKPOINT,
    )

    config = {
        "stage": (
            "v07_lookalike_aware_verifier"
        ),
        "temperature": float(
            temperature
        ),
        "lookalike_threshold": float(
            lookalike_threshold
        ),
        "confirmed_oil_threshold": float(
            confirmed_oil_threshold
        ),
        "decision_rule": {
            "LOOK_ALIKE": (
                "probability <= lookalike_threshold"
            ),
            "UNCERTAIN": (
                "lookalike_threshold < probability "
                "< confirmed_oil_threshold"
            ),
            "CONFIRMED_OIL": (
                "probability >= confirmed_oil_threshold"
            ),
        },
        "calibration_metrics": (
            calibration_metrics
        ),
        "gate_passed": bool(
            gate_passed
        ),
        "failed_checks": (
            failed_checks
        ),
        "model_checkpoint": relative(
            BEST_CHECKPOINT
        ),
        "source_verifier_config": (
            verifier_config
        ),
    }

    CALIBRATION_CONFIG.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    create_contact_sheet(
        calibration_predictions,
        calibration_frame,
    )

    summary = {
        "stage": (
            "v07_lookalike_aware_verifier_training"
        ),
        "actual_train_count": int(
            len(actual_train)
        ),
        "internal_validation_count": int(
            len(
                internal_validation
            )
        ),
        "calibration_count": int(
            len(
                calibration_frame
            )
        ),
        "actual_train_class_counts": {
            str(key): int(value)
            for key, value in actual_train.groupby(
                "class_name"
            ).size().to_dict().items()
        },
        "internal_validation_class_counts": {
            str(key): int(value)
            for key, value in internal_validation.groupby(
                "class_name"
            ).size().to_dict().items()
        },
        "calibration_class_counts": {
            str(key): int(value)
            for key, value in calibration_frame.groupby(
                "class_name"
            ).size().to_dict().items()
        },
        "scene_isolation_verified": True,
        "balanced_sampler": (
            "class-and-scene-balanced weighted sampling"
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_internal_metrics": (
            best_internal_metrics
        ),
        "temperature": float(
            temperature
        ),
        "calibration_metrics": (
            calibration_metrics
        ),
        "gate_passed": bool(
            gate_passed
        ),
        "failed_checks": (
            failed_checks
        ),
        "gate_criteria": {
            "minimum_confirmed_oil_precision": (
                args.minimum_confirmed_oil_precision
            ),
            "minimum_confirmed_oil_recall": (
                args.minimum_confirmed_oil_recall
            ),
            "minimum_lookalike_precision": (
                args.minimum_lookalike_precision
            ),
            "minimum_lookalike_recall": (
                args.minimum_lookalike_recall
            ),
            "max_negative_confirm_rate": (
                args.max_negative_confirm_rate
            ),
            "max_positive_lookalike_rate": (
                args.max_positive_lookalike_rate
            ),
        },
        "checkpoint": relative(
            BEST_CHECKPOINT
        ),
        "calibration_config": relative(
            CALIBRATION_CONFIG
        ),
        "training_history": relative(
            HISTORY_PATH
        ),
        "internal_predictions": relative(
            INTERNAL_PREDICTIONS
        ),
        "calibration_predictions": relative(
            CALIBRATION_PREDICTIONS
        ),
        "contact_sheet": (
            relative(
                CONTACT_SHEET
            )
            if CONTACT_SHEET.exists()
            else None
        ),
        "training_seconds": float(
            time.time()
            - start_time
        ),
        "fresh_end_to_end_holdout_used": False,
        "scientific_note": (
            "Checkpoint seçimi yalnız train içinden sahne bazında ayrılan "
            "internal validation ile yapılmıştır. Calibration split eğitim "
            "ve checkpoint seçiminde kullanılmamış; yalnız sıcaklık ölçekleme "
            "ve LOOK_ALIKE/UNCERTAIN/CONFIRMED_OIL eşiklerini dondurmak için "
            "tek kez kullanılmıştır. Taze uçtan uca holdout kullanılmamıştır."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "v0.7 VERIFIER EĞİTİM SONUCU"
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
        "Best epoch:",
        best_epoch,
    )

    print(
        "Temperature:",
        f"{temperature:.6f}",
    )

    print(
        "LOOK_ALIKE threshold:",
        f"{lookalike_threshold:.6f}",
    )

    print(
        "CONFIRMED_OIL threshold:",
        f"{confirmed_oil_threshold:.6f}",
    )

    print(
        "Confirmed precision:",
        f"{calibration_metrics['confirmed_oil_precision']:.4f}",
    )

    print(
        "Confirmed recall:",
        f"{calibration_metrics['confirmed_oil_recall']:.4f}",
    )

    print(
        "Look-alike precision:",
        f"{calibration_metrics['lookalike_precision']:.4f}",
    )

    print(
        "Look-alike recall:",
        f"{calibration_metrics['lookalike_recall']:.4f}",
    )

    print(
        "Overall uncertain rate:",
        f"{calibration_metrics['overall_uncertain_rate']:.4f}",
    )

    print(
        "Failed:",
        failed_checks,
    )

    print(
        "Checkpoint:",
        BEST_CHECKPOINT.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    if CONTACT_SHEET.exists():
        print(
            "Görsel:",
            CONTACT_SHEET.resolve(),
        )


if __name__ == "__main__":
    main()
