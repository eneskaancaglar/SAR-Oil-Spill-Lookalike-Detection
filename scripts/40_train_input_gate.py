from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v04_input_gate_dataset.csv"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "input_gate_v04"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v04_input_gate_training"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v04_input_gate_training.md"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best_input_gate.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "last_input_gate.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history.csv"
)

VALIDATION_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "validation_predictions.csv"
)

GROUP_METRICS_PATH = (
    OUTPUT_DIR
    / "validation_group_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "training_summary.json"
)

LOSS_PLOT_PATH = (
    OUTPUT_DIR
    / "loss_curve.png"
)

METRICS_PLOT_PATH = (
    OUTPUT_DIR
    / "validation_metrics.png"
)

NORMALIZATION_MEAN = (0.5,)
NORMALIZATION_STD = (0.25,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS benzeri deniz SAR görüntülerini kabul eden "
            "v0.4 input-gate modelini eğitir."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
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
        "--patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def select_device(
    requested: str,
) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA istendi fakat kullanılamıyor."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def resolve_path(value: Any) -> Path:
    text = str(value).strip()

    path = Path(text)

    if not path.is_absolute():
        path = ROOT / path

    return path


class RandomQuarterRotation:
    def __call__(
        self,
        image: Image.Image,
    ) -> Image.Image:
        angle = random.choice(
            [
                0,
                90,
                180,
                270,
            ]
        )

        if angle == 0:
            return image

        return image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
        )


class RandomGaussianNoise:
    def __init__(
        self,
        probability: float = 0.35,
        maximum_std: float = 0.035,
    ) -> None:
        self.probability = probability
        self.maximum_std = maximum_std

    def __call__(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() >= self.probability:
            return tensor

        noise_std = random.uniform(
            0.005,
            self.maximum_std,
        )

        noisy = tensor + (
            torch.randn_like(tensor)
            * noise_std
        )

        return noisy.clamp(
            0.0,
            1.0,
        )


class InputGateDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform,
    ) -> None:
        self.dataframe = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        str,
    ]:
        row = self.dataframe.iloc[index]

        image_path = resolve_path(
            row["image_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Görüntü bulunamadı: {image_path}"
            )

        with Image.open(image_path) as source:
            image = source.convert("L")

        image_tensor = self.transform(
            image
        )

        label = torch.tensor(
            float(row["binary_label"]),
            dtype=torch.float32,
        )

        return (
            image_tensor,
            label,
            str(row["sample_id"]),
        )


def build_transforms():
    training_transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 256),
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
            ),
            transforms.RandomResizedCrop(
                size=(224, 224),
                scale=(0.75, 1.00),
                ratio=(0.90, 1.10),
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
            ),
            RandomQuarterRotation(),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.RandomVerticalFlip(
                p=0.5
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.35,
                    )
                ],
                p=0.80,
            ),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(0.1, 1.0),
                    )
                ],
                p=0.20,
            ),
            transforms.ToTensor(),
            RandomGaussianNoise(
                probability=0.35,
                maximum_std=0.035,
            ),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN,
                std=NORMALIZATION_STD,
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN,
                std=NORMALIZATION_STD,
            ),
        ]
    )

    return (
        training_transform,
        evaluation_transform,
    )


def build_model(
    use_pretrained: bool,
) -> tuple[
    nn.Module,
    str,
]:
    pretrained_loaded = False
    initialization = "random"

    if use_pretrained:
        try:
            model = resnet18(
                weights=(
                    ResNet18_Weights.DEFAULT
                )
            )

            pretrained_loaded = True
            initialization = (
                "imagenet_pretrained"
            )

        except Exception as error:
            print(
                "UYARI: ImageNet ağırlıkları yüklenemedi."
            )

            print(
                f"{type(error).__name__}: {error}"
            )

            model = resnet18(
                weights=None
            )

    else:
        model = resnet18(
            weights=None
        )

    original_conv = model.conv1

    grayscale_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )

    if pretrained_loaded:
        with torch.no_grad():
            grayscale_conv.weight.copy_(
                original_conv.weight.mean(
                    dim=1,
                    keepdim=True,
                )
            )

    model.conv1 = grayscale_conv

    input_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(
            input_features,
            1,
        ),
    )

    return model, initialization


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, Any]:
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    labels = labels.astype(
        np.int64
    )

    true_positive = int(
        (
            (predictions == 1)
            & (labels == 1)
        ).sum()
    )

    true_negative = int(
        (
            (predictions == 0)
            & (labels == 0)
        ).sum()
    )

    false_positive = int(
        (
            (predictions == 1)
            & (labels == 0)
        ).sum()
    )

    false_negative = int(
        (
            (predictions == 0)
            & (labels == 1)
        ).sum()
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )

    accuracy = safe_divide(
        true_positive + true_negative,
        len(labels),
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "precision": float(precision),
        "supported_recall": float(recall),
        "unsupported_rejection_rate": float(
            specificity
        ),
        "f1": float(f1),
        "false_acceptance_rate": float(
            1.0 - specificity
        ),
        "false_rejection_rate": float(
            1.0 - recall
        ),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler,
    device: torch.device,
) -> float:
    model.train()

    running_loss = 0.0
    sample_count = 0

    amp_enabled = (
        device.type == "cuda"
    )

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

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(
                images
            ).flatten()

            loss = criterion(
                logits,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = int(
            images.shape[0]
        )

        running_loss += (
            float(loss.item())
            * batch_size
        )

        sample_count += batch_size

    return safe_divide(
        running_loss,
        sample_count,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.50,
) -> tuple[
    float,
    dict[str, Any],
    pd.DataFrame,
]:
    model.eval()

    running_loss = 0.0
    sample_count = 0

    labels_all: list[float] = []
    probabilities_all: list[float] = []
    logits_all: list[float] = []
    sample_ids_all: list[str] = []

    amp_enabled = (
        device.type == "cuda"
    )

    for images, labels, sample_ids in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels_device = labels.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(
                images
            ).flatten()

            loss = criterion(
                logits,
                labels_device,
            )

        probabilities = torch.sigmoid(
            logits
        )

        batch_size = int(
            images.shape[0]
        )

        running_loss += (
            float(loss.item())
            * batch_size
        )

        sample_count += batch_size

        labels_all.extend(
            labels.numpy().tolist()
        )

        logits_all.extend(
            logits
            .float()
            .cpu()
            .numpy()
            .tolist()
        )

        probabilities_all.extend(
            probabilities
            .float()
            .cpu()
            .numpy()
            .tolist()
        )

        sample_ids_all.extend(
            list(sample_ids)
        )

    labels_array = np.asarray(
        labels_all,
        dtype=np.float32,
    )

    probabilities_array = np.asarray(
        probabilities_all,
        dtype=np.float32,
    )

    metrics = calculate_metrics(
        labels_array,
        probabilities_array,
        threshold,
    )

    predictions = (
        probabilities_array >= threshold
    ).astype(np.int64)

    prediction_dataframe = pd.DataFrame(
        {
            "sample_id": sample_ids_all,
            "binary_label": (
                labels_array.astype(
                    np.int64
                )
            ),
            "raw_logit": np.asarray(
                logits_all,
                dtype=np.float32,
            ),
            "supported_probability": (
                probabilities_array
            ),
            "prediction": predictions,
            "correct": (
                predictions
                == labels_array.astype(
                    np.int64
                )
            ),
        }
    )

    return (
        safe_divide(
            running_loss,
            sample_count,
        ),
        metrics,
        prediction_dataframe,
    )


def save_checkpoint(
    destination: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    initialization: str,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "epoch": int(epoch),
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "validation_metrics": metrics,
        "architecture": (
            "resnet18_grayscale_input_gate"
        ),
        "initialization": initialization,
        "input_channels": 1,
        "input_size": 224,
        "normalization_mean": list(
            NORMALIZATION_MEAN
        ),
        "normalization_std": list(
            NORMALIZATION_STD
        ),
        "positive_class": (
            "supported_sea_sar"
        ),
        "negative_class": (
            "unsupported_input"
        ),
        "training_config": vars(args),
        "calibration_used": False,
        "test_locked_used": False,
    }

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    torch.save(
        payload,
        temporary,
    )

    temporary.replace(
        destination
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        payload = torch.load(
            path,
            map_location=device,
        )

    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )

    return payload


def calculate_group_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        source_dataset,
        source_group,
    ), subset in predictions.groupby(
        [
            "source_dataset",
            "source_group",
        ],
        dropna=False,
    ):
        labels = subset[
            "binary_label"
        ].to_numpy(
            dtype=np.int64
        )

        probabilities = subset[
            "supported_probability"
        ].to_numpy(
            dtype=np.float64
        )

        metrics = calculate_metrics(
            labels,
            probabilities,
            threshold=0.50,
        )

        rows.append(
            {
                "source_dataset": str(
                    source_dataset
                ),
                "source_group": str(
                    source_group
                ),
                "sample_count": int(
                    len(subset)
                ),
                "supported_count": int(
                    (labels == 1).sum()
                ),
                "unsupported_count": int(
                    (labels == 0).sum()
                ),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def save_plots(
    history: pd.DataFrame,
) -> None:
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Train loss",
    )

    plt.plot(
        history["epoch"],
        history["validation_loss"],
        marker="o",
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Input-Gate Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        LOSS_PLOT_PATH,
        dpi=180,
    )

    plt.close()

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["epoch"],
        history[
            "validation_supported_recall"
        ],
        marker="o",
        label="Supported recall",
    )

    plt.plot(
        history["epoch"],
        history[
            "validation_unsupported_rejection_rate"
        ],
        marker="o",
        label="Unsupported rejection",
    )

    plt.plot(
        history["epoch"],
        history[
            "validation_balanced_accuracy"
        ],
        marker="o",
        label="Balanced accuracy",
    )

    plt.plot(
        history["epoch"],
        history[
            "validation_f1"
        ],
        marker="o",
        label="F1",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.ylim(0.0, 1.02)
    plt.title("Input-Gate Validation Metrics")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        METRICS_PLOT_PATH,
        dpi=180,
    )

    plt.close()


def write_report(
    summary: dict[str, Any],
) -> None:
    metrics = summary[
        "best_validation_metrics"
    ]

    lines = [
        "# v0.4 Input-Gate Model Eğitimi",
        "",
        "## Amaç",
        "",
        "Petrol analizinden önce yüklenen görüntünün "
        "DARTIS benzeri deniz SAR veri alanında olup "
        "olmadığını belirlemek.",
        "",
        "## Kullanılan split'ler",
        "",
        f"- Train: {summary['train_samples']}",
        f"- Validation: {summary['validation_samples']}",
        "- Calibration: kullanılmadı",
        "- Test locked: kullanılmadı",
        "",
        "## Validation sonucu — geçici threshold 0.50",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}",
        f"- Desteklenen deniz SAR recall: "
        f"{metrics['supported_recall']:.4f}",
        f"- Desteklenmeyen giriş reddetme oranı: "
        f"{metrics['unsupported_rejection_rate']:.4f}",
        f"- False acceptance rate: "
        f"{metrics['false_acceptance_rate']:.4f}",
        f"- False rejection rate: "
        f"{metrics['false_rejection_rate']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        "",
        "## Bilimsel durum",
        "",
        "Bu aşamada kullanılan 0.50 eşiği nihai kabul eşiği değildir.",
        "",
        "Sonraki aşamada calibration split üzerinde muhafazakâr "
        "kabul, red ve belirsiz bölgeleri belirlenecektir.",
        "",
        "Kilitli testteki airplane, harbor, river ve runway "
        "kategorileri henüz kullanılmamıştır.",
        "",
        "Bu model tek başına denizin fiziksel varlığını garanti "
        "etmez. Sonraki aşamada sınıflandırıcı skoruna ek olarak "
        "desteklenen SAR dağılımına uzaklık kontrolü eklenecektir.",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    set_seed(
        args.seed
    )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Input-gate manifest bulunamadı: "
            f"{MANIFEST_PATH}"
        )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = {
        "sample_id",
        "binary_label",
        "split",
        "source_dataset",
        "source_group",
        "image_path",
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

    train_dataframe = dataframe[
        dataframe["split"].eq(
            "train"
        )
    ].copy()

    validation_dataframe = dataframe[
        dataframe["split"].eq(
            "validation"
        )
    ].copy()

    if train_dataframe.empty:
        raise RuntimeError(
            "Train split boş."
        )

    if validation_dataframe.empty:
        raise RuntimeError(
            "Validation split boş."
        )

    for split_name, subset in (
        ("train", train_dataframe),
        (
            "validation",
            validation_dataframe,
        ),
    ):
        labels = set(
            subset[
                "binary_label"
            ].astype(int).unique()
        )

        if labels != {0, 1}:
            raise RuntimeError(
                f"{split_name} split iki sınıfı "
                f"da içermiyor: {sorted(labels)}"
            )

    missing_files = []

    for path_text in pd.concat(
        [
            train_dataframe[
                "image_path"
            ],
            validation_dataframe[
                "image_path"
            ],
        ],
        ignore_index=True,
    ):
        path = resolve_path(
            path_text
        )

        if not path.exists():
            missing_files.append(
                str(path)
            )

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} görüntü bulunamadı.\n"
            + "\n".join(
                missing_files[:15]
            )
        )

    (
        training_transform,
        evaluation_transform,
    ) = build_transforms()

    train_dataset = InputGateDataset(
        train_dataframe,
        training_transform,
    )

    validation_dataset = InputGateDataset(
        validation_dataframe,
        evaluation_transform,
    )

    class_counts = (
        train_dataframe[
            "binary_label"
        ]
        .astype(int)
        .value_counts()
        .to_dict()
    )

    sample_weights_array = (
        train_dataframe[
            "binary_label"
        ]
        .astype(int)
        .map(
            lambda label:
            1.0 / float(
                class_counts[label]
            )
        )
        .to_numpy(
            dtype=np.float64,
            copy=True,
        )
    )

    sample_weights = torch.tensor(
        sample_weights_array,
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(
            train_dataset
        ),
        replacement=True,
        generator=(
            torch.Generator()
            .manual_seed(
                args.seed
            )
        ),
    )

    device = select_device(
        args.device
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.workers > 0
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.workers > 0
        ),
    )

    model, initialization = build_model(
        use_pretrained=(
            not args.no_pretrained
        )
    )

    model.to(device)

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    )

    amp_enabled = (
        device.type == "cuda"
    )

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )

    except (AttributeError, TypeError):
        scaler = (
            torch.cuda.amp.GradScaler(
                enabled=amp_enabled
            )
        )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.4"
    )
    print(
        "INPUT-GATE MODEL EĞİTİMİ"
    )
    print("=" * 78)

    print("Cihaz:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Model:",
        "ResNet18 grayscale input gate",
    )

    print(
        "Başlangıç:",
        initialization,
    )

    print(
        "Parametre:",
        f"{parameter_count:,}",
    )

    print(
        "Train:",
        len(train_dataset),
    )

    print(
        "Validation:",
        len(validation_dataset),
    )

    print(
        "Train desteklenen:",
        int(
            train_dataframe[
                "binary_label"
            ].eq(1).sum()
        ),
    )

    print(
        "Train desteklenmeyen:",
        int(
            train_dataframe[
                "binary_label"
            ].eq(0).sum()
        ),
    )

    print(
        "Calibration kullanımı: YOK"
    )

    print(
        "Test locked kullanımı: YOK"
    )

    print()

    history_rows = []

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
        )

        (
            validation_loss,
            validation_metrics,
            _,
        ) = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            threshold=0.50,
        )

        score = float(
            validation_metrics[
                "balanced_accuracy"
            ]
        )

        scheduler.step(score)

        learning_rate = float(
            optimizer.param_groups[
                0
            ]["lr"]
        )

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": float(
                train_loss
            ),
            "validation_loss": float(
                validation_loss
            ),
            "validation_accuracy": (
                validation_metrics[
                    "accuracy"
                ]
            ),
            "validation_balanced_accuracy": (
                validation_metrics[
                    "balanced_accuracy"
                ]
            ),
            "validation_supported_recall": (
                validation_metrics[
                    "supported_recall"
                ]
            ),
            "validation_unsupported_rejection_rate": (
                validation_metrics[
                    "unsupported_rejection_rate"
                ]
            ),
            "validation_false_acceptance_rate": (
                validation_metrics[
                    "false_acceptance_rate"
                ]
            ),
            "validation_false_rejection_rate": (
                validation_metrics[
                    "false_rejection_rate"
                ]
            ),
            "validation_f1": (
                validation_metrics[
                    "f1"
                ]
            ),
        }

        history_rows.append(row)

        save_checkpoint(
            LAST_CHECKPOINT,
            epoch,
            model,
            optimizer,
            scheduler,
            validation_metrics,
            args,
            initialization,
        )

        improved = (
            score > best_score + 1e-6
        )

        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                BEST_CHECKPOINT,
                epoch,
                model,
                optimizer,
                scheduler,
                validation_metrics,
                args,
                initialization,
            )

        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} "
            f"| train_loss={train_loss:.4f} "
            f"| val_loss={validation_loss:.4f} "
            f"| bal_acc="
            f"{validation_metrics['balanced_accuracy']:.4f} "
            f"| SAR_recall="
            f"{validation_metrics['supported_recall']:.4f} "
            f"| reject="
            f"{validation_metrics['unsupported_rejection_rate']:.4f} "
            f"| FAR="
            f"{validation_metrics['false_acceptance_rate']:.4f} "
            f"| lr={learning_rate:.2e}"
            f"{' BEST' if improved else ''}"
        )

        pd.DataFrame(
            history_rows
        ).to_csv(
            HISTORY_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print()
            print(
                "Early stopping:",
                f"{args.patience} epoch boyunca "
                "iyileşme olmadı.",
            )

            break

    checkpoint_payload = load_checkpoint(
        BEST_CHECKPOINT,
        model,
        device,
    )

    (
        best_validation_loss,
        best_validation_metrics,
        validation_predictions,
    ) = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        threshold=0.50,
    )

    validation_metadata = (
        validation_dataframe[
            [
                "sample_id",
                "source_dataset",
                "source_group",
                "scene_id",
                "image_path",
            ]
        ]
        .drop_duplicates(
            subset=["sample_id"]
        )
    )

    validation_predictions = (
        validation_predictions.merge(
            validation_metadata,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    )

    validation_predictions.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    group_metrics = calculate_group_metrics(
        validation_predictions
    )

    group_metrics.to_csv(
        GROUP_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    history_dataframe = pd.DataFrame(
        history_rows
    )

    save_plots(
        history_dataframe
    )

    summary = {
        "architecture": (
            "resnet18_grayscale_input_gate"
        ),
        "initialization": initialization,
        "parameter_count": int(
            parameter_count
        ),
        "device": str(device),
        "train_samples": int(
            len(train_dataset)
        ),
        "validation_samples": int(
            len(validation_dataset)
        ),
        "best_epoch": int(
            best_epoch
        ),
        "checkpoint_epoch": int(
            checkpoint_payload["epoch"]
        ),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "best_validation_metrics": (
            best_validation_metrics
        ),
        "temporary_threshold": 0.50,
        "calibration_used": False,
        "test_locked_used": False,
        "best_checkpoint": str(
            BEST_CHECKPOINT
        ),
        "validation_predictions": str(
            VALIDATION_PREDICTIONS_PATH
        ),
        "validation_group_metrics": str(
            GROUP_METRICS_PATH
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_report(summary)

    print()
    print("=" * 78)
    print(
        "INPUT-GATE EĞİTİM SONUCU"
    )
    print("=" * 78)

    print(
        "En iyi epoch:",
        best_epoch,
    )

    print(
        "Validation loss:",
        f"{best_validation_loss:.6f}",
    )

    print(
        "Balanced accuracy:",
        f"{best_validation_metrics['balanced_accuracy']:.4f}",
    )

    print(
        "Desteklenen SAR recall:",
        f"{best_validation_metrics['supported_recall']:.4f}",
    )

    print(
        "Desteklenmeyen reddetme:",
        f"{best_validation_metrics['unsupported_rejection_rate']:.4f}",
    )

    print(
        "False acceptance rate:",
        f"{best_validation_metrics['false_acceptance_rate']:.4f}",
    )

    print(
        "False rejection rate:",
        f"{best_validation_metrics['false_rejection_rate']:.4f}",
    )

    print()
    print(
        "Best checkpoint:",
        BEST_CHECKPOINT.resolve(),
    )

    print(
        "Validation tahminleri:",
        VALIDATION_PREDICTIONS_PATH.resolve(),
    )

    print(
        "Grup metrikleri:",
        GROUP_METRICS_PATH.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    print(
        "Rapor:",
        REPORT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
