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
    / "v03_verifier_dataset.csv"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "verifier_v03"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "verifier_v03"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "last.pth"
)

HISTORY_JSON = (
    OUTPUT_DIR
    / "training_history.json"
)

HISTORY_CSV = (
    OUTPUT_DIR
    / "training_history.csv"
)

VALIDATION_PREDICTIONS = (
    OUTPUT_DIR
    / "validation_predictions.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "training_summary.json"
)

MEAN = (0.449,)
STD = (0.226,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Robust Binary Oil Detector v0.3 "
            "ikinci aşama verifier modelini eğitir."
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
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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
        "--patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
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
        help=(
            "ImageNet ağırlıklarını kullanmadan "
            "ResNet18'i sıfırdan başlatır."
        ),
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def select_device(
    requested: str,
) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
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


def resolve_path(value: Any) -> Path:
    text = str(value).strip()

    path = Path(text)

    if not path.is_absolute():
        path = ROOT / path

    return path


class VerifierDataset(Dataset):
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
        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        str,
    ]:
        row = self.dataframe.iloc[
            index
        ]

        image_path = resolve_path(
            row["crop_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Crop bulunamadı: {image_path}"
            )

        with Image.open(
            image_path
        ) as source_image:
            image = source_image.convert(
                "L"
            )

        image_tensor = self.transform(
            image
        )

        label = torch.tensor(
            float(
                row[
                    "binary_label"
                ]
            ),
            dtype=torch.float32,
        )

        sample_id = str(
            row["sample_id"]
        )

        return (
            image_tensor,
            label,
            sample_id,
        )


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
            ),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.RandomVerticalFlip(
                p=0.5
            ),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.03, 0.03),
                scale=(0.97, 1.03),
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD,
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
                mean=MEAN,
                std=STD,
            ),
        ]
    )

    return (
        train_transform,
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
            print(
                "ImageNet ResNet18 ağırlıkları yükleniyor..."
            )

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
                "UYARI: ImageNet ağırlıkları "
                "yüklenemedi."
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                "Model rastgele ağırlıklarla "
                "başlatılacak."
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

    input_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        input_features,
        1,
    )

    return model, initialization


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    labels_int = labels.astype(
        np.int64
    )

    true_positive = int(
        (
            (predictions == 1)
            & (labels_int == 1)
        ).sum()
    )

    true_negative = int(
        (
            (predictions == 0)
            & (labels_int == 0)
        ).sum()
    )

    false_positive = int(
        (
            (predictions == 1)
            & (labels_int == 0)
        ).sum()
    )

    false_negative = int(
        (
            (predictions == 0)
            & (labels_int == 1)
        ).sum()
    )

    precision = safe_divide(
        true_positive,
        true_positive
        + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive
        + false_negative,
    )

    specificity = safe_divide(
        true_negative,
        true_negative
        + false_positive,
    )

    accuracy = safe_divide(
        true_positive
        + true_negative,
        len(labels_int),
    )

    f1 = safe_divide(
        2.0
        * precision
        * recall,
        precision
        + recall,
    )

    balanced_accuracy = (
        recall
        + specificity
    ) / 2.0

    false_positive_rate = safe_divide(
        false_positive,
        false_positive
        + true_negative,
    )

    false_negative_rate = safe_divide(
        false_negative,
        false_negative
        + true_positive,
    )

    return {
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "specificity": float(
            specificity
        ),
        "f1": float(
            f1
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
        "true_positive": int(
            true_positive
        ),
        "true_negative": int(
            true_negative
        ),
        "false_positive": int(
            false_positive
        ),
        "false_negative": int(
            false_negative
        ),
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

        batch_size = images.shape[0]

        running_loss += (
            float(
                loss.item()
            )
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
    threshold: float,
) -> tuple[
    float,
    dict[str, float],
    pd.DataFrame,
]:
    model.eval()

    amp_enabled = (
        device.type == "cuda"
    )

    running_loss = 0.0
    sample_count = 0

    labels_all: list[float] = []
    probabilities_all: list[float] = []
    sample_ids_all: list[str] = []

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

        batch_size = images.shape[0]

        running_loss += (
            float(
                loss.item()
            )
            * batch_size
        )

        sample_count += batch_size

        labels_all.extend(
            labels.numpy().tolist()
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

    probability_array = np.asarray(
        probabilities_all,
        dtype=np.float32,
    )

    metrics = calculate_metrics(
        labels_array,
        probability_array,
        threshold,
    )

    prediction_dataframe = pd.DataFrame(
        {
            "sample_id": (
                sample_ids_all
            ),
            "binary_label": (
                labels_array.astype(
                    np.int64
                )
            ),
            "probability": (
                probability_array
            ),
            "prediction": (
                probability_array
                >= threshold
            ).astype(
                np.int64
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
    metrics: dict[str, float],
    args: argparse.Namespace,
    initialization: str,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "epoch": int(
            epoch
        ),
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
            "resnet18_grayscale_binary"
        ),
        "initialization": (
            initialization
        ),
        "input_channels": 1,
        "input_size": 224,
        "normalization_mean": (
            list(MEAN)
        ),
        "normalization_std": (
            list(STD)
        ),
        "config": vars(args),
    }

    temporary_path = (
        destination.with_suffix(
            destination.suffix
            + ".part"
        )
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(
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
        payload[
            "model_state_dict"
        ]
    )

    return payload


def save_history_plots(
    history: list[dict[str, Any]],
) -> None:
    dataframe = pd.DataFrame(
        history
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        dataframe["epoch"],
        dataframe["train_loss"],
        marker="o",
        label="Train loss",
    )

    plt.plot(
        dataframe["epoch"],
        dataframe["validation_loss"],
        marker="o",
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Verifier Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "loss_curve.png",
        dpi=180,
    )

    plt.close()

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        dataframe["epoch"],
        dataframe[
            "validation_balanced_accuracy"
        ],
        marker="o",
        label="Balanced accuracy",
    )

    plt.plot(
        dataframe["epoch"],
        dataframe[
            "validation_f1"
        ],
        marker="o",
        label="F1",
    )

    plt.plot(
        dataframe["epoch"],
        dataframe[
            "validation_recall"
        ],
        marker="o",
        label="Recall",
    )

    plt.plot(
        dataframe["epoch"],
        dataframe[
            "validation_specificity"
        ],
        marker="o",
        label="Specificity",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.ylim(0.0, 1.02)
    plt.title(
        "Verifier Validation Metrics"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "validation_metrics.png",
        dpi=180,
    )

    plt.close()


def main() -> None:
    args = parse_args()

    set_seed(
        args.seed
    )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Verifier manifest bulunamadı: "
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

    dataframe = pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = {
        "sample_id",
        "binary_label",
        "split",
        "sampling_weight",
        "crop_path",
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

    missing_files = []

    for path_text in dataframe[
        "crop_path"
    ]:
        path = resolve_path(
            path_text
        )

        if not path.exists():
            missing_files.append(
                str(path)
            )

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} crop dosyası eksik.\n"
            + "\n".join(
                missing_files[:10]
            )
        )

    (
        train_transform,
        evaluation_transform,
    ) = build_transforms()

    train_dataset = VerifierDataset(
        train_dataframe,
        train_transform,
    )

    validation_dataset = VerifierDataset(
        validation_dataframe,
        evaluation_transform,
    )

    sampler_weights = torch.as_tensor(
        train_dataframe[
            "sampling_weight"
        ].astype(float).to_numpy(),
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(
            train_dataset
        ),
        replacement=True,
        generator=torch.Generator().manual_seed(
            args.seed
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

    model = model.to(
        device
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

    criterion = (
        nn.BCEWithLogitsLoss()
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
        "ROBUST BINARY OIL DETECTOR v0.3"
    )
    print(
        "VERIFIER MODEL EĞİTİMİ"
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
        "ResNet18 grayscale binary",
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
        "Train örneği:",
        len(train_dataset),
    )

    print(
        "Validation örneği:",
        len(validation_dataset),
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Epoch:",
        args.epochs,
    )

    print(
        "Threshold:",
        args.threshold,
    )

    print(
        "Calibration kullanımı: YOK"
    )

    print(
        "Test locked kullanımı: YOK"
    )

    print()

    history = []

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )

        (
            validation_loss,
            validation_metrics,
            _,
        ) = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
        )

        validation_score = (
            validation_metrics[
                "balanced_accuracy"
            ]
        )

        scheduler.step(
            validation_score
        )

        learning_rate = float(
            optimizer.param_groups[
                0
            ]["lr"]
        )

        history_row = {
            "epoch": epoch,
            "learning_rate": (
                learning_rate
            ),
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
            "validation_precision": (
                validation_metrics[
                    "precision"
                ]
            ),
            "validation_recall": (
                validation_metrics[
                    "recall"
                ]
            ),
            "validation_specificity": (
                validation_metrics[
                    "specificity"
                ]
            ),
            "validation_f1": (
                validation_metrics[
                    "f1"
                ]
            ),
            "validation_false_positive_rate": (
                validation_metrics[
                    "false_positive_rate"
                ]
            ),
            "validation_false_negative_rate": (
                validation_metrics[
                    "false_negative_rate"
                ]
            ),
        }

        history.append(
            history_row
        )

        save_checkpoint(
            destination=LAST_CHECKPOINT,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=validation_metrics,
            args=args,
            initialization=initialization,
        )

        improved = (
            validation_score
            > best_score
            + 1e-6
        )

        if improved:
            best_score = (
                validation_score
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                destination=(
                    BEST_CHECKPOINT
                ),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=(
                    validation_metrics
                ),
                args=args,
                initialization=(
                    initialization
                ),
            )

        else:
            epochs_without_improvement += 1

        marker = (
            " BEST"
            if improved
            else ""
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} "
            f"| train_loss={train_loss:.4f} "
            f"| val_loss={validation_loss:.4f} "
            f"| bal_acc="
            f"{validation_metrics['balanced_accuracy']:.4f} "
            f"| F1="
            f"{validation_metrics['f1']:.4f} "
            f"| P="
            f"{validation_metrics['precision']:.4f} "
            f"| R="
            f"{validation_metrics['recall']:.4f} "
            f"| Spec="
            f"{validation_metrics['specificity']:.4f} "
            f"| lr={learning_rate:.2e}"
            f"{marker}"
        )

        with HISTORY_JSON.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                history,
                file,
                indent=2,
                ensure_ascii=False,
            )

        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print()
            print(
                "Early stopping: "
                f"{args.patience} epoch boyunca "
                "iyileşme olmadı."
            )

            break

    save_history_plots(
        history
    )

    best_payload = load_checkpoint(
        BEST_CHECKPOINT,
        model,
        device,
    )

    (
        best_validation_loss,
        best_validation_metrics,
        best_predictions,
    ) = evaluate(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
        threshold=args.threshold,
    )

    validation_metadata = (
        validation_dataframe[
            [
                "sample_id",
                "scene_id",
                "source_group",
                "source_image_name",
                "crop_type",
                "is_hard_negative",
                "crop_path",
            ]
        ]
        .drop_duplicates(
            subset=[
                "sample_id"
            ]
        )
    )

    best_predictions = (
        best_predictions.merge(
            validation_metadata,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    )

    best_predictions.to_csv(
        VALIDATION_PREDICTIONS,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "architecture": (
            "resnet18_grayscale_binary"
        ),
        "initialization": (
            initialization
        ),
        "device": str(
            device
        ),
        "parameter_count": int(
            parameter_count
        ),
        "train_samples": int(
            len(train_dataset)
        ),
        "validation_samples": int(
            len(validation_dataset)
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_checkpoint_epoch": int(
            best_payload["epoch"]
        ),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "best_validation_metrics": (
            best_validation_metrics
        ),
        "threshold": float(
            args.threshold
        ),
        "calibration_used": False,
        "test_locked_used": False,
        "best_checkpoint": str(
            BEST_CHECKPOINT
        ),
        "last_checkpoint": str(
            LAST_CHECKPOINT
        ),
        "validation_predictions": str(
            VALIDATION_PREDICTIONS
        ),
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 78)
    print("VERIFIER EĞİTİM SONUCU")
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
        "Precision:",
        f"{best_validation_metrics['precision']:.4f}",
    )

    print(
        "Recall:",
        f"{best_validation_metrics['recall']:.4f}",
    )

    print(
        "Specificity:",
        f"{best_validation_metrics['specificity']:.4f}",
    )

    print(
        "F1:",
        f"{best_validation_metrics['f1']:.4f}",
    )

    print(
        "False-positive rate:",
        f"{best_validation_metrics['false_positive_rate']:.4f}",
    )

    print(
        "False-negative rate:",
        f"{best_validation_metrics['false_negative_rate']:.4f}",
    )

    print()
    print(
        "Best checkpoint:",
        BEST_CHECKPOINT.resolve(),
    )

    print(
        "Validation tahminleri:",
        VALIDATION_PREDICTIONS.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_JSON.resolve(),
    )


if __name__ == "__main__":
    main()
