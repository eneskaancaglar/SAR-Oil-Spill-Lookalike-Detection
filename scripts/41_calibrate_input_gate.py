from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v04_input_gate_dataset.csv"
)

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "input_gate_v04"
    / "best_input_gate.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v04_input_gate_calibration"
)

CONFIG_PATH = (
    ROOT
    / "checkpoints"
    / "input_gate_v04"
    / "input_gate_config.json"
)

PROTOTYPES_PATH = (
    ROOT
    / "checkpoints"
    / "input_gate_v04"
    / "supported_prototypes.npz"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "calibration_predictions.csv"
)

GROUP_METRICS_PATH = (
    OUTPUT_DIR
    / "calibration_group_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "calibration_summary.json"
)

SCATTER_PATH = (
    OUTPUT_DIR
    / "probability_similarity_scatter.png"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v04_input_gate_calibration.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Input-gate modelini calibration split üzerinde "
            "KABUL, BELİRSİZ ve RED kararları için ayarlar."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--maximum-false-accept-rate",
        type=float,
        default=0.0,
        help=(
            "Calibration optik görüntülerinin kabul edilmesine "
            "izin verilen en yüksek oran."
        ),
    )

    parser.add_argument(
        "--maximum-false-reject-rate",
        type=float,
        default=0.01,
        help=(
            "Calibration deniz SAR görüntülerinin doğrudan "
            "reddedilmesine izin verilen en yüksek oran."
        ),
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

    return parser.parse_args()


def resolve_path(value: Any) -> Path:
    path = Path(str(value).strip())

    if not path.is_absolute():
        path = ROOT / path

    return path


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


class GateDataset(Dataset):
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
    ):
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

        return (
            self.transform(image),
            torch.tensor(
                float(row["binary_label"]),
                dtype=torch.float32,
            ),
            str(row["sample_id"]),
            str(row["source_group"]),
        )


def build_model(
    device: torch.device,
):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {CHECKPOINT_PATH}"
        )

    try:
        payload = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        payload = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

    model = resnet18(
        weights=None
    )

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(
            model.fc.in_features,
            1,
        ),
    )

    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, payload


def forward_features(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)

    return torch.flatten(
        x,
        1,
    )


@torch.inference_mode()
def extract_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
]:
    logits_all = []
    features_all = []
    labels_all = []
    sample_ids_all = []
    groups_all = []

    amp_enabled = (
        device.type == "cuda"
    )

    for (
        images,
        labels,
        sample_ids,
        groups,
    ) in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            features = forward_features(
                model,
                images,
            )

            logits = model.fc(
                features
            ).flatten()

        normalized_features = F.normalize(
            features.float(),
            p=2,
            dim=1,
        )

        logits_all.append(
            logits
            .float()
            .cpu()
            .numpy()
        )

        features_all.append(
            normalized_features
            .cpu()
            .numpy()
        )

        labels_all.append(
            labels.numpy()
        )

        sample_ids_all.extend(
            list(sample_ids)
        )

        groups_all.extend(
            list(groups)
        )

    return (
        np.concatenate(
            logits_all
        ).astype(
            np.float64
        ),
        np.concatenate(
            features_all
        ).astype(
            np.float32
        ),
        np.concatenate(
            labels_all
        ).astype(
            np.int64
        ),
        sample_ids_all,
        groups_all,
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

    log_temperature = nn.Parameter(
        torch.zeros(
            (),
            device=device,
        )
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.05,
        max_iter=200,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        )

        loss = (
            F.binary_cross_entropy_with_logits(
                logits_tensor / temperature,
                labels_tensor,
            )
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    temperature = float(
        torch.exp(
            log_temperature
        )
        .detach()
        .cpu()
        .item()
    )

    return float(
        np.clip(
            temperature,
            0.05,
            20.0,
        )
    )


def sigmoid(
    values: np.ndarray,
) -> np.ndarray:
    values = np.clip(
        values,
        -60.0,
        60.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-values)
        )
    )


def build_prototypes(
    features: np.ndarray,
    groups: list[str],
) -> tuple[
    list[str],
    np.ndarray,
]:
    group_array = np.asarray(
        groups,
        dtype=str,
    )

    prototype_names = []
    prototype_vectors = []

    for group_name in sorted(
        np.unique(
            group_array
        )
    ):
        group_features = features[
            group_array == group_name
        ]

        if len(group_features) == 0:
            continue

        prototype = (
            group_features.mean(
                axis=0
            )
        )

        norm = float(
            np.linalg.norm(
                prototype
            )
        )

        if norm <= 0:
            continue

        prototype = (
            prototype
            / norm
        )

        prototype_names.append(
            str(group_name)
        )

        prototype_vectors.append(
            prototype.astype(
                np.float32
            )
        )

    if not prototype_vectors:
        raise RuntimeError(
            "Desteklenen SAR prototipi üretilemedi."
        )

    return (
        prototype_names,
        np.stack(
            prototype_vectors,
            axis=0,
        ),
    )


def maximum_similarity(
    features: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    similarities = (
        features
        @ prototypes.T
    )

    best_indices = np.argmax(
        similarities,
        axis=1,
    )

    best_values = similarities[
        np.arange(
            len(similarities)
        ),
        best_indices,
    ]

    return (
        best_values.astype(
            np.float64
        ),
        best_indices.astype(
            np.int64
        ),
    )


def create_similarity_candidates(
    similarities: np.ndarray,
) -> np.ndarray:
    quantiles = np.linspace(
        0.0,
        1.0,
        101,
    )

    candidates = np.quantile(
        similarities,
        quantiles,
    )

    candidates = np.concatenate(
        [
            candidates,
            np.asarray(
                [
                    -1.0,
                    0.0,
                    0.25,
                    0.50,
                    0.75,
                    1.0,
                ]
            ),
        ]
    )

    return np.unique(
        np.clip(
            candidates,
            -1.0,
            1.0,
        )
    )


def select_accept_thresholds(
    probabilities: np.ndarray,
    similarities: np.ndarray,
    labels: np.ndarray,
    maximum_false_accept_rate: float,
) -> tuple[
    float,
    float,
    dict[str, float],
]:
    positive_mask = (
        labels == 1
    )

    negative_mask = (
        labels == 0
    )

    probability_candidates = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.50,
                    0.999,
                    101,
                ),
                np.quantile(
                    probabilities,
                    np.linspace(
                        0.0,
                        1.0,
                        101,
                    ),
                ),
            ]
        )
    )

    similarity_candidates = (
        create_similarity_candidates(
            similarities
        )
    )

    best = None

    for probability_threshold in probability_candidates:
        probability_pass = (
            probabilities
            >= probability_threshold
        )

        for similarity_threshold in similarity_candidates:
            accepted = (
                probability_pass
                & (
                    similarities
                    >= similarity_threshold
                )
            )

            supported_acceptance = float(
                accepted[
                    positive_mask
                ].mean()
            )

            false_acceptance = float(
                accepted[
                    negative_mask
                ].mean()
            )

            if (
                false_acceptance
                > maximum_false_accept_rate
                + 1e-12
            ):
                continue

            score = (
                supported_acceptance,
                float(
                    probability_threshold
                ),
                float(
                    similarity_threshold
                ),
            )

            if (
                best is None
                or score > best["score"]
            ):
                best = {
                    "score": score,
                    "probability_threshold": float(
                        probability_threshold
                    ),
                    "similarity_threshold": float(
                        similarity_threshold
                    ),
                    "supported_acceptance_rate": (
                        supported_acceptance
                    ),
                    "false_acceptance_rate": (
                        false_acceptance
                    ),
                }

    if best is None:
        raise RuntimeError(
            "Kabul threshold kombinasyonu bulunamadı."
        )

    return (
        best[
            "probability_threshold"
        ],
        best[
            "similarity_threshold"
        ],
        best,
    )


def select_reject_thresholds(
    probabilities: np.ndarray,
    similarities: np.ndarray,
    labels: np.ndarray,
    accept_probability: float,
    accept_similarity: float,
    maximum_false_reject_rate: float,
) -> tuple[
    float,
    float,
    dict[str, float],
]:
    positive_mask = (
        labels == 1
    )

    negative_mask = (
        labels == 0
    )

    probability_candidates = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.001,
                    min(
                        0.50,
                        accept_probability
                        - 1e-6,
                    ),
                    101,
                ),
                np.quantile(
                    probabilities,
                    np.linspace(
                        0.0,
                        1.0,
                        101,
                    ),
                ),
            ]
        )
    )

    probability_candidates = (
        probability_candidates[
            probability_candidates
            < accept_probability
        ]
    )

    similarity_candidates = (
        create_similarity_candidates(
            similarities
        )
    )

    similarity_candidates = (
        similarity_candidates[
            similarity_candidates
            < accept_similarity
        ]
    )

    if len(
        similarity_candidates
    ) == 0:
        similarity_candidates = (
            np.asarray(
                [
                    accept_similarity
                    - 1e-6
                ]
            )
        )

    best = None

    for probability_threshold in probability_candidates:
        probability_reject = (
            probabilities
            <= probability_threshold
        )

        for similarity_threshold in similarity_candidates:
            rejected = (
                probability_reject
                | (
                    similarities
                    <= similarity_threshold
                )
            )

            unsupported_rejection = float(
                rejected[
                    negative_mask
                ].mean()
            )

            false_rejection = float(
                rejected[
                    positive_mask
                ].mean()
            )

            if (
                false_rejection
                > maximum_false_reject_rate
                + 1e-12
            ):
                continue

            score = (
                unsupported_rejection,
                -float(
                    probability_threshold
                ),
                -float(
                    similarity_threshold
                ),
            )

            if (
                best is None
                or score > best["score"]
            ):
                best = {
                    "score": score,
                    "probability_threshold": float(
                        probability_threshold
                    ),
                    "similarity_threshold": float(
                        similarity_threshold
                    ),
                    "unsupported_rejection_rate": (
                        unsupported_rejection
                    ),
                    "false_rejection_rate": (
                        false_rejection
                    ),
                }

    if best is None:
        raise RuntimeError(
            "Red threshold kombinasyonu bulunamadı."
        )

    return (
        best[
            "probability_threshold"
        ],
        best[
            "similarity_threshold"
        ],
        best,
    )


def assign_decisions(
    probabilities: np.ndarray,
    similarities: np.ndarray,
    accept_probability: float,
    accept_similarity: float,
    reject_probability: float,
    reject_similarity: float,
) -> np.ndarray:
    accepted = (
        (probabilities >= accept_probability)
        & (
            similarities
            >= accept_similarity
        )
    )

    rejected = (
        (probabilities <= reject_probability)
        | (
            similarities
            <= reject_similarity
        )
    )

    decisions = np.full(
        len(probabilities),
        "UNCERTAIN",
        dtype=object,
    )

    decisions[
        rejected
    ] = "REJECT"

    decisions[
        accepted
    ] = "ACCEPT"

    return decisions


def calculate_decision_metrics(
    labels: np.ndarray,
    decisions: np.ndarray,
) -> dict[str, Any]:
    positive_mask = (
        labels == 1
    )

    negative_mask = (
        labels == 0
    )

    return {
        "supported_samples": int(
            positive_mask.sum()
        ),
        "unsupported_samples": int(
            negative_mask.sum()
        ),
        "supported_accept_rate": float(
            (
                decisions[
                    positive_mask
                ]
                == "ACCEPT"
            ).mean()
        ),
        "supported_uncertain_rate": float(
            (
                decisions[
                    positive_mask
                ]
                == "UNCERTAIN"
            ).mean()
        ),
        "supported_false_reject_rate": float(
            (
                decisions[
                    positive_mask
                ]
                == "REJECT"
            ).mean()
        ),
        "unsupported_reject_rate": float(
            (
                decisions[
                    negative_mask
                ]
                == "REJECT"
            ).mean()
        ),
        "unsupported_uncertain_rate": float(
            (
                decisions[
                    negative_mask
                ]
                == "UNCERTAIN"
            ).mean()
        ),
        "unsupported_false_accept_rate": float(
            (
                decisions[
                    negative_mask
                ]
                == "ACCEPT"
            ).mean()
        ),
        "overall_accept_count": int(
            (
                decisions
                == "ACCEPT"
            ).sum()
        ),
        "overall_uncertain_count": int(
            (
                decisions
                == "UNCERTAIN"
            ).sum()
        ),
        "overall_reject_count": int(
            (
                decisions
                == "REJECT"
            ).sum()
        ),
    }


def group_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        source_dataset,
        source_group,
    ), subset in dataframe.groupby(
        [
            "source_dataset",
            "source_group",
        ],
        dropna=False,
    ):
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
                    subset[
                        "binary_label"
                    ].eq(1).sum()
                ),
                "unsupported_count": int(
                    subset[
                        "binary_label"
                    ].eq(0).sum()
                ),
                "accepted": int(
                    subset[
                        "decision"
                    ].eq(
                        "ACCEPT"
                    ).sum()
                ),
                "uncertain": int(
                    subset[
                        "decision"
                    ].eq(
                        "UNCERTAIN"
                    ).sum()
                ),
                "rejected": int(
                    subset[
                        "decision"
                    ].eq(
                        "REJECT"
                    ).sum()
                ),
                "mean_supported_probability": float(
                    subset[
                        "supported_probability"
                    ].mean()
                ),
                "mean_prototype_similarity": float(
                    subset[
                        "prototype_similarity"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def save_scatter(
    dataframe: pd.DataFrame,
    accept_probability: float,
    accept_similarity: float,
    reject_probability: float,
    reject_similarity: float,
) -> None:
    plt.figure(
        figsize=(9, 7)
    )

    for label_name, subset in dataframe.groupby(
        "binary_label_name"
    ):
        plt.scatter(
            subset[
                "supported_probability"
            ],
            subset[
                "prototype_similarity"
            ],
            alpha=0.65,
            s=24,
            label=str(
                label_name
            ),
        )

    plt.axvline(
        accept_probability,
        linestyle="--",
        label="Accept probability",
    )

    plt.axhline(
        accept_similarity,
        linestyle="--",
        label="Accept similarity",
    )

    plt.axvline(
        reject_probability,
        linestyle=":",
        label="Reject probability",
    )

    plt.axhline(
        reject_similarity,
        linestyle=":",
        label="Reject similarity",
    )

    plt.xlabel(
        "Calibrated supported-sea-SAR probability"
    )

    plt.ylabel(
        "Maximum DARTIS prototype similarity"
    )

    plt.title(
        "Input-Gate Calibration"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        SCATTER_PATH,
        dpi=180,
    )

    plt.close()


def write_report(
    summary: dict[str, Any],
) -> None:
    metrics = summary[
        "calibration_decision_metrics"
    ]

    lines = [
        "# v0.4 Input-Gate Kalibrasyonu",
        "",
        "## Amaç",
        "",
        "Yüklenen görüntüyü petrol analizine göndermeden önce "
        "muhafazakâr biçimde KABUL, BELİRSİZ veya RED olarak "
        "sınıflandırmak.",
        "",
        "## Karar sistemi",
        "",
        "### KABUL",
        "",
        "Hem deniz-SAR olasılığı hem de DARTIS prototip "
        "benzerliği kabul eşiklerini geçmelidir.",
        "",
        "### RED",
        "",
        "Deniz-SAR olasılığı veya prototip benzerliği red "
        "eşiğinin altında kalırsa görüntü reddedilir.",
        "",
        "### BELİRSİZ",
        "",
        "Kabul edilecek kadar güçlü olmayan ancak doğrudan "
        "reddedilecek kadar da uzak olmayan görüntülerdir. "
        "Petrol analizi çalıştırılmaz.",
        "",
        "## Threshold değerleri",
        "",
        f"- Temperature: "
        f"{summary['temperature']:.6f}",
        f"- Accept probability: "
        f"{summary['accept_probability_threshold']:.6f}",
        f"- Accept similarity: "
        f"{summary['accept_similarity_threshold']:.6f}",
        f"- Reject probability: "
        f"{summary['reject_probability_threshold']:.6f}",
        f"- Reject similarity: "
        f"{summary['reject_similarity_threshold']:.6f}",
        "",
        "## Calibration sonucu",
        "",
        f"- Desteklenen SAR kabul oranı: "
        f"{metrics['supported_accept_rate']:.4f}",
        f"- Desteklenen SAR belirsiz oranı: "
        f"{metrics['supported_uncertain_rate']:.4f}",
        f"- Desteklenen SAR yanlış red oranı: "
        f"{metrics['supported_false_reject_rate']:.4f}",
        f"- Desteklenmeyen giriş red oranı: "
        f"{metrics['unsupported_reject_rate']:.4f}",
        f"- Desteklenmeyen giriş belirsiz oranı: "
        f"{metrics['unsupported_uncertain_rate']:.4f}",
        f"- Desteklenmeyen giriş yanlış kabul oranı: "
        f"{metrics['unsupported_false_accept_rate']:.4f}",
        "",
        "## Bilimsel durum",
        "",
        "Kilitli test bu aşamada kullanılmamıştır.",
        "",
        "Airplane, harbor, river ve runway kategorileri "
        "sonraki aşamada ilk kez kullanılacaktır.",
        "",
        "Validation başarısının yüzde yüz olması sistemin "
        "evrensel biçimde güvenilir olduğunu göstermez. "
        "Asıl karar kilitli testten sonra verilecektir.",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    for required_path in (
        MANIFEST_PATH,
        CHECKPOINT_PATH,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: "
                f"{required_path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFIG_PATH.parent.mkdir(
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

    train_supported = dataframe[
        dataframe[
            "split"
        ].eq("train")
        & dataframe[
            "binary_label"
        ].eq(1)
    ].copy()

    calibration = dataframe[
        dataframe[
            "split"
        ].eq(
            "calibration"
        )
    ].copy()

    if train_supported.empty:
        raise RuntimeError(
            "Train supported SAR bölümü boş."
        )

    if calibration.empty:
        raise RuntimeError(
            "Calibration split boş."
        )

    calibration_labels = set(
        calibration[
            "binary_label"
        ].astype(int).unique()
    )

    if calibration_labels != {
        0,
        1,
    }:
        raise RuntimeError(
            "Calibration split iki sınıfı da içermiyor: "
            f"{sorted(calibration_labels)}"
        )

    device = select_device(
        args.device
    )

    model, checkpoint_payload = (
        build_model(
            device
        )
    )

    mean = tuple(
        checkpoint_payload.get(
            "normalization_mean",
            [0.5],
        )
    )

    std = tuple(
        checkpoint_payload.get(
            "normalization_std",
            [0.25],
        )
    )

    input_size = int(
        checkpoint_payload.get(
            "input_size",
            224,
        )
    )

    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    input_size,
                    input_size,
                ),
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    train_dataset = GateDataset(
        train_supported,
        transform,
    )

    calibration_dataset = GateDataset(
        calibration,
        transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.workers > 0
        ),
    )

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.workers > 0
        ),
    )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.4"
    )
    print(
        "INPUT-GATE CALIBRATION"
    )
    print("=" * 78)

    print(
        "Cihaz:",
        device,
    )

    print(
        "Prototip için train SAR:",
        len(train_dataset),
    )

    print(
        "Calibration örneği:",
        len(calibration_dataset),
    )

    print(
        "Test locked kullanımı: YOK"
    )

    print()

    (
        train_logits,
        train_features,
        train_labels,
        train_sample_ids,
        train_groups,
    ) = extract_outputs(
        model,
        train_loader,
        device,
    )

    (
        calibration_logits,
        calibration_features,
        calibration_labels,
        calibration_sample_ids,
        calibration_groups,
    ) = extract_outputs(
        model,
        calibration_loader,
        device,
    )

    (
        prototype_names,
        prototype_vectors,
    ) = build_prototypes(
        train_features,
        train_groups,
    )

    np.savez_compressed(
        PROTOTYPES_PATH,
        group_names=np.asarray(
            prototype_names
        ),
        centroids=prototype_vectors,
    )

    (
        calibration_similarities,
        best_prototype_indices,
    ) = maximum_similarity(
        calibration_features,
        prototype_vectors,
    )

    temperature = fit_temperature(
        calibration_logits,
        calibration_labels,
        device,
    )

    calibrated_probabilities = sigmoid(
        calibration_logits
        / temperature
    )

    (
        accept_probability,
        accept_similarity,
        accept_selection,
    ) = select_accept_thresholds(
        calibrated_probabilities,
        calibration_similarities,
        calibration_labels,
        args.maximum_false_accept_rate,
    )

    (
        reject_probability,
        reject_similarity,
        reject_selection,
    ) = select_reject_thresholds(
        calibrated_probabilities,
        calibration_similarities,
        calibration_labels,
        accept_probability,
        accept_similarity,
        args.maximum_false_reject_rate,
    )

    decisions = assign_decisions(
        calibrated_probabilities,
        calibration_similarities,
        accept_probability,
        accept_similarity,
        reject_probability,
        reject_similarity,
    )

    decision_metrics = (
        calculate_decision_metrics(
            calibration_labels,
            decisions,
        )
    )

    predictions = pd.DataFrame(
        {
            "sample_id": (
                calibration_sample_ids
            ),
            "binary_label": (
                calibration_labels
            ),
            "raw_logit": (
                calibration_logits
            ),
            "supported_probability": (
                calibrated_probabilities
            ),
            "prototype_similarity": (
                calibration_similarities
            ),
            "nearest_prototype": [
                prototype_names[index]
                for index
                in best_prototype_indices
            ],
            "decision": decisions,
        }
    )

    metadata = (
        calibration[
            [
                "sample_id",
                "binary_label_name",
                "source_dataset",
                "source_group",
                "scene_id",
                "image_path",
            ]
        ]
        .drop_duplicates(
            subset=[
                "sample_id"
            ]
        )
    )

    predictions = predictions.merge(
        metadata,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    grouped = group_metrics(
        predictions
    )

    grouped.to_csv(
        GROUP_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_scatter(
        predictions,
        accept_probability,
        accept_similarity,
        reject_probability,
        reject_similarity,
    )

    summary = {
        "temperature": float(
            temperature
        ),
        "accept_probability_threshold": float(
            accept_probability
        ),
        "accept_similarity_threshold": float(
            accept_similarity
        ),
        "reject_probability_threshold": float(
            reject_probability
        ),
        "reject_similarity_threshold": float(
            reject_similarity
        ),
        "accept_selection": (
            accept_selection
        ),
        "reject_selection": (
            reject_selection
        ),
        "calibration_decision_metrics": (
            decision_metrics
        ),
        "prototype_groups": (
            prototype_names
        ),
        "prototype_count": int(
            len(prototype_names)
        ),
        "train_supported_samples": int(
            len(train_supported)
        ),
        "calibration_samples": int(
            len(calibration)
        ),
        "maximum_false_accept_rate": float(
            args.maximum_false_accept_rate
        ),
        "maximum_false_reject_rate": float(
            args.maximum_false_reject_rate
        ),
        "calibration_used": True,
        "test_locked_used": False,
        "predictions_path": str(
            PREDICTIONS_PATH
        ),
        "prototypes_path": str(
            PROTOTYPES_PATH
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

    config = {
        "model_version": (
            "input_gate_v04"
        ),
        "architecture": (
            "resnet18_grayscale_input_gate"
        ),
        "checkpoint_path": str(
            CHECKPOINT_PATH
        ),
        "prototypes_path": str(
            PROTOTYPES_PATH
        ),
        "input_channels": 1,
        "input_size": input_size,
        "normalization_mean": list(
            mean
        ),
        "normalization_std": list(
            std
        ),
        "temperature": float(
            temperature
        ),
        "accept_probability_threshold": float(
            accept_probability
        ),
        "accept_similarity_threshold": float(
            accept_similarity
        ),
        "reject_probability_threshold": float(
            reject_probability
        ),
        "reject_similarity_threshold": float(
            reject_similarity
        ),
        "decision_policy": {
            "ACCEPT": (
                "probability >= accept_probability "
                "AND similarity >= accept_similarity"
            ),
            "REJECT": (
                "probability <= reject_probability "
                "OR similarity <= reject_similarity"
            ),
            "UNCERTAIN": (
                "All remaining samples"
            ),
        },
        "accepted_input_description": (
            "DARTIS-like sea or coastal SAR image"
        ),
        "rejected_input_description": (
            "Optical, aerial or unsupported image"
        ),
        "test_locked_used": False,
    }

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_report(
        summary
    )

    print("=" * 78)
    print(
        "INPUT-GATE CALIBRATION SONUCU"
    )
    print("=" * 78)

    print(
        "Temperature:",
        f"{temperature:.6f}",
    )

    print()
    print(
        "ACCEPT probability:",
        f"{accept_probability:.6f}",
    )

    print(
        "ACCEPT similarity:",
        f"{accept_similarity:.6f}",
    )

    print(
        "REJECT probability:",
        f"{reject_probability:.6f}",
    )

    print(
        "REJECT similarity:",
        f"{reject_similarity:.6f}",
    )

    print()
    print(
        "Desteklenen SAR kabul:",
        f"{decision_metrics['supported_accept_rate']:.4f}",
    )

    print(
        "Desteklenen SAR belirsiz:",
        f"{decision_metrics['supported_uncertain_rate']:.4f}",
    )

    print(
        "Desteklenen SAR yanlış red:",
        f"{decision_metrics['supported_false_reject_rate']:.4f}",
    )

    print(
        "Desteklenmeyen red:",
        f"{decision_metrics['unsupported_reject_rate']:.4f}",
    )

    print(
        "Desteklenmeyen belirsiz:",
        f"{decision_metrics['unsupported_uncertain_rate']:.4f}",
    )

    print(
        "Desteklenmeyen yanlış kabul:",
        f"{decision_metrics['unsupported_false_accept_rate']:.4f}",
    )

    print()
    print(
        "Test locked kullanımı: YOK"
    )

    print()
    print(
        "Config:",
        CONFIG_PATH.resolve(),
    )

    print(
        "Prototipler:",
        PROTOTYPES_PATH.resolve(),
    )

    print(
        "Tahminler:",
        PREDICTIONS_PATH.resolve(),
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
