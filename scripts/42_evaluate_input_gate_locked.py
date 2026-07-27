from __future__ import annotations

import argparse
import json
import shutil
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

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v04_input_gate_locked_test"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "locked_predictions.csv"
)

GROUP_METRICS_PATH = (
    OUTPUT_DIR
    / "locked_group_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "locked_summary.json"
)

SCATTER_PATH = (
    OUTPUT_DIR
    / "locked_probability_similarity.png"
)

ERROR_DIR = (
    OUTPUT_DIR
    / "review_examples"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v04_input_gate_locked_test.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.4 input-gate modelini kilitli DARTIS SAR "
            "ve görülmemiş UC Merced optik sınıflarında test eder."
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
        "--review-examples",
        type=int,
        default=20,
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
    path = Path(
        str(value).strip()
    )

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


def read_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON bulunamadı: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


class LockedDataset(Dataset):
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
    ):
        row = self.dataframe.iloc[
            index
        ]

        image_path = resolve_path(
            row["image_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Görüntü bulunamadı: {image_path}"
            )

        with Image.open(
            image_path
        ) as source:
            image = source.convert(
                "L"
            )

        return (
            self.transform(image),
            torch.tensor(
                float(
                    row[
                        "binary_label"
                    ]
                ),
                dtype=torch.float32,
            ),
            str(
                row["sample_id"]
            ),
        )


def build_model(
    device: torch.device,
) -> tuple[
    nn.Module,
    dict[str, Any],
]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: "
            f"{CHECKPOINT_PATH}"
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
        payload[
            "model_state_dict"
        ],
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
def collect_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    logits_all = []
    features_all = []
    labels_all = []
    sample_ids_all = []

    amp_enabled = (
        device.type == "cuda"
    )

    for (
        images,
        labels,
        sample_ids,
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
            features = (
                forward_features(
                    model,
                    images,
                )
            )

            logits = model.fc(
                features
            ).flatten()

        normalized_features = (
            F.normalize(
                features.float(),
                p=2,
                dim=1,
            )
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

    indices = np.argmax(
        similarities,
        axis=1,
    )

    values = similarities[
        np.arange(
            len(similarities)
        ),
        indices,
    ]

    return (
        values.astype(
            np.float64
        ),
        indices.astype(
            np.int64
        ),
    )


def assign_decisions(
    probabilities: np.ndarray,
    similarities: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    accept_probability = float(
        config[
            "accept_probability_threshold"
        ]
    )

    accept_similarity = float(
        config[
            "accept_similarity_threshold"
        ]
    )

    reject_probability = float(
        config[
            "reject_probability_threshold"
        ]
    )

    reject_similarity = float(
        config[
            "reject_similarity_threshold"
        ]
    )

    accepted = (
        (
            probabilities
            >= accept_probability
        )
        & (
            similarities
            >= accept_similarity
        )
    )

    rejected = (
        (
            probabilities
            <= reject_probability
        )
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


def calculate_metrics(
    labels: np.ndarray,
    decisions: np.ndarray,
) -> dict[str, Any]:
    supported = (
        labels == 1
    )

    unsupported = (
        labels == 0
    )

    def rate(
        mask: np.ndarray,
        decision: str,
    ) -> float:
        count = int(
            mask.sum()
        )

        if count <= 0:
            return 0.0

        return float(
            (
                decisions[
                    mask
                ]
                == decision
            ).mean()
        )

    return {
        "supported_samples": int(
            supported.sum()
        ),
        "unsupported_samples": int(
            unsupported.sum()
        ),
        "supported_accept_rate": rate(
            supported,
            "ACCEPT",
        ),
        "supported_uncertain_rate": rate(
            supported,
            "UNCERTAIN",
        ),
        "supported_false_reject_rate": rate(
            supported,
            "REJECT",
        ),
        "unsupported_reject_rate": rate(
            unsupported,
            "REJECT",
        ),
        "unsupported_uncertain_rate": rate(
            unsupported,
            "UNCERTAIN",
        ),
        "unsupported_false_accept_rate": rate(
            unsupported,
            "ACCEPT",
        ),
        "safe_unsupported_block_rate": float(
            rate(
                unsupported,
                "REJECT",
            )
            + rate(
                unsupported,
                "UNCERTAIN",
            )
        ),
        "total_accept": int(
            (
                decisions
                == "ACCEPT"
            ).sum()
        ),
        "total_uncertain": int(
            (
                decisions
                == "UNCERTAIN"
            ).sum()
        ),
        "total_reject": int(
            (
                decisions
                == "REJECT"
            ).sum()
        ),
    }


def calculate_group_metrics(
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
        labels = subset[
            "binary_label"
        ].to_numpy(
            dtype=np.int64
        )

        decisions = subset[
            "decision"
        ].to_numpy(
            dtype=object
        )

        supported_count = int(
            (
                labels == 1
            ).sum()
        )

        unsupported_count = int(
            (
                labels == 0
            ).sum()
        )

        accept_count = int(
            (
                decisions
                == "ACCEPT"
            ).sum()
        )

        uncertain_count = int(
            (
                decisions
                == "UNCERTAIN"
            ).sum()
        )

        reject_count = int(
            (
                decisions
                == "REJECT"
            ).sum()
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
                "supported_count": (
                    supported_count
                ),
                "unsupported_count": (
                    unsupported_count
                ),
                "accept_count": (
                    accept_count
                ),
                "uncertain_count": (
                    uncertain_count
                ),
                "reject_count": (
                    reject_count
                ),
                "accept_rate": float(
                    accept_count
                    / len(subset)
                ),
                "uncertain_rate": float(
                    uncertain_count
                    / len(subset)
                ),
                "reject_rate": float(
                    reject_count
                    / len(subset)
                ),
                "mean_probability": float(
                    subset[
                        "supported_probability"
                    ].mean()
                ),
                "maximum_probability": float(
                    subset[
                        "supported_probability"
                    ].max()
                ),
                "mean_similarity": float(
                    subset[
                        "prototype_similarity"
                    ].mean()
                ),
                "minimum_similarity": float(
                    subset[
                        "prototype_similarity"
                    ].min()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def copy_review_examples(
    dataframe: pd.DataFrame,
    maximum_examples: int,
) -> None:
    if ERROR_DIR.exists():
        shutil.rmtree(
            ERROR_DIR
        )

    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    conditions = {
        "unsupported_false_accept": (
            dataframe[
                "binary_label"
            ].eq(0)
            & dataframe[
                "decision"
            ].eq(
                "ACCEPT"
            )
        ),
        "unsupported_uncertain": (
            dataframe[
                "binary_label"
            ].eq(0)
            & dataframe[
                "decision"
            ].eq(
                "UNCERTAIN"
            )
        ),
        "supported_false_reject": (
            dataframe[
                "binary_label"
            ].eq(1)
            & dataframe[
                "decision"
            ].eq(
                "REJECT"
            )
        ),
        "supported_uncertain": (
            dataframe[
                "binary_label"
            ].eq(1)
            & dataframe[
                "decision"
            ].eq(
                "UNCERTAIN"
            )
        ),
    }

    for category, mask in conditions.items():
        subset = dataframe[
            mask
        ].copy()

        if category.startswith(
            "unsupported"
        ):
            subset = subset.sort_values(
                [
                    "supported_probability",
                    "prototype_similarity",
                ],
                ascending=False,
            )

        else:
            subset = subset.sort_values(
                [
                    "supported_probability",
                    "prototype_similarity",
                ],
                ascending=True,
            )

        destination = (
            ERROR_DIR
            / category
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, row in enumerate(
            subset.head(
                maximum_examples
            ).to_dict(
                orient="records"
            ),
            start=1,
        ):
            source = resolve_path(
                row["image_path"]
            )

            if not source.exists():
                continue

            filename = (
                f"{index:03d}"
                f"__{row['source_group']}"
                f"__p_{row['supported_probability']:.4f}"
                f"__s_{row['prototype_similarity']:.4f}"
                f"__{source.name}"
            )

            shutil.copy2(
                source,
                destination
                / filename,
            )


def save_scatter(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    plt.figure(
        figsize=(10, 7)
    )

    for group_name, subset in dataframe.groupby(
        "source_group"
    ):
        plt.scatter(
            subset[
                "supported_probability"
            ],
            subset[
                "prototype_similarity"
            ],
            s=22,
            alpha=0.65,
            label=str(
                group_name
            ),
        )

    plt.axvline(
        float(
            config[
                "accept_probability_threshold"
            ]
        ),
        linestyle="--",
        label=(
            "Accept probability"
        ),
    )

    plt.axhline(
        float(
            config[
                "accept_similarity_threshold"
            ]
        ),
        linestyle="--",
        label=(
            "Accept similarity"
        ),
    )

    plt.axvline(
        float(
            config[
                "reject_probability_threshold"
            ]
        ),
        linestyle=":",
        label=(
            "Reject probability"
        ),
    )

    plt.xlabel(
        "Calibrated supported-sea-SAR probability"
    )

    plt.ylabel(
        "Maximum DARTIS prototype similarity"
    )

    plt.title(
        "Input-Gate Locked Test"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend(
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()

    plt.savefig(
        SCATTER_PATH,
        dpi=180,
    )

    plt.close()


def write_report(
    summary: dict[str, Any],
    group_metrics: pd.DataFrame,
) -> None:
    metrics = summary[
        "locked_metrics"
    ]

    lines = [
        "# v0.4 Input-Gate Kilitli Test",
        "",
        "## Amaç",
        "",
        "Petrol analizinden önce yalnız desteklenen DARTIS "
        "benzeri deniz SAR görüntülerinin kabul edilip "
        "edilmediğini bağımsız kilitli veri üzerinde ölçmek.",
        "",
        "## Protokol",
        "",
        "- Model ağırlıkları donduruldu.",
        "- Temperature ve karar eşikleri donduruldu.",
        "- Kilitli test sonuçlarına göre eşik ayarlanmadı.",
        "- Airplane, harbor, river ve runway sınıfları "
        "eğitimde kullanılmadı.",
        "",
        "## Genel sonuçlar",
        "",
        f"- Kilitli desteklenen SAR: "
        f"{metrics['supported_samples']}",
        f"- Desteklenen SAR kabul oranı: "
        f"{metrics['supported_accept_rate']:.4f}",
        f"- Desteklenen SAR belirsiz oranı: "
        f"{metrics['supported_uncertain_rate']:.4f}",
        f"- Desteklenen SAR yanlış red oranı: "
        f"{metrics['supported_false_reject_rate']:.4f}",
        "",
        f"- Kilitli desteklenmeyen görüntü: "
        f"{metrics['unsupported_samples']}",
        f"- Desteklenmeyen red oranı: "
        f"{metrics['unsupported_reject_rate']:.4f}",
        f"- Desteklenmeyen belirsiz oranı: "
        f"{metrics['unsupported_uncertain_rate']:.4f}",
        f"- Desteklenmeyen yanlış kabul oranı: "
        f"{metrics['unsupported_false_accept_rate']:.4f}",
        f"- Güvenli engelleme oranı: "
        f"{metrics['safe_unsupported_block_rate']:.4f}",
        "",
        "## Grup sonuçları",
        "",
        "| Veri | Grup | Örnek | Kabul | Belirsiz | Red | "
        "Kabul oranı |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for row in group_metrics.to_dict(
        orient="records"
    ):
        lines.append(
            f"| {row['source_dataset']} "
            f"| {row['source_group']} "
            f"| {row['sample_count']} "
            f"| {row['accept_count']} "
            f"| {row['uncertain_count']} "
            f"| {row['reject_count']} "
            f"| {row['accept_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Başarı koşulu",
            "",
            "Airplane, harbor, river ve runway sınıflarındaki "
            "görüntülerin hiçbiri petrol analizine gönderilmemelidir.",
            "",
            "`REJECT` ve `UNCERTAIN` kararlarının ikisi de "
            "petrol pipeline'ını engeller.",
            "",
            "## Bilimsel sınır",
            "",
            "Bu input-gate yalnız eğitimde tanımlanan veri alanı "
            "için güvenlik katmanıdır. Dünyadaki bütün optik, SAR "
            "ve radar görüntülerini kapsayan evrensel bir "
            "görüntü tipi doğrulayıcısı değildir.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    for path in (
        MANIFEST_PATH,
        CHECKPOINT_PATH,
        CONFIG_PATH,
        PROTOTYPES_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: {path}"
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

    locked = dataframe[
        dataframe[
            "split"
        ].eq(
            "test_locked"
        )
    ].copy()

    if locked.empty:
        raise RuntimeError(
            "test_locked split boş."
        )

    locked_labels = set(
        locked[
            "binary_label"
        ].astype(int).unique()
    )

    if locked_labels != {
        0,
        1,
    }:
        raise RuntimeError(
            "Kilitli test iki sınıfı da içermiyor: "
            f"{sorted(locked_labels)}"
        )

    missing_files = []

    for path_text in locked[
        "image_path"
    ]:
        image_path = resolve_path(
            path_text
        )

        if not image_path.exists():
            missing_files.append(
                str(image_path)
            )

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} kilitli test "
            f"görüntüsü bulunamadı.\n"
            + "\n".join(
                missing_files[:15]
            )
        )

    config = read_json(
        CONFIG_PATH
    )

    device = select_device(
        args.device
    )

    model, checkpoint = (
        build_model(
            device
        )
    )

    prototype_data = np.load(
        PROTOTYPES_PATH,
        allow_pickle=False,
    )

    prototype_names = (
        prototype_data[
            "group_names"
        ].astype(str)
    )

    prototypes = (
        prototype_data[
            "centroids"
        ].astype(
            np.float32
        )
    )

    mean = tuple(
        checkpoint.get(
            "normalization_mean",
            [0.5],
        )
    )

    std = tuple(
        checkpoint.get(
            "normalization_std",
            [0.25],
        )
    )

    input_size = int(
        checkpoint.get(
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

    dataset = LockedDataset(
        locked,
        transform,
    )

    loader = DataLoader(
        dataset,
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
        "INPUT-GATE KİLİTLİ TEST"
    )
    print("=" * 78)

    print(
        "Cihaz:",
        device,
    )

    print(
        "Kilitli toplam:",
        len(dataset),
    )

    print(
        "Desteklenen SAR:",
        int(
            locked[
                "binary_label"
            ].eq(1).sum()
        ),
    )

    print(
        "Desteklenmeyen:",
        int(
            locked[
                "binary_label"
            ].eq(0).sum()
        ),
    )

    print(
        "Eşikler: DONDURULDU"
    )

    print()

    (
        logits,
        features,
        labels,
        sample_ids,
    ) = collect_outputs(
        model,
        loader,
        device,
    )

    temperature = float(
        config[
            "temperature"
        ]
    )

    probabilities = sigmoid(
        logits
        / temperature
    )

    (
        similarities,
        nearest_indices,
    ) = maximum_similarity(
        features,
        prototypes,
    )

    decisions = assign_decisions(
        probabilities,
        similarities,
        config,
    )

    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "binary_label": labels,
            "raw_logit": logits,
            "supported_probability": (
                probabilities
            ),
            "prototype_similarity": (
                similarities
            ),
            "nearest_prototype": [
                prototype_names[index]
                for index
                in nearest_indices
            ],
            "decision": decisions,
            "pipeline_allowed": (
                decisions == "ACCEPT"
            ),
        }
    )

    metadata = (
        locked[
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
            subset=["sample_id"]
        )
    )

    predictions = predictions.merge(
        metadata,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    predictions[
        "correct_safety_action"
    ] = np.where(
        predictions[
            "binary_label"
        ].eq(1),
        predictions[
            "decision"
        ].eq(
            "ACCEPT"
        ),
        predictions[
            "decision"
        ].ne(
            "ACCEPT"
        ),
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metrics = calculate_metrics(
        labels,
        decisions,
    )

    grouped = calculate_group_metrics(
        predictions
    )

    grouped.to_csv(
        GROUP_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    copy_review_examples(
        predictions,
        args.review_examples,
    )

    save_scatter(
        predictions,
        config,
    )

    locked_optical_groups = {
        "airplane",
        "harbor",
        "river",
        "runway",
    }

    optical_locked = predictions[
        predictions[
            "source_dataset"
        ].eq(
            "uc_merced"
        )
        & predictions[
            "source_group"
        ].isin(
            locked_optical_groups
        )
    ].copy()

    optical_false_accept_count = int(
        optical_locked[
            "decision"
        ].eq(
            "ACCEPT"
        ).sum()
    )

    airplane = predictions[
        predictions[
            "source_group"
        ].eq(
            "airplane"
        )
    ]

    airplane_accept_count = int(
        airplane[
            "decision"
        ].eq(
            "ACCEPT"
        ).sum()
    )

    summary = {
        "model_version": (
            "input_gate_v04"
        ),
        "locked_samples": int(
            len(predictions)
        ),
        "locked_metrics": metrics,
        "locked_optical_groups": sorted(
            locked_optical_groups
        ),
        "locked_optical_samples": int(
            len(optical_locked)
        ),
        "locked_optical_false_accept_count": (
            optical_false_accept_count
        ),
        "airplane_samples": int(
            len(airplane)
        ),
        "airplane_accept_count": (
            airplane_accept_count
        ),
        "airplane_safe_block_count": int(
            len(airplane)
            - airplane_accept_count
        ),
        "airplane_safe_block_rate": float(
            (
                len(airplane)
                - airplane_accept_count
            )
            / len(airplane)
        )
        if len(airplane) > 0
        else 0.0,
        "input_gate_passed_locked_safety_test": bool(
            optical_false_accept_count
            == 0
        ),
        "configuration_frozen": True,
        "test_locked_used": True,
        "predictions_path": str(
            PREDICTIONS_PATH
        ),
        "group_metrics_path": str(
            GROUP_METRICS_PATH
        ),
        "review_examples_path": str(
            ERROR_DIR
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

    write_report(
        summary,
        grouped,
    )

    print("=" * 78)
    print(
        "INPUT-GATE KİLİTLİ TEST SONUCU"
    )
    print("=" * 78)

    print(
        "Desteklenen SAR kabul:",
        f"{metrics['supported_accept_rate']:.4f}",
    )

    print(
        "Desteklenen SAR belirsiz:",
        f"{metrics['supported_uncertain_rate']:.4f}",
    )

    print(
        "Desteklenen SAR yanlış red:",
        f"{metrics['supported_false_reject_rate']:.4f}",
    )

    print()
    print(
        "Desteklenmeyen red:",
        f"{metrics['unsupported_reject_rate']:.4f}",
    )

    print(
        "Desteklenmeyen belirsiz:",
        f"{metrics['unsupported_uncertain_rate']:.4f}",
    )

    print(
        "Desteklenmeyen yanlış kabul:",
        f"{metrics['unsupported_false_accept_rate']:.4f}",
    )

    print(
        "Güvenli engelleme:",
        f"{metrics['safe_unsupported_block_rate']:.4f}",
    )

    print()
    print(
        "Airplane görüntüsü:",
        len(airplane),
    )

    print(
        "Airplane ACCEPT:",
        airplane_accept_count,
    )

    print(
        "Airplane güvenli engelleme:",
        f"{summary['airplane_safe_block_rate']:.4f}",
    )

    print()
    print(
        "Kilitli optik yanlış kabul:",
        optical_false_accept_count,
    )

    print(
        "Güvenlik testi:",
        (
            "BAŞARILI"
            if summary[
                "input_gate_passed_locked_safety_test"
            ]
            else "BAŞARISIZ"
        ),
    )

    print()
    print(
        "Tahminler:",
        PREDICTIONS_PATH.resolve(),
    )

    print(
        "Grup sonuçları:",
        GROUP_METRICS_PATH.resolve(),
    )

    print(
        "İnceleme örnekleri:",
        ERROR_DIR.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    print(
        "Rapor:",
        REPORT_PATH.resolve(),
    )

    print()
    print(
        "ÖNEMLİ: Kilitli test artık kullanıldı."
    )

    print(
        "Bu sonuçlara bakarak mevcut eşikler "
        "yeniden ayarlanmamalıdır."
    )


if __name__ == "__main__":
    main()
