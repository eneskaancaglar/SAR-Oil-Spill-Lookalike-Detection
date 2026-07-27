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
    / "v03_verifier_dataset.csv"
)

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "verifier_v03"
    / "best.pth"
)

THRESHOLD_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "verifier_v03"
    / "threshold_sweep"
    / "threshold_summary.json"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "verifier_v03"
    / "calibration"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "calibration_predictions.csv"
)

CURVE_PATH = (
    OUTPUT_DIR
    / "calibration_curve.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "calibration_summary.json"
)

DIAGRAM_PATH = (
    OUTPUT_DIR
    / "reliability_diagram.png"
)

DEPLOYMENT_CONFIG_PATH = (
    ROOT
    / "checkpoints"
    / "verifier_v03"
    / "calibration_config.json"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v03_verifier_calibration.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verifier modelinin olasılıklarını calibration "
            "split üzerinde temperature scaling ile kalibre eder."
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
        "--bins",
        type=int,
        default=15,
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


class CalibrationDataset(Dataset):
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
            row["crop_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Crop bulunamadı: {image_path}"
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


def load_checkpoint(
    device: torch.device,
) -> tuple[
    nn.Module,
    dict[str, Any],
]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Verifier checkpoint bulunamadı: "
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

    model.fc = nn.Linear(
        model.fc.in_features,
        1,
    )

    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, payload


@torch.inference_mode()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
]:
    logits_all: list[float] = []
    labels_all: list[float] = []
    sample_ids_all: list[str] = []

    amp_enabled = (
        device.type == "cuda"
    )

    for images, labels, sample_ids in loader:
        images = images.to(
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

        logits_all.extend(
            logits
            .float()
            .cpu()
            .numpy()
            .tolist()
        )

        labels_all.extend(
            labels.numpy().tolist()
        )

        sample_ids_all.extend(
            list(sample_ids)
        )

    return (
        np.asarray(
            logits_all,
            dtype=np.float64,
        ),
        np.asarray(
            labels_all,
            dtype=np.float64,
        ),
        sample_ids_all,
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
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.05,
        max_iter=200,
        tolerance_grad=1e-9,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        )

        loss = F.binary_cross_entropy_with_logits(
            logits_tensor / temperature,
            labels_tensor,
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

    if not math.isfinite(temperature):
        raise RuntimeError(
            "Temperature değeri sonlu değil."
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


def probability_to_logit(
    probability: float,
) -> float:
    probability = float(
        np.clip(
            probability,
            1e-8,
            1.0 - 1e-8,
        )
    )

    return math.log(
        probability
        / (
            1.0 - probability
        )
    )


def binary_nll(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    probabilities = np.clip(
        probabilities,
        1e-8,
        1.0 - 1e-8,
    )

    return float(
        -np.mean(
            labels
            * np.log(probabilities)
            + (
                1.0 - labels
            )
            * np.log(
                1.0 - probabilities
            )
        )
    )


def brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    return float(
        np.mean(
            (
                probabilities
                - labels
            )
            ** 2
        )
    )


def calibration_table(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int,
    probability_type: str,
) -> pd.DataFrame:
    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    rows = []

    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])

        if index == bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count > 0:
            mean_probability = float(
                probabilities[mask].mean()
            )

            observed_positive_rate = float(
                labels[mask].mean()
            )

            calibration_gap = abs(
                mean_probability
                - observed_positive_rate
            )

        else:
            mean_probability = float("nan")
            observed_positive_rate = float(
                "nan"
            )
            calibration_gap = float("nan")

        rows.append(
            {
                "probability_type": (
                    probability_type
                ),
                "bin_index": index,
                "lower_bound": lower,
                "upper_bound": upper,
                "sample_count": count,
                "mean_probability": (
                    mean_probability
                ),
                "observed_positive_rate": (
                    observed_positive_rate
                ),
                "absolute_gap": (
                    calibration_gap
                ),
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(
    calibration_dataframe: pd.DataFrame,
) -> float:
    total = int(
        calibration_dataframe[
            "sample_count"
        ].sum()
    )

    if total <= 0:
        return 0.0

    valid = calibration_dataframe[
        calibration_dataframe[
            "sample_count"
        ] > 0
    ]

    return float(
        (
            valid["sample_count"]
            / total
            * valid["absolute_gap"]
        ).sum()
    )


def decision_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
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

    def divide(
        numerator: float,
        denominator: float,
    ) -> float:
        return (
            numerator / denominator
            if denominator > 0
            else 0.0
        )

    precision = divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = divide(
        true_positive,
        true_positive + false_negative,
    )

    specificity = divide(
        true_negative,
        true_negative + false_positive,
    )

    f1 = divide(
        2.0 * precision * recall,
        precision + recall,
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(
            (
                recall
                + specificity
            )
            / 2.0
        ),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def save_reliability_diagram(
    raw_curve: pd.DataFrame,
    calibrated_curve: pd.DataFrame,
) -> None:
    raw_valid = raw_curve[
        raw_curve["sample_count"] > 0
    ]

    calibrated_valid = calibrated_curve[
        calibrated_curve[
            "sample_count"
        ] > 0
    ]

    plt.figure(
        figsize=(7, 7)
    )

    plt.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        raw_valid[
            "mean_probability"
        ],
        raw_valid[
            "observed_positive_rate"
        ],
        marker="o",
        label="Raw",
    )

    plt.plot(
        calibrated_valid[
            "mean_probability"
        ],
        calibrated_valid[
            "observed_positive_rate"
        ],
        marker="o",
        label="Temperature scaled",
    )

    plt.xlabel(
        "Mean predicted oil probability"
    )

    plt.ylabel(
        "Observed oil frequency"
    )

    plt.title(
        "Verifier Reliability Diagram"
    )

    plt.xlim(
        0.0,
        1.0,
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DIAGRAM_PATH,
        dpi=180,
    )

    plt.close()


def write_report(
    summary: dict[str, Any],
) -> None:
    before = summary[
        "before_calibration"
    ]

    after = summary[
        "after_calibration"
    ]

    decision = summary[
        "preserved_decision_metrics"
    ]

    lines = [
        "# Verifier Olasılık Kalibrasyonu",
        "",
        "## Amaç",
        "",
        "Verifier modelinin ürettiği skorları daha anlamlı "
        "petrol güven yüzdelerine dönüştürmek.",
        "",
        "Kalibrasyonda yalnız `calibration` split kullanılmıştır. "
        "`test_locked` kullanılmamıştır.",
        "",
        "## Temperature scaling",
        "",
        f"- Temperature: "
        f"`{summary['temperature']:.6f}`",
        f"- Ham verifier threshold: "
        f"`{summary['raw_threshold']:.6f}`",
        f"- Kalibre edilmiş eşdeğer threshold: "
        f"`{summary['calibrated_threshold']:.6f}`",
        "",
        "## Kalibrasyon metrikleri",
        "",
        "| Metrik | Önce | Sonra |",
        "|---|---:|---:|",
        f"| NLL | {before['nll']:.6f} "
        f"| {after['nll']:.6f} |",
        f"| Brier score | {before['brier']:.6f} "
        f"| {after['brier']:.6f} |",
        f"| ECE | {before['ece']:.6f} "
        f"| {after['ece']:.6f} |",
        "",
        "Düşük NLL, Brier score ve ECE daha iyi "
        "kalibrasyon anlamına gelir.",
        "",
        "## Korunan karar performansı",
        "",
        f"- Precision: {decision['precision']:.4f}",
        f"- Recall: {decision['recall']:.4f}",
        f"- Specificity: {decision['specificity']:.4f}",
        f"- F1: {decision['f1']:.4f}",
        f"- False positive: "
        f"{decision['false_positive']}",
        f"- False negative: "
        f"{decision['false_negative']}",
        "",
        "Temperature scaling monotonik olduğu için ham threshold "
        "kalibre edilmiş eşdeğer threshold'a dönüştürülmüş ve "
        "petrol/petrol değil kararları korunmuştur.",
        "",
        "## Kullanım",
        "",
        "Yeni bir aday crop için:",
        "",
        "```text",
        "raw_logit = verifier(crop)",
        "calibrated_probability = sigmoid(raw_logit / temperature)",
        "petrol = calibrated_probability >= calibrated_threshold",
        "```",
        "",
        "Bu güven değeri aday crop seviyesindedir. "
        "Bütün görüntü için nihai güven değeri, end-to-end "
        "pipeline aşamasında adayların sonuçları birleştirilerek "
        "hesaplanacaktır.",
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
        THRESHOLD_SUMMARY_PATH,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: "
                f"{required_path}"
            )

    if args.bins < 2:
        raise ValueError(
            "--bins en az 2 olmalıdır."
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

    calibration_dataframe = dataframe[
        dataframe["split"].eq(
            "calibration"
        )
    ].copy()

    if calibration_dataframe.empty:
        raise RuntimeError(
            "Calibration split boş."
        )

    missing_files = []

    for path_text in calibration_dataframe[
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
            f"{len(missing_files)} calibration crop'u eksik.\n"
            + "\n".join(
                missing_files[:10]
            )
        )

    device = select_device(
        args.device
    )

    model, checkpoint_payload = (
        load_checkpoint(
            device
        )
    )

    mean = tuple(
        checkpoint_payload.get(
            "normalization_mean",
            [0.449],
        )
    )

    std = tuple(
        checkpoint_payload.get(
            "normalization_std",
            [0.226],
        )
    )

    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
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

    dataset = CalibrationDataset(
        calibration_dataframe,
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
        "VERIFIER TEMPERATURE CALIBRATION"
    )
    print("=" * 78)
    print("Cihaz:", device)
    print(
        "Calibration örneği:",
        len(dataset),
    )
    print(
        "Test locked kullanımı: YOK"
    )
    print()

    (
        logits,
        labels,
        sample_ids,
    ) = collect_logits(
        model,
        loader,
        device,
    )

    temperature = fit_temperature(
        logits,
        labels,
        device,
    )

    raw_probabilities = sigmoid(
        logits
    )

    calibrated_logits = (
        logits
        / temperature
    )

    calibrated_probabilities = sigmoid(
        calibrated_logits
    )

    with THRESHOLD_SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        threshold_summary = json.load(
            file
        )

    raw_threshold = float(
        threshold_summary[
            "recommended_threshold"
        ]["threshold"]
    )

    raw_threshold_logit = (
        probability_to_logit(
            raw_threshold
        )
    )

    calibrated_threshold = float(
        sigmoid(
            np.asarray(
                [
                    raw_threshold_logit
                    / temperature
                ],
                dtype=np.float64,
            )
        )[0]
    )

    raw_curve = calibration_table(
        raw_probabilities,
        labels,
        args.bins,
        "raw",
    )

    calibrated_curve = calibration_table(
        calibrated_probabilities,
        labels,
        args.bins,
        "temperature_scaled",
    )

    calibration_curve = pd.concat(
        [
            raw_curve,
            calibrated_curve,
        ],
        ignore_index=True,
    )

    calibration_curve.to_csv(
        CURVE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    raw_ece = (
        expected_calibration_error(
            raw_curve
        )
    )

    calibrated_ece = (
        expected_calibration_error(
            calibrated_curve
        )
    )

    raw_metrics = decision_metrics(
        labels,
        raw_probabilities,
        raw_threshold,
    )

    calibrated_metrics = (
        decision_metrics(
            labels,
            calibrated_probabilities,
            calibrated_threshold,
        )
    )

    raw_decisions = (
        raw_probabilities
        >= raw_threshold
    )

    calibrated_decisions = (
        calibrated_probabilities
        >= calibrated_threshold
    )

    changed_decision_count = int(
        (
            raw_decisions
            != calibrated_decisions
        ).sum()
    )

    prediction_dataframe = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "binary_label": (
                labels.astype(int)
            ),
            "raw_logit": logits,
            "raw_probability": (
                raw_probabilities
            ),
            "calibrated_logit": (
                calibrated_logits
            ),
            "calibrated_probability": (
                calibrated_probabilities
            ),
            "raw_prediction": (
                raw_decisions.astype(int)
            ),
            "calibrated_prediction": (
                calibrated_decisions.astype(
                    int
                )
            ),
        }
    )

    metadata = (
        calibration_dataframe[
            [
                "sample_id",
                "scene_id",
                "source_group",
                "source_image_name",
                "crop_type",
                "binary_label_name",
                "crop_path",
            ]
        ]
        .drop_duplicates(
            subset=["sample_id"]
        )
    )

    prediction_dataframe = (
        prediction_dataframe.merge(
            metadata,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    )

    prediction_dataframe.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_reliability_diagram(
        raw_curve,
        calibrated_curve,
    )

    before_calibration = {
        "nll": binary_nll(
            raw_probabilities,
            labels,
        ),
        "brier": brier_score(
            raw_probabilities,
            labels,
        ),
        "ece": float(raw_ece),
    }

    after_calibration = {
        "nll": binary_nll(
            calibrated_probabilities,
            labels,
        ),
        "brier": brier_score(
            calibrated_probabilities,
            labels,
        ),
        "ece": float(
            calibrated_ece
        ),
    }

    calibration_improved_nll = (
        after_calibration["nll"]
        <= before_calibration["nll"]
        + 1e-10
    )

    recommended_probability_mode = (
        "temperature_scaled"
        if calibration_improved_nll
        else "raw"
    )

    summary = {
        "calibration_samples": int(
            len(labels)
        ),
        "oil_samples": int(
            (labels == 1).sum()
        ),
        "non_oil_samples": int(
            (labels == 0).sum()
        ),
        "temperature": float(
            temperature
        ),
        "raw_threshold": float(
            raw_threshold
        ),
        "raw_threshold_logit": float(
            raw_threshold_logit
        ),
        "calibrated_threshold": float(
            calibrated_threshold
        ),
        "before_calibration": (
            before_calibration
        ),
        "after_calibration": (
            after_calibration
        ),
        "calibration_improved_nll": bool(
            calibration_improved_nll
        ),
        "recommended_probability_mode": (
            recommended_probability_mode
        ),
        "raw_decision_metrics": (
            raw_metrics
        ),
        "preserved_decision_metrics": (
            calibrated_metrics
        ),
        "changed_decision_count": int(
            changed_decision_count
        ),
        "calibration_split_used": True,
        "validation_used_for_threshold": True,
        "test_locked_used": False,
        "checkpoint_path": str(
            CHECKPOINT_PATH
        ),
        "predictions_path": str(
            PREDICTIONS_PATH
        ),
        "curve_path": str(
            CURVE_PATH
        ),
        "diagram_path": str(
            DIAGRAM_PATH
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

    deployment_config = {
        "architecture": (
            "resnet18_grayscale_binary"
        ),
        "checkpoint_path": str(
            CHECKPOINT_PATH
        ),
        "input_channels": 1,
        "input_size": 224,
        "normalization_mean": list(
            mean
        ),
        "normalization_std": list(
            std
        ),
        "temperature": float(
            temperature
        ),
        "raw_threshold": float(
            raw_threshold
        ),
        "calibrated_threshold": float(
            calibrated_threshold
        ),
        "probability_mode": (
            recommended_probability_mode
        ),
        "positive_class": "oil",
        "negative_class": "non_oil",
        "test_locked_used": False,
    }

    with DEPLOYMENT_CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            deployment_config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_report(
        summary
    )

    print("=" * 78)
    print("KALİBRASYON SONUCU")
    print("=" * 78)

    print(
        "Temperature:",
        f"{temperature:.6f}",
    )

    print(
        "Ham threshold:",
        f"{raw_threshold:.6f}",
    )

    print(
        "Kalibre threshold:",
        f"{calibrated_threshold:.6f}",
    )

    print()
    print(
        "NLL önce:",
        f"{before_calibration['nll']:.6f}",
    )

    print(
        "NLL sonra:",
        f"{after_calibration['nll']:.6f}",
    )

    print(
        "Brier önce:",
        f"{before_calibration['brier']:.6f}",
    )

    print(
        "Brier sonra:",
        f"{after_calibration['brier']:.6f}",
    )

    print(
        "ECE önce:",
        f"{before_calibration['ece']:.6f}",
    )

    print(
        "ECE sonra:",
        f"{after_calibration['ece']:.6f}",
    )

    print()
    print(
        "Değişen karar sayısı:",
        changed_decision_count,
    )

    print(
        "Önerilen probability mode:",
        recommended_probability_mode,
    )

    print(
        "Test locked kullanımı: YOK"
    )

    print()
    print(
        "Calibration config:",
        DEPLOYMENT_CONFIG_PATH.resolve(),
    )

    print(
        "Tahminler:",
        PREDICTIONS_PATH.resolve(),
    )

    print(
        "Reliability diagram:",
        DIAGRAM_PATH.resolve(),
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
