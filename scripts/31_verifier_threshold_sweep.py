from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT
    / "outputs"
    / "verifier_v03"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "verifier_v03"
    / "threshold_sweep"
)

SWEEP_CSV = (
    OUTPUT_DIR
    / "threshold_sweep.csv"
)

GROUP_METRICS_CSV = (
    OUTPUT_DIR
    / "recommended_threshold_group_metrics.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "threshold_summary.json"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v03_verifier_threshold_selection.md"
)

PLOT_PATH = (
    OUTPUT_DIR
    / "threshold_metrics.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verifier validation tahminleri üzerinde "
            "karar eşiği taraması yapar."
        )
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--end",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=0.95,
        help=(
            "Safety threshold seçimi için gereken "
            "minimum petrol recall değeri."
        ),
    )

    return parser.parse_args()


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
    threshold: float,
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

    positive_count = (
        true_positive
        + false_negative
    )

    negative_count = (
        true_negative
        + false_positive
    )

    precision = safe_divide(
        true_positive,
        true_positive
        + false_positive,
    )

    recall = safe_divide(
        true_positive,
        positive_count,
    )

    specificity = safe_divide(
        true_negative,
        negative_count,
    )

    accuracy = safe_divide(
        true_positive
        + true_negative,
        len(labels),
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
        negative_count,
    )

    false_negative_rate = safe_divide(
        false_negative,
        positive_count,
    )

    predicted_positive_rate = safe_divide(
        true_positive
        + false_positive,
        len(labels),
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
        "predicted_positive_rate": float(
            predicted_positive_rate
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
        "positive_count": int(
            positive_count
        ),
        "negative_count": int(
            negative_count
        ),
    }


def row_to_dictionary(
    row: pd.Series,
) -> dict[str, Any]:
    result = {}

    for key, value in row.to_dict().items():
        if isinstance(
            value,
            (np.integer,),
        ):
            result[str(key)] = int(
                value
            )

        elif isinstance(
            value,
            (np.floating,),
        ):
            result[str(key)] = float(
                value
            )

        else:
            result[str(key)] = value

    return result


def choose_best_row(
    dataframe: pd.DataFrame,
    primary_columns: list[str],
    ascending: list[bool],
) -> pd.Series:
    ordered = dataframe.sort_values(
        primary_columns,
        ascending=ascending,
        kind="mergesort",
    )

    return ordered.iloc[0]


def calculate_group_metrics(
    dataframe: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    output_rows = []

    grouping_columns = []

    if "source_group" in dataframe.columns:
        grouping_columns.append(
            "source_group"
        )

    if "crop_type" in dataframe.columns:
        grouping_columns.append(
            "crop_type"
        )

    for grouping_column in grouping_columns:
        for group_value, subset in dataframe.groupby(
            grouping_column,
            dropna=False,
        ):
            labels = (
                subset["binary_label"]
                .astype(int)
                .to_numpy()
            )

            probabilities = (
                subset["probability"]
                .astype(float)
                .to_numpy()
            )

            metrics = calculate_metrics(
                labels,
                probabilities,
                threshold,
            )

            output_rows.append(
                {
                    "grouping_column": (
                        grouping_column
                    ),
                    "group_value": str(
                        group_value
                    ),
                    "sample_count": int(
                        len(subset)
                    ),
                    "oil_count": int(
                        (
                            labels == 1
                        ).sum()
                    ),
                    "non_oil_count": int(
                        (
                            labels == 0
                        ).sum()
                    ),
                    "accuracy": (
                        metrics["accuracy"]
                    ),
                    "precision": (
                        metrics["precision"]
                    ),
                    "recall": (
                        metrics["recall"]
                    ),
                    "specificity": (
                        metrics["specificity"]
                    ),
                    "false_positive_rate": (
                        metrics[
                            "false_positive_rate"
                        ]
                    ),
                    "false_negative_rate": (
                        metrics[
                            "false_negative_rate"
                        ]
                    ),
                    "predicted_positive_rate": (
                        metrics[
                            "predicted_positive_rate"
                        ]
                    ),
                    "true_positive": (
                        metrics["true_positive"]
                    ),
                    "true_negative": (
                        metrics["true_negative"]
                    ),
                    "false_positive": (
                        metrics["false_positive"]
                    ),
                    "false_negative": (
                        metrics["false_negative"]
                    ),
                }
            )

    return pd.DataFrame(
        output_rows
    )


def save_plot(
    sweep: pd.DataFrame,
) -> None:
    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        sweep["threshold"],
        sweep["precision"],
        label="Precision",
    )

    plt.plot(
        sweep["threshold"],
        sweep["recall"],
        label="Recall",
    )

    plt.plot(
        sweep["threshold"],
        sweep["specificity"],
        label="Specificity",
    )

    plt.plot(
        sweep["threshold"],
        sweep["f1"],
        label="F1",
    )

    plt.plot(
        sweep["threshold"],
        sweep["balanced_accuracy"],
        label="Balanced accuracy",
    )

    plt.xlabel(
        "Verifier threshold"
    )

    plt.ylabel(
        "Metric"
    )

    plt.title(
        "Verifier Validation Threshold Sweep"
    )

    plt.ylim(
        0.0,
        1.02,
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=180,
    )

    plt.close()


def format_metric(
    value: float,
) -> str:
    return f"{value:.4f}"


def write_report(
    summary: dict[str, Any],
    group_metrics: pd.DataFrame,
) -> None:
    balanced = summary[
        "best_balanced_accuracy"
    ]

    best_f1 = summary[
        "best_f1"
    ]

    safety = summary.get(
        "safety_threshold"
    )

    recommended = summary[
        "recommended_threshold"
    ]

    lines = [
        "# Verifier Threshold Seçimi",
        "",
        "## Amaç",
        "",
        "İkinci aşama verifier modelinin petrol ve petrol değil "
        "kararını vereceği eşik validation verisi üzerinde seçilmiştir.",
        "",
        "Calibration ve kilitli test verileri kullanılmamıştır.",
        "",
        "## Aday eşikler",
        "",
        "| Seçim | Threshold | Precision | Recall | "
        "Specificity | F1 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| En iyi balanced accuracy "
            f"| {balanced['threshold']:.2f} "
            f"| {balanced['precision']:.4f} "
            f"| {balanced['recall']:.4f} "
            f"| {balanced['specificity']:.4f} "
            f"| {balanced['f1']:.4f} "
            f"| {balanced['balanced_accuracy']:.4f} |"
        ),
        (
            f"| En iyi F1 "
            f"| {best_f1['threshold']:.2f} "
            f"| {best_f1['precision']:.4f} "
            f"| {best_f1['recall']:.4f} "
            f"| {best_f1['specificity']:.4f} "
            f"| {best_f1['f1']:.4f} "
            f"| {best_f1['balanced_accuracy']:.4f} |"
        ),
    ]

    if safety is not None:
        lines.append(
            f"| Safety — recall ≥ "
            f"{summary['minimum_recall']:.2f} "
            f"| {safety['threshold']:.2f} "
            f"| {safety['precision']:.4f} "
            f"| {safety['recall']:.4f} "
            f"| {safety['specificity']:.4f} "
            f"| {safety['f1']:.4f} "
            f"| {safety['balanced_accuracy']:.4f} |"
        )

    else:
        lines.append(
            f"| Safety — recall ≥ "
            f"{summary['minimum_recall']:.2f} "
            f"| Bulunamadı | - | - | - | - | - |"
        )

    lines.extend(
        [
            "",
            "## Önerilen eşik",
            "",
            f"- Seçim yöntemi: "
            f"`{summary['recommended_selection_method']}`",
            f"- Verifier threshold: "
            f"`{recommended['threshold']:.2f}`",
            f"- Precision: "
            f"{recommended['precision']:.4f}",
            f"- Recall: "
            f"{recommended['recall']:.4f}",
            f"- Specificity: "
            f"{recommended['specificity']:.4f}",
            f"- F1: "
            f"{recommended['f1']:.4f}",
            f"- Balanced accuracy: "
            f"{recommended['balanced_accuracy']:.4f}",
            f"- False-positive rate: "
            f"{recommended['false_positive_rate']:.4f}",
            f"- False-negative rate: "
            f"{recommended['false_negative_rate']:.4f}",
            "",
            "## Confusion matrix",
            "",
            f"- True positive: "
            f"{recommended['true_positive']}",
            f"- True negative: "
            f"{recommended['true_negative']}",
            f"- False positive: "
            f"{recommended['false_positive']}",
            f"- False negative: "
            f"{recommended['false_negative']}",
            "",
        ]
    )

    if not group_metrics.empty:
        lines.extend(
            [
                "## Grup bazlı değerlendirme",
                "",
                "| Grup türü | Grup | Örnek | Petrol | Petrol değil | "
                "Recall | Specificity | False-positive rate |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for row in group_metrics.to_dict(
            orient="records"
        ):
            lines.append(
                f"| {row['grouping_column']} "
                f"| {row['group_value']} "
                f"| {row['sample_count']} "
                f"| {row['oil_count']} "
                f"| {row['non_oil_count']} "
                f"| {row['recall']:.4f} "
                f"| {row['specificity']:.4f} "
                f"| {row['false_positive_rate']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Bilimsel not",
            "",
            "Bu eşik yalnız validation verisiyle seçilmiştir. "
            "Sonraki aşamada calibration split kullanılarak "
            "verifier olasılıkları kalibre edilecektir. "
            "Kilitli test verisi henüz kullanılmayacaktır.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Validation tahmin dosyası bulunamadı: "
            f"{PREDICTIONS_PATH}"
        )

    if args.step <= 0:
        raise ValueError(
            "--step sıfırdan büyük olmalıdır."
        )

    if not (
        0.0
        <= args.start
        < args.end
        <= 1.0
    ):
        raise ValueError(
            "Threshold aralığı 0 ile 1 arasında olmalıdır."
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
        PREDICTIONS_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = {
        "binary_label",
        "probability",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise RuntimeError(
            "Validation prediction dosyasında eksik sütunlar: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe.dropna(
        subset=[
            "binary_label",
            "probability",
        ]
    ).copy()

    dataframe[
        "binary_label"
    ] = (
        pd.to_numeric(
            dataframe[
                "binary_label"
            ],
            errors="raise",
        )
        .astype(int)
    )

    dataframe[
        "probability"
    ] = pd.to_numeric(
        dataframe[
            "probability"
        ],
        errors="raise",
    )

    invalid_probabilities = (
        (
            dataframe[
                "probability"
            ]
            < 0.0
        )
        | (
            dataframe[
                "probability"
            ]
            > 1.0
        )
    )

    if invalid_probabilities.any():
        raise RuntimeError(
            "0 ile 1 dışında olasılık bulundu."
        )

    label_values = set(
        dataframe[
            "binary_label"
        ].unique()
    )

    if not label_values.issubset(
        {0, 1}
    ):
        raise RuntimeError(
            f"Geçersiz etiketler: {sorted(label_values)}"
        )

    labels = (
        dataframe[
            "binary_label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    probabilities = (
        dataframe[
            "probability"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    thresholds = np.arange(
        args.start,
        args.end
        + args.step / 2.0,
        args.step,
    )

    thresholds = np.clip(
        thresholds,
        0.0,
        1.0,
    )

    thresholds = np.unique(
        np.round(
            thresholds,
            10,
        )
    )

    sweep_rows = [
        calculate_metrics(
            labels,
            probabilities,
            float(threshold),
        )
        for threshold in thresholds
    ]

    sweep = pd.DataFrame(
        sweep_rows
    )

    sweep.to_csv(
        SWEEP_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    best_balanced_row = choose_best_row(
        sweep,
        primary_columns=[
            "balanced_accuracy",
            "f1",
            "recall",
            "specificity",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
    )

    best_f1_row = choose_best_row(
        sweep,
        primary_columns=[
            "f1",
            "balanced_accuracy",
            "recall",
            "precision",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
    )

    safety_candidates = sweep[
        sweep[
            "recall"
        ]
        >= args.minimum_recall
    ].copy()

    safety_row = None

    if not safety_candidates.empty:
        safety_row = choose_best_row(
            safety_candidates,
            primary_columns=[
                "specificity",
                "precision",
                "balanced_accuracy",
                "f1",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )

        recommended_row = safety_row

        recommended_method = (
            "safety_recall_constraint"
        )

    else:
        recommended_row = (
            best_balanced_row
        )

        recommended_method = (
            "balanced_accuracy_fallback"
        )

    recommended_threshold = float(
        recommended_row[
            "threshold"
        ]
    )

    group_metrics = calculate_group_metrics(
        dataframe,
        recommended_threshold,
    )

    group_metrics.to_csv(
        GROUP_METRICS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    save_plot(
        sweep
    )

    summary = {
        "validation_samples": int(
            len(dataframe)
        ),
        "oil_samples": int(
            (
                labels == 1
            ).sum()
        ),
        "non_oil_samples": int(
            (
                labels == 0
            ).sum()
        ),
        "threshold_start": float(
            args.start
        ),
        "threshold_end": float(
            args.end
        ),
        "threshold_step": float(
            args.step
        ),
        "minimum_recall": float(
            args.minimum_recall
        ),
        "best_balanced_accuracy": (
            row_to_dictionary(
                best_balanced_row
            )
        ),
        "best_f1": (
            row_to_dictionary(
                best_f1_row
            )
        ),
        "safety_threshold": (
            row_to_dictionary(
                safety_row
            )
            if safety_row is not None
            else None
        ),
        "recommended_selection_method": (
            recommended_method
        ),
        "recommended_threshold": (
            row_to_dictionary(
                recommended_row
            )
        ),
        "calibration_used": False,
        "test_locked_used": False,
        "threshold_sweep_csv": str(
            SWEEP_CSV
        ),
        "group_metrics_csv": str(
            GROUP_METRICS_CSV
        ),
        "plot_path": str(
            PLOT_PATH
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

    write_report(
        summary,
        group_metrics,
    )

    recommended = summary[
        "recommended_threshold"
    ]

    print("=" * 78)
    print(
        "VERIFIER THRESHOLD SWEEP"
    )
    print("=" * 78)

    print(
        "Validation örneği:",
        summary[
            "validation_samples"
        ],
    )

    print(
        "Petrol:",
        summary[
            "oil_samples"
        ],
    )

    print(
        "Petrol değil:",
        summary[
            "non_oil_samples"
        ],
    )

    print()
    print(
        "En iyi balanced threshold:",
        f"{summary['best_balanced_accuracy']['threshold']:.2f}",
    )

    print(
        "En iyi F1 threshold:",
        f"{summary['best_f1']['threshold']:.2f}",
    )

    if summary[
        "safety_threshold"
    ] is not None:
        print(
            "Safety threshold:",
            f"{summary['safety_threshold']['threshold']:.2f}",
        )

    else:
        print(
            "Safety threshold:",
            "Bulunamadı",
        )

    print()
    print(
        "ÖNERİLEN THRESHOLD:",
        f"{recommended['threshold']:.2f}",
    )

    print(
        "Seçim yöntemi:",
        summary[
            "recommended_selection_method"
        ],
    )

    print(
        "Precision:",
        f"{recommended['precision']:.4f}",
    )

    print(
        "Recall:",
        f"{recommended['recall']:.4f}",
    )

    print(
        "Specificity:",
        f"{recommended['specificity']:.4f}",
    )

    print(
        "F1:",
        f"{recommended['f1']:.4f}",
    )

    print(
        "Balanced accuracy:",
        f"{recommended['balanced_accuracy']:.4f}",
    )

    print(
        "False positive:",
        recommended[
            "false_positive"
        ],
    )

    print(
        "False negative:",
        recommended[
            "false_negative"
        ],
    )

    print()
    print(
        "Sweep CSV:",
        SWEEP_CSV.resolve(),
    )

    print(
        "Grup metrikleri:",
        GROUP_METRICS_CSV.resolve(),
    )

    print(
        "Grafik:",
        PLOT_PATH.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_JSON.resolve(),
    )

    print(
        "Rapor:",
        REPORT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
