from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOS_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset_manifest.csv"
)

DARTIS_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "metadata"
    / "dartis_no_oil_manifest.csv"
)

DARTIS_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

DARTIS_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "dartis_stress_test"
    / "dartis_no_oil_per_image_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "domain_shift_analysis"
)


def resolve_project_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def percentile_from_histogram(
    histogram: np.ndarray,
    percentile: float,
) -> float:
    cumulative = np.cumsum(histogram)
    target = percentile / 100.0 * cumulative[-1]

    index = int(
        np.searchsorted(
            cumulative,
            target,
            side="left",
        )
    )

    return index / 255.0


def analyze_image(
    image_path: Path,
) -> tuple[dict[str, float | int], np.ndarray]:
    with Image.open(image_path) as image:
        image = image.convert("L")
        array = np.asarray(image, dtype=np.uint8)

    histogram = np.bincount(
        array.reshape(-1),
        minlength=256,
    ).astype(np.int64)

    values = np.arange(256, dtype=np.float64)
    pixel_count = int(histogram.sum())

    mean_raw = float(
        np.sum(histogram * values) / pixel_count
    )

    variance_raw = float(
        np.sum(
            histogram * (values - mean_raw) ** 2
        )
        / pixel_count
    )

    standard_deviation_raw = float(
        np.sqrt(variance_raw)
    )

    metrics = {
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "pixel_count": pixel_count,
        "mean_intensity": mean_raw / 255.0,
        "standard_deviation": (
            standard_deviation_raw / 255.0
        ),
        "p05": percentile_from_histogram(
            histogram,
            5.0,
        ),
        "p50": percentile_from_histogram(
            histogram,
            50.0,
        ),
        "p95": percentile_from_histogram(
            histogram,
            95.0,
        ),
        "dark_pixel_percentage": float(
            histogram[:33].sum()
            / pixel_count
            * 100.0
        ),
        "bright_pixel_percentage": float(
            histogram[224:].sum()
            / pixel_count
            * 100.0
        ),
    }

    return metrics, histogram


def global_histogram_summary(
    histogram: np.ndarray,
) -> dict[str, float | int]:
    values = np.arange(256, dtype=np.float64)
    pixel_count = int(histogram.sum())

    mean_raw = float(
        np.sum(histogram * values) / pixel_count
    )

    return {
        "pixel_count": pixel_count,
        "mean_intensity": mean_raw / 255.0,
        "p05": percentile_from_histogram(
            histogram,
            5.0,
        ),
        "p50": percentile_from_histogram(
            histogram,
            50.0,
        ),
        "p95": percentile_from_histogram(
            histogram,
            95.0,
        ),
        "dark_pixel_percentage": float(
            histogram[:33].sum()
            / pixel_count
            * 100.0
        ),
        "bright_pixel_percentage": float(
            histogram[224:].sum()
            / pixel_count
            * 100.0
        ),
    }


def process_sos(
    output_rows: list[dict[str, Any]],
    group_histograms: dict[str, np.ndarray],
) -> None:
    dataframe = pd.read_csv(
        SOS_MANIFEST,
        encoding="utf-8-sig",
    )

    split_values = (
        dataframe["split"]
        .astype(str)
        .str.lower()
    )

    validation = dataframe[
        split_values.isin(
            {"val", "valid", "validation"}
        )
    ].copy()

    if validation.empty:
        raise RuntimeError(
            "SOS manifestinde validation satiri bulunamadi."
        )

    print(
        f"SOS validation görüntüsü: {len(validation)}"
    )

    for index, row in enumerate(
        validation.itertuples(index=False),
        start=1,
    ):
        image_path = resolve_project_path(
            str(row.image_path)
        )

        metrics, histogram = analyze_image(
            image_path
        )

        sensor = str(row.sensor)
        sensor_group = f"SOS_{sensor}"

        output_rows.append(
            {
                "dataset_group": "SOS_validation",
                "subgroup": sensor_group,
                "image_name": str(row.sample_name),
                "image_path": str(image_path),
                **metrics,
            }
        )

        group_histograms[
            "SOS_validation"
        ] += histogram

        group_histograms[
            sensor_group
        ] += histogram

        if index % 500 == 0:
            print(
                f"SOS: {index}/{len(validation)}"
            )


def process_dartis(
    output_rows: list[dict[str, Any]],
    group_histograms: dict[str, np.ndarray],
) -> None:
    dataframe = pd.read_csv(
        DARTIS_MANIFEST,
        encoding="utf-8-sig",
    )

    print(
        f"DARTIS no-oil görüntüsü: {len(dataframe)}"
    )

    for index, row in enumerate(
        dataframe.itertuples(index=False),
        start=1,
    ):
        image_path = (
            DARTIS_RAW_DIR
            / str(row.image_set)
            / str(row.image_name)
        )

        metrics, histogram = analyze_image(
            image_path
        )

        image_set = str(row.image_set)
        group_name = f"DARTIS_{image_set}"

        output_rows.append(
            {
                "dataset_group": "DARTIS_no_oil",
                "subgroup": group_name,
                "image_name": str(row.image_name),
                "image_path": str(image_path),
                **metrics,
            }
        )

        group_histograms[
            "DARTIS_no_oil"
        ] += histogram

        group_histograms[
            group_name
        ] += histogram

        if index % 500 == 0:
            print(
                f"DARTIS: {index}/{len(dataframe)}"
            )


def save_distribution_plot(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(1, 1, 1)

    for group_name in (
        "SOS_validation",
        "DARTIS_nw",
        "DARTIS_nc",
    ):
        if group_name == "SOS_validation":
            values = dataframe.loc[
                dataframe["dataset_group"]
                == "SOS_validation",
                "mean_intensity",
            ]
        else:
            values = dataframe.loc[
                dataframe["subgroup"]
                == group_name,
                "mean_intensity",
            ]

        axis.hist(
            values,
            bins=40,
            density=True,
            histtype="step",
            linewidth=2,
            label=group_name,
        )

    axis.set_title(
        "Görüntü başına ortalama yoğunluk dağılımı"
    )
    axis.set_xlabel(
        "Ortalama piksel yoğunluğu (0-1)"
    )
    axis.set_ylabel("Yoğunluk")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_global_histogram_plot(
    group_histograms: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(1, 1, 1)

    x_values = np.arange(256) / 255.0

    for group_name in (
        "SOS_validation",
        "DARTIS_nw",
        "DARTIS_nc",
    ):
        histogram = group_histograms[group_name]

        normalized = (
            histogram / histogram.sum()
        )

        axis.plot(
            x_values,
            normalized,
            label=group_name,
        )

    axis.set_title(
        "Global piksel yoğunluğu dağılımı"
    )
    axis.set_xlabel("Piksel yoğunluğu (0-1)")
    axis.set_ylabel("Piksel oranı")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def analyze_false_alarm_correlation(
    statistics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, float | int]:
    if not DARTIS_RESULTS.exists():
        return {
            "available": 0,
        }

    results = pd.read_csv(
        DARTIS_RESULTS,
        encoding="utf-8-sig",
    )

    dartis_statistics = statistics[
        statistics["dataset_group"]
        == "DARTIS_no_oil"
    ].copy()

    merged = dartis_statistics.merge(
        results[
            [
                "image_name",
                "t080_positive_ratio",
            ]
        ],
        on="image_name",
        how="inner",
    )

    mean_correlation = float(
        merged["mean_intensity"].corr(
            merged["t080_positive_ratio"]
        )
    )

    std_correlation = float(
        merged["standard_deviation"].corr(
            merged["t080_positive_ratio"]
        )
    )

    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(1, 1, 1)

    axis.scatter(
        merged["mean_intensity"],
        merged["t080_positive_ratio"] * 100.0,
        s=10,
        alpha=0.4,
    )

    axis.set_title(
        "DARTIS yoğunluğu ve yanlış alarm ilişkisi"
    )
    axis.set_xlabel(
        "Görüntünün ortalama yoğunluğu"
    )
    axis.set_ylabel(
        "Threshold 0.80 yanlış pozitif alanı (%)"
    )
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_dir
        / "intensity_vs_false_alarm.png",
        dpi=160,
    )
    plt.close(figure)

    return {
        "available": 1,
        "matched_image_count": len(merged),
        "mean_intensity_correlation": (
            mean_correlation
        ),
        "standard_deviation_correlation": (
            std_correlation
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_histograms: dict[str, np.ndarray] = (
        defaultdict(
            lambda: np.zeros(
                256,
                dtype=np.int64,
            )
        )
    )

    output_rows: list[dict[str, Any]] = []

    print("=" * 75)
    print("SOS - DARTIS DOMAIN SHIFT ANALIZI")
    print("=" * 75)

    process_sos(
        output_rows,
        group_histograms,
    )

    process_dartis(
        output_rows,
        group_histograms,
    )

    statistics = pd.DataFrame(output_rows)

    statistics_path = (
        OUTPUT_DIR
        / "per_image_intensity_statistics.csv"
    )

    statistics.to_csv(
        statistics_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary: dict[str, Any] = {
        "groups": {},
    }

    for group_name in (
        "SOS_validation",
        "DARTIS_no_oil",
        "DARTIS_nw",
        "DARTIS_nc",
    ):
        if group_name in {
            "SOS_validation",
            "DARTIS_no_oil",
        }:
            group_rows = statistics[
                statistics["dataset_group"]
                == group_name
            ]
        else:
            group_rows = statistics[
                statistics["subgroup"]
                == group_name
            ]

        summary["groups"][group_name] = {
            "image_count": len(group_rows),
            "mean_of_image_means": float(
                group_rows[
                    "mean_intensity"
                ].mean()
            ),
            "median_of_image_means": float(
                group_rows[
                    "mean_intensity"
                ].median()
            ),
            "mean_image_standard_deviation": float(
                group_rows[
                    "standard_deviation"
                ].mean()
            ),
            "global_pixels": global_histogram_summary(
                group_histograms[group_name]
            ),
        }

    summary["false_alarm_correlation"] = (
        analyze_false_alarm_correlation(
            statistics,
            OUTPUT_DIR,
        )
    )

    summary_path = (
        OUTPUT_DIR
        / "domain_shift_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    save_distribution_plot(
        statistics,
        OUTPUT_DIR
        / "image_mean_distribution.png",
    )

    save_global_histogram_plot(
        group_histograms,
        OUTPUT_DIR
        / "global_pixel_distribution.png",
    )

    print()
    print("=" * 75)
    print("GRUP OZETLERI")
    print("=" * 75)

    for group_name, group_summary in (
        summary["groups"].items()
    ):
        global_summary = (
            group_summary["global_pixels"]
        )

        print()
        print(group_name)
        print(
            "  Goruntu:",
            group_summary["image_count"],
        )
        print(
            "  Ortalama yogunluk:",
            f"{global_summary['mean_intensity']:.4f}",
        )
        print(
            "  P05 / P50 / P95:",
            f"{global_summary['p05']:.4f} / "
            f"{global_summary['p50']:.4f} / "
            f"{global_summary['p95']:.4f}",
        )
        print(
            "  Koyu piksel orani:",
            f"%{global_summary['dark_pixel_percentage']:.2f}",
        )

    correlation = summary[
        "false_alarm_correlation"
    ]

    if correlation.get("available") == 1:
        print()
        print(
            "Ortalama yogunluk - FP alan "
            "korelasyonu:",
            f"{correlation['mean_intensity_correlation']:.4f}",
        )

        print(
            "Standart sapma - FP alan "
            "korelasyonu:",
            f"{correlation['standard_deviation_correlation']:.4f}",
        )

    print()
    print("CSV: ", statistics_path.resolve())
    print("JSON:", summary_path.resolve())
    print(
        "Grafikler:",
        OUTPUT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
