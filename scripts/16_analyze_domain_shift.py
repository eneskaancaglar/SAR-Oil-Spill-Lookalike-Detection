from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

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


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def analyze_image(
    image_path: Path,
) -> tuple[dict[str, float | int], np.ndarray]:

    with Image.open(image_path) as image:
        image = image.convert("L")
        array = np.asarray(image, dtype=np.uint8)

    normalized = array.astype(np.float32) / 255.0

    histogram = np.bincount(
        array.reshape(-1),
        minlength=256,
    ).astype(np.int64)

    statistics = {
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "mean_intensity": float(normalized.mean()),
        "standard_deviation": float(normalized.std()),
        "p05": float(np.percentile(normalized, 5)),
        "p50": float(np.percentile(normalized, 50)),
        "p95": float(np.percentile(normalized, 95)),
        "dark_pixel_percentage": float(
            (normalized < 0.13).mean() * 100.0
        ),
        "bright_pixel_percentage": float(
            (normalized > 0.87).mean() * 100.0
        ),
    }

    return statistics, histogram


def process_sos(
    rows: list[dict],
    histograms: dict[str, np.ndarray],
) -> None:

    dataframe = pd.read_csv(
        SOS_MANIFEST,
        encoding="utf-8-sig",
    )

    split_text = (
        dataframe["split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    validation = dataframe[
        split_text.isin(
            ["val", "valid", "validation"]
        )
    ].copy()

    if validation.empty:
        raise RuntimeError(
            "SOS manifestinde validation verisi bulunamadi."
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

        if not image_path.exists():
            raise FileNotFoundError(
                f"SOS görüntüsü bulunamadi: {image_path}"
            )

        statistics, histogram = analyze_image(
            image_path
        )

        sensor = str(row.sensor)
        subgroup = f"SOS_{sensor}"

        rows.append(
            {
                "dataset_group": "SOS_validation",
                "subgroup": subgroup,
                "image_name": str(row.sample_name),
                "image_path": str(image_path),
                **statistics,
            }
        )

        histograms["SOS_validation"] += histogram
        histograms[subgroup] += histogram

        if index % 500 == 0:
            print(
                f"SOS: {index}/{len(validation)}"
            )


def process_dartis(
    rows: list[dict],
    histograms: dict[str, np.ndarray],
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
        image_set = str(row.image_set)
        image_name = str(row.image_name)

        image_path = (
            DARTIS_RAW_DIR
            / image_set
            / image_name
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"DARTIS görüntüsü bulunamadi: {image_path}"
            )

        statistics, histogram = analyze_image(
            image_path
        )

        subgroup = f"DARTIS_{image_set}"

        rows.append(
            {
                "dataset_group": "DARTIS_no_oil",
                "subgroup": subgroup,
                "image_name": image_name,
                "image_path": str(image_path),
                **statistics,
            }
        )

        histograms["DARTIS_no_oil"] += histogram
        histograms[subgroup] += histogram

        if index % 500 == 0:
            print(
                f"DARTIS: {index}/{len(dataframe)}"
            )


def summarize_group(
    rows: pd.DataFrame,
    histogram: np.ndarray,
) -> dict:

    pixel_values = np.arange(256) / 255.0
    pixel_count = int(histogram.sum())

    if pixel_count == 0:
        raise RuntimeError(
            "Histogram bos olamaz."
        )

    global_mean = float(
        np.sum(
            pixel_values * histogram
        )
        / pixel_count
    )

    cumulative = np.cumsum(histogram)

    def histogram_percentile(
        percentile: float,
    ) -> float:
        target = (
            percentile / 100.0
            * cumulative[-1]
        )

        index = int(
            np.searchsorted(
                cumulative,
                target,
            )
        )

        return index / 255.0

    return {
        "image_count": int(len(rows)),
        "mean_of_image_means": float(
            rows["mean_intensity"].mean()
        ),
        "median_of_image_means": float(
            rows["mean_intensity"].median()
        ),
        "mean_image_standard_deviation": float(
            rows["standard_deviation"].mean()
        ),
        "mean_dark_pixel_percentage": float(
            rows["dark_pixel_percentage"].mean()
        ),
        "global_mean_intensity": global_mean,
        "global_p05": histogram_percentile(5),
        "global_p50": histogram_percentile(50),
        "global_p95": histogram_percentile(95),
    }


def save_mean_distribution_plot(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:

    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(1, 1, 1)

    groups = {
        "SOS validation": dataframe[
            dataframe["dataset_group"]
            == "SOS_validation"
        ],
        "DARTIS water (nw)": dataframe[
            dataframe["subgroup"]
            == "DARTIS_nw"
        ],
        "DARTIS coast (nc)": dataframe[
            dataframe["subgroup"]
            == "DARTIS_nc"
        ],
    }

    for label, group in groups.items():
        axis.hist(
            group["mean_intensity"],
            bins=40,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
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
    histograms: dict[str, np.ndarray],
    output_path: Path,
) -> None:

    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(1, 1, 1)

    x_values = np.arange(256) / 255.0

    groups = {
        "SOS validation": histograms[
            "SOS_validation"
        ],
        "DARTIS water (nw)": histograms[
            "DARTIS_nw"
        ],
        "DARTIS coast (nc)": histograms[
            "DARTIS_nc"
        ],
    }

    for label, histogram in groups.items():
        normalized = (
            histogram / histogram.sum()
        )

        axis.plot(
            x_values,
            normalized,
            label=label,
        )

    axis.set_title(
        "Global piksel yoğunluğu dağılımı"
    )
    axis.set_xlabel(
        "Piksel yoğunluğu (0-1)"
    )
    axis.set_ylabel("Piksel oranı")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def analyze_false_alarm_correlation(
    statistics: pd.DataFrame,
    output_path: Path,
) -> dict:

    if not DARTIS_RESULTS.exists():
        return {
            "available": False,
        }

    results = pd.read_csv(
        DARTIS_RESULTS,
        encoding="utf-8-sig",
    )

    required_column = "t080_positive_ratio"

    if required_column not in results.columns:
        return {
            "available": False,
            "reason": (
                f"{required_column} bulunamadi."
            ),
        }

    dartis_statistics = statistics[
        statistics["dataset_group"]
        == "DARTIS_no_oil"
    ].copy()

    merged = dartis_statistics.merge(
        results[
            [
                "image_name",
                required_column,
            ]
        ],
        on="image_name",
        how="inner",
    )

    mean_correlation = float(
        merged["mean_intensity"].corr(
            merged[required_column]
        )
    )

    contrast_correlation = float(
        merged["standard_deviation"].corr(
            merged[required_column]
        )
    )

    dark_pixel_correlation = float(
        merged["dark_pixel_percentage"].corr(
            merged[required_column]
        )
    )

    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(1, 1, 1)

    axis.scatter(
        merged["mean_intensity"],
        merged[required_column] * 100.0,
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
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return {
        "available": True,
        "matched_image_count": int(len(merged)),
        "mean_intensity_correlation": (
            mean_correlation
        ),
        "standard_deviation_correlation": (
            contrast_correlation
        ),
        "dark_pixel_correlation": (
            dark_pixel_correlation
        ),
    }


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []

    histograms = defaultdict(
        lambda: np.zeros(
            256,
            dtype=np.int64,
        )
    )

    print("=" * 75)
    print("SOS - DARTIS DOMAIN SHIFT ANALIZI")
    print("=" * 75)

    process_sos(
        rows,
        histograms,
    )

    process_dartis(
        rows,
        histograms,
    )

    statistics = pd.DataFrame(rows)

    statistics_path = (
        OUTPUT_DIR
        / "per_image_intensity_statistics.csv"
    )

    statistics.to_csv(
        statistics_path,
        index=False,
        encoding="utf-8-sig",
    )

    group_filters = {
        "SOS_validation": (
            statistics["dataset_group"]
            == "SOS_validation"
        ),
        "DARTIS_no_oil": (
            statistics["dataset_group"]
            == "DARTIS_no_oil"
        ),
        "DARTIS_nw": (
            statistics["subgroup"]
            == "DARTIS_nw"
        ),
        "DARTIS_nc": (
            statistics["subgroup"]
            == "DARTIS_nc"
        ),
    }

    summary = {
        "groups": {},
    }

    for group_name, group_filter in (
        group_filters.items()
    ):
        group_rows = statistics[group_filter]

        summary["groups"][group_name] = (
            summarize_group(
                group_rows,
                histograms[group_name],
            )
        )

    summary[
        "false_alarm_correlation"
    ] = analyze_false_alarm_correlation(
        statistics,
        OUTPUT_DIR
        / "intensity_vs_false_alarm.png",
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

    save_mean_distribution_plot(
        statistics,
        OUTPUT_DIR
        / "image_mean_distribution.png",
    )

    save_global_histogram_plot(
        histograms,
        OUTPUT_DIR
        / "global_pixel_distribution.png",
    )

    print()
    print("=" * 75)
    print("GRUP SONUCLARI")
    print("=" * 75)

    for group_name, result in (
        summary["groups"].items()
    ):
        print()
        print(group_name)
        print(
            f"  Görüntü sayısı:       "
            f"{result['image_count']}"
        )
        print(
            f"  Ortalama yoğunluk:    "
            f"{result['global_mean_intensity']:.4f}"
        )
        print(
            f"  Ortalama kontrast:    "
            f"{result['mean_image_standard_deviation']:.4f}"
        )
        print(
            f"  Koyu piksel ort.:     "
            f"%{result['mean_dark_pixel_percentage']:.2f}"
        )
        print(
            f"  Global P05/P50/P95:   "
            f"{result['global_p05']:.4f} / "
            f"{result['global_p50']:.4f} / "
            f"{result['global_p95']:.4f}"
        )

    correlation = summary[
        "false_alarm_correlation"
    ]

    if correlation.get("available"):
        print()
        print("=" * 75)
        print("YANLIS ALARM KORELASYONLARI")
        print("=" * 75)

        print(
            "Ortalama yoğunluk - FP alan: ",
            f"{correlation['mean_intensity_correlation']:.4f}",
        )

        print(
            "Kontrast - FP alan:          ",
            f"{correlation['standard_deviation_correlation']:.4f}",
        )

        print(
            "Koyu piksel - FP alan:       ",
            f"{correlation['dark_pixel_correlation']:.4f}",
        )

    print()
    print("CSV: ", statistics_path.resolve())
    print("JSON:", summary_path.resolve())
    print("Grafikler:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
