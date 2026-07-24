from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASELINE_CSV = (
    ROOT
    / "outputs"
    / "external_test_baseline"
    / "dartis_no_oil_per_image_results.csv"
)

FINETUNED_CSV = (
    ROOT
    / "outputs"
    / "external_test_finetuned"
    / "dartis_no_oil_per_image_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "finetune_improvement_analysis"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "hard_negative_finetune_comparison.md"
)

THRESHOLD_COLUMN = "t060_positive_ratio"


def load_results(
    path: Path,
    prefix: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Sonuç CSV dosyası bulunamadı: {path}"
        )

    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required = {
        "image_set",
        "image_name",
        THRESHOLD_COLUMN,
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise RuntimeError(
            f"{path.name} içinde eksik sütunlar var: "
            f"{sorted(missing)}"
        )

    return dataframe[
        [
            "image_set",
            "image_name",
            THRESHOLD_COLUMN,
        ]
    ].rename(
        columns={
            THRESHOLD_COLUMN: (
                f"{prefix}_positive_ratio"
            )
        }
    )


def summarize_group(
    dataframe: pd.DataFrame,
) -> dict:
    baseline = dataframe[
        "baseline_positive_ratio"
    ]

    finetuned = dataframe[
        "finetuned_positive_ratio"
    ]

    improvement = (
        baseline - finetuned
    )

    return {
        "image_count": int(len(dataframe)),
        "baseline_mean_fp_pixel_percentage": float(
            baseline.mean() * 100.0
        ),
        "finetuned_mean_fp_pixel_percentage": float(
            finetuned.mean() * 100.0
        ),
        "baseline_alarm_ge_1_percentage": float(
            (baseline >= 0.01).mean() * 100.0
        ),
        "finetuned_alarm_ge_1_percentage": float(
            (finetuned >= 0.01).mean() * 100.0
        ),
        "improved_image_percentage": float(
            (improvement > 0).mean() * 100.0
        ),
        "worsened_image_percentage": float(
            (improvement < 0).mean() * 100.0
        ),
        "unchanged_image_percentage": float(
            (improvement == 0).mean() * 100.0
        ),
        "mean_absolute_reduction_points": float(
            improvement.mean() * 100.0
        ),
        "median_absolute_reduction_points": float(
            improvement.median() * 100.0
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = load_results(
        BASELINE_CSV,
        "baseline",
    )

    finetuned = load_results(
        FINETUNED_CSV,
        "finetuned",
    )

    merged = baseline.merge(
        finetuned,
        on=[
            "image_set",
            "image_name",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 493:
        raise RuntimeError(
            "Beklenen 493 external-test görüntüsünün "
            f"tamamı eşleşmedi. Eşleşen: {len(merged)}"
        )

    merged[
        "reduction_ratio"
    ] = (
        merged["baseline_positive_ratio"]
        - merged["finetuned_positive_ratio"]
    )

    merged[
        "reduction_percentage_points"
    ] = (
        merged["reduction_ratio"]
        * 100.0
    )

    merged[
        "result"
    ] = "unchanged"

    merged.loc[
        merged["reduction_ratio"] > 0,
        "result",
    ] = "improved"

    merged.loc[
        merged["reduction_ratio"] < 0,
        "result",
    ] = "worsened"

    summary = {
        "threshold": 0.60,
        "groups": {},
    }

    group_filters = {
        "overall": pd.Series(
            True,
            index=merged.index,
        ),
        "nc": merged["image_set"] == "nc",
        "nw": merged["image_set"] == "nw",
    }

    for group_name, group_filter in (
        group_filters.items()
    ):
        summary["groups"][
            group_name
        ] = summarize_group(
            merged[group_filter]
        )

    merged.to_csv(
        OUTPUT_DIR
        / "per_image_finetune_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    merged.sort_values(
        "finetuned_positive_ratio",
        ascending=False,
    ).head(30).to_csv(
        OUTPUT_DIR
        / "top_remaining_false_alarms.csv",
        index=False,
        encoding="utf-8-sig",
    )

    merged.sort_values(
        "reduction_ratio",
        ascending=False,
    ).head(30).to_csv(
        OUTPUT_DIR
        / "largest_improvements.csv",
        index=False,
        encoding="utf-8-sig",
    )

    merged.sort_values(
        "reduction_ratio",
        ascending=True,
    ).head(30).to_csv(
        OUTPUT_DIR
        / "largest_regressions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        OUTPUT_DIR
        / "finetune_comparison_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    figure = plt.figure(
        figsize=(8, 8)
    )
    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.scatter(
        merged["baseline_positive_ratio"]
        * 100.0,
        merged["finetuned_positive_ratio"]
        * 100.0,
        s=14,
        alpha=0.5,
    )

    maximum = max(
        merged[
            "baseline_positive_ratio"
        ].max(),
        merged[
            "finetuned_positive_ratio"
        ].max(),
    ) * 100.0

    axis.plot(
        [0, maximum],
        [0, maximum],
        linestyle="--",
    )

    axis.set_xlabel(
        "Baseline yanlış pozitif alanı (%)"
    )
    axis.set_ylabel(
        "Fine-tuned yanlış pozitif alanı (%)"
    )
    axis.set_title(
        "External test: görüntü bazında model karşılaştırması"
    )
    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR
        / "baseline_vs_finetuned_scatter.png",
        dpi=180,
    )
    plt.close(figure)

    figure = plt.figure(
        figsize=(9, 6)
    )
    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.hist(
        merged[
            "reduction_percentage_points"
        ],
        bins=50,
    )

    axis.axvline(
        0,
        linestyle="--",
    )

    axis.set_xlabel(
        "Yanlış pozitif alanındaki azalma "
        "(yüzde puan)"
    )
    axis.set_ylabel(
        "Görüntü sayısı"
    )
    axis.set_title(
        "Fine-tuning iyileştirme dağılımı"
    )
    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR
        / "improvement_distribution.png",
        dpi=180,
    )
    plt.close(figure)

    lines = [
        "# Hard-Negative Fine-Tuning Karşılaştırması",
        "",
        "Seçilen threshold: **0.60**",
        "",
        "| Grup | N | Baseline FP piksel | Fine-tuned FP piksel | "
        "Baseline >=%1 alarm | Fine-tuned >=%1 alarm | "
        "İyileşen görüntü | Kötüleşen görüntü |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for group_name in (
        "overall",
        "nc",
        "nw",
    ):
        result = summary[
            "groups"
        ][group_name]

        lines.append(
            f"| {group_name} "
            f"| {result['image_count']} "
            f"| %{result['baseline_mean_fp_pixel_percentage']:.4f} "
            f"| %{result['finetuned_mean_fp_pixel_percentage']:.4f} "
            f"| %{result['baseline_alarm_ge_1_percentage']:.2f} "
            f"| %{result['finetuned_alarm_ge_1_percentage']:.2f} "
            f"| %{result['improved_image_percentage']:.2f} "
            f"| %{result['worsened_image_percentage']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Sonuç",
            "",
            "- Fine-tuning açık deniz `nw` örneklerinde "
            "yanlış alarmları güçlü biçimde azaltmıştır.",
            "- Kıyı içeren `nc` örnekleri temel hata kaynağı "
            "olarak kalmıştır.",
            "- Bir sonraki deney kara-deniz ayrımı veya kıyı "
            "bölgesi baskılama yöntemlerini incelemelidir.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 72)
    print("FINE-TUNING IYILESTIRME ANALIZI")
    print("=" * 72)

    for group_name in (
        "overall",
        "nc",
        "nw",
    ):
        result = summary[
            "groups"
        ][group_name]

        print()
        print(group_name)
        print(
            f"  Görüntü:            "
            f"{result['image_count']}"
        )
        print(
            f"  Baseline FP pixel:   "
            f"%{result['baseline_mean_fp_pixel_percentage']:.4f}"
        )
        print(
            f"  Fine-tuned FP pixel: "
            f"%{result['finetuned_mean_fp_pixel_percentage']:.4f}"
        )
        print(
            f"  İyileşen görüntü:    "
            f"%{result['improved_image_percentage']:.2f}"
        )
        print(
            f"  Kötüleşen görüntü:   "
            f"%{result['worsened_image_percentage']:.2f}"
        )

    print()
    print("Rapor:", REPORT_PATH.resolve())
    print("Çıktılar:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
