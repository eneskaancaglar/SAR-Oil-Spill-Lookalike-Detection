from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LOCKED_PREDICTIONS = (
    ROOT
    / "outputs"
    / "v04_input_gate_locked_test"
    / "locked_predictions.csv"
)

LOCKED_SUMMARY = (
    ROOT
    / "outputs"
    / "v04_input_gate_locked_test"
    / "locked_summary.json"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v05_hard_negative_development.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v05_hard_negative_postmortem"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

GROUP_SUMMARY_PATH = (
    OUTPUT_DIR
    / "group_summary.csv"
)

REVIEW_DIR = (
    OUTPUT_DIR
    / "review_images"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v05_hard_negative_postmortem.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.4 kilitli testinde kullanılan desteklenmeyen "
            "görüntüleri v0.5 geliştirme hard-negative "
            "kataloğuna dönüştürür."
        )
    )

    parser.add_argument(
        "--false-accept-weight",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--uncertain-weight",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--correct-reject-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--copy-review-images",
        action="store_true",
    )

    return parser.parse_args()


def resolve_path(value: Any) -> Path:
    text = str(value).strip()

    path = Path(text)

    if not path.is_absolute():
        path = ROOT / path

    return path


def safe_filename(value: str) -> str:
    value = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        value,
    )

    return value.strip(
        "._"
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


def assign_weight(
    decision: str,
    args: argparse.Namespace,
) -> float:
    if decision == "ACCEPT":
        return float(
            args.false_accept_weight
        )

    if decision == "UNCERTAIN":
        return float(
            args.uncertain_weight
        )

    return float(
        args.correct_reject_weight
    )


def assign_priority(
    decision: str,
) -> int:
    if decision == "ACCEPT":
        return 3

    if decision == "UNCERTAIN":
        return 2

    return 1


def assign_difficulty(
    decision: str,
) -> str:
    if decision == "ACCEPT":
        return "false_accept_hard_negative"

    if decision == "UNCERTAIN":
        return "uncertain_hard_negative"

    return "correctly_rejected_negative"


def copy_review_images(
    dataframe: pd.DataFrame,
) -> None:
    if REVIEW_DIR.exists():
        shutil.rmtree(
            REVIEW_DIR
        )

    REVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_dataframe = dataframe[
        dataframe[
            "decision"
        ].isin(
            [
                "ACCEPT",
                "UNCERTAIN",
            ]
        )
    ].copy()

    for row in review_dataframe.to_dict(
        orient="records"
    ):
        source = resolve_path(
            row["image_path"]
        )

        if not source.exists():
            continue

        category = (
            "false_accept"
            if row["decision"] == "ACCEPT"
            else "uncertain"
        )

        destination_directory = (
            REVIEW_DIR
            / category
            / str(
                row["source_group"]
            )
        )

        destination_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_name = safe_filename(
            f"{row['source_group']}"
            f"__p_{row['supported_probability']:.6f}"
            f"__s_{row['prototype_similarity']:.6f}"
            f"__{source.name}"
        )

        shutil.copy2(
            source,
            destination_directory
            / destination_name,
        )


def write_report(
    summary: dict[str, Any],
    group_summary: pd.DataFrame,
) -> None:
    lines = [
        "# v0.5 Input-Gate Hard-Negative Postmortem",
        "",
        "## v0.4 sonucu",
        "",
        f"- Desteklenmeyen kilitli görüntü: "
        f"{summary['unsupported_samples']}",
        f"- Yanlış kabul: "
        f"{summary['false_accept_count']}",
        f"- Belirsiz: "
        f"{summary['uncertain_count']}",
        f"- Doğru red: "
        f"{summary['correct_reject_count']}",
        f"- Güvenli engelleme oranı: "
        f"{summary['safe_block_rate']:.4f}",
        "",
        "v0.4 güvenlik testi başarısız olduğu için model "
        "uygulamaya eklenmemiştir.",
        "",
        "## v0.5 kullanımı",
        "",
        "Bu görüntüler artık bağımsız test verisi değildir. "
        "Tüketilmiş tanı seti olarak v0.5 model geliştirmesinde "
        "hard-negative örnekler şeklinde kullanılacaktır.",
        "",
        "| Karar | Eğitim ağırlığı | Öncelik |",
        "|---|---:|---:|",
        f"| ACCEPT | "
        f"{summary['weights']['false_accept']} | 3 |",
        f"| UNCERTAIN | "
        f"{summary['weights']['uncertain']} | 2 |",
        f"| REJECT | "
        f"{summary['weights']['correct_reject']} | 1 |",
        "",
        "## Grup dağılımı",
        "",
        "| Grup | Örnek | Yanlış kabul | Belirsiz | Doğru red |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in group_summary.to_dict(
        orient="records"
    ):
        lines.append(
            f"| {row['source_group']} "
            f"| {row['sample_count']} "
            f"| {row['false_accept_count']} "
            f"| {row['uncertain_count']} "
            f"| {row['correct_reject_count']} |"
        )

    lines.extend(
        [
            "",
            "## Bilimsel kural",
            "",
            "v0.5 değerlendirmesinde bu görüntüler tekrar "
            "kilitli test olarak raporlanmayacaktır.",
            "",
            "v0.5 için farklı kaynaklardan yeni ve dokunulmamış "
            "bir external-test seti oluşturulmalıdır.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if not LOCKED_PREDICTIONS.exists():
        raise FileNotFoundError(
            "v0.4 kilitli tahmin dosyası bulunamadı: "
            f"{LOCKED_PREDICTIONS}"
        )

    locked_summary = read_json(
        LOCKED_SUMMARY
    )

    if not locked_summary.get(
        "test_locked_used",
        False,
    ):
        raise RuntimeError(
            "v0.4 test_locked kullanılmış olarak "
            "işaretlenmemiş."
        )

    OUTPUT_MANIFEST.parent.mkdir(
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
        LOCKED_PREDICTIONS,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = {
        "sample_id",
        "binary_label",
        "source_dataset",
        "source_group",
        "image_path",
        "supported_probability",
        "prototype_similarity",
        "decision",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Eksik sütunlar: "
            f"{sorted(missing_columns)}"
        )

    unsupported = dataframe[
        dataframe[
            "binary_label"
        ].astype(int).eq(0)
    ].copy()

    if unsupported.empty:
        raise RuntimeError(
            "Desteklenmeyen kilitli görüntü bulunamadı."
        )

    valid_decisions = {
        "ACCEPT",
        "UNCERTAIN",
        "REJECT",
    }

    unexpected_decisions = (
        set(
            unsupported[
                "decision"
            ].astype(str).unique()
        )
        - valid_decisions
    )

    if unexpected_decisions:
        raise RuntimeError(
            "Beklenmeyen kararlar: "
            f"{sorted(unexpected_decisions)}"
        )

    missing_files = []

    for value in unsupported[
        "image_path"
    ]:
        path = resolve_path(
            value
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

    unsupported[
        "hard_negative_weight"
    ] = unsupported[
        "decision"
    ].map(
        lambda decision:
        assign_weight(
            str(decision),
            args,
        )
    )

    unsupported[
        "hardness_priority"
    ] = unsupported[
        "decision"
    ].map(
        lambda decision:
        assign_priority(
            str(decision)
        )
    )

    unsupported[
        "difficulty_type"
    ] = unsupported[
        "decision"
    ].map(
        lambda decision:
        assign_difficulty(
            str(decision)
        )
    )

    unsupported[
        "v05_usage"
    ] = (
        "development_hard_negative"
    )

    unsupported[
        "original_evaluation_role"
    ] = (
        "v04_test_locked_consumed"
    )

    unsupported[
        "eligible_for_v05_locked_test"
    ] = False

    unsupported[
        "target_binary_label"
    ] = 0

    unsupported[
        "target_binary_label_name"
    ] = "unsupported_input"

    unsupported = (
        unsupported
        .sort_values(
            [
                "hardness_priority",
                "supported_probability",
                "prototype_similarity",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    unsupported.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    group_rows = []

    for group_name, subset in unsupported.groupby(
        "source_group",
        dropna=False,
    ):
        group_rows.append(
            {
                "source_group": str(
                    group_name
                ),
                "sample_count": int(
                    len(subset)
                ),
                "false_accept_count": int(
                    subset[
                        "decision"
                    ].eq(
                        "ACCEPT"
                    ).sum()
                ),
                "uncertain_count": int(
                    subset[
                        "decision"
                    ].eq(
                        "UNCERTAIN"
                    ).sum()
                ),
                "correct_reject_count": int(
                    subset[
                        "decision"
                    ].eq(
                        "REJECT"
                    ).sum()
                ),
                "maximum_supported_probability": float(
                    subset[
                        "supported_probability"
                    ].max()
                ),
                "maximum_prototype_similarity": float(
                    subset[
                        "prototype_similarity"
                    ].max()
                ),
            }
        )

    group_summary = pd.DataFrame(
        group_rows
    ).sort_values(
        [
            "false_accept_count",
            "uncertain_count",
            "maximum_supported_probability",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    group_summary.to_csv(
        GROUP_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    false_accepts = unsupported[
        unsupported[
            "decision"
        ].eq(
            "ACCEPT"
        )
    ]

    uncertain = unsupported[
        unsupported[
            "decision"
        ].eq(
            "UNCERTAIN"
        )
    ]

    correct_rejects = unsupported[
        unsupported[
            "decision"
        ].eq(
            "REJECT"
        )
    ]

    summary = {
        "source_model": (
            "input_gate_v04"
        ),
        "v04_safety_test_passed": False,
        "unsupported_samples": int(
            len(unsupported)
        ),
        "false_accept_count": int(
            len(false_accepts)
        ),
        "uncertain_count": int(
            len(uncertain)
        ),
        "correct_reject_count": int(
            len(correct_rejects)
        ),
        "safe_block_rate": float(
            (
                len(uncertain)
                + len(correct_rejects)
            )
            / len(unsupported)
        ),
        "weights": {
            "false_accept": float(
                args.false_accept_weight
            ),
            "uncertain": float(
                args.uncertain_weight
            ),
            "correct_reject": float(
                args.correct_reject_weight
            ),
        },
        "v04_locked_data_consumed": True,
        "eligible_for_v05_locked_test": False,
        "manifest_path": str(
            OUTPUT_MANIFEST
        ),
        "group_summary_path": str(
            GROUP_SUMMARY_PATH
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

    if args.copy_review_images:
        copy_review_images(
            unsupported
        )

    write_report(
        summary,
        group_summary,
    )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.5"
    )
    print(
        "HARD-NEGATIVE POSTMORTEM"
    )
    print("=" * 78)

    print(
        "Tüketilmiş negatif örnek:",
        len(unsupported),
    )

    print(
        "Yanlış kabul:",
        len(false_accepts),
    )

    print(
        "Belirsiz:",
        len(uncertain),
    )

    print(
        "Doğru red:",
        len(correct_rejects),
    )

    print(
        "Güvenli engelleme:",
        f"{summary['safe_block_rate']:.4f}",
    )

    print()
    print(
        "YANLIŞ KABUL EDİLEN GRUPLAR"
    )

    if false_accepts.empty:
        print(
            "Yanlış kabul bulunamadı."
        )

    else:
        columns = [
            "source_group",
            "supported_probability",
            "prototype_similarity",
            "nearest_prototype",
            "image_path",
        ]

        print(
            false_accepts[
                columns
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "Manifest:",
        OUTPUT_MANIFEST.resolve(),
    )

    print(
        "Grup özeti:",
        GROUP_SUMMARY_PATH.resolve(),
    )

    if args.copy_review_images:
        print(
            "İnceleme görüntüleri:",
            REVIEW_DIR.resolve(),
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
        "Bu görüntüler v0.5 testinde "
        "tekrar kullanılmayacak."
    )


if __name__ == "__main__":
    main()
