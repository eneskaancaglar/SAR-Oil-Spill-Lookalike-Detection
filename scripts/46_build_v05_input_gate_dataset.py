from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

V04_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v04_input_gate_dataset.csv"
)

UC_HARD_NEGATIVE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v05_hard_negative_development.csv"
)

EUROSAT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v05_eurosat_inventory.csv"
)

OPENSARURBAN_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v05_opensarurban_inventory.csv"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v05_input_gate_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v05_input_gate_dataset"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "dataset_summary.json"
)

SOURCE_COUNTS_PATH = (
    OUTPUT_DIR
    / "source_split_counts.csv"
)

GROUP_ASSIGNMENT_PATH = (
    OUTPUT_DIR
    / "opensarurban_group_assignments.csv"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v05_input_gate_dataset.md"
)


def stable_hash(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest bulunamadı: {path}"
        )

    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if dataframe.empty:
        raise RuntimeError(
            f"Manifest boş: {path}"
        )

    return dataframe


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "<na>",
    }:
        return ""

    return text


def resolve_path(value: Any) -> Path:
    path = Path(
        clean_text(value)
    )

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def make_record_id(
    source_dataset: str,
    source_sample_id: str,
) -> str:
    return hashlib.sha256(
        (
            source_dataset
            + "|"
            + source_sample_id
        ).encode("utf-8")
    ).hexdigest()[:24]


def deterministic_order(
    dataframe: pd.DataFrame,
    key_column: str = "sample_id",
) -> pd.DataFrame:
    ordered = dataframe.copy()

    ordered["_stable_rank"] = ordered[
        key_column
    ].astype(str).map(
        stable_hash
    )

    ordered = (
        ordered
        .sort_values("_stable_rank")
        .drop(columns=["_stable_rank"])
        .reset_index(drop=True)
    )

    return ordered


def create_row(
    *,
    source_dataset: str,
    source_group: str,
    source_sample_id: str,
    split: str,
    binary_label: int,
    image_path: str,
    sample_weight: float,
    difficulty_type: str,
    split_group_id: str,
    scene_id: str,
    development_role: str,
    eligible_for_locked_test: bool,
) -> dict[str, Any]:
    return {
        "sample_id": make_record_id(
            source_dataset,
            source_sample_id,
        ),
        "source_sample_id": source_sample_id,
        "binary_label": int(
            binary_label
        ),
        "binary_label_name": (
            "supported_sea_sar"
            if int(binary_label) == 1
            else "unsupported_input"
        ),
        "split": split,
        "source_dataset": source_dataset,
        "source_group": source_group,
        "scene_id": scene_id,
        "split_group_id": split_group_id,
        "image_path": image_path,
        "sample_weight": float(
            sample_weight
        ),
        "difficulty_type": difficulty_type,
        "input_policy": (
            "accept_for_oil_pipeline"
            if int(binary_label) == 1
            else "reject_before_oil_pipeline"
        ),
        "development_role": development_role,
        "eligible_for_locked_test": bool(
            eligible_for_locked_test
        ),
    }


def collect_dartis_supported() -> list[dict[str, Any]]:
    dataframe = read_manifest(
        V04_MANIFEST
    )

    required = {
        "sample_id",
        "binary_label",
        "split",
        "source_group",
        "scene_id",
        "split_group_id",
        "image_path",
    }

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise RuntimeError(
            "v0.4 manifest eksik sütunlar: "
            f"{sorted(missing)}"
        )

    supported = dataframe[
        dataframe[
            "binary_label"
        ].astype(int).eq(1)
        & dataframe[
            "split"
        ].isin(
            [
                "train",
                "validation",
                "calibration",
            ]
        )
    ].copy()

    rows = []

    for row in supported.to_dict(
        orient="records"
    ):
        rows.append(
            create_row(
                source_dataset="dartis",
                source_group=clean_text(
                    row["source_group"]
                ),
                source_sample_id=clean_text(
                    row["sample_id"]
                ),
                split=clean_text(
                    row["split"]
                ),
                binary_label=1,
                image_path=clean_text(
                    row["image_path"]
                ),
                sample_weight=1.0,
                difficulty_type=(
                    "supported_sea_coastal_sar"
                ),
                split_group_id=clean_text(
                    row["split_group_id"]
                ),
                scene_id=clean_text(
                    row["scene_id"]
                ),
                development_role=(
                    "supported_input"
                ),
                eligible_for_locked_test=False,
            )
        )

    return rows


def collect_uc_merced_hard_negatives() -> list[
    dict[str, Any]
]:
    dataframe = read_manifest(
        UC_HARD_NEGATIVE_MANIFEST
    )

    required = {
        "sample_id",
        "source_group",
        "image_path",
        "decision",
    }

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise RuntimeError(
            "UC hard-negative manifest eksik: "
            f"{sorted(missing)}"
        )

    rows = []

    for row in dataframe.to_dict(
        orient="records"
    ):
        decision = clean_text(
            row["decision"]
        )

        default_weight = {
            "ACCEPT": 6.0,
            "UNCERTAIN": 4.0,
            "REJECT": 1.5,
        }.get(
            decision,
            1.0,
        )

        weight_value = row.get(
            "hard_negative_weight",
            default_weight,
        )

        try:
            weight = float(
                weight_value
            )

        except (TypeError, ValueError):
            weight = default_weight

        source_sample_id = clean_text(
            row["sample_id"]
        )

        source_group = clean_text(
            row["source_group"]
        )

        rows.append(
            create_row(
                source_dataset="uc_merced_consumed",
                source_group=source_group,
                source_sample_id=source_sample_id,
                split="train",
                binary_label=0,
                image_path=clean_text(
                    row["image_path"]
                ),
                sample_weight=weight,
                difficulty_type=(
                    "uc_merced_"
                    + decision.lower()
                    + "_hard_negative"
                ),
                split_group_id=(
                    "uc-consumed:"
                    + source_sample_id
                ),
                scene_id=(
                    source_group
                    + ":"
                    + source_sample_id
                ),
                development_role=(
                    "consumed_v04_test_now_training"
                ),
                eligible_for_locked_test=False,
            )
        )

    return rows


def split_eurosat_class(
    class_dataframe: pd.DataFrame,
    class_name: str,
) -> dict[str, pd.DataFrame]:
    ordered = deterministic_order(
        class_dataframe
    )

    if class_name == "SeaLake":
        return {
            "test_negative_locked": (
                ordered.head(1200)
            )
        }

    if class_name == "River":
        counts = {
            "train": 1800,
            "validation": 300,
            "calibration": 300,
        }

    else:
        counts = {
            "train": 1000,
            "validation": 250,
            "calibration": 250,
        }

    result = {}
    cursor = 0

    for split_name in (
        "train",
        "validation",
        "calibration",
    ):
        count = counts[
            split_name
        ]

        result[
            split_name
        ] = ordered.iloc[
            cursor:cursor + count
        ].copy()

        cursor += count

    return result


def collect_eurosat() -> list[dict[str, Any]]:
    dataframe = read_manifest(
        EUROSAT_MANIFEST
    )

    required = {
        "sample_id",
        "source_group",
        "image_path",
    }

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise RuntimeError(
            "EuroSAT manifest eksik: "
            f"{sorted(missing)}"
        )

    rows = []

    for class_name, subset in dataframe.groupby(
        "source_group",
        dropna=False,
    ):
        class_name = clean_text(
            class_name
        )

        split_frames = split_eurosat_class(
            subset,
            class_name,
        )

        for split_name, split_dataframe in (
            split_frames.items()
        ):
            for row in split_dataframe.to_dict(
                orient="records"
            ):
                source_sample_id = clean_text(
                    row["sample_id"]
                )

                try:
                    weight = float(
                        row.get(
                            "hard_negative_weight",
                            1.0,
                        )
                    )

                except (TypeError, ValueError):
                    weight = 1.0

                rows.append(
                    create_row(
                        source_dataset="eurosat_rgb",
                        source_group=class_name,
                        source_sample_id=(
                            source_sample_id
                        ),
                        split=split_name,
                        binary_label=0,
                        image_path=clean_text(
                            row["image_path"]
                        ),
                        sample_weight=weight,
                        difficulty_type=clean_text(
                            row.get(
                                "difficulty_type",
                                "optical_negative",
                            )
                        ),
                        split_group_id=(
                            "eurosat:"
                            + source_sample_id
                        ),
                        scene_id=(
                            class_name
                            + ":"
                            + source_sample_id
                        ),
                        development_role=(
                            "new_negative_locked"
                            if split_name
                            == "test_negative_locked"
                            else "v05_development"
                        ),
                        eligible_for_locked_test=(
                            split_name
                            == "test_negative_locked"
                        ),
                    )
                )

    return rows


def assign_opensarurban_groups(
    groups: list[str],
) -> dict[str, str]:
    ordered_groups = sorted(
        groups,
        key=lambda value: stable_hash(
            "opensarurban-group|"
            + value
        ),
    )

    if len(ordered_groups) < 8:
        raise RuntimeError(
            "OpenSARUrban için en az 8 grup "
            "gerekiyor. Bulunan: "
            f"{len(ordered_groups)}"
        )

    assignments = {}

    for index, group_name in enumerate(
        ordered_groups
    ):
        if index < 3:
            split = "test_negative_locked"

        elif index < 5:
            split = "calibration"

        elif index < 7:
            split = "validation"

        else:
            split = "train"

        assignments[
            group_name
        ] = split

    return assignments


def collect_opensarurban() -> tuple[
    list[dict[str, Any]],
    pd.DataFrame,
]:
    dataframe = read_manifest(
        OPENSARURBAN_MANIFEST
    )

    required = {
        "sample_id",
        "source_group",
        "image_path",
    }

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise RuntimeError(
            "OpenSARUrban manifest eksik: "
            f"{sorted(missing)}"
        )

    groups = sorted(
        dataframe[
            "source_group"
        ].astype(str).unique()
    )

    assignments = (
        assign_opensarurban_groups(
            groups
        )
    )

    assignment_rows = []

    for group_name in groups:
        group_count = int(
            dataframe[
                "source_group"
            ].astype(str).eq(
                group_name
            ).sum()
        )

        assignment_rows.append(
            {
                "source_group": group_name,
                "split": assignments[
                    group_name
                ],
                "sample_count": group_count,
            }
        )

    rows = []

    for group_name, subset in dataframe.groupby(
        "source_group",
        dropna=False,
    ):
        group_name = clean_text(
            group_name
        )

        split_name = assignments[
            group_name
        ]

        ordered = deterministic_order(
            subset
        ).head(500)

        for row in ordered.to_dict(
            orient="records"
        ):
            source_sample_id = clean_text(
                row["sample_id"]
            )

            rows.append(
                create_row(
                    source_dataset="opensarurban",
                    source_group=group_name,
                    source_sample_id=(
                        source_sample_id
                    ),
                    split=split_name,
                    binary_label=0,
                    image_path=clean_text(
                        row["image_path"]
                    ),
                    sample_weight=4.0,
                    difficulty_type=(
                        "non_sea_real_sar"
                    ),
                    split_group_id=(
                        "opensarurban-group:"
                        + group_name
                    ),
                    scene_id=(
                        group_name
                        + ":"
                        + source_sample_id
                    ),
                    development_role=(
                        "new_negative_locked"
                        if split_name
                        == "test_negative_locked"
                        else "v05_development"
                    ),
                    eligible_for_locked_test=(
                        split_name
                        == "test_negative_locked"
                    ),
                )
            )

    return (
        rows,
        pd.DataFrame(
            assignment_rows
        ),
    )


def validate_paths(
    dataframe: pd.DataFrame,
) -> None:
    missing = []

    for value in dataframe[
        "image_path"
    ]:
        path = resolve_path(
            value
        )

        if not path.exists():
            missing.append(
                str(path)
            )

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} görüntü bulunamadı.\n"
            + "\n".join(
                missing[:20]
            )
        )


def validate_split_leakage(
    dataframe: pd.DataFrame,
) -> None:
    leakage = (
        dataframe
        .groupby(
            "split_group_id"
        )["split"]
        .nunique()
    )

    leaking_groups = leakage[
        leakage > 1
    ]

    if not leaking_groups.empty:
        raise RuntimeError(
            "Split leakage tespit edildi:\n"
            + leaking_groups.head(20)
            .to_string()
        )


def create_source_counts(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return (
        dataframe
        .groupby(
            [
                "split",
                "binary_label_name",
                "source_dataset",
                "source_group",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="sample_count"
        )
        .sort_values(
            [
                "split",
                "binary_label_name",
                "source_dataset",
                "source_group",
            ]
        )
    )


def write_report(
    dataframe: pd.DataFrame,
    summary: dict[str, Any],
    assignments: pd.DataFrame,
) -> None:
    lines = [
        "# v0.5 Input-Gate Veri Seti",
        "",
        "## Amaç",
        "",
        "Yalnız DARTIS benzeri deniz/kıyı SAR "
        "görüntülerinin petrol pipeline'ına "
        "gönderilmesini sağlamak.",
        "",
        "## Veri kaynakları",
        "",
        "- DARTIS: desteklenen deniz/kıyı SAR",
        "- UC Merced: tüketilmiş optik hard-negative",
        "- EuroSAT RGB: optik kara, nehir ve su",
        "- OpenSARUrban: gerçek fakat deniz olmayan SAR",
        "",
        "## Split dağılımı",
        "",
        "| Split | Desteklenen | Desteklenmeyen | Toplam |",
        "|---|---:|---:|---:|",
    ]

    split_order = [
        "train",
        "validation",
        "calibration",
        "test_negative_locked",
    ]

    for split_name in split_order:
        subset = dataframe[
            dataframe[
                "split"
            ].eq(
                split_name
            )
        ]

        supported = int(
            subset[
                "binary_label"
            ].eq(1).sum()
        )

        unsupported = int(
            subset[
                "binary_label"
            ].eq(0).sum()
        )

        lines.append(
            f"| {split_name} "
            f"| {supported} "
            f"| {unsupported} "
            f"| {len(subset)} |"
        )

    locked_groups = assignments[
        assignments[
            "split"
        ].eq(
            "test_negative_locked"
        )
    ]["source_group"].astype(
        str
    ).tolist()

    lines.extend(
        [
            "",
            "## Yeni negatif kilitli test",
            "",
            "- EuroSAT `SeaLake` sınıfı",
            "- OpenSARUrban'dan tamamen ayrılmış gruplar:",
            "",
        ]
    )

    for group_name in locked_groups:
        lines.append(
            f"  - `{group_name}`"
        )

    lines.extend(
        [
            "",
            "Bu örnekler model eğitiminde, validation'da "
            "ve calibration'da kullanılmayacaktır.",
            "",
            "## Önemli sınırlama",
            "",
            "v0.5 için yeni bağımsız desteklenen deniz-SAR "
            "test kaynağı henüz bulunmamaktadır. Bu nedenle "
            "kilitli test yalnız negatif güvenlik performansını "
            "ölçecektir.",
            "",
            f"- Toplam örnek: "
            f"{summary['total_samples']}",
            f"- Split leakage: "
            f"{summary['split_leakage_groups']}",
            f"- Eksik görüntü: "
            f"{summary['missing_image_count']}",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
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

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.5"
    )
    print(
        "BİRLEŞİK INPUT-GATE VERİ SETİ"
    )
    print("=" * 78)

    rows = []

    dartis_rows = (
        collect_dartis_supported()
    )

    uc_rows = (
        collect_uc_merced_hard_negatives()
    )

    eurosat_rows = (
        collect_eurosat()
    )

    (
        opensar_rows,
        group_assignments,
    ) = collect_opensarurban()

    rows.extend(
        dartis_rows
    )

    rows.extend(
        uc_rows
    )

    rows.extend(
        eurosat_rows
    )

    rows.extend(
        opensar_rows
    )

    dataframe = pd.DataFrame(
        rows
    )

    duplicated_ids = dataframe[
        "sample_id"
    ].duplicated(
        keep=False
    )

    if duplicated_ids.any():
        raise RuntimeError(
            "Tekrarlı sample_id bulundu:\n"
            + dataframe[
                duplicated_ids
            ][
                [
                    "sample_id",
                    "source_dataset",
                    "image_path",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    validate_split_leakage(
        dataframe
    )

    validate_paths(
        dataframe
    )

    dataframe = (
        dataframe
        .sort_values(
            [
                "split",
                "binary_label",
                "source_dataset",
                "source_group",
                "sample_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    dataframe.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    group_assignments.to_csv(
        GROUP_ASSIGNMENT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    source_counts = create_source_counts(
        dataframe
    )

    source_counts.to_csv(
        SOURCE_COUNTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    split_summary = {}

    for split_name, subset in dataframe.groupby(
        "split"
    ):
        split_summary[
            str(split_name)
        ] = {
            "total": int(
                len(subset)
            ),
            "supported": int(
                subset[
                    "binary_label"
                ].eq(1).sum()
            ),
            "unsupported": int(
                subset[
                    "binary_label"
                ].eq(0).sum()
            ),
        }

    summary = {
        "dataset_version": (
            "v05_input_gate"
        ),
        "total_samples": int(
            len(dataframe)
        ),
        "supported_samples": int(
            dataframe[
                "binary_label"
            ].eq(1).sum()
        ),
        "unsupported_samples": int(
            dataframe[
                "binary_label"
            ].eq(0).sum()
        ),
        "split_summary": (
            split_summary
        ),
        "source_counts": {
            "dartis_supported": int(
                len(dartis_rows)
            ),
            "uc_merced_consumed": int(
                len(uc_rows)
            ),
            "eurosat_rgb": int(
                len(eurosat_rows)
            ),
            "opensarurban": int(
                len(opensar_rows)
            ),
        },
        "opensarurban_group_assignments": (
            group_assignments
            .to_dict(
                orient="records"
            )
        ),
        "negative_locked_sources": [
            "eurosat_rgb:SeaLake",
            "opensarurban:held_out_groups",
        ],
        "positive_locked_test_available": False,
        "split_leakage_groups": 0,
        "missing_image_count": 0,
        "v04_dartis_test_locked_excluded": True,
        "v04_uc_merced_test_consumed_as_training": True,
        "manifest_path": str(
            OUTPUT_MANIFEST.resolve()
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
        dataframe,
        summary,
        group_assignments,
    )

    print()
    print("=" * 78)
    print(
        "v0.5 VERİ SETİ SONUCU"
    )
    print("=" * 78)

    print(
        "DARTIS supported:",
        len(dartis_rows),
    )

    print(
        "UC Merced hard-negative:",
        len(uc_rows),
    )

    print(
        "EuroSAT:",
        len(eurosat_rows),
    )

    print(
        "OpenSARUrban:",
        len(opensar_rows),
    )

    print()

    for split_name in (
        "train",
        "validation",
        "calibration",
        "test_negative_locked",
    ):
        subset = dataframe[
            dataframe[
                "split"
            ].eq(
                split_name
            )
        ]

        print(
            f"{split_name:20s}: "
            f"{len(subset):5d} "
            f"| supported="
            f"{int(subset['binary_label'].eq(1).sum()):5d} "
            f"| unsupported="
            f"{int(subset['binary_label'].eq(0).sum()):5d}"
        )

    print()
    print(
        "Split leakage: 0"
    )

    print(
        "Eksik görüntü: 0"
    )

    print()
    print(
        "Manifest:",
        OUTPUT_MANIFEST.resolve(),
    )

    print(
        "Kaynak dağılımı:",
        SOURCE_COUNTS_PATH.resolve(),
    )

    print(
        "OpenSAR grup splitleri:",
        GROUP_ASSIGNMENT_PATH.resolve(),
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
