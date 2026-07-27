from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

POSITIVE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_positive_verifier_crops.csv"
)

NEGATIVE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_negative_verifier_crops.csv"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v03_verifier_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v03_verifier_dataset"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "verifier_dataset_summary.json"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v03_verifier_dataset.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pozitif ve negatif DARTIS crop'larını "
            "sahne bazlı verifier veri setinde birleştirir."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--positive-train-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--positive-validation-ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--positive-calibration-ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--negative-validation-ratio",
        type=float,
        default=0.50,
        help=(
            "hard_negative_val sahnelerinin validation'a "
            "ayrılacak oranı. Kalan calibration olur."
        ),
    )

    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=1.50,
    )

    return parser.parse_args()


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
    text = clean_text(value)

    if not text:
        return Path()

    path = Path(text)

    if not path.is_absolute():
        path = ROOT / path

    return path


def stable_sample_id(
    label: int,
    crop_path: str,
) -> str:
    content = (
        f"{label}|{crop_path.lower()}"
    ).encode(
        "utf-8",
        errors="replace",
    )

    digest = hashlib.sha256(
        content
    ).hexdigest()[:20]

    prefix = (
        "oil"
        if label == 1
        else "non_oil"
    )

    return f"{prefix}_{digest}"


def scene_key(
    scene_id: Any,
    source_image_name: Any,
) -> str:
    scene = clean_text(
        scene_id
    )

    if scene:
        return scene

    image_name = clean_text(
        source_image_name
    )

    if not image_name:
        raise RuntimeError(
            "Hem scene_id hem source_image_name eksik."
        )

    return (
        "IMAGE_FALLBACK:"
        + image_name
    )


def load_positive() -> pd.DataFrame:
    if not POSITIVE_MANIFEST.exists():
        raise FileNotFoundError(
            "Pozitif crop manifesti bulunamadı: "
            f"{POSITIVE_MANIFEST}"
        )

    dataframe = pd.read_csv(
        POSITIVE_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = {
        "image_set",
        "scene_id",
        "source_image_name",
        "crop_path",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise RuntimeError(
            "Pozitif manifestte eksik sütunlar: "
            f"{sorted(missing)}"
        )

    rows = []

    for row in dataframe.to_dict(
        orient="records"
    ):
        crop_path = resolve_path(
            row["crop_path"]
        )

        current_scene = scene_key(
            row.get("scene_id"),
            row.get(
                "source_image_name"
            ),
        )

        rows.append(
            {
                "source_dataset": "DARTIS",
                "binary_label": 1,
                "binary_label_name": "oil",
                "source_group": clean_text(
                    row.get("image_set")
                ).lower(),
                "scene_id": current_scene,
                "source_image_name": clean_text(
                    row.get(
                        "source_image_name"
                    )
                ),
                "source_image_path": clean_text(
                    row.get(
                        "source_image_path"
                    )
                ),
                "crop_type": (
                    "positive_bbox_context"
                ),
                "is_hard_negative": False,
                "original_source_split": (
                    "unassigned_positive"
                ),
                "crop_path": str(
                    crop_path
                ),
                "crop_exists": (
                    crop_path.exists()
                    and crop_path.is_file()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def load_negative() -> pd.DataFrame:
    if not NEGATIVE_MANIFEST.exists():
        raise FileNotFoundError(
            "Negatif crop manifesti bulunamadı: "
            f"{NEGATIVE_MANIFEST}"
        )

    dataframe = pd.read_csv(
        NEGATIVE_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = {
        "image_set",
        "scene_id",
        "source_image_name",
        "source_split",
        "negative_type",
        "crop_path",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise RuntimeError(
            "Negatif manifestte eksik sütunlar: "
            f"{sorted(missing)}"
        )

    rows = []

    for row in dataframe.to_dict(
        orient="records"
    ):
        crop_path = resolve_path(
            row["crop_path"]
        )

        negative_type = clean_text(
            row.get(
                "negative_type"
            )
        ).lower()

        current_scene = scene_key(
            row.get("scene_id"),
            row.get(
                "source_image_name"
            ),
        )

        rows.append(
            {
                "source_dataset": "DARTIS",
                "binary_label": 0,
                "binary_label_name": (
                    "non_oil"
                ),
                "source_group": clean_text(
                    row.get("image_set")
                ).lower(),
                "scene_id": current_scene,
                "source_image_name": clean_text(
                    row.get(
                        "source_image_name"
                    )
                ),
                "source_image_path": clean_text(
                    row.get(
                        "source_image_path"
                    )
                ),
                "crop_type": negative_type,
                "is_hard_negative": (
                    negative_type == "hard"
                ),
                "original_source_split": clean_text(
                    row.get(
                        "source_split"
                    )
                ),
                "crop_path": str(
                    crop_path
                ),
                "crop_exists": (
                    crop_path.exists()
                    and crop_path.is_file()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def shuffled(
    values: set[str],
    seed: int,
) -> list[str]:
    output = sorted(
        values
    )

    generator = random.Random(
        seed
    )

    generator.shuffle(
        output
    )

    return output


def allocate_positive_scenes(
    scenes: set[str],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    calibration_ratio: float,
) -> dict[str, str]:
    ratio_sum = (
        train_ratio
        + validation_ratio
        + calibration_ratio
    )

    if ratio_sum >= 1.0:
        raise ValueError(
            "Pozitif train + validation + calibration "
            "oranları toplamı 1'den küçük olmalı. "
            "Kalan bölüm test_locked olacaktır."
        )

    scene_list = shuffled(
        scenes,
        seed,
    )

    total = len(
        scene_list
    )

    train_count = int(
        round(
            total * train_ratio
        )
    )

    validation_count = int(
        round(
            total * validation_ratio
        )
    )

    calibration_count = int(
        round(
            total * calibration_ratio
        )
    )

    used_count = (
        train_count
        + validation_count
        + calibration_count
    )

    if used_count > total:
        overflow = (
            used_count - total
        )

        calibration_count = max(
            0,
            calibration_count
            - overflow,
        )

    train_end = train_count

    validation_end = (
        train_end
        + validation_count
    )

    calibration_end = (
        validation_end
        + calibration_count
    )

    mapping = {}

    for scene in scene_list[
        :train_end
    ]:
        mapping[scene] = "train"

    for scene in scene_list[
        train_end:validation_end
    ]:
        mapping[scene] = (
            "validation"
        )

    for scene in scene_list[
        validation_end:calibration_end
    ]:
        mapping[scene] = (
            "calibration"
        )

    for scene in scene_list[
        calibration_end:
    ]:
        mapping[scene] = (
            "test_locked"
        )

    return mapping


def build_scene_mapping(
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, str]:
    mapping: dict[
        str,
        str,
    ] = {}

    negative_train_scenes = set(
        negative.loc[
            negative[
                "original_source_split"
            ].eq(
                "hard_negative_train"
            ),
            "scene_id",
        ]
    )

    negative_val_pool_scenes = set(
        negative.loc[
            negative[
                "original_source_split"
            ].eq(
                "hard_negative_val"
            ),
            "scene_id",
        ]
    )

    overlap = (
        negative_train_scenes
        & negative_val_pool_scenes
    )

    if overlap:
        examples = sorted(
            overlap
        )[:10]

        raise RuntimeError(
            "Aynı negatif sahne hem train hem "
            "hard_negative_val içinde bulundu: "
            f"{examples}"
        )

    for scene in (
        negative_train_scenes
    ):
        mapping[scene] = "train"

    val_pool_list = shuffled(
        negative_val_pool_scenes,
        args.seed + 1,
    )

    validation_count = int(
        round(
            len(val_pool_list)
            * args.negative_validation_ratio
        )
    )

    if (
        len(val_pool_list) >= 2
    ):
        validation_count = min(
            len(val_pool_list) - 1,
            max(
                1,
                validation_count,
            ),
        )

    for scene in val_pool_list[
        :validation_count
    ]:
        mapping[scene] = (
            "validation"
        )

    for scene in val_pool_list[
        validation_count:
    ]:
        mapping[scene] = (
            "calibration"
        )

    positive_scenes = set(
        positive[
            "scene_id"
        ]
    )

    unassigned_positive_scenes = (
        positive_scenes
        - set(mapping)
    )

    positive_mapping = (
        allocate_positive_scenes(
            scenes=(
                unassigned_positive_scenes
            ),
            seed=args.seed + 2,
            train_ratio=(
                args.positive_train_ratio
            ),
            validation_ratio=(
                args.positive_validation_ratio
            ),
            calibration_ratio=(
                args.positive_calibration_ratio
            ),
        )
    )

    mapping.update(
        positive_mapping
    )

    return mapping


def add_sampling_weights(
    dataframe: pd.DataFrame,
    hard_negative_weight: float,
) -> tuple[
    pd.DataFrame,
    dict[str, float],
]:
    dataframe = dataframe.copy()

    dataframe[
        "sampling_weight"
    ] = 1.0

    train = dataframe[
        dataframe["split"].eq(
            "train"
        )
    ]

    label_counts = (
        train[
            "binary_label"
        ]
        .value_counts()
        .to_dict()
    )

    total_train = int(
        len(train)
    )

    class_weights = {}

    for label in (
        0,
        1,
    ):
        count = int(
            label_counts.get(
                label,
                0,
            )
        )

        if count <= 0:
            raise RuntimeError(
                f"Train split içinde {label} "
                "etiketli örnek bulunamadı."
            )

        class_weights[
            str(label)
        ] = (
            total_train
            / (
                2.0
                * count
            )
        )

    train_mask = dataframe[
        "split"
    ].eq(
        "train"
    )

    negative_mask = (
        train_mask
        & dataframe[
            "binary_label"
        ].eq(0)
    )

    positive_mask = (
        train_mask
        & dataframe[
            "binary_label"
        ].eq(1)
    )

    dataframe.loc[
        negative_mask,
        "sampling_weight",
    ] = class_weights["0"]

    dataframe.loc[
        positive_mask,
        "sampling_weight",
    ] = class_weights["1"]

    hard_mask = (
        train_mask
        & dataframe[
            "is_hard_negative"
        ].eq(True)
    )

    dataframe.loc[
        hard_mask,
        "sampling_weight",
    ] *= hard_negative_weight

    train_weight_mean = float(
        dataframe.loc[
            train_mask,
            "sampling_weight",
        ].mean()
    )

    dataframe.loc[
        train_mask,
        "sampling_weight",
    ] /= train_weight_mean

    return dataframe, class_weights


def validate_no_scene_leakage(
    dataframe: pd.DataFrame,
) -> None:
    scene_split_counts = (
        dataframe[
            [
                "scene_id",
                "split",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "scene_id"
        )["split"]
        .nunique()
    )

    leaked_scenes = (
        scene_split_counts[
            scene_split_counts > 1
        ]
    )

    if not leaked_scenes.empty:
        raise RuntimeError(
            "Sahne leakage tespit edildi. "
            f"Sahne sayısı: {len(leaked_scenes)}"
        )


def nested_counts(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    result = {}

    for split_name in (
        "train",
        "validation",
        "calibration",
        "test_locked",
    ):
        subset = dataframe[
            dataframe["split"].eq(
                split_name
            )
        ]

        result[split_name] = {
            "samples": int(
                len(subset)
            ),
            "scenes": int(
                subset[
                    "scene_id"
                ].nunique()
            ),
            "source_images": int(
                subset[
                    "source_image_name"
                ].nunique()
            ),
            "oil": int(
                subset[
                    "binary_label"
                ].eq(1).sum()
            ),
            "non_oil": int(
                subset[
                    "binary_label"
                ].eq(0).sum()
            ),
            "hard_negative": int(
                subset[
                    "is_hard_negative"
                ].eq(True).sum()
            ),
            "easy_negative": int(
                (
                    subset[
                        "binary_label"
                    ].eq(0)
                    & subset[
                        "is_hard_negative"
                    ].eq(False)
                ).sum()
            ),
        }

    return result


def write_report(
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Robust Binary Oil Detector v0.3 — Verifier Veri Seti",
        "",
        "## Amaç",
        "",
        "DARTIS petrol bounding box crop'ları ile petrolsüz "
        "zor ve kolay negatif crop'ları birleştirilmiştir.",
        "",
        "Verifier modelinin çıktısı:",
        "",
        "- `1`: Petrol",
        "- `0`: Petrol değil",
        "",
        "## Bilimsel veri ayrımı",
        "",
        "- Ayrım kaynak sahne bazında yapılmıştır.",
        "- Aynı sahne birden fazla split içinde bulunmaz.",
        "- DARTIS `external_test` eğitimde kullanılmamıştır.",
        "- `calibration`, güven yüzdesini kalibre etmek için ayrılmıştır.",
        "- `test_locked`, model seçimi bitene kadar kullanılmayacaktır.",
        "",
        "## Genel sonuç",
        "",
        f"- Toplam örnek: {summary['total_samples']}",
        f"- Petrol örneği: {summary['oil_samples']}",
        f"- Petrol değil örneği: {summary['non_oil_samples']}",
        f"- Benzersiz sahne: {summary['unique_scenes']}",
        f"- Eksik crop dosyası: {summary['missing_crop_files']}",
        f"- Scene leakage: {summary['scene_leakage_count']}",
        "",
        "## Split dağılımı",
        "",
        "| Split | Örnek | Sahne | Petrol | Petrol değil | "
        "Hard negatif | Easy negatif |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for split_name in (
        "train",
        "validation",
        "calibration",
        "test_locked",
    ):
        result = summary[
            "splits"
        ][split_name]

        lines.append(
            f"| {split_name} "
            f"| {result['samples']} "
            f"| {result['scenes']} "
            f"| {result['oil']} "
            f"| {result['non_oil']} "
            f"| {result['hard_negative']} "
            f"| {result['easy_negative']} |"
        )

    lines.extend(
        [
            "",
            "## Train sınıf ağırlıkları",
            "",
            f"- Petrol değil: "
            f"{summary['train_class_weights']['0']:.6f}",
            f"- Petrol: "
            f"{summary['train_class_weights']['1']:.6f}",
            f"- Hard-negative çarpanı: "
            f"{summary['hard_negative_weight']:.2f}",
            "",
            "## Test notu",
            "",
            "`test_locked` bölümünde yalnızca kilitli pozitif "
            "petrol crop'ları bulunabilir. Nihai negatif test, "
            "daha önce saklanan DARTIS `external_test` "
            "görüntülerinden model seçimi tamamlandıktan sonra "
            "oluşturulacaktır.",
            "",
            "## Sonraki aşama",
            "",
            "Verifier sınıflandırma modeli eğitilecek ve "
            "validation sonuçlarına göre en iyi checkpoint seçilecektir.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.3"
    )
    print(
        "VERIFIER VERİ SETİ HAZIRLIĞI"
    )
    print("=" * 78)

    positive = load_positive()
    negative = load_negative()

    print(
        "Pozitif crop:",
        len(positive),
    )

    print(
        "Negatif crop:",
        len(negative),
    )

    dataframe = pd.concat(
        [
            positive,
            negative,
        ],
        ignore_index=True,
    )

    before_duplicates = len(
        dataframe
    )

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=[
                "binary_label",
                "crop_path",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    duplicates_removed = (
        before_duplicates
        - len(dataframe)
    )

    missing_crop_files = int(
        (
            ~dataframe[
                "crop_exists"
            ]
        ).sum()
    )

    if missing_crop_files > 0:
        examples = dataframe.loc[
            ~dataframe[
                "crop_exists"
            ],
            "crop_path",
        ].head(10)

        raise FileNotFoundError(
            f"{missing_crop_files} crop dosyası eksik.\n"
            + "\n".join(
                examples.tolist()
            )
        )

    scene_mapping = (
        build_scene_mapping(
            positive=positive,
            negative=negative,
            args=args,
        )
    )

    dataframe[
        "split"
    ] = dataframe[
        "scene_id"
    ].map(
        scene_mapping
    )

    if dataframe[
        "split"
    ].isna().any():
        raise RuntimeError(
            "Bazı sahnelere split atanamadı."
        )

    validate_no_scene_leakage(
        dataframe
    )

    dataframe, class_weights = (
        add_sampling_weights(
            dataframe,
            args.hard_negative_weight,
        )
    )

    dataframe[
        "sample_id"
    ] = dataframe.apply(
        lambda row: stable_sample_id(
            int(
                row[
                    "binary_label"
                ]
            ),
            str(
                row[
                    "crop_path"
                ]
            ),
        ),
        axis=1,
    )

    output_columns = [
        "sample_id",
        "binary_label",
        "binary_label_name",
        "split",
        "scene_id",
        "source_dataset",
        "source_group",
        "source_image_name",
        "source_image_path",
        "crop_type",
        "is_hard_negative",
        "original_source_split",
        "sampling_weight",
        "crop_path",
    ]

    dataframe = dataframe[
        output_columns
    ].sort_values(
        [
            "split",
            "binary_label",
            "scene_id",
            "sample_id",
        ]
    ).reset_index(drop=True)

    dataframe.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    split_counts = nested_counts(
        dataframe
    )

    scene_split_check = (
        dataframe[
            [
                "scene_id",
                "split",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "scene_id"
        )["split"]
        .nunique()
    )

    scene_leakage_count = int(
        (
            scene_split_check > 1
        ).sum()
    )

    summary = {
        "seed": int(
            args.seed
        ),
        "total_samples": int(
            len(dataframe)
        ),
        "oil_samples": int(
            dataframe[
                "binary_label"
            ].eq(1).sum()
        ),
        "non_oil_samples": int(
            dataframe[
                "binary_label"
            ].eq(0).sum()
        ),
        "unique_scenes": int(
            dataframe[
                "scene_id"
            ].nunique()
        ),
        "duplicates_removed": int(
            duplicates_removed
        ),
        "missing_crop_files": int(
            missing_crop_files
        ),
        "scene_leakage_count": int(
            scene_leakage_count
        ),
        "external_test_used": False,
        "hard_negative_weight": float(
            args.hard_negative_weight
        ),
        "train_class_weights": {
            key: float(value)
            for key, value
            in class_weights.items()
        },
        "splits": split_counts,
        "source_groups": {
            str(key): int(value)
            for key, value in (
                dataframe[
                    "source_group"
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        },
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
        summary
    )

    print()
    print("=" * 78)
    print("VERİ SETİ SONUCU")
    print("=" * 78)

    print(
        "Toplam örnek:",
        summary[
            "total_samples"
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

    print(
        "Benzersiz sahne:",
        summary[
            "unique_scenes"
        ],
    )

    print(
        "Scene leakage:",
        summary[
            "scene_leakage_count"
        ],
    )

    print()

    for split_name in (
        "train",
        "validation",
        "calibration",
        "test_locked",
    ):
        result = summary[
            "splits"
        ][split_name]

        print(
            f"{split_name:12s}: "
            f"{result['samples']:5d} örnek "
            f"| petrol={result['oil']:4d} "
            f"| değil={result['non_oil']:4d} "
            f"| sahne={result['scenes']:4d}"
        )

    print()
    print(
        "Manifest:",
        OUTPUT_MANIFEST.resolve(),
    )

    print(
        "Özet:   ",
        SUMMARY_PATH.resolve(),
    )

    print(
        "Rapor:  ",
        REPORT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
