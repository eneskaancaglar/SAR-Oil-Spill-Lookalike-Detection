from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "metadata"
    / "dartis_catalog_raw.csv"
)

DEFAULT_NO_OIL_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "metadata"
    / "dartis_no_oil_manifest.csv"
)

DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dartis_no_oil_scene_split.csv"
)

DEFAULT_OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "dartis_scene_split_summary.json"
)

SPLIT_NAMES = (
    "hard_negative_train",
    "hard_negative_val",
    "external_test",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS no-oil görüntülerini kaynak SAR sahnesi "
            "bazında train, validation ve test gruplarına ayırır."
        )
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_NO_OIL_MANIFEST,
    )

    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_MANIFEST,
    )

    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def clean_text_column(
    series: pd.Series,
) -> pd.Series:
    cleaned = (
        series
        .astype("string")
        .str.strip()
    )

    cleaned = cleaned.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        }
    )

    return cleaned


def validate_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, float]:
    ratios = {
        "hard_negative_train": train_ratio,
        "hard_negative_val": val_ratio,
        "external_test": test_ratio,
    }

    for name, value in ratios.items():
        if value <= 0.0:
            raise ValueError(
                f"{name} oranı sıfırdan büyük olmalı."
            )

    ratio_sum = sum(ratios.values())

    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            "Train, validation ve test oranlarının toplamı 1 olmalı. "
            f"Mevcut toplam: {ratio_sum}"
        )

    return ratios


def load_scene_mapping(
    catalog_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"DARTIS katalog dosyası bulunamadı: {catalog_path}"
        )

    catalog = pd.read_csv(
        catalog_path,
        encoding="utf-8-sig",
    )

    required_columns = {
        "Image set",
        "IMAGE",
        "ID_3",
    }

    missing_columns = required_columns - set(catalog.columns)

    if missing_columns:
        raise RuntimeError(
            "Katalogda gerekli sütunlar eksik: "
            f"{sorted(missing_columns)}"
        )

    catalog = catalog.rename(
        columns={
            "Image set": "image_set",
            "IMAGE": "image_name",
        }
    )

    catalog["image_set"] = (
        catalog["image_set"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    catalog["image_name"] = clean_text_column(
        catalog["image_name"]
    )

    catalog["ID_3"] = clean_text_column(
        catalog["ID_3"]
    )

    if "ID_2" in catalog.columns:
        catalog["ID_2"] = clean_text_column(
            catalog["ID_2"]
        )
    else:
        catalog["ID_2"] = pd.NA

    no_oil = catalog[
        catalog["image_set"].isin({"nc", "nw"})
        & catalog["image_name"].notna()
    ].copy()

    no_oil["scene_source"] = "ID_3"
    no_oil["scene_id"] = no_oil["ID_3"]

    missing_id3 = no_oil["scene_id"].isna()

    no_oil.loc[
        missing_id3,
        "scene_id",
    ] = no_oil.loc[
        missing_id3,
        "ID_2",
    ]

    no_oil.loc[
        missing_id3
        & no_oil["ID_2"].notna(),
        "scene_source",
    ] = "ID_2"

    still_missing = no_oil["scene_id"].isna()

    no_oil.loc[
        still_missing,
        "scene_id",
    ] = (
        "IMAGE_FALLBACK:"
        + no_oil.loc[
            still_missing,
            "image_name",
        ].astype(str)
    )

    no_oil.loc[
        still_missing,
        "scene_source",
    ] = "image_name_fallback"

    image_scene_counts = (
        no_oil
        .groupby(
            ["image_set", "image_name"]
        )["scene_id"]
        .nunique()
    )

    inconsistent = image_scene_counts[
        image_scene_counts > 1
    ]

    if not inconsistent.empty:
        raise RuntimeError(
            "Aynı görüntü birden fazla kaynak sahneyle eşleşiyor. "
            f"İlk örnekler: {inconsistent.head(10).index.tolist()}"
        )

    mapping = (
        no_oil[
            [
                "image_set",
                "image_name",
                "scene_id",
                "scene_source",
                "ID_3",
                "ID_2",
            ]
        ]
        .drop_duplicates(
            subset=["image_set", "image_name"],
            keep="first",
        )
        .copy()
    )

    source_counts = (
        mapping["scene_source"]
        .value_counts()
        .to_dict()
    )

    return mapping, {
        str(key): int(value)
        for key, value in source_counts.items()
    }


def load_and_merge_manifest(
    manifest_path: Path,
    scene_mapping: pd.DataFrame,
) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"DARTIS no-oil manifesti bulunamadı: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
    )

    required_columns = {
        "image_set",
        "image_name",
        "category",
        "width",
        "height",
        "expected_has_oil",
    }

    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise RuntimeError(
            "No-oil manifestinde gerekli sütunlar eksik: "
            f"{sorted(missing_columns)}"
        )

    manifest["image_set"] = (
        manifest["image_set"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    manifest["image_name"] = (
        manifest["image_name"]
        .astype(str)
        .str.strip()
    )

    merged = manifest.merge(
        scene_mapping,
        on=["image_set", "image_name"],
        how="left",
        validate="one_to_one",
    )

    missing_scene_rows = merged[
        merged["scene_id"].isna()
    ]

    if not missing_scene_rows.empty:
        raise RuntimeError(
            f"{len(missing_scene_rows)} görüntünün sahne bilgisi bulunamadı. "
            f"İlk örnek: {missing_scene_rows.iloc[0]['image_name']}"
        )

    if merged["image_name"].duplicated().any():
        duplicated = merged.loc[
            merged["image_name"].duplicated(),
            "image_name",
        ].head(10).tolist()

        raise RuntimeError(
            "Birden fazla kez bulunan görüntü isimleri var: "
            f"{duplicated}"
        )

    return merged


def create_scene_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    working = dataframe.copy()

    working["is_nc"] = (
        working["image_set"] == "nc"
    ).astype(int)

    working["is_nw"] = (
        working["image_set"] == "nw"
    ).astype(int)

    scene_statistics = (
        working
        .groupby(
            "scene_id",
            as_index=False,
        )
        .agg(
            total_images=(
                "image_name",
                "size",
            ),
            nc_images=(
                "is_nc",
                "sum",
            ),
            nw_images=(
                "is_nw",
                "sum",
            ),
        )
    )

    scene_statistics["contains_nc"] = (
        scene_statistics["nc_images"] > 0
    ).astype(int)

    scene_statistics["contains_nw"] = (
        scene_statistics["nw_images"] > 0
    ).astype(int)

    return scene_statistics


def assign_scenes(
    scene_statistics: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    totals = {
        "total_images": int(
            scene_statistics["total_images"].sum()
        ),
        "nc_images": int(
            scene_statistics["nc_images"].sum()
        ),
        "nw_images": int(
            scene_statistics["nw_images"].sum()
        ),
        "scene_count": int(
            len(scene_statistics)
        ),
    }

    targets: dict[str, dict[str, float]] = {}

    for split_name, ratio in ratios.items():
        targets[split_name] = {
            key: value * ratio
            for key, value in totals.items()
        }

    current: dict[str, dict[str, int]] = {
        split_name: {
            "total_images": 0,
            "nc_images": 0,
            "nw_images": 0,
            "scene_count": 0,
        }
        for split_name in SPLIT_NAMES
    }

    records = scene_statistics.to_dict(
        orient="records"
    )

    random_generator = random.Random(seed)
    random_generator.shuffle(records)

    records.sort(
        key=lambda item: (
            int(item["total_images"]),
            max(
                int(item["nc_images"]),
                int(item["nw_images"]),
            ),
        ),
        reverse=True,
    )

    field_weights = {
        "total_images": 3.0,
        "nc_images": 2.0,
        "nw_images": 1.5,
        "scene_count": 0.5,
    }

    assignments: dict[str, str] = {}

    for record in records:
        best_split = None
        best_score = None

        for candidate_split in SPLIT_NAMES:
            hypothetical = {
                split_name: values.copy()
                for split_name, values in current.items()
            }

            hypothetical[candidate_split][
                "total_images"
            ] += int(record["total_images"])

            hypothetical[candidate_split][
                "nc_images"
            ] += int(record["nc_images"])

            hypothetical[candidate_split][
                "nw_images"
            ] += int(record["nw_images"])

            hypothetical[candidate_split][
                "scene_count"
            ] += 1

            score = 0.0

            for split_name in SPLIT_NAMES:
                for field_name, weight in (
                    field_weights.items()
                ):
                    target = max(
                        targets[split_name][field_name],
                        1.0,
                    )

                    difference = (
                        hypothetical[split_name][field_name]
                        - target
                    ) / target

                    score += (
                        weight
                        * difference
                        * difference
                    )

            fill_ratio = (
                hypothetical[candidate_split][
                    "total_images"
                ]
                / max(
                    targets[candidate_split][
                        "total_images"
                    ],
                    1.0,
                )
            )

            comparison_value = (
                score,
                fill_ratio,
                SPLIT_NAMES.index(
                    candidate_split
                ),
            )

            if (
                best_score is None
                or comparison_value < best_score
            ):
                best_score = comparison_value
                best_split = candidate_split

        if best_split is None:
            raise RuntimeError(
                "Kaynak sahne için split seçilemedi."
            )

        scene_id = str(record["scene_id"])
        assignments[scene_id] = best_split

        current[best_split][
            "total_images"
        ] += int(record["total_images"])

        current[best_split][
            "nc_images"
        ] += int(record["nc_images"])

        current[best_split][
            "nw_images"
        ] += int(record["nw_images"])

        current[best_split][
            "scene_count"
        ] += 1

    return assignments


def check_scene_leakage(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    scene_sets = {
        split_name: set(
            dataframe.loc[
                dataframe["split"] == split_name,
                "scene_id",
            ].astype(str)
        )
        for split_name in SPLIT_NAMES
    }

    pairwise_overlaps: dict[str, int] = {}

    for first_index, first_split in enumerate(
        SPLIT_NAMES
    ):
        for second_split in SPLIT_NAMES[
            first_index + 1 :
        ]:
            overlap = (
                scene_sets[first_split]
                & scene_sets[second_split]
            )

            key = (
                f"{first_split}__{second_split}"
            )

            pairwise_overlaps[key] = len(overlap)

    total_overlap = sum(
        pairwise_overlaps.values()
    )

    if total_overlap != 0:
        raise RuntimeError(
            "Kaynak sahne sızıntısı tespit edildi: "
            f"{pairwise_overlaps}"
        )

    return {
        "pairwise_scene_overlaps": (
            pairwise_overlaps
        ),
        "total_scene_overlap_count": (
            total_overlap
        ),
    }


def build_summary(
    dataframe: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
    source_counts: dict[str, int],
    leakage_result: dict[str, Any],
) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}

    total_images = len(dataframe)
    total_scenes = dataframe["scene_id"].nunique()

    for split_name in SPLIT_NAMES:
        subset = dataframe[
            dataframe["split"] == split_name
        ]

        split_summary[split_name] = {
            "image_count": int(len(subset)),
            "image_percentage": (
                len(subset)
                / total_images
                * 100.0
            ),
            "scene_count": int(
                subset["scene_id"].nunique()
            ),
            "scene_percentage": (
                subset["scene_id"].nunique()
                / total_scenes
                * 100.0
            ),
            "nc_image_count": int(
                (subset["image_set"] == "nc").sum()
            ),
            "nw_image_count": int(
                (subset["image_set"] == "nw").sum()
            ),
        }

    scene_categories = (
        dataframe[
            ["scene_id", "image_set"]
        ]
        .drop_duplicates()
        .groupby("scene_id")["image_set"]
        .nunique()
    )

    return {
        "seed": seed,
        "requested_ratios": ratios,
        "total_image_count": int(total_images),
        "total_scene_count": int(total_scenes),
        "mean_images_per_scene": float(
            total_images / total_scenes
        ),
        "mixed_nc_nw_scene_count": int(
            (scene_categories > 1).sum()
        ),
        "scene_source_counts": source_counts,
        "splits": split_summary,
        "leakage_check": leakage_result,
    }


def main() -> None:
    args = parse_args()

    ratios = validate_ratios(
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    print("=" * 75)
    print("DARTIS SAHNE BAZLI SPLIT")
    print("=" * 75)

    scene_mapping, source_counts = (
        load_scene_mapping(args.catalog)
    )

    dataframe = load_and_merge_manifest(
        args.manifest,
        scene_mapping,
    )

    scene_statistics = (
        create_scene_statistics(dataframe)
    )

    print(
        f"Toplam görüntü:       {len(dataframe)}"
    )
    print(
        "Benzersiz kaynak sahne:",
        scene_statistics["scene_id"].nunique(),
    )
    print(
        "Ortalama görüntü/sahne:",
        f"{len(dataframe) / len(scene_statistics):.2f}",
    )

    assignments = assign_scenes(
        scene_statistics=scene_statistics,
        ratios=ratios,
        seed=args.seed,
    )

    dataframe["split"] = (
        dataframe["scene_id"]
        .astype(str)
        .map(assignments)
    )

    if dataframe["split"].isna().any():
        raise RuntimeError(
            "Bazı görüntülere split atanamadı."
        )

    leakage_result = check_scene_leakage(
        dataframe
    )

    output_columns = [
        "source_dataset",
        "catalog_index",
        "image_set",
        "category",
        "image_name",
        "width",
        "height",
        "expected_has_oil",
        "scene_id",
        "scene_source",
        "ID_3",
        "ID_2",
        "split",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in dataframe.columns
    ]

    dataframe = dataframe[
        output_columns
    ].sort_values(
        by=[
            "split",
            "image_set",
            "scene_id",
            "image_name",
        ]
    )

    summary = build_summary(
        dataframe=dataframe,
        ratios=ratios,
        seed=args.seed,
        source_counts=source_counts,
        leakage_result=leakage_result,
    )

    args.output_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        args.output_manifest,
        index=False,
        encoding="utf-8-sig",
    )

    with args.output_summary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 75)
    print("SPLIT SONUÇLARI")
    print("=" * 75)

    for split_name in SPLIT_NAMES:
        result = summary["splits"][
            split_name
        ]

        print()
        print(split_name)
        print(
            f"  Görüntü: "
            f"{result['image_count']} "
            f"(%{result['image_percentage']:.2f})"
        )
        print(
            f"  Sahne:   "
            f"{result['scene_count']} "
            f"(%{result['scene_percentage']:.2f})"
        )
        print(
            f"  nc:      "
            f"{result['nc_image_count']}"
        )
        print(
            f"  nw:      "
            f"{result['nw_image_count']}"
        )

    print()
    print(
        "Sahne overlap:",
        summary[
            "leakage_check"
        ]["total_scene_overlap_count"],
    )

    print(
        "Karışık nc/nw sahnesi:",
        summary[
            "mixed_nc_nw_scene_count"
        ],
    )

    print()
    print(
        "Manifest:",
        args.output_manifest.resolve(),
    )
    print(
        "Özet:   ",
        args.output_summary.resolve(),
    )


if __name__ == "__main__":
    main()
