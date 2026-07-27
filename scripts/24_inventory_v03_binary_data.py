from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOS_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dataset_manifest.csv"
)

DARTIS_CATALOG = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "metadata"
    / "dartis_catalog_raw.csv"
)

DARTIS_RAW = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v03_binary_data_inventory"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v03_binary_data_inventory.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "inventory_summary.json"
)

OUTPUT_REPORT = (
    ROOT
    / "reports"
    / "v03_binary_data_inventory.md"
)


DARTIS_GROUP_INFO = {
    "ow": {
        "binary_label": 1,
        "description": "Petrollü açık deniz",
    },
    "oc": {
        "binary_label": 1,
        "description": "Petrollü kıyı",
    },
    "nw": {
        "binary_label": 0,
        "description": "Petrolsüz açık deniz",
    },
    "nc": {
        "binary_label": 0,
        "description": "Petrolsüz kıyı/kara",
    },
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV bulunamadı: {path}"
        )

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )


def normalize_text(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    normalized_columns = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = (
            candidate.strip().lower()
        )

        if normalized_candidate in normalized_columns:
            return normalized_columns[
                normalized_candidate
            ]

    if required:
        raise RuntimeError(
            "Gerekli sütun bulunamadı. "
            f"Arananlar: {candidates}. "
            f"Mevcut sütunlar: "
            f"{list(dataframe.columns)}"
        )

    return None


def extract_filename(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    parsed = urlparse(text)

    if parsed.scheme:
        path_text = parsed.path
    else:
        path_text = text

    return Path(path_text).name


def detect_reference_type(
    reference: Any,
) -> str:
    if pd.isna(reference):
        return "missing"

    text = str(reference).strip()

    if not text:
        return "missing"

    suffix = Path(
        urlparse(text).path
    ).suffix.lower()

    if suffix == ".xml":
        return "xml_annotation"

    if suffix in {
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
    }:
        return "possible_pixel_mask"

    if suffix in {
        ".jpg",
        ".jpeg",
    }:
        return "image_reference"

    return (
        f"other_reference:{suffix}"
        if suffix
        else "reference_without_suffix"
    )


def find_local_image(
    group_name: str,
    image_name: str,
) -> Path | None:
    if not image_name:
        return None

    direct_path = (
        DARTIS_RAW
        / group_name
        / image_name
    )

    if direct_path.exists():
        return direct_path

    group_directory = (
        DARTIS_RAW
        / group_name
    )

    if not group_directory.exists():
        return None

    stem = Path(image_name).stem

    for suffix in (
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
    ):
        candidate = (
            group_directory
            / f"{stem}{suffix}"
        )

        if candidate.exists():
            return candidate

    return None


def build_sos_inventory() -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    dataframe = read_csv(
        SOS_MANIFEST
    )

    required_columns = {
        "sample_name",
        "split",
        "sensor",
        "image_path",
        "mask_path",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "SOS manifestinde eksik sütunlar: "
            f"{sorted(missing_columns)}"
        )

    if "has_oil" in dataframe.columns:
        has_oil = (
            pd.to_numeric(
                dataframe["has_oil"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    elif (
        "oil_pixel_ratio"
        in dataframe.columns
    ):
        has_oil = (
            pd.to_numeric(
                dataframe[
                    "oil_pixel_ratio"
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .gt(0.0)
            .astype(int)
        )

    else:
        raise RuntimeError(
            "SOS manifestinde has_oil veya "
            "oil_pixel_ratio sütunu bulunamadı."
        )

    inventory = pd.DataFrame(
        {
            "source_dataset": "SOS",
            "sample_name": (
                dataframe["sample_name"]
                .astype(str)
            ),
            "binary_label": has_oil,
            "binary_label_name": (
                has_oil.map(
                    {
                        1: "oil",
                        0: "non_oil",
                    }
                )
            ),
            "source_group": (
                "segmentation_pair"
            ),
            "source_group_description": (
                "SOS görüntü-mask çifti"
            ),
            "split": (
                dataframe["split"]
                .astype(str)
            ),
            "sensor": (
                dataframe["sensor"]
                .astype(str)
            ),
            "scene_id": pd.NA,
            "image_reference": (
                dataframe["image_path"]
                .astype(str)
            ),
            "annotation_reference": (
                dataframe["mask_path"]
                .astype(str)
            ),
            "annotation_type": (
                "pixel_segmentation_mask"
            ),
            "local_image_exists": (
                dataframe["image_path"]
                .map(
                    lambda value:
                    (
                        ROOT
                        / str(value)
                    ).exists()
                    if not Path(
                        str(value)
                    ).is_absolute()
                    else Path(
                        str(value)
                    ).exists()
                )
            ),
            "local_annotation_exists": (
                dataframe["mask_path"]
                .map(
                    lambda value:
                    (
                        ROOT
                        / str(value)
                    ).exists()
                    if not Path(
                        str(value)
                    ).is_absolute()
                    else Path(
                        str(value)
                    ).exists()
                )
            ),
        }
    )

    summary = {
        "total_samples": int(
            len(inventory)
        ),
        "oil_samples": int(
            (
                inventory[
                    "binary_label"
                ]
                == 1
            ).sum()
        ),
        "non_oil_samples": int(
            (
                inventory[
                    "binary_label"
                ]
                == 0
            ).sum()
        ),
        "splits": {
            str(key): int(value)
            for key, value in (
                inventory["split"]
                .value_counts()
                .to_dict()
                .items()
            )
        },
        "sensors": {
            str(key): int(value)
            for key, value in (
                inventory["sensor"]
                .value_counts()
                .to_dict()
                .items()
            )
        },
        "local_images_available": int(
            inventory[
                "local_image_exists"
            ].sum()
        ),
        "local_masks_available": int(
            inventory[
                "local_annotation_exists"
            ].sum()
        ),
    }

    return inventory, summary


def build_dartis_inventory() -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    dataframe = read_csv(
        DARTIS_CATALOG
    )

    group_column = find_column(
        dataframe,
        ["Image set", "image_set"],
    )

    image_column = find_column(
        dataframe,
        ["IMAGE", "image", "image_name"],
    )

    scene_column = find_column(
        dataframe,
        ["ID_3", "scene_id"],
        required=False,
    )

    binary_column = find_column(
        dataframe,
        ["Binary", "binary"],
        required=False,
    )

    dataframe = dataframe.copy()

    dataframe["group_value"] = (
        normalize_text(
            dataframe[group_column]
        )
        .str.lower()
    )

    dataframe["image_reference_value"] = (
        normalize_text(
            dataframe[image_column]
        )
    )

    dataframe["image_name_value"] = (
        dataframe[
            "image_reference_value"
        ].map(extract_filename)
    )

    if scene_column is not None:
        dataframe["scene_id_value"] = (
            normalize_text(
                dataframe[
                    scene_column
                ]
            )
        )
    else:
        dataframe["scene_id_value"] = pd.NA

    if binary_column is not None:
        dataframe[
            "annotation_reference_value"
        ] = normalize_text(
            dataframe[binary_column]
        )
    else:
        dataframe[
            "annotation_reference_value"
        ] = pd.NA

    valid = dataframe[
        dataframe["group_value"].isin(
            DARTIS_GROUP_INFO.keys()
        )
        & dataframe[
            "image_name_value"
        ].ne("")
    ].copy()

    unique_images = (
        valid
        .sort_index()
        .drop_duplicates(
            subset=[
                "group_value",
                "image_name_value",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    local_paths = []

    for row in unique_images.itertuples(
        index=False
    ):
        local_path = find_local_image(
            str(row.group_value),
            str(row.image_name_value),
        )

        local_paths.append(
            str(local_path)
            if local_path is not None
            else ""
        )

    unique_images[
        "local_image_path_value"
    ] = local_paths

    inventory_rows = []

    for row in unique_images.itertuples(
        index=False
    ):
        group_name = str(
            row.group_value
        )

        group_info = (
            DARTIS_GROUP_INFO[
                group_name
            ]
        )

        annotation_reference = (
            row.annotation_reference_value
        )

        local_image_path = str(
            row.local_image_path_value
        )

        inventory_rows.append(
            {
                "source_dataset": (
                    "DARTIS"
                ),
                "sample_name": (
                    str(row.image_name_value)
                ),
                "binary_label": (
                    group_info[
                        "binary_label"
                    ]
                ),
                "binary_label_name": (
                    "oil"
                    if group_info[
                        "binary_label"
                    ] == 1
                    else "non_oil"
                ),
                "source_group": (
                    group_name
                ),
                "source_group_description": (
                    group_info[
                        "description"
                    ]
                ),
                "split": "unassigned_v03",
                "sensor": "Sentinel-1",
                "scene_id": (
                    row.scene_id_value
                ),
                "image_reference": (
                    row.image_reference_value
                ),
                "annotation_reference": (
                    annotation_reference
                ),
                "annotation_type": (
                    detect_reference_type(
                        annotation_reference
                    )
                ),
                "local_image_exists": (
                    bool(
                        local_image_path
                    )
                ),
                "local_annotation_exists": (
                    False
                ),
            }
        )

    inventory = pd.DataFrame(
        inventory_rows
    )

    group_summary: dict[
        str,
        Any,
    ] = {}

    for group_name in (
        "ow",
        "oc",
        "nw",
        "nc",
    ):
        subset = inventory[
            inventory["source_group"]
            == group_name
        ]

        group_summary[group_name] = {
            "description": (
                DARTIS_GROUP_INFO[
                    group_name
                ]["description"]
            ),
            "binary_label": (
                DARTIS_GROUP_INFO[
                    group_name
                ]["binary_label"]
            ),
            "unique_image_count": int(
                len(subset)
            ),
            "unique_scene_count": int(
                subset["scene_id"]
                .dropna()
                .nunique()
            ),
            "local_image_count": int(
                subset[
                    "local_image_exists"
                ].sum()
            ),
            "annotation_types": {
                str(key): int(value)
                for key, value in (
                    subset[
                        "annotation_type"
                    ]
                    .value_counts()
                    .to_dict()
                    .items()
                )
            },
        }

    candidate_columns = [
        str(column)
        for column in dataframe.columns
        if any(
            token in str(
                column
            ).lower()
            for token in (
                "binary",
                "mask",
                "xml",
                "annot",
                "label",
            )
        )
    ]

    candidate_samples: dict[
        str,
        list[str],
    ] = {}

    for column in candidate_columns:
        values = (
            dataframe[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = values[
            values.ne("")
        ]

        candidate_samples[column] = (
            values
            .drop_duplicates()
            .head(5)
            .tolist()
        )

    summary = {
        "catalog_row_count": int(
            len(dataframe)
        ),
        "unique_binary_images": int(
            len(inventory)
        ),
        "oil_images": int(
            (
                inventory[
                    "binary_label"
                ]
                == 1
            ).sum()
        ),
        "non_oil_images": int(
            (
                inventory[
                    "binary_label"
                ]
                == 0
            ).sum()
        ),
        "local_images_available": int(
            inventory[
                "local_image_exists"
            ].sum()
        ),
        "groups": group_summary,
        "annotation_candidate_columns": (
            candidate_columns
        ),
        "annotation_reference_samples": (
            candidate_samples
        ),
    }

    return inventory, summary


def write_report(
    summary: dict[str, Any],
) -> None:
    sos = summary["SOS"]
    dartis = summary["DARTIS"]

    lines = [
        "# Robust Binary Oil Detector v0.3 — Veri Envanteri",
        "",
        "## Hedef etiket şeması",
        "",
        "- `1`: Petrol",
        "- `0`: Petrol değil",
        "",
        "Kara, kıyı, rüzgâr izi, alg, biyolojik film, "
        "sığ su, gemi izi ve diğer tüm yapılar "
        "`petrol değil` sınıfındadır.",
        "",
        "## SOS",
        "",
        f"- Toplam örnek: {sos['total_samples']}",
        f"- Petrol örneği: {sos['oil_samples']}",
        f"- Petrolsüz örnek: {sos['non_oil_samples']}",
        f"- Yerel görüntü: {sos['local_images_available']}",
        f"- Yerel maske: {sos['local_masks_available']}",
        "",
        "### SOS split dağılımı",
        "",
    ]

    for split_name, count in (
        sos["splits"].items()
    ):
        lines.append(
            f"- {split_name}: {count}"
        )

    lines.extend(
        [
            "",
            "### SOS sensör dağılımı",
            "",
        ]
    )

    for sensor_name, count in (
        sos["sensors"].items()
    ):
        lines.append(
            f"- {sensor_name}: {count}"
        )

    lines.extend(
        [
            "",
            "## DARTIS",
            "",
            f"- Katalog satırı: "
            f"{dartis['catalog_row_count']}",
            f"- Benzersiz görüntü: "
            f"{dartis['unique_binary_images']}",
            f"- Petrol görüntüsü: "
            f"{dartis['oil_images']}",
            f"- Petrolsüz görüntü: "
            f"{dartis['non_oil_images']}",
            f"- Yerelde bulunan görüntü: "
            f"{dartis['local_images_available']}",
            "",
            "### DARTIS grupları",
            "",
            "| Grup | Açıklama | Etiket | Görüntü | Sahne | Yerelde |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    for group_name in (
        "ow",
        "oc",
        "nw",
        "nc",
    ):
        result = dartis[
            "groups"
        ][group_name]

        lines.append(
            f"| {group_name} "
            f"| {result['description']} "
            f"| {result['binary_label']} "
            f"| {result['unique_image_count']} "
            f"| {result['unique_scene_count']} "
            f"| {result['local_image_count']} |"
        )

    lines.extend(
        [
            "",
            "## Anotasyonla ilişkili katalog sütunları",
            "",
        ]
    )

    candidate_columns = dartis[
        "annotation_candidate_columns"
    ]

    if candidate_columns:
        for column in candidate_columns:
            lines.append(
                f"- `{column}`"
            )
    else:
        lines.append(
            "- Anotasyon adını açıkça taşıyan sütun bulunamadı."
        )

    lines.extend(
        [
            "",
            "## Sonraki işlem",
            "",
            "1. DARTIS `ow` ve `oc` petrollü görüntülerinin "
            "yerel durumunu doğrulamak.",
            "2. Pozitif anotasyonların XML, binary görüntü veya "
            "başka formatta olup olmadığını kesinleştirmek.",
            "3. Pozitif DARTIS görüntü ve anotasyonlarını indirmek.",
            "4. Bütün pozitif anotasyonları piksel maskesine çevirmek.",
            "5. Yeni sahne bazlı train/validation/calibration/test "
            "ayrımı oluşturmak.",
            "",
        ]
    )

    OUTPUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("ROBUST BINARY OIL DETECTOR v0.3")
    print("VERİ ENVANTERİ")
    print("=" * 78)

    sos_inventory, sos_summary = (
        build_sos_inventory()
    )

    dartis_inventory, dartis_summary = (
        build_dartis_inventory()
    )

    combined_inventory = pd.concat(
        [
            sos_inventory,
            dartis_inventory,
        ],
        ignore_index=True,
    )

    combined_inventory.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "label_map": {
            "0": "non_oil",
            "1": "oil",
        },
        "SOS": sos_summary,
        "DARTIS": dartis_summary,
        "combined": {
            "total_samples": int(
                len(
                    combined_inventory
                )
            ),
            "oil_samples": int(
                (
                    combined_inventory[
                        "binary_label"
                    ]
                    == 1
                ).sum()
            ),
            "non_oil_samples": int(
                (
                    combined_inventory[
                        "binary_label"
                    ]
                    == 0
                ).sum()
            ),
        },
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_report(summary)

    print()
    print("SOS")
    print(
        f"  Toplam:      "
        f"{sos_summary['total_samples']}"
    )
    print(
        f"  Petrol:      "
        f"{sos_summary['oil_samples']}"
    )
    print(
        f"  Petrol değil:"
        f" {sos_summary['non_oil_samples']}"
    )

    print()
    print("DARTIS")

    for group_name in (
        "ow",
        "oc",
        "nw",
        "nc",
    ):
        result = dartis_summary[
            "groups"
        ][group_name]

        print(
            f"  {group_name}: "
            f"{result['unique_image_count']} görüntü "
            f"| {result['unique_scene_count']} sahne "
            f"| yerelde "
            f"{result['local_image_count']}"
        )

    print()
    print(
        "DARTIS petrol görüntüsü:",
        dartis_summary[
            "oil_images"
        ],
    )

    print(
        "DARTIS petrolsüz görüntü:",
        dartis_summary[
            "non_oil_images"
        ],
    )

    print()
    print(
        "Anotasyonla ilişkili sütunlar:",
        dartis_summary[
            "annotation_candidate_columns"
        ],
    )

    print()
    print(
        "Manifest:",
        OUTPUT_MANIFEST.resolve(),
    )

    print(
        "JSON:    ",
        OUTPUT_JSON.resolve(),
    )

    print(
        "Rapor:   ",
        OUTPUT_REPORT.resolve(),
    )


if __name__ == "__main__":
    main()
