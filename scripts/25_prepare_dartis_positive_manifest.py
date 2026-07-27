from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "metadata"
    / "dartis_catalog_raw.csv"
)

RAW_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

ANNOTATION_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "annotations"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_positive_oil_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v03_dartis_positive_inventory"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "dartis_positive_summary.json"
)

OUTPUT_REPORT = (
    ROOT
    / "reports"
    / "v03_dartis_positive_inventory.md"
)

POSITIVE_GROUPS = {
    "ow": "Petrollü açık deniz",
    "oc": "Petrollü kıyı",
}


def read_catalog() -> pd.DataFrame:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"DARTIS katalog dosyası bulunamadı: {CATALOG_PATH}"
        )

    return pd.read_csv(
        CATALOG_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    column_map = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in column_map:
            return column_map[key]

    if required:
        raise RuntimeError(
            "Gerekli sütun bulunamadı. "
            f"Aranan sütunlar: {candidates}. "
            f"Mevcut sütunlar: {list(dataframe.columns)}"
        )

    return None


def clean_value(value: Any) -> str:
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


def extract_filename(reference: str) -> str:
    reference = clean_value(reference)

    if not reference:
        return ""

    parsed = urlparse(reference)
    path_text = parsed.path if parsed.path else reference

    filename = Path(path_text).name

    if filename:
        return filename

    return reference.replace("/", "_").replace("\\", "_")


def reference_type(reference: str) -> str:
    reference = clean_value(reference)

    if not reference:
        return "missing"

    suffix = Path(
        urlparse(reference).path
    ).suffix.lower()

    if suffix == ".xml":
        return "xml"

    if suffix in {
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
    }:
        return "pixel_mask_or_image"

    if suffix in {
        ".jpg",
        ".jpeg",
    }:
        return "jpeg_image"

    if suffix:
        return f"other:{suffix}"

    return "no_suffix"


def find_existing_file(
    root: Path,
    group_name: str,
    filename: str,
) -> Path | None:
    if not filename:
        return None

    group_directory = root / group_name

    direct_path = group_directory / filename

    if direct_path.exists():
        return direct_path

    if not group_directory.exists():
        return None

    stem = Path(filename).stem

    for suffix in (
        ".jpg",
        ".jpeg",
        ".png",
        ".xml",
        ".tif",
        ".tiff",
    ):
        candidate = group_directory / f"{stem}{suffix}"

        if candidate.exists():
            return candidate

    return None


def unique_join(values: pd.Series) -> str:
    unique_values = []

    for value in values:
        text = clean_value(value)

        if text and text not in unique_values:
            unique_values.append(text)

    return json.dumps(
        unique_values,
        ensure_ascii=False,
    )


def first_non_empty(values: pd.Series) -> str:
    for value in values:
        text = clean_value(value)

        if text:
            return text

    return ""


def build_manifest(
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    group_column = find_column(
        catalog,
        ["Image set", "image_set"],
    )

    image_column = find_column(
        catalog,
        ["IMAGE", "image", "image_name"],
    )

    annotation_column = find_column(
        catalog,
        [
            "Binary",
            "binary",
            "mask",
            "annotation",
        ],
        required=False,
    )

    scene_column = find_column(
        catalog,
        ["ID_3", "scene_id"],
        required=False,
    )

    fallback_scene_column = find_column(
        catalog,
        ["ID_2"],
        required=False,
    )

    working = catalog.copy()

    working["group_name"] = (
        working[group_column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    working = working[
        working["group_name"].isin(
            POSITIVE_GROUPS.keys()
        )
    ].copy()

    working["image_reference_value"] = (
        working[image_column]
        .map(clean_value)
    )

    working["image_name_value"] = (
        working["image_reference_value"]
        .map(extract_filename)
    )

    if annotation_column is not None:
        working["annotation_reference_value"] = (
            working[annotation_column]
            .map(clean_value)
        )
    else:
        working["annotation_reference_value"] = ""

    if scene_column is not None:
        working["scene_id_primary"] = (
            working[scene_column]
            .map(clean_value)
        )
    else:
        working["scene_id_primary"] = ""

    if fallback_scene_column is not None:
        working["scene_id_fallback"] = (
            working[fallback_scene_column]
            .map(clean_value)
        )
    else:
        working["scene_id_fallback"] = ""

    working = working[
        working["image_name_value"] != ""
    ].copy()

    working["scene_id_value"] = (
        working["scene_id_primary"]
    )

    missing_scene = (
        working["scene_id_value"] == ""
    )

    working.loc[
        missing_scene,
        "scene_id_value",
    ] = working.loc[
        missing_scene,
        "scene_id_fallback",
    ]

    still_missing_scene = (
        working["scene_id_value"] == ""
    )

    working.loc[
        still_missing_scene,
        "scene_id_value",
    ] = (
        "IMAGE_FALLBACK:"
        + working.loc[
            still_missing_scene,
            "image_name_value",
        ]
    )

    grouped = (
        working
        .groupby(
            [
                "group_name",
                "image_name_value",
            ],
            as_index=False,
        )
        .agg(
            image_reference=(
                "image_reference_value",
                first_non_empty,
            ),
            primary_annotation_reference=(
                "annotation_reference_value",
                first_non_empty,
            ),
            annotation_references_json=(
                "annotation_reference_value",
                unique_join,
            ),
            scene_id=(
                "scene_id_value",
                first_non_empty,
            ),
            catalog_row_count=(
                "image_name_value",
                "size",
            ),
        )
    )

    manifest_rows = []

    for _, row in grouped.iterrows():
        group_name = str(
            row["group_name"]
        )

        image_name = str(
            row["image_name_value"]
        )

        annotation_reference = str(
            row[
                "primary_annotation_reference"
            ]
        )

        annotation_name = extract_filename(
            annotation_reference
        )

        local_image = find_existing_file(
            RAW_ROOT,
            group_name,
            image_name,
        )

        local_annotation = find_existing_file(
            ANNOTATION_ROOT,
            group_name,
            annotation_name,
        )

        try:
            annotation_reference_list = json.loads(
                str(
                    row[
                        "annotation_references_json"
                    ]
                )
            )
        except json.JSONDecodeError:
            annotation_reference_list = []

        manifest_rows.append(
            {
                "source_dataset": "DARTIS",
                "binary_label": 1,
                "binary_label_name": "oil",
                "image_set": group_name,
                "category": POSITIVE_GROUPS[
                    group_name
                ],
                "image_name": image_name,
                "image_reference": str(
                    row["image_reference"]
                ),
                "annotation_name": (
                    annotation_name
                ),
                "primary_annotation_reference": (
                    annotation_reference
                ),
                "annotation_references_json": (
                    json.dumps(
                        annotation_reference_list,
                        ensure_ascii=False,
                    )
                ),
                "annotation_reference_count": (
                    len(
                        annotation_reference_list
                    )
                ),
                "annotation_type": reference_type(
                    annotation_reference
                ),
                "scene_id": str(
                    row["scene_id"]
                ),
                "catalog_row_count": int(
                    row["catalog_row_count"]
                ),
                "local_image_exists": (
                    local_image is not None
                ),
                "local_image_path": (
                    str(local_image)
                    if local_image is not None
                    else ""
                ),
                "local_annotation_exists": (
                    local_annotation is not None
                ),
                "local_annotation_path": (
                    str(local_annotation)
                    if local_annotation is not None
                    else ""
                ),
                "split": "unassigned_v03",
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    ).sort_values(
        [
            "image_set",
            "scene_id",
            "image_name",
        ]
    ).reset_index(drop=True)

    group_summaries = {}

    for group_name in (
        "ow",
        "oc",
    ):
        subset = manifest[
            manifest["image_set"]
            == group_name
        ]

        group_summaries[
            group_name
        ] = {
            "description": (
                POSITIVE_GROUPS[group_name]
            ),
            "image_count": int(
                len(subset)
            ),
            "scene_count": int(
                subset["scene_id"].nunique()
            ),
            "catalog_object_row_count": int(
                subset[
                    "catalog_row_count"
                ].sum()
            ),
            "local_image_count": int(
                subset[
                    "local_image_exists"
                ].sum()
            ),
            "local_annotation_count": int(
                subset[
                    "local_annotation_exists"
                ].sum()
            ),
            "annotation_type_counts": {
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

    summary = {
        "catalog_path": str(
            CATALOG_PATH
        ),
        "catalog_total_rows": int(
            len(catalog)
        ),
        "positive_catalog_rows": int(
            len(working)
        ),
        "positive_unique_images": int(
            len(manifest)
        ),
        "positive_unique_scenes": int(
            manifest["scene_id"].nunique()
        ),
        "local_positive_images": int(
            manifest[
                "local_image_exists"
            ].sum()
        ),
        "local_positive_annotations": int(
            manifest[
                "local_annotation_exists"
            ].sum()
        ),
        "annotation_column": (
            annotation_column
        ),
        "groups": group_summaries,
        "annotation_type_counts": {
            str(key): int(value)
            for key, value in (
                manifest[
                    "annotation_type"
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        },
        "sample_image_references": (
            manifest[
                "image_reference"
            ]
            .dropna()
            .astype(str)
            .loc[
                lambda values:
                values.str.strip() != ""
            ]
            .drop_duplicates()
            .head(5)
            .tolist()
        ),
        "sample_annotation_references": (
            manifest[
                "primary_annotation_reference"
            ]
            .dropna()
            .astype(str)
            .loc[
                lambda values:
                values.str.strip() != ""
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        ),
    }

    return manifest, summary


def write_report(
    summary: dict[str, Any],
) -> None:
    lines = [
        "# DARTIS Pozitif Petrol Verisi Envanteri",
        "",
        "## Amaç",
        "",
        "DARTIS `ow` ve `oc` gruplarındaki petrollü "
        "görüntüleri Robust Binary Oil Detector v0.3 "
        "eğitimi için hazırlamak.",
        "",
        "## Genel sonuç",
        "",
        f"- Pozitif katalog satırı: "
        f"{summary['positive_catalog_rows']}",
        f"- Benzersiz petrollü görüntü: "
        f"{summary['positive_unique_images']}",
        f"- Benzersiz kaynak sahne: "
        f"{summary['positive_unique_scenes']}",
        f"- Yerelde bulunan görüntü: "
        f"{summary['local_positive_images']}",
        f"- Yerelde bulunan anotasyon: "
        f"{summary['local_positive_annotations']}",
        f"- Anotasyon sütunu: "
        f"`{summary['annotation_column']}`",
        "",
        "## Grup sonuçları",
        "",
        "| Grup | Açıklama | Görüntü | Sahne | "
        "Katalog nesne satırı | Yerel görüntü | "
        "Yerel anotasyon |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for group_name in (
        "ow",
        "oc",
    ):
        group = summary[
            "groups"
        ][group_name]

        lines.append(
            f"| {group_name} "
            f"| {group['description']} "
            f"| {group['image_count']} "
            f"| {group['scene_count']} "
            f"| {group['catalog_object_row_count']} "
            f"| {group['local_image_count']} "
            f"| {group['local_annotation_count']} |"
        )

    lines.extend(
        [
            "",
            "## Anotasyon türleri",
            "",
        ]
    )

    for annotation_type, count in (
        summary[
            "annotation_type_counts"
        ].items()
    ):
        lines.append(
            f"- `{annotation_type}`: {count}"
        )

    lines.extend(
        [
            "",
            "## Sonraki aşama",
            "",
            "1. Petrollü `ow` ve `oc` görüntülerini indirmek.",
            "2. Anotasyon dosyalarını indirmek.",
            "3. Anotasyon biçimini incelemek.",
            "4. Anotasyonları 640x640 binary petrol maskesine çevirmek.",
            "5. Görüntü-mask eşleşmelerini doğrulamak.",
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
    print("DARTIS POZİTİF PETROL ENVANTERİ")
    print("=" * 78)

    catalog = read_catalog()

    manifest, summary = build_manifest(
        catalog
    )

    manifest.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    with OUTPUT_SUMMARY.open(
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

    print(
        "Pozitif katalog satırı:",
        summary[
            "positive_catalog_rows"
        ],
    )

    print(
        "Benzersiz petrollü görüntü:",
        summary[
            "positive_unique_images"
        ],
    )

    print(
        "Benzersiz kaynak sahne:",
        summary[
            "positive_unique_scenes"
        ],
    )

    print(
        "Yerel pozitif görüntü:",
        summary[
            "local_positive_images"
        ],
    )

    print(
        "Yerel pozitif anotasyon:",
        summary[
            "local_positive_annotations"
        ],
    )

    print()
    print("GRUPLAR")

    for group_name in (
        "ow",
        "oc",
    ):
        result = summary[
            "groups"
        ][group_name]

        print(
            f"  {group_name}: "
            f"{result['image_count']} görüntü "
            f"| {result['scene_count']} sahne "
            f"| {result['catalog_object_row_count']} "
            f"katalog nesne satırı"
        )

    print()
    print(
        "Anotasyon sütunu:",
        summary[
            "annotation_column"
        ],
    )

    print(
        "Anotasyon türleri:",
        summary[
            "annotation_type_counts"
        ],
    )

    print()
    print("Örnek görüntü referansları:")

    for reference in summary[
        "sample_image_references"
    ]:
        print(" ", reference)

    print()
    print("Örnek anotasyon referansları:")

    for reference in summary[
        "sample_annotation_references"
    ]:
        print(" ", reference)

    print()
    print(
        "Manifest:",
        OUTPUT_MANIFEST.resolve(),
    )

    print(
        "JSON:    ",
        OUTPUT_SUMMARY.resolve(),
    )

    print(
        "Rapor:   ",
        OUTPUT_REPORT.resolve(),
    )


if __name__ == "__main__":
    main()
