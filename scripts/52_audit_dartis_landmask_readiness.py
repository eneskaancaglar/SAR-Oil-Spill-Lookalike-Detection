from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DARTIS_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_landmask_readiness"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_semantic_manifest.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v06_dartis_landmask_readiness.md"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}

GROUP_MAPPING = {
    "oc": {
        "oil_label": 1,
        "class_name": "oil",
        "surface_context": "coast",
        "land_mask_required": True,
    },
    "ow": {
        "oil_label": 1,
        "class_name": "oil",
        "surface_context": "water",
        "land_mask_required": False,
    },
    "nc": {
        "oil_label": 0,
        "class_name": "non_oil_marine",
        "surface_context": "coast",
        "land_mask_required": True,
    },
    "nw": {
        "oil_label": 0,
        "class_name": "non_oil_marine",
        "surface_context": "water",
        "land_mask_required": False,
    },
}

REQUIRED_COORDINATE_COLUMNS = {
    "patch_ul_lon",
    "patch_ul_lat",
    "patch_ur_lon",
    "patch_ur_lat",
    "patch_br_lon",
    "patch_br_lat",
    "patch_bl_lon",
    "patch_bl_lat",
}


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")

    except ValueError:
        return str(
            path.resolve()
        )


def read_table(
    path: Path,
) -> pd.DataFrame:
    errors = []

    for encoding in (
        "utf-8",
        "utf-8-sig",
        "latin-1",
    ):
        try:
            return pd.read_csv(
                path,
                sep="\t",
                encoding=encoding,
                low_memory=False,
            )

        except Exception as error:
            errors.append(
                f"{encoding}: "
                f"{type(error).__name__}: {error}"
            )

    raise RuntimeError(
        f"Tablo okunamadı: {path}\n"
        + "\n".join(errors)
    )


def find_images(
    group_directory: Path,
) -> list[Path]:
    if not group_directory.exists():
        return []

    return sorted(
        path
        for path in group_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
    )


def build_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    raw_root = DARTIS_ROOT / "raw"

    if not raw_root.exists():
        raise FileNotFoundError(
            f"DARTIS raw klasörü bulunamadı: "
            f"{raw_root}"
        )

    for group_name, group_info in (
        GROUP_MAPPING.items()
    ):
        group_directory = (
            raw_root
            / group_name
        )

        for image_path in find_images(
            group_directory
        ):
            rows.append(
                {
                    "sample_id": (
                        f"{group_name}:"
                        f"{image_path.stem}"
                    ),
                    "group_name": group_name,
                    "oil_label": int(
                        group_info[
                            "oil_label"
                        ]
                    ),
                    "class_name": group_info[
                        "class_name"
                    ],
                    "surface_context": group_info[
                        "surface_context"
                    ],
                    "land_mask_required": bool(
                        group_info[
                            "land_mask_required"
                        ]
                    ),
                    "image_path": relative(
                        image_path
                    ),
                    "annotation_type": (
                        "oil_object_annotation_expected"
                        if group_info[
                            "oil_label"
                        ] == 1
                        else "image_level_non_oil_only"
                    ),
                    "lookalike_semantics": (
                        "broad_non_oil_marine"
                        if group_info[
                            "oil_label"
                        ] == 0
                        else "not_applicable"
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.6"
    )
    print(
        "DARTIS LAND-MASK HAZIRLIK KONTROLÜ"
    )
    print("=" * 78)

    if not DARTIS_ROOT.exists():
        raise FileNotFoundError(
            f"DARTIS klasörü bulunamadı: "
            f"{DARTIS_ROOT}"
        )

    manifest = build_manifest()

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    table_candidates = sorted(
        {
            *ROOT.rglob(
                "DARTIS_2019.tab"
            ),
            *DARTIS_ROOT.rglob(
                "*.tab"
            ),
        }
    )

    table_results = []

    for table_path in table_candidates:
        try:
            dataframe = read_table(
                table_path
            )

            normalized_columns = {
                str(column).strip()
                for column
                in dataframe.columns
            }

            missing_coordinates = sorted(
                REQUIRED_COORDINATE_COLUMNS
                - normalized_columns
            )

            table_results.append(
                {
                    "path": relative(
                        table_path
                    ),
                    "row_count": int(
                        len(dataframe)
                    ),
                    "column_count": int(
                        len(dataframe.columns)
                    ),
                    "coordinate_columns_complete": (
                        len(
                            missing_coordinates
                        ) == 0
                    ),
                    "missing_coordinate_columns": (
                        missing_coordinates
                    ),
                    "columns": [
                        str(column)
                        for column
                        in dataframe.columns
                    ],
                }
            )

        except Exception as error:
            table_results.append(
                {
                    "path": relative(
                        table_path
                    ),
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                    "coordinate_columns_complete": (
                        False
                    ),
                }
            )

    shapefiles = sorted(
        ROOT.rglob(
            "*.shp"
        )
    )

    xml_files = sorted(
        DARTIS_ROOT.rglob(
            "*.xml"
        )
    )

    group_counts = (
        manifest
        .groupby(
            [
                "group_name",
                "class_name",
                "surface_context",
                "land_mask_required",
            ]
        )
        .size()
        .reset_index(
            name="image_count"
        )
        .sort_values(
            "group_name"
        )
    )

    coordinate_table_ready = any(
        result.get(
            "coordinate_columns_complete",
            False,
        )
        for result in table_results
    )

    summary = {
        "stage": (
            "v06_dartis_landmask_readiness"
        ),
        "dartis_root": relative(
            DARTIS_ROOT
        ),
        "total_images": int(
            len(manifest)
        ),
        "oil_images": int(
            manifest[
                "oil_label"
            ].eq(1).sum()
        ),
        "non_oil_marine_images": int(
            manifest[
                "oil_label"
            ].eq(0).sum()
        ),
        "coastal_images": int(
            manifest[
                "surface_context"
            ].eq(
                "coast"
            ).sum()
        ),
        "open_water_images": int(
            manifest[
                "surface_context"
            ].eq(
                "water"
            ).sum()
        ),
        "land_mask_required_images": int(
            manifest[
                "land_mask_required"
            ].sum()
        ),
        "table_candidates": table_results,
        "coordinate_table_ready": bool(
            coordinate_table_ready
        ),
        "shapefile_count": int(
            len(shapefiles)
        ),
        "shapefiles": [
            relative(path)
            for path in shapefiles
        ],
        "xml_annotation_count": int(
            len(xml_files)
        ),
        "locked_test_used": False,
        "training_performed": False,
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

    report_lines = [
        "# v0.6 DARTIS Land-Mask Hazırlık Kontrolü",
        "",
        "## Doğrulanmış grup anlamları",
        "",
        "- `oc`: oil / coast",
        "- `ow`: oil / water",
        "- `nc`: no-oil / coast",
        "- `nw`: no-oil / water",
        "",
        "## Yerel veri",
        "",
        f"- Toplam görüntü: {len(manifest)}",
        f"- Petrol: "
        f"{int(manifest['oil_label'].eq(1).sum())}",
        f"- Non-oil marine: "
        f"{int(manifest['oil_label'].eq(0).sum())}",
        f"- Kıyılı görüntü: "
        f"{int(manifest['surface_context'].eq('coast').sum())}",
        f"- Açık su görüntüsü: "
        f"{int(manifest['surface_context'].eq('water').sum())}",
        "",
        "## Land-mask hazırlığı",
        "",
        f"- DARTIS koordinat tablosu hazır: "
        f"{coordinate_table_ready}",
        f"- Shapefile sayısı: "
        f"{len(shapefiles)}",
        f"- XML anotasyonu: "
        f"{len(xml_files)}",
        "",
        "Bu aşamada eğitim yapılmamış ve kilitli "
        "test kullanılmamıştır.",
    ]

    REPORT_PATH.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print()
    print("Grup dağılımı:")
    print(
        group_counts.to_string(
            index=False
        )
    )

    print()
    print(
        "Toplam görüntü:",
        len(manifest),
    )

    print(
        "Kıyılı görüntü:",
        int(
            manifest[
                "surface_context"
            ].eq(
                "coast"
            ).sum()
        ),
    )

    print(
        "Açık su görüntüsü:",
        int(
            manifest[
                "surface_context"
            ].eq(
                "water"
            ).sum()
        ),
    )

    print()
    print(
        "DARTIS_2019.tab adayı:",
        len(table_candidates),
    )

    for result in table_results:
        print(
            " -",
            result.get(
                "path"
            ),
            "| koordinatlar:",
            result.get(
                "coordinate_columns_complete",
                False,
            ),
        )

        missing = result.get(
            "missing_coordinate_columns",
            [],
        )

        if missing:
            print(
                "   Eksik:",
                ", ".join(missing),
            )

    print(
        "Shapefile:",
        len(shapefiles),
    )

    for shapefile in shapefiles:
        print(
            " -",
            relative(shapefile),
        )

    print(
        "XML anotasyonu:",
        len(xml_files),
    )

    print()
    print(
        "Koordinat tablosu hazır:",
        coordinate_table_ready,
    )

    print()
    print(
        "Manifest:",
        MANIFEST_PATH.resolve(),
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
        "Kilitli test kullanımı: YOK"
    )


if __name__ == "__main__":
    main()
